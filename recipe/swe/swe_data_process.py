#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Build and sanitize Verl-compatible SWE datasets.

The CLI accepts either a local dataset/parquet path or a Hugging Face dataset
identifier. Generated Parquet may record the operator-provided image archive
root because node-aware preloading consumes that path; public source packages
must never include the generated Parquet itself.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer

if __package__:
    from .env_server.swe_utils import get_swe_eval_format_image_name, render_template, resolve_swe_repo_dir
    from .image_download import archive_directory, archive_name, resolve_image
    from .instance_routing import get_assigned_group, get_repo_routing_map
else:  # Support `python recipe/swe/swe_data_process.py ...`.
    from env_server.swe_utils import get_swe_eval_format_image_name, render_template, resolve_swe_repo_dir
    from image_download import archive_directory, archive_name, resolve_image
    from instance_routing import get_assigned_group, get_repo_routing_map


TRANSFER_FIELDS = (
    "instance_id",
    "repo",
    "base_commit",
    "patch",
    "test_patch",
    "problem_statement",
    "hints_text",
    "created_at",
    "version",
    "environment_setup_commit",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "meta",
    "install_config",
    "requirements",
    "environment",
    "license_name",
    "docker_image",
    "difficulty",
    "interface",
    "repo_language",
    "issue_specificity",
    "issue_categories",
    "before_repo_set_cmd",
    "selected_test_files_to_run",
    "dockerhub_tag",
)
REQUIRED_FIELDS = ("instance_id", "repo", "base_commit", "test_patch", "problem_statement")
LIST_FIELDS = ("FAIL_TO_PASS", "PASS_TO_PASS")
FIELD_DEFAULTS: dict[str, Any] = {
    "patch": "",
    "hints_text": "",
    "created_at": "",
    "version": "",
    "environment_setup_commit": "",
    "FAIL_TO_PASS": [],
    "PASS_TO_PASS": [],
    "meta": {},
    "install_config": {},
    "requirements": "",
    "environment": "",
    "license_name": "",
    "docker_image": "",
    "difficulty": "",
    "interface": "",
    "repo_language": "",
    "issue_specificity": "",
    "issue_categories": "",
    "before_repo_set_cmd": "",
    "selected_test_files_to_run": "",
    "dockerhub_tag": "",
}
FIELD_ALIASES = {
    "FAIL_TO_PASS": ("FAIL_TO_PASS", "fail_to_pass"),
    "PASS_TO_PASS": ("PASS_TO_PASS", "pass_to_pass"),
    "repo_language": ("repo_language", "language"),
    "license_name": ("license_name", "license"),
}
JSON_FIELDS = {"meta", "install_config"}
HOST_PATH_KEYS = {
    "local_tar_path",
    "local_image_tar_path",
    "host_image_path",
    "pro_run_script_path",
    "pro_parser_path",
}


def _load_source(source: str, split: str, revision: str | None = None):
    path = Path(source).expanduser()
    if path.is_file():
        suffix_to_builder = {
            ".parquet": "parquet",
            ".json": "json",
            ".jsonl": "json",
            ".csv": "csv",
        }
        try:
            builder = suffix_to_builder[path.suffix.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported local dataset file: {path}") from exc
        return load_dataset(builder, data_files={split: str(path)}, split=split)

    if path.is_dir() and ((path / "state.json").exists() or (path / "dataset_dict.json").exists()):
        dataset = load_from_disk(str(path))
        if isinstance(dataset, DatasetDict):
            if split not in dataset:
                raise ValueError(f"Split {split!r} is not present in {path}")
            return dataset[split]
        return dataset

    kwargs = {"split": split}
    if revision:
        kwargs["revision"] = revision
    return load_dataset(source, **kwargs)


def _parse_test_list(value: Any, *, instance_id: str, field: str) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
    if isinstance(value, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(value)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
    raise ValueError(f"{instance_id}: {field} must be a list, got {type(value).__name__}")


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


def _canonical_json_object(value: Any, *, instance_id: str, field: str) -> str:
    if _is_missing(value) or value == "":
        value = {}
    elif isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError(f"{instance_id}: {field} must be a JSON object")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_instance_ids(path: str | None) -> set[str] | None:
    if not path:
        return None
    source = Path(path).expanduser()
    if source.suffix.lower() == ".parquet":
        frame = pd.read_parquet(source, columns=["extra_info"])
        ids: set[str] = set()
        for value in frame["extra_info"]:
            if hasattr(value, "as_py"):
                value = value.as_py()
            if isinstance(value, str):
                value = json.loads(value)
            if not isinstance(value, dict) or not value.get("instance_id"):
                raise ValueError(f"{source}: every extra_info row must contain instance_id")
            ids.add(str(value["instance_id"]))
        return ids

    text = source.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [line.strip() for line in text.splitlines() if line.strip()]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("Instance-id file must contain a JSON string list or one ID per line")
    return set(value)


class PromptLengthComputer:
    def __init__(self, tokenizer_path: str, agent_config_path: str, tokenizer_revision: str | None):
        config = yaml.safe_load(Path(agent_config_path).read_text(encoding="utf-8"))["agent"]
        self.system_template = config["system_template"]
        self.instance_template = config["instance_template"]
        kwargs: dict[str, Any] = {"use_fast": True}
        if tokenizer_revision:
            kwargs["revision"] = tokenizer_revision
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **kwargs)

    def __call__(self, instance: dict[str, Any], data_docker_source: str) -> int:
        problem_statement = build_agent_problem_statement(instance, data_docker_source)
        messages = [
            {"role": "system", "content": render_template(self.system_template)},
            {
                "role": "user",
                "content": render_template(self.instance_template, task=problem_statement),
            },
        ]
        kwargs = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": False,
        }
        try:
            token_ids = self.tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            token_ids = self.tokenizer.apply_chat_template(messages, **kwargs)
        return len(token_ids)


def _copy_instance_fields(instance: dict[str, Any]) -> dict[str, Any]:
    instance_id = instance.get("instance_id", "<missing-instance-id>")
    missing = [
        field
        for field in REQUIRED_FIELDS
        if field not in instance or _is_missing(instance[field])
    ]
    if missing:
        raise ValueError(f"{instance_id}: missing required field(s): {', '.join(missing)}")

    result: dict[str, Any] = {}
    for field in TRANSFER_FIELDS:
        value = None
        for source_field in FIELD_ALIASES.get(field, (field,)):
            if source_field in instance and not _is_missing(instance[source_field]):
                value = copy.deepcopy(instance[source_field])
                break
        if value is None:
            value = copy.deepcopy(FIELD_DEFAULTS[field])
        if field in LIST_FIELDS:
            value = _parse_test_list(value, instance_id=instance_id, field=field)
        elif field in JSON_FIELDS:
            value = _canonical_json_object(value, instance_id=instance_id, field=field)
        result[field] = value
    return result


def _is_swebench_pro(data_docker_source: str) -> bool:
    return data_docker_source.strip().lower().replace("_", "-") == "swe-bench-pro"


def build_agent_problem_statement(instance: dict[str, Any], data_docker_source: str) -> str:
    problem = str(instance["problem_statement"])
    if not _is_swebench_pro(data_docker_source):
        return problem
    return (
        f"{problem}\n\nRequirements:\n{instance.get('requirements', '')}\n\n"
        f"New interfaces introduced:\n{instance.get('interface', '')}"
    )


def build_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    dataset = _load_source(args.source, args.split, args.revision)
    chosen_ids = _read_instance_ids(args.instance_ids)
    excluded_ids: set[str] = set()
    for path in args.exclude_instance_ids or []:
        excluded_ids.update(_read_instance_ids(path) or set())

    excluded_repos: set[str] = set()
    if args.exclude_repos_from:
        excluded = _load_source(
            args.exclude_repos_from,
            args.exclude_repos_split,
            args.exclude_repos_revision,
        )
        excluded_repos = {str(item["repo"]) for item in excluded}

    selected: list[tuple[int, dict[str, Any]]] = []
    for index, raw_instance in enumerate(dataset):
        instance = dict(raw_instance)
        instance_id = instance.get("instance_id")
        if chosen_ids is not None and instance_id not in chosen_ids:
            continue
        if instance_id in excluded_ids:
            continue
        if instance.get("repo") in excluded_repos:
            continue
        selected.append((index, instance))

    if not selected:
        raise ValueError("No instances remain after filtering")

    # Preserve the retained 0815 routing protocol: compute repository routing
    # on the selected source population before filtering absent archives or
    # long prompts, then keep each row's assignment through seeded sampling.
    routing = get_repo_routing_map(
        [str(instance["instance_id"]) for _, instance in selected],
        num_nodes=args.num_groups,
    )

    prompt_length_computer = None
    if args.tokenizer:
        if not args.agent_config:
            raise ValueError("--agent-config is required with --tokenizer")
        prompt_length_computer = PromptLengthComputer(
            args.tokenizer, args.agent_config, args.tokenizer_revision
        )
    elif args.max_prompt_length is not None:
        raise ValueError("--max-prompt-length requires --tokenizer and --agent-config")

    candidates: list[dict[str, Any]] = []
    for source_index, instance in selected:
        extra_info = _copy_instance_fields(instance)
        instance_id = str(extra_info["instance_id"])
        docker_image_name = resolve_image(instance, args.data_docker_source)
        tar_file_name = archive_name(docker_image_name, args.data_docker_source)
        tar_path = archive_directory(args.image_root, args.dataset_name) / tar_file_name
        if not args.keep_missing_images and not tar_path.is_file():
            continue

        prompt_length = (
            prompt_length_computer(instance, args.data_docker_source)
            if prompt_length_computer
            else 0
        )
        if args.max_prompt_length is not None and prompt_length > args.max_prompt_length:
            continue

        is_pro = _is_swebench_pro(args.data_docker_source)
        image_name = (
            docker_image_name
            if is_pro or args.data_docker_source.strip().lower().replace("_", "-") == "swe-rebench-v2"
            else get_swe_eval_format_image_name(instance_id)
        )
        extra_info["data_source"] = args.dataset_name
        extra_info["data_docker_source"] = args.data_docker_source
        extra_info["repo_dir"] = "/app" if is_pro else resolve_swe_repo_dir(extra_info)
        extra_info["evaluator_type"] = "swe_bench_pro" if is_pro else "swe_bench"
        extra_info["agent_problem_statement"] = build_agent_problem_statement(
            instance, args.data_docker_source
        )
        if is_pro:
            if not args.swe_bench_pro_run_scripts_dir:
                raise ValueError("--swe-bench-pro-run-scripts-dir is required for SWE-bench Pro")
            pro_root = Path(args.swe_bench_pro_run_scripts_dir).expanduser().resolve()
            pro_run_script = pro_root / instance_id / "run_script.sh"
            pro_parser = pro_root / instance_id / "parser.py"
            if not pro_run_script.is_file() or not pro_parser.is_file():
                raise FileNotFoundError(
                    f"{instance_id}: missing SWE-bench Pro evaluator under {pro_root}"
                )
            extra_info["pro_run_script_path"] = str(pro_run_script)
            extra_info["pro_parser_path"] = str(pro_parser)
        else:
            extra_info["pro_run_script_path"] = ""
            extra_info["pro_parser_path"] = ""

        candidates.append(
            {
                "source_index": source_index,
                "instance_id": instance_id,
                "image_name": image_name,
                "docker_image_name": docker_image_name,
                "tar_file_name": tar_file_name,
                "tar_path": tar_path,
                "prompt_length": prompt_length,
                "extra_info": extra_info,
            }
        )

    if not candidates:
        raise ValueError("No instances remain after image and prompt-length filtering")

    if args.part_size is not None:
        random.Random(args.seed).shuffle(candidates)
        num_parts = (len(candidates) + args.part_size - 1) // args.part_size
        if args.part_index > num_parts:
            raise ValueError(
                f"--part-index {args.part_index} exceeds the {num_parts} generated part(s)"
            )
        start = (args.part_index - 1) * args.part_size
        end = min(start + args.part_size, len(candidates))
        candidates = candidates[start:end]
    elif args.max_count is not None and len(candidates) > args.max_count:
        candidates = random.Random(args.seed).sample(candidates, args.max_count)

    records = []
    for candidate in candidates:
        extra_info = candidate["extra_info"]
        instance_id = candidate["instance_id"]
        extra_info.update(
            {
                "split": args.split,
                "index": candidate["source_index"],
                "image_name": candidate["image_name"],
                "docker_image_name": candidate["docker_image_name"],
                "tar_file_name": candidate["tar_file_name"],
                "local_tar_path": str(candidate["tar_path"]),
                "group_id": get_assigned_group(
                    instance_id,
                    routing,
                ),
                "prompt_length": candidate["prompt_length"],
            }
        )
        records.append(
            {
                "data_source": args.dataset_name,
                "agent_name": "swe_agent",
                "prompt": [{"role": "user", "content": instance_id}],
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": extra_info,
            }
        )

    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_output(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)


def command_build(args: argparse.Namespace) -> None:
    output = Path(args.output).expanduser()
    _prepare_output(output, force=args.force)
    records = build_records(args)
    pd.DataFrame(records).to_parquet(output, index=False)
    print(json.dumps({"rows": len(records), "output": str(output), "sha256": _sha256(output)}))


def _clean_extra_info(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    cleaned = copy.deepcopy(value)
    for key in HOST_PATH_KEYS:
        cleaned.pop(key, None)
    return cleaned


def _normalize_prompt(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        return value
    normalized = []
    for item in value:
        if isinstance(item, dict):
            item = dict(item)
            item["role"] = "user"
        normalized.append(item)
    return normalized


def command_clean(args: argparse.Namespace) -> None:
    source = Path(args.input).expanduser()
    output = source if args.in_place else Path(args.output).expanduser()
    if not args.in_place and output == source:
        raise ValueError("Use --in-place to overwrite the input parquet")
    _prepare_output(output, force=args.force or args.in_place)

    frame = pd.read_parquet(source)
    if "extra_info" in frame.columns:
        frame["extra_info"] = frame["extra_info"].apply(_clean_extra_info)
    if "prompt" in frame.columns and args.normalize_prompt_roles:
        frame["prompt"] = frame["prompt"].apply(_normalize_prompt)
    frame.to_parquet(output, index=False)
    print(json.dumps({"rows": len(frame), "output": str(output), "sha256": _sha256(output)}))


def build_parser() -> argparse.ArgumentParser:
    canopy_root = os.environ.get("CANOPY_ROOT")
    default_agent_config = (
        str(Path(canopy_root) / "recipe/swe/config/swe_agent_xml_config.yaml")
        if canopy_root
        else None
    )

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build a verl-compatible SWE parquet")
    build.add_argument("--source", required=True, help="Local path or Hugging Face dataset ID")
    build.add_argument("--output", required=True)
    build.add_argument("--split", default="test")
    build.add_argument("--revision", help="Dataset revision/commit")
    build.add_argument("--dataset-name", default="SWE-bench_Verified")
    build.add_argument("--data-docker-source", default="SWE-bench")
    build.add_argument("--num-groups", type=int, default=8)
    build.add_argument("--max-count", type=int)
    build.add_argument("--part-size", type=int, help="Deterministically select one shuffled part")
    build.add_argument("--part-index", type=int, default=1, help="One-based part number")
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--instance-ids", help="JSON list or newline-delimited instance IDs")
    build.add_argument(
        "--exclude-instance-ids",
        action="append",
        help=(
            "Parquet, JSON list, or newline-delimited IDs to remove; repeat for "
            "train/eval decontamination"
        ),
    )
    build.add_argument("--exclude-repos-from", help="Dataset whose repositories should be excluded")
    build.add_argument("--exclude-repos-split", default="test")
    build.add_argument("--exclude-repos-revision")
    build.add_argument("--image-root", required=True, help="Root created by image_download.py")
    build.add_argument(
        "--keep-missing-images",
        action="store_true",
        help="Keep rows whose local image archive is absent (default: filter them)",
    )
    build.add_argument(
        "--swe-bench-pro-run-scripts-dir",
        help="Official per-instance run_script.sh/parser.py root required by SWE-bench Pro",
    )
    build.add_argument("--tokenizer", help="Tokenizer path or Hugging Face model ID")
    build.add_argument("--tokenizer-revision")
    build.add_argument("--agent-config", default=default_agent_config)
    build.add_argument("--max-prompt-length", type=int)
    build.add_argument("--force", action="store_true")
    build.set_defaults(func=command_build)

    clean = subparsers.add_parser("clean", help="Remove host paths from an existing verl parquet")
    clean.add_argument("--input", required=True)
    output_group = clean.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output")
    output_group.add_argument("--in-place", action="store_true")
    clean.add_argument(
        "--normalize-prompt-roles",
        action="store_true",
        help="Explicitly rewrite every prompt message role to user",
    )
    clean.add_argument("--force", action="store_true")
    clean.set_defaults(func=command_clean)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "num_groups", 1) <= 0:
        raise ValueError("--num-groups must be positive")
    if getattr(args, "max_count", None) is not None and args.max_count <= 0:
        raise ValueError("--max-count must be positive")
    if getattr(args, "part_size", None) is not None and args.part_size <= 0:
        raise ValueError("--part-size must be positive")
    if getattr(args, "part_index", 1) <= 0:
        raise ValueError("--part-index must be positive")
    if getattr(args, "part_size", None) is None and getattr(args, "part_index", 1) != 1:
        raise ValueError("--part-index requires --part-size")
    if getattr(args, "part_size", None) is not None and getattr(args, "max_count", None) is not None:
        raise ValueError("--part-size and --max-count are mutually exclusive")
    args.func(args)


if __name__ == "__main__":
    main()
