#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run one real SWE environment evaluation from a processed Parquet row.

This is a manual integration smoke, not a pytest test. It supports the six
0815 datasets used by the SWE training pipeline:

* SWE-bench Verified
* SWE-bench Multilingual
* SWE-rebench V1
* SWE-rebench V2
* SWE-bench Pro
* SWE-rebench Leaderboard

Typical usage from the verl repository root::

    python -m recipe.swe.tests.smoke_swe_env_evaluation \
      --dataset rebench-v2 \
      --image-name docker.io/swerebenchv2/virtuslab-git-machete:179-9749145 \
      --patch gold

``--patch empty`` passes a real empty string. Legacy/V2 already treat it as a
no-op patch; the Pro adapter explicitly skips ``git apply`` for an empty patch.
The real test command, parser, and reward computation still run.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# Support both ``python -m ...`` and direct absolute-path execution.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pyarrow.parquet as pq
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from recipe.swe.env_server.config import AgentConfig
from recipe.swe.env_server.swe_env_client import SWEEnvClient, load_env_config
from recipe.swe.env_server.swe_utils import (
    get_instance_docker_image,
    load_yaml_config_from_file_path,
    resolve_swe_repo_dir,
)
from recipe.swe.env_server.test_spec import (
    make_swebench_pro_test_spec,
    make_test_spec,
)


DEFAULT_AGENT_CONFIG_PATH = str(
    Path(__file__).resolve().parents[1]
    / "config"
    / "swe_agent_qwen3_native_toolcall_nothink.yaml"
)

DATASET_PRESETS: dict[str, dict[str, str]] = {
    "verified": {
        "example_image": (
            "docker.io/swebench/"
            "sweb.eval.x86_64.astropy_1776_astropy-12907:latest"
        ),
    },
    "multilingual": {
        "example_image": (
            "docker.io/swebench/"
            "sweb.eval.x86_64.babel_1776_babel-13928:latest"
        ),
    },
    "rebench": {
        "example_image": (
            "docker.io/swerebench/"
            "sweb.eval.x86_64.0b01001001_1776_spectree-64:latest"
        ),
    },
    "rebench-v2": {
        "example_image": (
            "docker.io/swerebenchv2/"
            "virtuslab-git-machete:179-9749145"
        ),
    },
    "pro": {
        "example_image": (
            "jefzda/sweap-images:qutebrowser.qutebrowser-"
            "qutebrowser__qutebrowser-"
            "0833b5f6f140d04200ec91605f88704dd18e2970-"
            "v059c6fdc75567943479b23ebca7c07b5e9a7f"
        ),
    },
    "leaderboard": {
        "example_image": (
            "docker.io/swerebench/"
            "sweb.eval.x86_64.pgmpy_1776_pgmpy-3137:latest"
        ),
    },
}


def _normalize_image_name(image_name: str) -> str:
    value = str(image_name or "").strip()
    if value.startswith("docker.io/"):
        value = value[len("docker.io/") :]
    return value


def _normalize_source(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def infer_dataset_name(instance: dict[str, Any]) -> str:
    evaluator_type = _normalize_source(instance.get("evaluator_type"))
    data_source = _normalize_source(instance.get("data_source"))
    docker_source = _normalize_source(instance.get("data_docker_source"))

    if evaluator_type == "swe-bench-pro" or docker_source == "swe-bench-pro":
        return "pro"
    if "multilingual" in data_source or docker_source == "swe-bench-multilingual":
        return "multilingual"
    if "leaderboard" in data_source:
        return "leaderboard"
    if docker_source == "swe-rebench-v2":
        return "rebench-v2"
    if docker_source == "swe-rebench":
        return "rebench"
    if "verified" in data_source or docker_source == "swe-bench":
        return "verified"
    raise ValueError(
        "Cannot infer dataset from row: "
        f"data_source={instance.get('data_source')!r}, "
        f"data_docker_source={instance.get('data_docker_source')!r}, "
        f"evaluator_type={instance.get('evaluator_type')!r}"
    )


def load_instance_from_parquet(
    data_path: str,
    *,
    image_name: str | None = None,
    instance_id: str | None = None,
) -> tuple[dict[str, Any], int]:
    """Find exactly one processed row without recomputing its group routing."""

    if not image_name and not instance_id:
        raise ValueError("Pass --image-name, --instance-id, or both")
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Processed Parquet does not exist: {path}")

    parquet_file = pq.ParquetFile(path)
    required_columns = {"data_source", "extra_info"}
    missing_columns = required_columns - set(parquet_file.schema_arrow.names)
    if missing_columns:
        raise ValueError(
            "The input must be a processed Verl Parquet with columns "
            f"{sorted(required_columns)}; missing={sorted(missing_columns)}"
        )

    normalized_target_image = (
        _normalize_image_name(image_name) if image_name else None
    )
    matches: list[tuple[dict[str, Any], int]] = []
    global_row_index = 0
    for batch in parquet_file.iter_batches(
        batch_size=256,
        columns=["data_source", "extra_info"],
    ):
        for row in batch.to_pylist():
            info = dict(row.get("extra_info") or {})
            current_index = global_row_index
            global_row_index += 1

            image_matches = True
            if normalized_target_image is not None:
                row_images = {
                    _normalize_image_name(info.get("docker_image_name", "")),
                    _normalize_image_name(info.get("image_name", "")),
                }
                image_matches = normalized_target_image in row_images
            id_matches = (
                instance_id is None
                or str(info.get("instance_id")) == str(instance_id)
            )
            if not image_matches or not id_matches:
                continue

            top_level_source = row.get("data_source")
            if top_level_source:
                info["data_source"] = top_level_source
            matches.append((info, current_index))

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one matching row, "
            f"found={len(matches)}, image_name={image_name!r}, "
            f"instance_id={instance_id!r}, data_path={data_path!r}"
        )
    return matches[0]


def select_patch(instance: dict[str, Any], patch_mode: str) -> tuple[str, str]:
    if patch_mode == "gold":
        patch = str(instance.get("patch") or "")
        if not patch.strip():
            raise ValueError(
                f"Gold patch is empty for instance_id={instance['instance_id']!r}"
            )
        return patch, "gold"
    if patch_mode == "empty":
        return "", "empty-no-op"
    raise ValueError(f"Unsupported patch mode: {patch_mode!r}")


def _inspect_image_on_node(executable: str, image_name: str) -> dict[str, Any]:
    exists = subprocess.run(
        [executable, "image", "exists", image_name],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if exists.returncode != 0:
        return {
            "exists": False,
            "returncode": exists.returncode,
            "stderr": (exists.stderr or "").strip(),
        }
    inspected = subprocess.run(
        [executable, "image", "inspect", image_name],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if inspected.returncode != 0:
        return {
            "exists": False,
            "returncode": inspected.returncode,
            "stderr": (inspected.stderr or "").strip(),
        }
    item = json.loads(inspected.stdout)[0]
    config = item.get("Config") or {}
    return {
        "exists": True,
        "id": item.get("Id"),
        "size": item.get("Size"),
        "created": item.get("Created"),
        "working_dir": config.get("WorkingDir"),
        "repo_tags": item.get("RepoTags"),
    }


def _remove_container_on_node(
    executable: str,
    container_id: str,
) -> dict[str, Any]:
    inspected = subprocess.run(
        [executable, "container", "inspect", container_id],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if inspected.returncode != 0:
        return {
            "removed": False,
            "already_absent": True,
            "container_id": container_id,
        }
    item = json.loads(inspected.stdout)[0]
    removed = subprocess.run(
        [executable, "rm", "-f", container_id],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return {
        "removed": removed.returncode == 0,
        "already_absent": False,
        "container_id": item.get("Id", container_id),
        "container_name": str(item.get("Name") or "").lstrip("/"),
        "image": item.get("ImageName") or item.get("Image"),
        "returncode": removed.returncode,
        "stdout": (removed.stdout or "").strip(),
        "stderr": (removed.stderr or "").strip(),
    }


def _resolve_unique_group_node(group_id: int) -> dict[str, Any]:
    resource_key = f"group_{group_id}"
    matching_nodes = [
        node
        for node in ray.nodes()
        if node.get("Alive") and resource_key in (node.get("Resources") or {})
    ]
    if len(matching_nodes) != 1:
        raise RuntimeError(
            f"Expected exactly one live Ray node with {resource_key!r}, "
            f"found={len(matching_nodes)}"
        )
    node = matching_nodes[0]
    return {
        "resource_key": resource_key,
        "node_id": node["NodeID"],
        "node_ip": node["NodeManagerAddress"],
        "capacity": node["Resources"][resource_key],
    }


async def run_smoke(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    result: dict[str, Any] = {
        "requested_dataset": args.dataset,
        "patch_mode": args.patch_mode,
        "smoke_ok": False,
    }
    client: SWEEnvClient | None = None
    container_id = ""
    group_node: dict[str, Any] | None = None
    exit_code = 2

    try:
        if not args.data_path:
            raise ValueError("--data-path is required")
        data_path = args.data_path

        instance, row_index = load_instance_from_parquet(
            data_path,
            image_name=args.image_name,
            instance_id=args.instance_id,
        )
        inferred_dataset = infer_dataset_name(instance)
        if args.dataset != "auto" and args.dataset != inferred_dataset:
            raise ValueError(
                f"Dataset mismatch: requested={args.dataset!r}, "
                f"row_inferred={inferred_dataset!r}"
            )

        instance_id = str(instance["instance_id"])
        data_docker_source = str(instance.get("data_docker_source") or "")
        image_name = get_instance_docker_image(
            instance_id=instance_id,
            data_docker_source=data_docker_source,
            instance=instance,
        )
        stored_image_name = str(
            instance.get("docker_image_name")
            or instance.get("image_name")
            or ""
        )
        if _normalize_image_name(image_name) != _normalize_image_name(
            stored_image_name
        ):
            raise ValueError(
                "Resolved image does not match processed Parquet: "
                f"resolved={image_name!r}, stored={stored_image_name!r}"
            )

        cwd = resolve_swe_repo_dir(instance)
        # The 0815 V2 parquet still contains /testbed as a legacy placeholder.
        # Match the production agent loop by overwriting it before spec build.
        instance["repo_dir"] = cwd
        if inferred_dataset == "pro":
            test_spec = make_swebench_pro_test_spec(instance)
        else:
            test_spec = make_test_spec(instance)

        group_id = int(instance["group_id"])
        if args.resource_tokens <= 0:
            raise ValueError("--resource-tokens must be > 0")
        if not ray.is_initialized():
            ray.init(address=args.ray_address)
        group_node = _resolve_unique_group_node(group_id)
        if args.resource_tokens > float(group_node["capacity"]):
            raise ValueError(
                f"--resource-tokens={args.resource_tokens} exceeds "
                f"{group_node['resource_key']} capacity={group_node['capacity']}"
            )
        scheduling_strategy = NodeAffinitySchedulingStrategy(
            node_id=group_node["node_id"],
            soft=False,
        )
        inspect_ref = ray.remote(_inspect_image_on_node).options(
            num_cpus=0,
            scheduling_strategy=scheduling_strategy,
        ).remote("podman", image_name)
        image_inspect = ray.get(inspect_ref, timeout=120)
        if not image_inspect.get("exists"):
            raise RuntimeError(
                f"Image is not loaded on {group_node['resource_key']} "
                f"({group_node['node_ip']}): {image_name}; "
                f"inspect={image_inspect}"
            )

        result.update(
            {
                "dataset": inferred_dataset,
                "data_path": data_path,
                "row_index": row_index,
                "instance_id": instance_id,
                "data_source": instance.get("data_source"),
                "data_docker_source": data_docker_source,
                "image_name": image_name,
                "cwd": cwd,
                "group_id": group_id,
                "node_ip": group_node["node_ip"],
                "image_inspect": image_inspect,
                "evaluator_type": getattr(test_spec, "evaluator_type", "swe_bench"),
            }
        )

        if args.dry_run:
            result.update(
                {
                    "dry_run": True,
                    "smoke_ok": True,
                    "message": "Row, TestSpec, routing, and loaded image validated",
                }
            )
            exit_code = 0
        else:
            env_config = load_env_config(image=image_name)
            env_config.data_source = str(instance.get("data_source") or "")
            env_config.cwd = cwd
            env_config.group_id = group_id
            env_config.env_resource_tokens = args.resource_tokens
            env_config.env_cpu_limit = str(args.cpu_limit)
            env_config.env_mem_limit = args.mem_limit
            env_config.ray_env_actor_num_cpus = args.ray_actor_cpus
            env_config.enable_lxcfs_cpu_view = args.enable_lxcfs
            env_config.map_testbed_to_tmpfs = args.map_testbed_to_tmpfs
            if env_config.map_testbed_to_tmpfs and cwd != "/testbed":
                raise ValueError(
                    "--map-testbed-to-tmpfs is only supported for /testbed; "
                    f"resolved cwd is {cwd!r}"
                )

            agent_config = AgentConfig(
                **load_yaml_config_from_file_path(args.agent_config)["agent"]
            )
            client = SWEEnvClient(
                instance,
                agent_config=agent_config,
                env_config=env_config,
                test_spec=test_spec,
            )
            try:
                init_response = await asyncio.wait_for(
                    client.initialize(
                        timeout=args.init_timeout,
                        max_start_sleep_seconds=0.0,
                        log_prefix=f"[{inferred_dataset} smoke init]",
                    ),
                    timeout=args.schedule_timeout + args.init_timeout,
                )
            except TimeoutError as exc:
                raise RuntimeError(
                    "Timed out while scheduling or initializing the "
                    f"environment after {args.schedule_timeout + args.init_timeout}s"
                ) from exc
            container_id = client.container_id
            result["container_id"] = container_id
            result["init"] = {
                "success": init_response.success,
                "message": init_response.msg,
                "duration": init_response.duration,
            }
            if not init_response.success:
                raise RuntimeError(
                    f"Environment initialization failed: {init_response.msg}"
                )

            patch, patch_transport = select_patch(instance, args.patch_mode)
            eval_timeout = args.eval_timeout
            if eval_timeout is None:
                eval_timeout = 3600 if inferred_dataset == "pro" else 1200
            overall_eval_timeout = (
                args.overall_eval_timeout
                if args.overall_eval_timeout is not None
                else eval_timeout + 600
            )
            try:
                response = await asyncio.wait_for(
                    client.evaluate(
                        patch,
                        timeout=eval_timeout,
                        cwd=cwd,
                        log_prefix=(
                            f"[{inferred_dataset} {args.patch_mode} eval]"
                        ),
                    ),
                    timeout=overall_eval_timeout,
                )
            except TimeoutError as exc:
                raise RuntimeError(
                    "Overall evaluation exceeded "
                    f"{overall_eval_timeout}s"
                ) from exc
            report = response.report or {}
            failure_code = str(report.get("eval_failure_code") or "")
            resolved = bool(report.get("resolved", False))
            infrastructure_ok = bool(
                response.success
                and response.did_real_eval
                and not response.apply_patch_failed
                and not failure_code
            )
            smoke_ok = infrastructure_ok
            expected_reward = args.expect_reward
            if expected_reward is None and args.patch_mode == "gold":
                expected_reward = 1.0
            if expected_reward is not None:
                smoke_ok = smoke_ok and abs(
                    float(response.reward_score) - expected_reward
                ) < 1e-9
            if args.patch_mode == "gold":
                smoke_ok = smoke_ok and resolved

            test_log = str(report.get("test_log") or "")
            result.update(
                {
                    "dry_run": False,
                    "patch_transport": patch_transport,
                    "patch_length": len(patch),
                    "patch_sha256": hashlib.sha256(
                        patch.encode("utf-8")
                    ).hexdigest(),
                    "expected_reward": expected_reward,
                    "evaluation": {
                        "success": response.success,
                        "did_real_eval": response.did_real_eval,
                        "apply_patch_failed": response.apply_patch_failed,
                        "eval_failure_code": failure_code,
                        "reward_score": response.reward_score,
                        "resolved": resolved,
                        "f2p_rate": report.get("f2p_rate"),
                        "p2p_rate": report.get("p2p_rate"),
                        "tests_status": report.get("tests_status"),
                        "parser_total_tests": report.get("parser_total_tests"),
                        "message": response.msg,
                        "duration": response.duration,
                        "time_info": response.time_info,
                        "test_log_tail": (
                            ""
                            if args.test_log_tail == 0
                            else test_log[-args.test_log_tail :]
                        ),
                    },
                    "smoke_ok": smoke_ok,
                }
            )
            exit_code = 0 if smoke_ok else 1
    except Exception as exc:
        result.update(
            {
                "smoke_ok": False,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            }
        )
        exit_code = 2
    finally:
        # RayDockerEnviroment.cleanup is intentionally a no-op in production.
        # Kill only this actor, then remove only its exact container on the
        # already resolved node. Never use a broad podman cleanup command.
        if client is not None and client.env is not None:
            if not container_id:
                try:
                    container_id = await asyncio.wait_for(
                        client.env.get_container_id.remote(),
                        timeout=10,
                    )
                    if container_id:
                        result["container_id"] = container_id
                except Exception as exc:
                    result["container_id_lookup_error"] = str(exc)
            try:
                ray.kill(client.env, no_restart=True)
                result["ray_actor_killed"] = True
            except Exception as exc:
                result["ray_actor_killed"] = False
                result["ray_actor_cleanup_error"] = str(exc)

        if container_id and group_node is not None:
            if args.keep_container:
                result["container_cleanup"] = {
                    "removed": False,
                    "kept_for_debugging": True,
                    "container_id": container_id,
                    "node_ip": group_node["node_ip"],
                }
            else:
                try:
                    cleanup_strategy = NodeAffinitySchedulingStrategy(
                        node_id=group_node["node_id"],
                        soft=False,
                    )
                    cleanup_ref = ray.remote(_remove_container_on_node).options(
                        num_cpus=0,
                        scheduling_strategy=cleanup_strategy,
                    ).remote("podman", container_id)
                    result["container_cleanup"] = ray.get(
                        cleanup_ref,
                        timeout=120,
                    )
                    cleanup = result["container_cleanup"]
                    if not cleanup.get("removed") and not cleanup.get(
                        "already_absent"
                    ):
                        result["smoke_ok"] = False
                        exit_code = 2
                except Exception as exc:
                    result["container_cleanup"] = {
                        "removed": False,
                        "error": str(exc),
                        "container_id": container_id,
                        "node_ip": group_node["node_ip"],
                    }
                    result["smoke_ok"] = False
                    exit_code = 2
    return exit_code, result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a real SWEEnvClient evaluation for one processed Parquet row"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["auto", *DATASET_PRESETS.keys()],
        default="auto",
        help="Dataset adapter; auto infers it from the processed row",
    )
    parser.add_argument(
        "--data-path",
        required=False,
        help="Processed Verl Parquet (required unless --list-presets is used)",
    )
    parser.add_argument(
        "--image-name",
        help=(
            "Exact docker_image_name/image_name; an optional docker.io/ "
            "prefix is treated as equivalent"
        ),
    )
    parser.add_argument("--instance-id", help="Exact instance_id (optional alternative)")
    parser.add_argument(
        "--patch",
        "--patch-mode",
        dest="patch_mode",
        choices=["gold", "empty"],
        default="gold",
        help="Gold solution or semantically empty no-op patch",
    )
    parser.add_argument(
        "--expect-reward",
        type=float,
        help="Override reward assertion; gold defaults to 1, empty has none",
    )
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--agent-config", default=DEFAULT_AGENT_CONFIG_PATH)
    parser.add_argument(
        "--schedule-timeout",
        type=int,
        default=300,
        help="Maximum time to wait for the group resource before init starts",
    )
    parser.add_argument("--init-timeout", type=int, default=360)
    parser.add_argument(
        "--eval-timeout",
        type=int,
        help=(
            "Per internal evaluation command; defaults to 3600 for Pro "
            "and 1200 for other datasets"
        ),
    )
    parser.add_argument(
        "--overall-eval-timeout",
        type=int,
        help="Whole evaluate call; defaults to --eval-timeout plus 600s",
    )
    parser.add_argument("--resource-tokens", type=float, default=12.5)
    parser.add_argument("--cpu-limit", default="2")
    parser.add_argument("--mem-limit", default="6g")
    parser.add_argument("--ray-actor-cpus", type=float, default=0.01)
    parser.add_argument("--enable-lxcfs", action="store_true")
    parser.add_argument("--map-testbed-to-tmpfs", action="store_true")
    parser.add_argument("--test-log-tail", type=int, default=2000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-container", action="store_true")
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print the supported dataset adapters and example image names",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.list_presets:
        print(json.dumps(DATASET_PRESETS, indent=2, ensure_ascii=False))
        return 0
    if not args.data_path:
        parser.error("--data-path is required")
    if not args.image_name and not args.instance_id:
        if args.dataset == "auto":
            parser.error(
                "one of --image-name or --instance-id is required "
                "when --dataset=auto"
            )
        args.image_name = DATASET_PRESETS[args.dataset]["example_image"]
    if args.test_log_tail < 0:
        parser.error("--test-log-tail must be >= 0")
    if args.schedule_timeout <= 0 or args.init_timeout <= 0:
        parser.error("--schedule-timeout and --init-timeout must be > 0")
    if args.eval_timeout is not None and args.eval_timeout <= 0:
        parser.error("--eval-timeout must be > 0")
    if args.overall_eval_timeout is not None and args.overall_eval_timeout <= 0:
        parser.error("--overall-eval-timeout must be > 0")

    exit_code, result = asyncio.run(run_smoke(args))
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
