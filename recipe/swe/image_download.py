#!/usr/bin/env python3
"""Create and execute manifest-driven SWE container-image download tasks.

The public workflow deliberately has no object-storage integration. Images are
pulled with Docker, saved as gzip-compressed tar archives, and later loaded on
Ray workers by ``preload_images.py``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset, load_from_disk

if __package__:
    from .env_server.swe_utils import get_instance_docker_image
else:
    from env_server.swe_utils import get_instance_docker_image


def _load_source(source: str, split: str, revision: str | None):
    path = Path(source).expanduser()
    if path.is_file():
        builders = {".parquet": "parquet", ".json": "json", ".jsonl": "json", ".csv": "csv"}
        try:
            builder = builders[path.suffix.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported dataset file: {path}") from exc
        return load_dataset(builder, data_files={split: str(path)}, split=split)
    if path.is_dir() and ((path / "state.json").exists() or (path / "dataset_dict.json").exists()):
        dataset = load_from_disk(str(path))
        if isinstance(dataset, DatasetDict):
            return dataset[split]
        return dataset
    kwargs: dict[str, Any] = {"split": split}
    if revision:
        kwargs["revision"] = revision
    return load_dataset(source, **kwargs)


def archive_name(image_name: str, data_docker_source: str) -> str:
    """Return the archive basename used by the retained SWE data layout.

    Legacy SWE-bench/ReBench archives omit Docker Hub's explicit
    ``docker.io/`` prefix. ReBench V2 and SWE-bench Pro retain the complete
    image spelling supplied by their dataset rows.
    """

    normalized = data_docker_source.strip().lower().replace("_", "-")
    archive_image = image_name
    if normalized not in {"swe-rebench-v2", "swe-bench-pro"}:
        archive_image = archive_image.removeprefix("docker.io/")
    return archive_image.replace("/", "_").replace(":", "_") + ".tar"


def archive_directory(image_root: str | Path, dataset_name: str) -> Path:
    """Resolve one dataset archive directory without allowing root escape."""

    root = Path(image_root).expanduser().resolve()
    directory = (root / dataset_name).resolve()
    if not directory.is_relative_to(root):
        raise ValueError(f"Dataset archive directory is outside --image-root: {directory}")
    return directory


def resolve_image(instance: dict[str, Any], data_docker_source: str) -> str:
    instance_id = str(instance["instance_id"])
    normalized = data_docker_source.strip().lower().replace("_", "-")
    if normalized == "swe-bench-pro":
        image = instance.get("docker_image_name") or instance.get("image_name")
        if not image and instance.get("dockerhub_tag"):
            image = f"jefzda/sweap-images:{instance['dockerhub_tag']}"
        if not image:
            raise ValueError(f"{instance_id}: SWE-bench Pro row has no image field")
        return str(image)
    return get_instance_docker_image(
        instance_id=instance_id,
        data_docker_source=data_docker_source,
        instance=instance,
    )


def command_generate(args: argparse.Namespace) -> None:
    dataset = _load_source(args.source, args.split, args.revision)
    image_root = Path(args.image_root).expanduser().resolve()
    archive_dir = archive_directory(image_root, args.dataset_name)
    archive_dir.mkdir(parents=True, exist_ok=True)
    task_file = Path(args.task_file).expanduser() if args.task_file else image_root / f"tasks-{args.dataset_name}.json"
    if task_file.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {task_file}; pass --force")

    tasks: list[dict[str, str]] = []
    seen_images: set[str] = set()
    for raw in dataset:
        instance = dict(raw)
        instance_id = str(instance["instance_id"])
        image = resolve_image(instance, args.data_docker_source)
        if image in seen_images:
            continue
        seen_images.add(image)
        tar_path = archive_dir / archive_name(image, args.data_docker_source)
        tasks.append(
            {
                "dataset_name": args.dataset_name,
                "data_docker_source": args.data_docker_source,
                "instance_id": instance_id,
                "docker_image_name": image,
                "tar_file_name": tar_path.name,
                "local_image_tar_path": str(tar_path),
            }
        )

    task_file.parent.mkdir(parents=True, exist_ok=True)
    task_file.write_text(json.dumps(tasks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"tasks": len(tasks), "task_file": str(task_file), "archive_dir": str(archive_dir)}))


def _archive_is_valid(path: Path, min_bytes: int) -> bool:
    if not path.is_file() or path.stat().st_size < min_bytes:
        return False
    try:
        # Final archives are installed with os.replace only after both Docker
        # and gzip exit successfully. Checking the magic bytes here avoids
        # decompressing thousands of very large archives on every resume.
        with path.open("rb") as stream:
            return stream.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def _run_checked(command: list[str], timeout: int | None = None) -> None:
    subprocess.run(command, check=True, timeout=timeout)


def _save_image(engine: str, image: str, destination: Path) -> None:
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        partial.unlink()
    process = subprocess.Popen([engine, "save", image], stdout=subprocess.PIPE)
    assert process.stdout is not None
    try:
        with gzip.open(partial, "wb", compresslevel=1) as output:
            shutil.copyfileobj(process.stdout, output, length=1024 * 1024)
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, [engine, "save", image])
        os.replace(partial, destination)
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.kill()
            process.wait()
        if partial.exists():
            partial.unlink()


def _run_task(task: dict[str, str], args: argparse.Namespace) -> dict[str, str]:
    instance_id = task["instance_id"]
    image = task["docker_image_name"]
    destination = Path(task["local_image_tar_path"])
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if _archive_is_valid(destination, args.min_archive_bytes):
            return {"instance_id": instance_id, "status": "skipped", "archive": str(destination)}
        if destination.exists():
            destination.unlink()
        _run_checked([args.engine, "pull", image], timeout=args.pull_timeout)
        _save_image(args.engine, image, destination)
        if not _archive_is_valid(destination, args.min_archive_bytes):
            raise RuntimeError(f"Archive verification failed: {destination}")
        if args.remove_image_after_save:
            _run_checked([args.engine, "image", "rm", image])
        return {"instance_id": instance_id, "status": "saved", "archive": str(destination)}
    except Exception as exc:
        return {"instance_id": instance_id, "status": "failed", "error": str(exc)}


def command_run(args: argparse.Namespace) -> None:
    task_file = Path(args.task_file).expanduser()
    tasks = json.loads(task_file.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError("Task manifest must contain a JSON list")
    image_root = Path(args.image_root).expanduser().resolve()
    required_fields = {
        "instance_id",
        "docker_image_name",
        "local_image_tar_path",
    }
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"Task {index} is not a JSON object")
        missing = sorted(required_fields.difference(task))
        if missing:
            raise ValueError(f"Task {index} is missing: {', '.join(missing)}")
        destination = Path(str(task["local_image_tar_path"])).expanduser().resolve()
        if not destination.is_relative_to(image_root):
            raise ValueError(f"Task {index} archive is outside --image-root: {destination}")
    results: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        pending = {executor.submit(_run_task, task, args): task for task in tasks}
        for future in as_completed(pending):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

    counts = {status: sum(item["status"] == status for item in results) for status in ("saved", "skipped", "failed")}
    print(json.dumps({"task_file": str(task_file), **counts}))
    if counts["failed"]:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Create a download task manifest")
    generate.add_argument("--source", required=True, help="Dataset ID or local dataset path")
    generate.add_argument("--revision", help="Pinned dataset revision")
    generate.add_argument("--split", required=True)
    generate.add_argument("--dataset-name", required=True)
    generate.add_argument("--data-docker-source", required=True)
    generate.add_argument("--image-root", required=True)
    generate.add_argument("--task-file")
    generate.add_argument("--force", action="store_true")
    generate.set_defaults(func=command_generate)

    run = subparsers.add_parser("run", help="Pull and save all tasks")
    run.add_argument("--task-file", required=True)
    run.add_argument("--image-root", required=True, help="Allowed archive root from the generate step")
    run.add_argument("--engine", default="docker")
    run.add_argument("--max-workers", type=int, default=8)
    run.add_argument("--pull-timeout", type=int, default=3600)
    run.add_argument("--min-archive-bytes", type=int, default=1_000_000)
    run.add_argument(
        "--remove-image-after-save",
        action="store_true",
        help="Explicitly remove each Docker image after its archive is verified",
    )
    run.set_defaults(func=command_run)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "max_workers", 1) <= 0:
        raise ValueError("--max-workers must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
