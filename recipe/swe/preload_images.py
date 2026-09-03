#!/usr/bin/env python3
"""Preload processed SWE image archives on their assigned Ray nodes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd
import ray


@ray.remote
def load_image(
    instance_id: str,
    image: str,
    tar_path: str,
    engine: str,
    inspect_timeout: int,
    load_timeout: int,
) -> dict[str, Any]:
    node_ip = ray.util.get_node_ip_address()
    started = time.monotonic()
    if not os.path.isfile(tar_path):
        return {"status": "missing", "instance_id": instance_id, "node_ip": node_ip, "tar_path": tar_path}
    try:
        exists = subprocess.run(
            [engine, "image", "exists", image],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=inspect_timeout,
        )
        if exists.returncode == 0:
            return {"status": "present", "instance_id": instance_id, "node_ip": node_ip, "image": image}
        subprocess.run(
            [engine, "load", "-i", tar_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=load_timeout,
        )
        verification = subprocess.run(
            [engine, "image", "exists", image],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=inspect_timeout,
        )
        if verification.returncode != 0:
            raise RuntimeError("image load returned success but the expected image tag is absent")
        return {
            "status": "loaded",
            "instance_id": instance_id,
            "node_ip": node_ip,
            "image": image,
            "seconds": round(time.monotonic() - started, 2),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "instance_id": instance_id,
            "node_ip": node_ip,
            "image": image,
            "error": str(exc),
        }


def _as_extra_info(value: Any, path: Path) -> dict[str, Any]:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: extra_info must be a struct or JSON object")
    return value


def collect_tasks(parquet_paths: list[str]) -> list[dict[str, Any]]:
    tasks: dict[tuple[int, str, str], dict[str, Any]] = {}
    for raw_path in parquet_paths:
        path = Path(raw_path).expanduser()
        frame = pd.read_parquet(path, columns=["extra_info"])
        for value in frame["extra_info"]:
            info = _as_extra_info(value, path)
            required = ("instance_id", "group_id", "docker_image_name", "local_tar_path")
            missing = [key for key in required if info.get(key) in (None, "")]
            if missing:
                raise ValueError(f"{path}: row is missing {', '.join(missing)}")
            task = {
                "instance_id": str(info["instance_id"]),
                "group_id": int(info["group_id"]),
                "docker_image_name": str(info["docker_image_name"]),
                "local_tar_path": str(info["local_tar_path"]),
            }
            if task["group_id"] < 0:
                raise ValueError(f"{path}: group_id must be non-negative")
            key = (task["group_id"], task["docker_image_name"], task["local_tar_path"])
            tasks.setdefault(key, task)
    return list(tasks.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", nargs="+", help="Processed train/eval Parquet files")
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--engine", default="podman")
    parser.add_argument(
        "--resource-units",
        type=float,
        default=100.0,
        help="Ray group capacity consumed per load (100 means at most 10 loads per 1000-unit node)",
    )
    parser.add_argument("--inspect-timeout", type=int, default=60)
    parser.add_argument("--load-timeout", type=int, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.resource_units <= 0:
        parser.error("--resource-units must be positive")
    if args.inspect_timeout <= 0 or args.load_timeout <= 0:
        parser.error("--inspect-timeout and --load-timeout must be positive")

    tasks = collect_tasks(args.parquet)
    by_group: dict[int, int] = {}
    for task in tasks:
        by_group[task["group_id"]] = by_group.get(task["group_id"], 0) + 1
    print(json.dumps({"unique_images": len(tasks), "tasks_by_group": by_group}, sort_keys=True))
    if args.dry_run:
        return 0
    if not tasks:
        return 0

    ray.init(address=args.ray_address)
    cluster_resources = ray.cluster_resources()
    required_groups = {f"group_{task['group_id']}" for task in tasks}
    missing_groups = sorted(group for group in required_groups if group not in cluster_resources)
    insufficient_groups = sorted(
        group
        for group in required_groups
        if 0 < float(cluster_resources.get(group, 0)) < args.resource_units
    )
    if missing_groups or insufficient_groups:
        raise RuntimeError(
            "Ray group resources do not match the Parquet: "
            f"missing={missing_groups}, insufficient={insufficient_groups}"
        )
    futures = []
    for task in tasks:
        resource = f"group_{task['group_id']}"
        futures.append(
            load_image.options(
                num_cpus=0,
                resources={resource: args.resource_units},
                max_retries=0,
            ).remote(
                task["instance_id"],
                task["docker_image_name"],
                task["local_tar_path"],
                args.engine,
                args.inspect_timeout,
                args.load_timeout,
            )
        )

    failed = 0
    completed = 0
    while futures:
        ready, futures = ray.wait(futures, num_returns=1, timeout=30)
        if not ready:
            print(json.dumps({"status": "waiting", "remaining": len(futures)}), flush=True)
            continue
        result = ray.get(ready[0])
        completed += 1
        if result["status"] in {"missing", "failed"}:
            failed += 1
        print(json.dumps({"completed": completed, "total": len(tasks), **result}, ensure_ascii=False), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
