#!/usr/bin/env python3
"""Generate verl-compatible AppWorld parquet metadata from a local install.

This utility intentionally contains no AppWorld task IDs or task content. The
caller must point it at an authorized AppWorld installation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Callable, Iterable


SUPPORTED_SPLITS = ("train", "dev", "test_normal", "test_challenge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--appworld-root",
        default=os.getenv("APPWORLD_ROOT"),
        help="Authorized AppWorld root (default: APPWORLD_ROOT).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SUPPORTED_SPLITS,
        default=list(SUPPORTED_SPLITS),
    )
    parser.add_argument(
        "--by-difficulty",
        action="store_true",
        help="Also emit one parquet per difficulty level (1, 2, and 3).",
    )
    parser.add_argument(
        "--task-prefix-file",
        type=Path,
        help="Optional newline-delimited allowlist of task-ID prefixes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated parquet files.",
    )
    return parser.parse_args()


def load_prefixes(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"task-prefix allowlist not found: {path}")
    prefixes = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not prefixes:
        raise ValueError(f"task-prefix allowlist is empty: {path}")
    return prefixes


def build_dataframe(
    split: str,
    task_ids: Iterable[str],
    get_task_difficulty: Callable[[str], int],
    fixed_difficulty: int | None = None,
    allowed_prefixes: set[str] | None = None,
) -> Any:
    import pandas as pd

    rows = []
    for source_index, task_id in enumerate(task_ids):
        prefix = task_id.split("_", maxsplit=1)[0]
        if allowed_prefixes is not None and prefix not in allowed_prefixes:
            continue
        difficulty = (
            fixed_difficulty
            if fixed_difficulty is not None
            else int(get_task_difficulty(task_id))
        )
        rows.append(
            {
                "data_source": f"appworld_{split}",
                "agent_name": "appworld_env_agent",
                # The current AppWorld agent loop reads the task identifier
                # directly from raw_prompt.  Keep the historical/current data
                # schema's explicit role marker instead of presenting the ID as
                # a natural-language user message.
                "prompt": [{"role": "task_id", "content": task_id}],
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "task_id": task_id,
                    "split": split,
                    "index": source_index,
                    "difficulty": difficulty,
                },
            }
        )
    return pd.DataFrame(rows)


def write_parquet(frame: Any, destination: Path, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite {destination}; pass --overwrite explicitly"
        )
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"wrote {len(frame)} rows to {destination}")


def main() -> None:
    args = parse_args()
    if not args.appworld_root:
        raise SystemExit("--appworld-root or APPWORLD_ROOT is required")

    appworld_root = Path(args.appworld_root).expanduser().resolve()
    if not appworld_root.is_dir():
        raise FileNotFoundError(f"AppWorld root is not a directory: {appworld_root}")
    os.environ["APPWORLD_ROOT"] = str(appworld_root)

    # Import only after APPWORLD_ROOT is set; AppWorld reads it at import time.
    from appworld import get_task_difficulty, load_task_ids

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_prefixes = load_prefixes(args.task_prefix_file)

    for split in args.splits:
        task_ids = load_task_ids(split)
        frame = build_dataframe(
            split,
            task_ids,
            get_task_difficulty,
            allowed_prefixes=allowed_prefixes,
        )
        write_parquet(frame, output_dir / f"{split}.parquet", args.overwrite)

        if args.by_difficulty:
            for difficulty in (1, 2, 3):
                difficulty_ids = load_task_ids(split, difficulty=difficulty)
                frame = build_dataframe(
                    split,
                    difficulty_ids,
                    get_task_difficulty,
                    fixed_difficulty=difficulty,
                    allowed_prefixes=allowed_prefixes,
                )
                write_parquet(
                    frame,
                    output_dir / f"{split}_level{difficulty}.parquet",
                    args.overwrite,
                )


if __name__ == "__main__":
    main()
