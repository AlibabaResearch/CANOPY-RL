#!/usr/bin/env python3
"""Fail closed when a CANOPY public source tree contains review-only material."""

from __future__ import annotations

import argparse
import ast
import hashlib
import re
import subprocess
import sys
import tomllib
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CANONICAL_REPOSITORY = "https://github.com/AlibabaResearch/SignalCoverageRL"
PAPER_ARXIV_ID = "2609.01245"
PAPER_TITLE_FRAGMENT = "Outcome-Only Reinforcement Learning Can Suffice"
PAPER_AUTHORS = ("Liming", "Xiaoxia", "Yifu", "Teng", "Bin")
APPWORLD_PROMPT_REVISION = "ba33afb327152803956fdc16f2c3b94a88377453"
APPWORLD_PROMPT_SHA256 = "f9dbebbc906109e98e9742f10a549e498c5a145e2b47783aa793169a1e846ec8"
MODIFIED_VERL_FILES = {
    "verl/experimental/agent_loop/__init__.py",
    "verl/experimental/agent_loop/agent_loop.py",
    "verl/trainer/runtime_env.yaml",
    "verl/utils/megatron/router_replay_patch.py",
    "verl/utils/model.py",
    "verl/utils/net_utils.py",
    "verl/utils/transferqueue_utils.py",
    "verl/workers/engine/megatron/transformer_impl.py",
    "verl/workers/rollout/sglang_rollout/async_sglang_server.py",
}
REQUIRED = {
    "LICENSE",
    "NOTICE",
    "README.md",
    "README.zh-CN.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "THIRD_PARTY_COMPONENTS.yml",
    "docs/TESTED_ENVIRONMENT.md",
    "docs/VERL_COMPATIBILITY.md",
    "patches/README.md",
    "patches/verl-canopy.patch",
    "verl/experimental/fully_async_policy/shell/grpo_qwen3_235b_megatron_npu.sh",
    "pyproject.toml",
    "recipe/appworld/README.md",
    "recipe/appworld/REQUIRED_VERL.txt",
    "recipe/appworld/env_server/prompts.py",
    "recipe/swe/README.md",
    "recipe/swe/REQUIRED_VERL.txt",
    "recipe/swe/image_download.py",
    "recipe/swe/env_server/container_gc.py",
    "recipe/swe/env_server/dependency_mirrors.py",
    "recipe/swe/preload_images.py",
    "recipe/swe/swe_data_process.py",
    "run_scripts/README.md",
    "run_scripts/appworld_readme.md",
    "run_scripts/appworld_readme.zh-CN.md",
    "run_scripts/swe_readme.md",
    "run_scripts/swe_readme.zh-CN.md",
    "run_scripts/appworld/train/appworld_grpo_qwen3_14b_8nodes_0118.sh",
    "run_scripts/swe/cluster/start_head.sh",
    "run_scripts/swe/cluster/start_worker.sh",
    "run_scripts/swe/cluster/podman.sh",
    "run_scripts/swe/cluster/storage.conf",
    "run_scripts/swe/train/swe_qwen36_35b_a3b_12nodes_0817_toolcall_nothink_rebenchv1v2p1_16k_b120_val4bench.sh",
    "run_scripts/swe/train/swe_qwen36_35b_a3b_12nodes_0819_toolcall_nothink_rebenchv1v2p1_16k_gc_offload.sh",
    "tests/utils/megatron/test_router_replay_live_registry_on_cpu.py",
    "tests/utils/test_net_utils.py",
    "THIRD_PARTY_LICENSES/verl/LICENSE",
    "THIRD_PARTY_LICENSES/verl/Notice.txt",
    "THIRD_PARTY_LICENSES/AppWorld/LICENSE",
    "THIRD_PARTY_LICENSES/AppWorld/NOTICE.md",
    "THIRD_PARTY_LICENSES/mini-swe-agent/LICENSE.md",
    "THIRD_PARTY_LICENSES/SWE-bench/LICENSE",
    "THIRD_PARTY_LICENSES/PyTorch/LICENSE",
}

FORBIDDEN_EXACT = {
    "tools/appworld_bundle.py",
    "tools/build_release.sh",
    "tools/check_release.py",
}

FORBIDDEN_BASENAMES = {
    "open_issues.md",
    "provenance.md",
    "release_checklist.md",
    "review_only.md",
    "sca_submission.md",
    "verl_component.json",
}

FORBIDDEN_ROOT_INSTALL_EXACT = {
    "pipfile",
    "pipfile.lock",
    "conda.yaml",
    "conda.yml",
    "conda-lock.yml",
    "environment.yaml",
    "environment.yml",
    "pixi.lock",
    "pixi.toml",
    "pdm.lock",
    "poetry.lock",
    "pylock.toml",
    "setup.cfg",
    "setup.py",
    "uv.lock",
}

FORBIDDEN_ROOT_INSTALL_PATTERNS = (
    re.compile(r"constraints(?:[-_.].*)?\.(?:in|txt)", re.IGNORECASE),
    re.compile(r"requirements(?:[-_.].*)?\.(?:in|txt)", re.IGNORECASE),
)

FORBIDDEN_TOP_LEVEL = {
    ".git",
    "data",
    "logs",
    "outputs",
    "release",
    "runtime",
    "requirements",
    "verl_data",
}

FORBIDDEN_SUFFIXES = {
    ".7z",
    ".bundle",
    ".ckpt",
    ".gz",
    ".orig",
    ".parquet",
    ".pyc",
    ".pt",
    ".pth",
    ".safetensors",
    ".tar",
    ".tgz",
    ".zip",
}

SENSITIVE_PATTERNS = {
    "personal user/path": re.compile(r"liming\.plm|/Users/|/mnt/nas", re.IGNORECASE),
    "private host or storage": re.compile(
        r"\bmfdsw\b|8\.130\.92\.255|alibaba-inc\.com|aliyuncs\.com|oss://|/mnt/oss",
        re.IGNORECASE,
    ),
    "private cluster identity": re.compile(
        r"DSW_INSTANCE_ID|\bdsw-[a-z0-9]{10,}|\bmf_dsw_\d",
        re.IGNORECASE,
    ),
    "private job identifier": re.compile(r"raysubmit_[A-Za-z0-9]+"),
    "anonymous-review marker": re.compile(r"Anonymous Authors|anonymous artifact|review-only", re.IGNORECASE),
    "obvious credential": re.compile(
        r"(?:api[_-]?key|access[_-]?token|secret[_-]?key|password)\s*[:=]\s*['\"][^'\"]+['\"]",
        re.IGNORECASE,
    ),
}

TEXT_SUFFIXES = {
    "",
    ".cff",
    ".cfg",
    ".conf",
    ".ini",
    ".json",
    ".md",
    ".patch",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".typed",
    ".yaml",
    ".yml",
}


def git_tracked_files(root: Path) -> list[Path] | None:
    top_level = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if top_level.returncode != 0 or Path(top_level.stdout.strip()).resolve() != root:
        return None
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return [root / raw.decode() for raw in result.stdout.split(b"\0") if raw]


def candidate_files(root: Path) -> list[Path]:
    tracked = git_tracked_files(root)
    if tracked is not None:
        return sorted(path for path in tracked if path.is_file() or path.is_symlink())
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts
    )


def validate_source_only_pyproject(root: Path) -> list[str]:
    """Ensure the root TOML remains tooling-only, not an install manifest."""

    path = root / "pyproject.toml"
    if not path.is_file():
        return []
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        return [f"invalid pyproject.toml: {exc}"]

    errors: list[str] = []
    if "build-system" in data or "project" in data:
        errors.append("pyproject.toml must not define a build system or installable project")
    unexpected_top = set(data) - {"tool"}
    if unexpected_top:
        errors.append(f"unexpected pyproject.toml tables: {', '.join(sorted(unexpected_top))}")
    unexpected_tools = set(data.get("tool", {})) - {"ruff"}
    if unexpected_tools:
        errors.append(f"non-lint pyproject.toml tools require review: {', '.join(sorted(unexpected_tools))}")
    return errors


def validate_project_metadata(root: Path) -> list[str]:
    """Keep public paper and repository metadata aligned with the canonical release."""

    errors: list[str] = []
    for relative in ("README.md", "README.zh-CN.md"):
        path = root / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if PAPER_ARXIV_ID not in text or PAPER_TITLE_FRAGMENT not in text:
            errors.append(f"{relative} must link the final paper title and arXiv:{PAPER_ARXIV_ID}")

    citation_path = root / "CITATION.cff"
    if citation_path.is_file():
        citation = citation_path.read_text(encoding="utf-8")
        for required in (CANONICAL_REPOSITORY, PAPER_ARXIV_ID, PAPER_TITLE_FRAGMENT, "preferred-citation:"):
            if required not in citation:
                errors.append(f"CITATION.cff is missing canonical metadata: {required}")
        for author in PAPER_AUTHORS:
            if f'given-names: "{author}"' not in citation:
                errors.append(f"CITATION.cff is missing paper author: {author}")

    security_path = root / "SECURITY.md"
    if security_path.is_file():
        security = security_path.read_text(encoding="utf-8")
        if f"{CANONICAL_REPOSITORY}/security/advisories/new" not in security:
            errors.append("SECURITY.md must use the canonical private-reporting URL")
    return errors


def validate_appworld_paper_prompt(root: Path) -> list[str]:
    """Keep the bundled AppWorld prompt aligned with the paper experiment."""

    path = root / "recipe/appworld/env_server/prompts.py"
    if not path.is_file():
        return ["missing bundled AppWorld paper prompt"]

    try:
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        return [f"cannot parse bundled AppWorld paper prompt: {exc}"]

    template: str | None = None
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "_PAPER_SYSTEM_PROMPT_TEMPLATE"
            for target in node.targets
        ):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                value = None
            if isinstance(value, str):
                template = value
            break

    errors: list[str] = []
    if template is None:
        return ["recipe/appworld/env_server/prompts.py must define a literal paper prompt"]
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
    if digest != APPWORLD_PROMPT_SHA256:
        errors.append(
            "bundled AppWorld prompt no longer matches the paper experiment: "
            f"expected {APPWORLD_PROMPT_SHA256}, got {digest}"
        )
    source_text = path.read_text(encoding="utf-8")
    if APPWORLD_PROMPT_REVISION not in source_text:
        errors.append("bundled AppWorld prompt lacks its pinned upstream revision")
    for required_default in (
        "return _PAPER_SYSTEM_PROMPT_TEMPLATE",
        "SYSTEM_PROMPT_TEMPLATE = load_system_prompt_template()",
    ):
        if required_default not in source_text:
            errors.append(
                "bundled AppWorld prompt is not the runtime default: "
                f"missing {required_default}"
            )
    return errors


def validate_required_verl(root: Path) -> list[str]:
    expected = {
        "MODE": "reproduction_commit",
        "REPRODUCTION_COMMIT": "19c6af5de10de2b5272c83c0e82aa715c8c621f3",
        "ROLLING_COMPATIBILITY": "best-effort-unverified",
    }
    errors: list[str] = []
    for relative in ("recipe/appworld/REQUIRED_VERL.txt", "recipe/swe/REQUIRED_VERL.txt"):
        path = root / relative
        if not path.is_file():
            continue
        values: dict[str, str] = {}
        for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.fullmatch(r"([A-Z][A-Z0-9_]*)=(.*)", line)
            if not match:
                errors.append(f"invalid {relative} line {number}: {raw_line}")
                continue
            key, value = match.groups()
            if key in values:
                errors.append(f"duplicate {relative} key: {key}")
            values[key] = value
        for key, expected_value in expected.items():
            if values.get(key) != expected_value:
                errors.append(f"{relative} must set {key}={expected_value}")
    return errors


def validate_verl_modification_notices(root: Path) -> list[str]:
    errors: list[str] = []
    for relative in sorted(MODIFIED_VERL_FILES):
        path = root / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if relative.endswith(".py"):
            has_notice = "Modified by CANOPY contributors in 2026" in text
        else:
            has_notice = "Canopy modification" in text
        if not has_notice:
            errors.append(f"modified Verl file lacks a CANOPY modification notice: {relative}")
    return errors


def validate_swe_runtime_defaults(root: Path) -> list[str]:
    """Keep the public SWE environment fail-closed after selective syncs."""

    errors: list[str] = []
    config_path = root / "recipe/swe/env_server/config.py"
    trainer_path = root / "recipe/swe/config/swe_agent_megatron.yaml"
    runtime_paths = (
        root / "recipe/swe/env_server/dependency_mirrors.py",
        root / "recipe/swe/env_server/environments.py",
        root / "recipe/swe/env_server/swe_env_client.py",
        root / "recipe/swe/swe_agent_loop.py",
        config_path,
        trainer_path,
    )

    config_text = config_path.read_text(encoding="utf-8") if config_path.is_file() else ""
    trainer_text = trainer_path.read_text(encoding="utf-8") if trainer_path.is_file() else ""
    if "network_mode: str = \"none\"" not in config_text:
        errors.append("SWE Docker network_mode must default to none")
    if "dependency_mirror_enabled: bool = False" not in config_text:
        errors.append("SWE dependency mirrors must default to disabled")
    if not re.search(
        r"^[ ]{4}dependency_mirror_enabled:[ ]+False[ ]*$",
        trainer_text,
        flags=re.MULTILINE,
    ):
        errors.append("SWE trainer dependency mirrors must default to False")

    unsafe_patterns = {
        "hard-coded host networking": re.compile(
            r"['\"]--network['\"]\s*[:,]\s*['\"]host['\"]"
        ),
        "disabled seccomp profile": re.compile(r"seccomp=unconfined"),
        "private dependency endpoint": re.compile(
            r"aliyuncs\.com|100\.100\.2\.148", re.IGNORECASE
        ),
    }
    for path in runtime_paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for label, pattern in unsafe_patterns.items():
            if pattern.search(text):
                errors.append(f"{label} found in {path.relative_to(root).as_posix()}")
    return errors


def validate_third_party_inventory(root: Path, relative_files: set[str]) -> list[str]:
    """Validate the fixed inventory schema and its tracked path lists."""

    inventory = root / "THIRD_PARTY_COMPONENTS.yml"
    if not inventory.is_file():
        return []
    errors: list[str] = []
    text = inventory.read_text(encoding="utf-8")
    if "\t" in text:
        errors.append("THIRD_PARTY_COMPONENTS.yml must not contain tab indentation")
    for required_line in (
        "schema_version: 1",
        "project:",
        "components:",
        "runtime_environment:",
        "  install_manifest_distributed: false",
        "  packages_bundled: false",
    ):
        if required_line not in text.splitlines():
            errors.append(f"THIRD_PARTY_COMPONENTS.yml is missing: {required_line}")

    expected_components = {
        "verl",
        "PyTorch copied portions",
        "AppWorld",
        "mini-swe-agent",
        "SWE-bench",
        "SWE-rebench-SWE-bench-fork",
    }
    component_names = re.findall(r"^  - name: (.+)$", text, flags=re.MULTILINE)
    if len(component_names) != len(set(component_names)):
        errors.append("THIRD_PARTY_COMPONENTS.yml contains duplicate component names")
    missing_components = expected_components - set(component_names)
    unexpected_components = set(component_names) - expected_components
    if missing_components:
        errors.append(f"third-party inventory is missing components: {', '.join(sorted(missing_components))}")
    if unexpected_components:
        errors.append(f"third-party inventory has unreviewed components: {', '.join(sorted(unexpected_components))}")

    component_starts = list(re.finditer(r"^  - name: (.+)$", text, flags=re.MULTILINE))
    for index, start in enumerate(component_starts):
        end = component_starts[index + 1].start() if index + 1 < len(component_starts) else len(text)
        block = text[start.start() : end]
        keys = re.findall(r"^    ([a-z_]+):", block, flags=re.MULTILINE)
        if len(keys) != len(set(keys)):
            errors.append(f"duplicate fields for third-party component: {start.group(1)}")
        required_keys = {"role", "source", "license", "bundled_source", "distributed_paths", "license_files", "modified"}
        missing_keys = required_keys - set(keys)
        if missing_keys:
            errors.append(
                f"third-party component {start.group(1)} is missing fields: {', '.join(sorted(missing_keys))}"
            )

    active_list = False
    for line in text.splitlines():
        if re.fullmatch(r"\s{4}(?:distributed_paths|license_files):", line):
            active_list = True
            continue
        match = re.fullmatch(r"\s{6}-\s+(.+)", line) if active_list else None
        if match:
            declared = match.group(1).strip().strip("'\"")
            if declared.endswith("/"):
                present = any(candidate.startswith(declared) for candidate in relative_files)
            else:
                present = declared in relative_files
            if not present:
                errors.append(f"declared third-party path is not tracked: {declared}")
            continue
        if active_list and line.strip() and len(line) - len(line.lstrip()) <= 4:
            active_list = False
    return errors


def check(root: Path) -> list[str]:
    errors: list[str] = []
    files = candidate_files(root)
    relative_files = {path.relative_to(root).as_posix() for path in files}

    for required in sorted(REQUIRED - relative_files):
        errors.append(f"missing required file: {required}")

    errors.extend(validate_source_only_pyproject(root))
    errors.extend(validate_project_metadata(root))
    errors.extend(validate_appworld_paper_prompt(root))
    errors.extend(validate_verl_modification_notices(root))
    errors.extend(validate_required_verl(root))
    errors.extend(validate_swe_runtime_defaults(root))
    errors.extend(validate_third_party_inventory(root, relative_files))

    for path in files:
        relative = path.relative_to(root)
        rel = relative.as_posix()
        if any(unicodedata.category(character) == "Cf" for character in rel):
            errors.append(f"Unicode format character found in public path: {rel!r}")
        if rel in FORBIDDEN_EXACT:
            errors.append(f"review/internal file is public: {rel}")
        if relative.name.casefold() in FORBIDDEN_BASENAMES:
            errors.append(f"non-public filename is public: {rel}")
        if len(relative.parts) == 1 and (
            rel.casefold() in FORBIDDEN_ROOT_INSTALL_EXACT
            or any(pattern.fullmatch(rel) for pattern in FORBIDDEN_ROOT_INSTALL_PATTERNS)
        ):
            errors.append(f"standalone root install manifest contradicts source-only release: {rel}")
        if relative.parts and relative.parts[0].casefold() in FORBIDDEN_TOP_LEVEL:
            errors.append(f"forbidden public directory: {rel}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden artifact type: {rel}")
        if path.is_symlink():
            errors.append(f"symbolic link requires manual review: {rel}")
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            errors.append(f"unapproved public file type: {rel}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"non-UTF-8 text file: {rel}")
            continue
        if rel == "tools/check_public_release.py":
            # This file necessarily contains the deny-list literals.
            continue
        for label, pattern in SENSITIVE_PATTERNS.items():
            candidate_text = text
            if label == "obvious credential":
                # Upstream verl contains this documented placeholder only.
                candidate_text = candidate_text.replace('api_key="123-abc"', "")
            if pattern.search(candidate_text):
                errors.append(f"{label} found in {rel}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    errors = check(root)
    if errors:
        print("Public release check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"Public release check passed: {len(candidate_files(root))} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
