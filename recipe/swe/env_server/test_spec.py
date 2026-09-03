import ast
import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union, cast

from swebench.harness.constants import (DEFAULT_DOCKER_SPECS, END_TEST_OUTPUT,
                                        KEY_INSTANCE_ID, LATEST,
                                        MAP_REPO_TO_EXT,
                                        MAP_REPO_VERSION_TO_SPECS,
                                        START_TEST_OUTPUT, SWEbenchInstance)
from swebench.harness.dockerfiles import (get_dockerfile_base,
                                          get_dockerfile_env,
                                          get_dockerfile_instance)
from swebench.harness.test_spec.create_scripts import (make_env_script_list,
                                                       make_eval_script_list,
                                                       make_repo_script_list)
from swebench.harness.test_spec.test_spec import TestSpec
from swebench.harness.test_spec.utils import (get_test_cmds,
                                              make_eval_script_list_common)
from unidiff.errors import UnidiffParseError

from recipe.swe.env_server.swe_utils import resolve_swe_repo_dir


def _decode_string_list(value: Any, *, key: str, instance_id: str) -> list[str]:
    """Decode list-like Parquet values without using ``eval``."""

    if value is None:
        return []
    if isinstance(value, str):
        if not value:
            return []
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = ast.literal_eval(value)
    elif hasattr(value, "tolist"):
        value = value.tolist()
    elif isinstance(value, tuple):
        value = list(value)

    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise TypeError(
            f"{key} must decode to list[str] for instance_id={instance_id!r}, "
            f"got {type(value).__name__}: {value!r}"
        )
    return value


@dataclass(frozen=True)
class SWEProTestSpec:
    """Runtime contract for the official SWE-bench Pro public evaluator."""

    instance_id: str
    repo: str
    base_commit: str
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    repo_directory: str
    before_repo_set_cmd: str
    selected_test_files_to_run: list[str]
    run_script_path: str
    parser_path: str
    evaluator_type: str = "swe_bench_pro"


def make_swebench_pro_test_spec(instance: dict) -> SWEProTestSpec:
    """Build a Pro spec while keeping its per-instance official scripts."""

    instance_id = str(instance[KEY_INSTANCE_ID])
    run_script_path = str(instance.get("pro_run_script_path", ""))
    parser_path = str(instance.get("pro_parser_path", ""))
    for label, path in (
        ("run script", run_script_path),
        ("parser", parser_path),
    ):
        if not path or not Path(path).is_file():
            raise FileNotFoundError(
                f"SWE-bench Pro {label} is missing for "
                f"instance_id={instance_id!r}: {path!r}"
            )

    before_repo_set_cmd = str(instance.get("before_repo_set_cmd", "")).strip()
    # Match the official evaluator, which executes the final command from this
    # multi-line setup field after applying the model patch.
    before_repo_set_cmd = (
        before_repo_set_cmd.splitlines()[-1] if before_repo_set_cmd else ""
    )

    return SWEProTestSpec(
        instance_id=instance_id,
        repo=str(instance["repo"]),
        base_commit=str(instance["base_commit"]),
        FAIL_TO_PASS=_decode_string_list(
            instance.get("FAIL_TO_PASS", []),
            key="FAIL_TO_PASS",
            instance_id=instance_id,
        ),
        PASS_TO_PASS=_decode_string_list(
            instance.get("PASS_TO_PASS", []),
            key="PASS_TO_PASS",
            instance_id=instance_id,
        ),
        repo_directory=resolve_swe_repo_dir(instance),
        before_repo_set_cmd=before_repo_set_cmd,
        selected_test_files_to_run=_decode_string_list(
            instance.get("selected_test_files_to_run", []),
            key="selected_test_files_to_run",
            instance_id=instance_id,
        ),
        run_script_path=run_script_path,
        parser_path=parser_path,
    )


def _make_rebench_v2_eval_script_list(
    instance, specs, env_name, repo_directory, base_commit, test_patch
) -> list[str]:
    """Build the runtime-only V2 test script for a prebuilt image.

    The common builder already has the semantics we need: restore the gold
    test files, apply ``test_patch``, emit the parser markers, and execute the
    V2-provided ``test_cmd``.  In particular it does not rerun the image-build
    ``install`` commands or activate the legacy ``testbed`` conda environment.

    One public V2 patch contains unquoted spaces in a newly added path and
    cannot be parsed by ``unidiff``.  For that case, follow the official V2
    evaluator more closely and apply the test patch without precomputing the
    files to reset.  The evaluation container is disposable.
    """

    try:
        return make_eval_script_list_common(
            instance,
            specs,
            env_name,
            repo_directory,
            base_commit,
            test_patch,
        )
    except UnidiffParseError:
        instance_id = instance.get("instance_id", "unknown")
        warnings.warn(
            "Falling back to the no-preparse SWE-rebench-V2 eval script for "
            f"instance_id={instance_id!r}; test_patch paths could not be "
            "parsed by unidiff.",
            RuntimeWarning,
            stacklevel=2,
        )
        test_commands = get_test_cmds(instance)
        if (
            not isinstance(test_commands, list)
            or not test_commands
            or not all(
                isinstance(command, str) and command.strip()
                for command in test_commands
            )
        ):
            raise ValueError(
                "SWE-rebench-V2 test_cmd must resolve to a non-empty list of "
                f"commands for instance_id={instance_id!r}: {test_commands!r}"
            )

        heredoc_delimiter = "EOF_114329324912"
        apply_test_patch_command = (
            "git apply --verbose --reject - "
            f"<<'{heredoc_delimiter}'\n"
            f"{test_patch}\n"
            f"{heredoc_delimiter}"
        )
        return [
            f"cd {repo_directory}",
            f"git config --global --add safe.directory {repo_directory}",
            f"cd {repo_directory}",
            apply_test_patch_command,
            f": '{START_TEST_OUTPUT}'",
            *test_commands,
            f": '{END_TEST_OUTPUT}'",
        ]


def get_test_specs_from_dataset(
    dataset: Union[list[SWEbenchInstance], list[TestSpec]],
    namespace: Optional[str] = None,
    instance_image_tag: str = LATEST,
    env_image_tag: str = LATEST,
) -> list[TestSpec]:
    """
    Idempotent function that converts a list of SWEbenchInstance objects to a list of TestSpec objects.
    """
    if isinstance(dataset[0], TestSpec):
        return cast(list[TestSpec], dataset)
    return list(
        map(
            lambda x: make_test_spec(x, namespace, instance_image_tag, env_image_tag),
            cast(list[SWEbenchInstance], dataset),
        )
    )


def make_test_spec(
    instance,
    namespace: Optional[str] = None,
    base_image_tag: str = LATEST,
    env_image_tag: str = LATEST,
    instance_image_tag: str = LATEST,
    arch: str = "x86_64",
) -> TestSpec:
    if isinstance(instance, TestSpec):
        return instance
    assert base_image_tag is not None, "base_image_tag cannot be None"
    assert env_image_tag is not None, "env_image_tag cannot be None"
    assert instance_image_tag is not None, "instance_image_tag cannot be None"
    instance_id = instance[KEY_INSTANCE_ID]
    repo = instance["repo"]
    version = instance.get("version")
    base_commit = instance["base_commit"]
    problem_statement = instance.get("problem_statement")
    hints_text = instance.get("hints_text")  # Unused
    test_patch = instance["test_patch"]

    def _from_json_or_obj(key: str, default: Any) -> Any:
        """If key points to string, load with json"""
        if key not in instance:
            return default
        value = instance[key]
        if isinstance(value, str):
            if not value:
                return default
            return json.loads(value)
        return value

    pass_to_pass = _from_json_or_obj("PASS_TO_PASS", [])
    fail_to_pass = _from_json_or_obj("FAIL_TO_PASS", [])

    env_name = "testbed"
    data_docker_source = (
        str(instance.get("data_docker_source", ""))
        .strip()
        .lower()
        .replace("_", "-")
    )
    is_rebench_v2 = data_docker_source == "swe-rebench-v2"
    repo_directory = resolve_swe_repo_dir(instance)
    # specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
    install_config = _from_json_or_obj("install_config", {})
    if not isinstance(install_config, dict):
        raise TypeError(
            "install_config must decode to a dict, "
            f"got {type(install_config).__name__} for {instance_id}"
        )
    if install_config and not install_config.get("fake", False):
        specs = install_config
    else:
        install_config = None
        specs = MAP_REPO_VERSION_TO_SPECS[repo][version]

    docker_specs = specs.get("docker_specs") or {}

    # repo_script_list = make_repo_script_list(
    #     specs, repo, repo_directory, base_commit, env_name
    # )
    repo_script_list = []
    # 不去做env，因为已经有环境了。
    # env_script_list = make_env_script_list(instance, specs, env_name)
    env_script_list = []

    # Parquet stores install_config as a canonical JSON string so heterogeneous
    # datasets share one Arrow schema. Upstream script builders still expect the
    # instance field itself to be a mapping, so pass a decoded shallow copy.
    script_instance = dict(instance)
    script_instance["install_config"] = install_config or {}
    if is_rebench_v2:
        eval_script_list = _make_rebench_v2_eval_script_list(
            script_instance,
            specs,
            env_name,
            repo_directory,
            base_commit,
            test_patch,
        )
    else:
        eval_script_list = make_eval_script_list(
            script_instance,
            specs,
            env_name,
            repo_directory,
            base_commit,
            test_patch,
        )
    return TestSpec(
        instance_id=instance_id,
        repo=repo,
        env_script_list=env_script_list,
        repo_script_list=repo_script_list,
        eval_script_list=eval_script_list,
        version=version,
        arch=arch,
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        language=MAP_REPO_TO_EXT[repo],
        docker_specs=docker_specs,
        namespace=namespace,
        base_image_tag=base_image_tag,
        env_image_tag=env_image_tag,
        instance_image_tag=instance_image_tag,
        install_config=install_config,
        image_name=instance.get("image_name", None),
    )
