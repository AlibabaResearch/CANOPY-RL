import asyncio
import json
from types import SimpleNamespace

from recipe.swe.env_server.schemas import EnvStepResponse
from recipe.swe.env_server.config import DockerEnvironmentConfig
from recipe.swe.env_server.environments import DockerEnvironment
from recipe.swe.env_server.swe_env_client import SWEEnvClient
from recipe.swe.env_server.swe_utils import get_instance_docker_image
from recipe.swe.env_server.test_spec import (
    SWEProTestSpec,
    make_swebench_pro_test_spec,
)


def _pro_instance(tmp_path):
    run_script = tmp_path / "run_script.sh"
    parser = tmp_path / "parser.py"
    run_script.write_text("#!/bin/bash\necho tests\n", encoding="utf-8")
    parser.write_text("print('parser')\n", encoding="utf-8")
    return {
        "instance_id": "instance_owner__repo-deadbeef-v1",
        "repo": "owner/repo",
        "base_commit": "abc123",
        "FAIL_TO_PASS": ["f2p-test"],
        "PASS_TO_PASS": ["p2p-test"],
        "data_docker_source": "SWE-bench-pro",
        "evaluator_type": "swe_bench_pro",
        "repo_dir": "/app",
        "before_repo_set_cmd": "git reset --hard abc123\ngit checkout deadbeef -- tests",
        "selected_test_files_to_run": '["tests/a.py", "tests/b.py"]',
        "pro_run_script_path": str(run_script),
        "pro_parser_path": str(parser),
        "docker_image_name": "jefzda/sweap-images:owner.repo-deadbeef",
    }


def test_make_pro_spec_uses_official_per_instance_contract(tmp_path):
    instance = _pro_instance(tmp_path)
    spec = make_swebench_pro_test_spec(instance)

    assert spec.repo_directory == "/app"
    assert spec.before_repo_set_cmd == "git checkout deadbeef -- tests"
    assert spec.selected_test_files_to_run == ["tests/a.py", "tests/b.py"]
    assert spec.FAIL_TO_PASS == ["f2p-test"]
    assert spec.PASS_TO_PASS == ["p2p-test"]


def test_pro_image_name_is_not_reconstructed(tmp_path):
    instance = _pro_instance(tmp_path)
    assert get_instance_docker_image(
        instance_id=instance["instance_id"],
        data_docker_source=instance["data_docker_source"],
        instance=instance,
    ) == instance["docker_image_name"]


def test_pro_container_overrides_image_entrypoint():
    config = DockerEnvironmentConfig(
        image="pro-image",
        data_source="SWE-bench_Pro",
        cwd="/app",
        run_args=[],
        enable_lxcfs_cpu_view=False,
    )
    environment = DockerEnvironment(config)
    _, command, _, _ = environment.prepare_container_start_cmd()

    assert command[command.index("--entrypoint") + 1] == "/bin/bash"
    assert not any(item.startswith("GOPROXY=") for item in command)
    image_index = command.index("pro-image")
    assert command[image_index + 1 :] == ["-c", "exec sleep 4h"]


def test_pro_container_injects_explicit_go_proxy_only():
    config = DockerEnvironmentConfig(
        image="pro-image",
        data_source="SWE-bench_Pro",
        cwd="/app",
        run_args=[],
        go_proxy="https://proxy.example.invalid,direct",
        enable_lxcfs_cpu_view=False,
    )
    environment = DockerEnvironment(config)
    _, command, _, _ = environment.prepare_container_start_cmd()

    assert "GOPROXY=https://proxy.example.invalid,direct" in command


def test_pro_grading_requires_all_f2p_and_p2p():
    client = SWEEnvClient.__new__(SWEEnvClient)
    client.test_spec = SWEProTestSpec(
        instance_id="iid",
        repo="owner/repo",
        base_commit="abc123",
        FAIL_TO_PASS=["f2p"],
        PASS_TO_PASS=["p2p"],
        repo_directory="/app",
        before_repo_set_cmd="",
        selected_test_files_to_run=[],
        run_script_path="run.sh",
        parser_path="parser.py",
    )

    resolved = client._compute_pro_eval_response(
        {
            "tests": [
                {"name": "f2p", "status": "PASSED"},
                {"name": "p2p", "status": "PASSED"},
            ]
        },
        "patch",
    )
    unresolved = client._compute_pro_eval_response(
        {"tests": [{"name": "f2p", "status": "PASSED"}]},
        "patch",
    )

    assert resolved.reward_score == 1.0
    assert resolved.report["resolved"] is True
    assert unresolved.reward_score == 0.0
    assert unresolved.report["p2p_rate"] == 0.0


def test_pro_runtime_writes_scripts_runs_parser_and_grades(tmp_path):
    spec = make_swebench_pro_test_spec(_pro_instance(tmp_path))
    client = SWEEnvClient.__new__(SWEEnvClient)
    client.test_spec = spec
    client.env_config = SimpleNamespace(cwd="/app")
    writes = {}
    commands = []

    async def write_content(content, remote_path, log_prefix=""):
        writes[str(remote_path)] = content
        return {"returncode": 0}

    async def execute(command, **kwargs):
        commands.append(command)
        if command == "cat /workspace/output.json":
            return EnvStepResponse(
                success=True,
                execute_success=True,
                env_raw_output_str=json.dumps(
                    {
                        "tests": [
                            {"name": "f2p-test", "status": "PASSED"},
                            {"name": "p2p-test", "status": "PASSED"},
                        ]
                    }
                ),
            )
        return EnvStepResponse(success=True, execute_success=True)

    async def run_eval_script(script, **kwargs):
        commands.append(script)
        return EnvStepResponse(success=True, execute_success=True)

    client.write_content_to_container = write_content
    client.execute_command = execute
    client.run_eval_script = run_eval_script

    response = asyncio.run(
        client.evaluate_swebench_pro("diff --git a/a b/a\n", cwd="/app")
    )

    assert response.reward_score == 1.0
    assert set(writes) == {
        "/workspace/patch.diff",
        "/workspace/run_script.sh",
        "/workspace/parser.py",
    }
    assert commands[0] == "git reset --hard abc123 && git checkout abc123"
    assert commands[1] == "git apply -v /workspace/patch.diff"
    assert "git checkout deadbeef -- tests" in commands[2]
    assert "tests/a.py,tests/b.py" in commands[2]
    assert commands[3] == "cat /workspace/output.json"
