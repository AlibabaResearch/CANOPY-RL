import asyncio
from types import SimpleNamespace

import pytest

# Import the verl parser package before recipe.loop_schema.  The agent-loop
# package registers SWEAgentLoop from its __init__, so the reverse order creates
# a standalone-pytest import cycle.
from verl.experimental.agent_loop.tool_parser import FunctionCall  # noqa: F401

from recipe.swe.loop_schema import AgentData, AgentState
from recipe.swe.env_server.swe_env_client import SWEEnvClient
from recipe.swe.step_diagnostics import new_trajectory_diagnostics
from recipe.swe.swe_agent_loop import (
    SWEAgentLoop,
    _append_tool_exception,
    _build_action_timeout_notice,
)


def _step_response(
    *, timeout=False, killed_by=None, execute_success=True, output_text="command output"
):
    success = execute_success and not timeout
    return SimpleNamespace(
        output={
            "output": output_text,
            "returncode": 0 if execute_success else -1,
            "exception_info": "",
            "extra": {"killed_by": killed_by} if killed_by else {},
        },
        env_raw_output_str=output_text,
        duration=0.1,
        time_info="100ms",
        timeout=timeout,
        msg="",
        success=success,
        execute_success=execute_success,
    )


class _FakeEnv:
    def __init__(self, response_factory):
        self.response_factory = response_factory
        self.execute_calls = 0

    async def execute_action(self, **kwargs):
        self.execute_calls += 1
        return self.response_factory()

    @staticmethod
    def check_finished_and_extract_predict_patch(output, log_prefix=""):
        return False, ""


def _make_loop(
    env,
    *,
    warning_threshold=2,
    termination_threshold=3,
):
    loop = SWEAgentLoop.__new__(SWEAgentLoop)
    loop.env = env
    loop.agent_config = SimpleNamespace(
        use_tool_call=True,
        enforce_exact_submission_command=True,
        enforce_valid_submission_patch=True,
        observation_template="{{ output.exception_info }}|{{ output.output }}",
    )
    loop.tool_call_parser_name = "qwen3_coder"
    loop.action_execute_timeout = 120
    loop.max_env_timeout_cnt = 3
    loop.default_timeout_reward_score = -0.2
    loop.default_no_eval_reward_score = 0.0
    loop.repeated_action_warning_threshold = warning_threshold
    loop.repeated_action_termination_threshold = termination_threshold
    loop.enable_thinking = False
    loop.apply_chat_template_kwargs = {}
    loop.tokenizer = object()
    loop.response_length = 4096
    loop.max_assistant_turns = 200
    loop.min_valid_patch_length = 10
    return loop


def _make_agent(command="pytest -q", *, is_validation=False):
    agent_data = AgentData(messages=[], metrics={}, request_id="guard-test")
    agent_data.group_id = 0
    agent_data.container_id = "container"
    agent_data.cwd = "/testbed"
    agent_data.assistant_turns = 1
    agent_data.is_validation = is_validation
    agent_data.swe_diagnostics = new_trajectory_diagnostics(True, "validation")
    agent_data.action_parse_response = SimpleNamespace(
        success=True,
        actions=[{"command": command, "tool_call_id": "bash_0"}],
        observation="",
        msg="",
    )
    return agent_data


async def _run_tool_state(loop, agent_data, monkeypatch):
    monkeypatch.setattr(
        "recipe.swe.swe_agent_loop.verl_apply_chat_template",
        lambda *args, **kwargs: [101, 102],
    )
    loop.loop = asyncio.get_running_loop()
    return await loop._handle_processing_tools_state(agent_data)


def test_timeout_notice_preserves_original_exception():
    output = {"exception_info": "Command Timed out after 120 seconds"}
    _append_tool_exception(output, _build_action_timeout_notice(120, 1, 3, -0.2))
    assert "Command Timed out after 120 seconds" in output["exception_info"]
    assert "hard limit of 120 seconds" in output["exception_info"]
    assert "Timeout count: 1/3" in output["exception_info"]
    assert "remaining before termination: 2" in output["exception_info"]
    assert "Terminal reward at the limit: -0.2" in output["exception_info"]


@pytest.mark.parametrize("is_validation", [False, True])
def test_repeated_action_warns_on_second_and_terminates_after_third(
    monkeypatch, is_validation
):
    env = _FakeEnv(lambda: _step_response())
    loop = _make_loop(env)
    agent_data = _make_agent(
        "sleep 60 && check-port",
        is_validation=is_validation,
    )

    async def scenario():
        first = await _run_tool_state(loop, agent_data, monkeypatch)
        second = await _run_tool_state(loop, agent_data, monkeypatch)
        second_observation = agent_data.messages[-1]["content"]
        third = await _run_tool_state(loop, agent_data, monkeypatch)
        return first, second, second_observation, third

    first, second, second_observation, third = asyncio.run(scenario())
    assert first == AgentState.GENERATING
    assert second == AgentState.GENERATING
    assert "same command produced the same result 2 consecutive times" in second_observation
    assert "If the next execution produces the same result" in second_observation
    assert third == AgentState.TIMEOUT
    assert env.execute_calls == 3
    assert agent_data.swe_diagnostics["events"]["action_execute_attempt"] == 3
    assert agent_data.swe_diagnostics["events"]["action_execute_success"] == 3
    assert agent_data.swe_diagnostics["timing"]["env_step_timed_count"] == 3
    assert agent_data.swe_diagnostics["terminal_outcome"] == "repeated_action_limit"
    assert agent_data.swe_diagnostics["events"]["repeated_action"] == 2


def test_same_command_with_changing_results_is_not_terminated(monkeypatch):
    outputs = iter(("progress 1", "progress 2", "progress 3"))
    env = _FakeEnv(lambda: _step_response(output_text=next(outputs)))
    loop = _make_loop(env)
    agent_data = _make_agent("check-background-job", is_validation=True)

    async def scenario():
        return [
            await _run_tool_state(loop, agent_data, monkeypatch),
            await _run_tool_state(loop, agent_data, monkeypatch),
            await _run_tool_state(loop, agent_data, monkeypatch),
        ]

    states = asyncio.run(scenario())
    assert states == [AgentState.GENERATING] * 3
    assert env.execute_calls == 3
    assert "repeated_action" not in agent_data.swe_diagnostics["events"]


def test_wall_timeout_feedback_and_three_timeout_termination(monkeypatch):
    env = _FakeEnv(
        lambda: _step_response(timeout=True, killed_by="timeout", execute_success=False)
    )
    loop = _make_loop(env, warning_threshold=0, termination_threshold=0)
    agent_data = _make_agent(is_validation=True)

    async def scenario():
        first = await _run_tool_state(loop, agent_data, monkeypatch)
        first_observation = agent_data.messages[-1]["content"]
        second = await _run_tool_state(loop, agent_data, monkeypatch)
        third = await _run_tool_state(loop, agent_data, monkeypatch)
        return first, first_observation, second, third

    first, first_observation, second, third = asyncio.run(scenario())
    assert first == AgentState.GENERATING
    assert "Timeout count: 1/3" in first_observation
    assert second == AgentState.GENERATING
    assert third == AgentState.TIMEOUT
    assert agent_data.env_timeout_cnt == 3
    assert agent_data.swe_diagnostics["events"]["action_execute_attempt"] == 3
    assert agent_data.swe_diagnostics["events"]["action_timeout"] == 3
    assert agent_data.swe_diagnostics["timing"]["env_step_timed_count"] == 3
    assert "action_nonzero_exit" not in agent_data.swe_diagnostics["events"]
    assert agent_data.swe_diagnostics["terminal_outcome"] == "action_timeout_limit"


def test_output_limit_does_not_consume_wall_timeout_budget(monkeypatch):
    env = _FakeEnv(
        lambda: _step_response(timeout=True, killed_by="size", execute_success=False)
    )
    loop = _make_loop(env, warning_threshold=0, termination_threshold=0)
    agent_data = _make_agent()

    state = asyncio.run(_run_tool_state(loop, agent_data, monkeypatch))
    assert state == AgentState.GENERATING
    assert agent_data.env_timeout_cnt == 0
    assert agent_data.swe_diagnostics["events"]["action_execute_attempt"] == 1
    assert agent_data.swe_diagnostics["events"]["action_output_limit"] == 1
    assert agent_data.swe_diagnostics["timing"]["env_step_timed_count"] == 1
    assert "action_timeout" not in agent_data.swe_diagnostics["events"]


def test_cancelled_core_loop_keeps_live_agent_progress():
    loop = SWEAgentLoop.__new__(SWEAgentLoop)
    loop.use_sparse_reward = True
    live = _make_agent()

    async def slow_pending(agent_data, sampling_params):
        agent_data.assistant_turns = 17
        agent_data.user_turns = 16
        agent_data.server_duration = 123.0
        await asyncio.sleep(60)

    loop._handle_pending_state = slow_pending

    async def scenario():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                loop.run_core_success_loop(
                    SimpleNamespace(),
                    SimpleNamespace(success=True, msg=""),
                    SimpleNamespace(),
                    {},
                    True,
                    "guard-test",
                    0,
                    live.swe_diagnostics,
                    live,
                ),
                timeout=0.01,
            )

    asyncio.run(scenario())
    assert live.assistant_turns == 17
    assert live.user_turns == 16
    assert live.server_duration == 123.0


def test_eval_environment_cleanup_clears_live_reference():
    class EvalEnv:
        def __init__(self):
            self.kill_calls = 0

        async def kill_ray_actor(self, **_kwargs):
            self.kill_calls += 1

    loop = SWEAgentLoop.__new__(SWEAgentLoop)
    live = _make_agent(is_validation=True)
    eval_env = EvalEnv()
    live.active_eval_env = eval_env

    asyncio.run(loop._cleanup_eval_env(live, eval_env))
    assert eval_env.kill_calls == 1
    assert live.active_eval_env is None


def test_evaluate_uses_full_wall_time_for_early_failure_response():
    response = SimpleNamespace(duration=0.0)

    async def evaluate_v2(*args, **kwargs):
        await asyncio.sleep(0.01)
        return response

    client = SimpleNamespace(evaluate_v2=evaluate_v2)
    result = asyncio.run(SWEEnvClient.evaluate(client, "patch"))
    assert result.duration >= 0.01


def test_action_exception_records_dispatch_and_partial_wall_time(monkeypatch):
    class FailingEnv(_FakeEnv):
        async def execute_action(self, **kwargs):
            self.execute_calls += 1
            await asyncio.sleep(0.01)
            raise RuntimeError("ray action failed")

    env = FailingEnv(lambda: _step_response())
    loop = _make_loop(env)
    agent_data = _make_agent()

    with pytest.raises(RuntimeError, match="ray action failed"):
        asyncio.run(_run_tool_state(loop, agent_data, monkeypatch))
    assert agent_data.swe_diagnostics["events"]["action_execute_attempt"] == 1
    assert agent_data.swe_diagnostics["events"]["action_system_error"] == 1
    assert agent_data.swe_diagnostics["timing"]["env_step_timed_count"] == 1
    assert agent_data.swe_diagnostics["timing"]["env_step_seconds"] >= 0.01
