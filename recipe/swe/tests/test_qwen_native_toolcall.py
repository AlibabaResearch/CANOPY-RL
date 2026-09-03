import asyncio
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from transformers import AutoTokenizer

# Import verl's parser package before the recipe-level Hermes wrapper. The
# local agent_loop package registers SWEAgentLoop from __init__, so reversing
# this order creates an import cycle during standalone pytest collection.
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.chat_template import apply_chat_template

from recipe.swe.env_server.actions_toolcall import (
    BASH_TOOL,
    EXACT_SUBMISSION_COMMAND,
    is_exact_submission_command,
    is_git_diff_patch,
    parse_toolcall_actions,
)
from recipe.swe.env_server.config import AgentConfig
from recipe.swe.env_server.exceptions import FormatError
from recipe.swe.hermes_tool_parser import CanopyHermesToolParser
from recipe.swe.qwen_native_toolcall import validate_complete_qwen3_tool_calls


MODEL_PATH = os.environ.get(
    "QWEN_NATIVE_TOOLCALL_TEST_MODEL",
    "",
)
QWEN35_MODEL_PATH = os.environ.get(
    "QWEN35_NATIVE_TOOLCALL_TEST_MODEL",
    "",
)
FORMAT_ERROR_TEMPLATE = "{{ error }} actions={{ actions|length }}"
NATIVE_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "swe_agent_qwen3_native_toolcall_nothink.yaml"
)
NATIVE_THINK_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "swe_agent_qwen3_native_toolcall_think.yaml"
)
QWEN35_NATIVE_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "swe_agent_qwen35_9b_native_toolcall_nothink.yaml"
)


@pytest.fixture(scope="module")
def tokenizer():
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Tokenizer path does not exist: {MODEL_PATH}")
    return AutoTokenizer.from_pretrained(MODEL_PATH)


@pytest.fixture(scope="module")
def qwen35_tokenizer():
    if not os.path.exists(QWEN35_MODEL_PATH):
        pytest.skip(f"Tokenizer path does not exist: {QWEN35_MODEL_PATH}")
    return AutoTokenizer.from_pretrained(QWEN35_MODEL_PATH)


def test_qwen_native_single_bash_multiline(tokenizer):
    text = """<tool_call>
<function=bash>
<parameter=command>
python - <<'PY'
print("native tool call")
PY
</parameter>
</function>
</tool_call>"""
    assert validate_complete_qwen3_tool_calls(text) == []

    parser = ToolParser.get_tool_parser("qwen3_coder", tokenizer)
    schema = [OpenAIFunctionToolSchema.model_validate(BASH_TOOL)]
    _, calls = asyncio.run(
        parser.extract_tool_calls(tokenizer.encode(text, add_special_tokens=False), schema)
    )
    actions = parse_toolcall_actions(
        calls,
        format_error_template=FORMAT_ERROR_TEMPLATE,
        toolcall_extract_error_msgs=[],
        enforce_single_tool_call=True,
    )
    assert actions[0]["command"] == "python - <<'PY'\nprint(\"native tool call\")\nPY"


def test_qwen_thinking_native_bash_call_parses(tokenizer):
    text = """I should inspect the current directory first.
</think>

<tool_call>
<function=bash>
<parameter=command>pwd</parameter>
</function>
</tool_call>"""
    assert validate_complete_qwen3_tool_calls(
        text, require_thinking_close=True
    ) == []
    parser = ToolParser.get_tool_parser("qwen3_coder", tokenizer)
    schema = [OpenAIFunctionToolSchema.model_validate(BASH_TOOL)]
    _, calls = asyncio.run(
        parser.extract_tool_calls(tokenizer.encode(text, add_special_tokens=False), schema)
    )
    actions = parse_toolcall_actions(
        calls,
        format_error_template=FORMAT_ERROR_TEMPLATE,
        toolcall_extract_error_msgs=[],
        enforce_single_tool_call=True,
    )
    assert actions[0]["command"] == "pwd"


@pytest.mark.parametrize(
    "text",
    [
        (
            "reasoning without a close\n"
            "<tool_call><function=bash><parameter=command>pwd</parameter>"
            "</function></tool_call>"
        ),
        (
            "<think>duplicate open</think>\n"
            "<tool_call><function=bash><parameter=command>pwd</parameter>"
            "</function></tool_call>"
        ),
    ],
)
def test_qwen_invalid_thinking_boundary_not_executable(text):
    errors = validate_complete_qwen3_tool_calls(
        text, require_thinking_close=True
    )
    assert errors
    assert any("think" in error for error in errors)


def test_qwen_native_thinking_agent_config_loads():
    data = yaml.safe_load(NATIVE_THINK_CONFIG_PATH.read_text())["agent"]
    config = AgentConfig(**data)
    assert config.use_tool_call
    assert config.enforce_single_tool_call
    assert config.enforce_exact_bash_arguments
    assert config.enforce_exact_submission_command
    assert config.enforce_valid_submission_patch
    assert "thinking block" in config.system_template


def test_qwen_native_nonthinking_agent_config_loads():
    data = yaml.safe_load(NATIVE_CONFIG_PATH.read_text())["agent"]
    config = AgentConfig(**data)
    assert config.use_tool_call
    assert config.enforce_single_tool_call
    assert config.enforce_exact_bash_arguments
    assert config.enforce_exact_submission_command
    assert config.enforce_valid_submission_patch


def test_qwen35_native_agent_config_loads_and_requests_direct_calls():
    data = yaml.safe_load(QWEN35_NATIVE_CONFIG_PATH.read_text())["agent"]
    config = AgentConfig(**data)
    normalized_system_template = " ".join(config.system_template.split())
    normalized_instance_template = " ".join(config.instance_template.split())
    assert config.use_tool_call
    assert config.enforce_single_tool_call
    assert config.enforce_exact_bash_arguments
    assert config.enforce_exact_submission_command
    assert config.enforce_valid_submission_patch
    assert "emit no prose before it" in normalized_system_template
    assert "never repeat an unchanged command" in normalized_system_template
    assert "submit the best valid current patch" in normalized_instance_template


@pytest.mark.parametrize(
    "text",
    [
        "<tool_call><function=bash><parameter=command>pwd</parameter></function>",
        "<tool_call><function=bash><parameter=command>pwd</function></tool_call>",
        "<tool_call><function=bash><parameter=command>pwd</parameter></tool_call>",
    ],
)
def test_qwen_incomplete_xml_not_executable(text):
    assert validate_complete_qwen3_tool_calls(text)


def test_qwen_text_after_tool_call_not_executable():
    text = (
        "<tool_call><function=bash><parameter=command>pwd</parameter>"
        "</function></tool_call>do something else"
    )
    assert "no text is allowed" in " ".join(validate_complete_qwen3_tool_calls(text))


def test_qwen_text_before_call_and_stop_token_suffix_accepted():
    text = (
        "I will inspect the repository.\n"
        "<tool_call><function=bash><parameter=command>pwd</parameter>"
        "</function></tool_call>\n<|im_end|>"
    )
    assert validate_complete_qwen3_tool_calls(text) == []


def test_zero_call_rejected_by_swe_policy():
    with pytest.raises(FormatError):
        parse_toolcall_actions(
            [],
            format_error_template=FORMAT_ERROR_TEMPLATE,
            toolcall_extract_error_msgs=[],
            enforce_single_tool_call=True,
        )


def test_multiple_calls_rejected_by_swe_policy():
    calls = [
        FunctionCall(name="bash", arguments='{"command":"pwd"}'),
        FunctionCall(name="bash", arguments='{"command":"ls"}'),
    ]
    with pytest.raises(FormatError):
        parse_toolcall_actions(
            calls,
            format_error_template=FORMAT_ERROR_TEMPLATE,
            toolcall_extract_error_msgs=[],
            enforce_single_tool_call=True,
        )


@pytest.mark.parametrize(
    "call",
    [
        FunctionCall(name="unknown", arguments='{"command":"pwd"}'),
        FunctionCall(name="bash", arguments="{}"),
    ],
)
def test_unknown_tool_and_missing_command_rejected(call):
    with pytest.raises(FormatError):
        parse_toolcall_actions(
            [call],
            format_error_template=FORMAT_ERROR_TEMPLATE,
            toolcall_extract_error_msgs=[],
            enforce_single_tool_call=True,
        )


def test_unexpected_bash_arguments_are_opt_in_rejected():
    call = FunctionCall(
        name="bash",
        arguments=json.dumps({"command": "pwd", "default": "ignored"}),
    )

    actions = parse_toolcall_actions(
        [call],
        format_error_template=FORMAT_ERROR_TEMPLATE,
        toolcall_extract_error_msgs=[],
        enforce_single_tool_call=True,
    )
    assert actions[0]["command"] == "pwd"

    with pytest.raises(FormatError) as exc_info:
        parse_toolcall_actions(
            [call],
            format_error_template=FORMAT_ERROR_TEMPLATE,
            toolcall_extract_error_msgs=[],
            enforce_single_tool_call=True,
            enforce_exact_bash_arguments=True,
        )
    assert "Only 'command' is allowed" in exc_info.value.messages[0]["content"]


@pytest.mark.parametrize("command", [None, "", "  \n\t"])
def test_null_or_empty_command_rejected(command):
    call = FunctionCall(
        name="bash",
        arguments=json.dumps({"command": command}),
    )
    with pytest.raises(FormatError) as exc_info:
        parse_toolcall_actions(
            [call],
            format_error_template=FORMAT_ERROR_TEMPLATE,
            toolcall_extract_error_msgs=[],
            enforce_single_tool_call=True,
        )
    assert "non-empty string" in exc_info.value.messages[0]["content"]


def test_exact_native_submission_command_accepted():
    calls = [
        FunctionCall(
            name="bash",
            arguments=json.dumps({"command": EXACT_SUBMISSION_COMMAND}),
        )
    ]
    actions = parse_toolcall_actions(
        calls,
        format_error_template=FORMAT_ERROR_TEMPLATE,
        toolcall_extract_error_msgs=[],
        enforce_single_tool_call=True,
        enforce_exact_submission_command=True,
    )
    assert actions[0]["command"] == EXACT_SUBMISSION_COMMAND


def test_augmented_native_submission_command_rejected():
    calls = [
        FunctionCall(
            name="bash",
            arguments=json.dumps(
                {"command": EXACT_SUBMISSION_COMMAND + " 2>/dev/null || git diff"}
            ),
        )
    ]
    with pytest.raises(FormatError):
        parse_toolcall_actions(
            calls,
            format_error_template=FORMAT_ERROR_TEMPLATE,
            toolcall_extract_error_msgs=[],
            enforce_single_tool_call=True,
            enforce_exact_submission_command=True,
        )


def test_submission_error_explains_that_rejected_command_was_not_executed():
    template = yaml.safe_load(NATIVE_CONFIG_PATH.read_text())["agent"][
        "format_error_template"
    ]
    combined_command = (
        "git diff -- pkg/a.py > patch.txt && " + EXACT_SUBMISSION_COMMAND
    )
    call = FunctionCall(
        name="bash",
        arguments=json.dumps({"command": combined_command}),
    )
    with pytest.raises(FormatError) as exc_info:
        parse_toolcall_actions(
            [call],
            format_error_template=template,
            toolcall_extract_error_msgs=[],
            enforce_single_tool_call=True,
            enforce_exact_submission_command=True,
        )
    content = exc_info.value.messages[0]["content"]
    assert "The rejected command was not executed." in content
    assert "must create it only" in content
    assert "never use an absolute path" in content


def test_ordinary_format_error_omits_submission_recovery_instructions():
    template = yaml.safe_load(NATIVE_CONFIG_PATH.read_text())["agent"][
        "format_error_template"
    ]
    with pytest.raises(FormatError) as exc_info:
        parse_toolcall_actions(
            [],
            format_error_template=template,
            toolcall_extract_error_msgs=[],
            enforce_single_tool_call=True,
            enforce_exact_submission_command=True,
        )
    content = exc_info.value.messages[0]["content"]
    assert "The rejected command was not executed." not in content


def test_exact_submission_policy_does_not_restrict_ordinary_bash():
    calls = [FunctionCall(name="bash", arguments='{"command":"git diff --stat"}')]
    actions = parse_toolcall_actions(
        calls,
        format_error_template=FORMAT_ERROR_TEMPLATE,
        toolcall_extract_error_msgs=[],
        enforce_single_tool_call=True,
        enforce_exact_submission_command=True,
    )
    assert actions[0]["command"] == "git diff --stat"


def test_shell_constructed_submission_marker_fails_final_gate():
    constructed_marker = 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_"OUTPUT" && cat patch.txt'
    # Static parsing cannot safely predict all shell expansions. The execution
    # path therefore checks this predicate again before accepting final output.
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" not in constructed_marker
    assert not is_exact_submission_command(constructed_marker)
    assert is_exact_submission_command(f"\n{EXACT_SUBMISSION_COMMAND}\n")


@pytest.mark.parametrize("patch", ["", "   \n", "patch.txt not created yet", "garbage\n"])
def test_invalid_submission_patch_rejected(patch):
    assert not is_git_diff_patch(patch)


def test_git_format_submission_patch_accepted():
    patch = """\n diff --git a/pkg/a.py b/pkg/a.py
--- a/pkg/a.py
+++ b/pkg/a.py
@@ -1 +1 @@
-old
+new
"""
    assert is_git_diff_patch(patch)


@pytest.mark.parametrize(
    ("command", "patch", "expected_error_tag"),
    [
        (
            'echo COMPLETE_TASK_AND_SUBMIT_FINAL_"OUTPUT" && cat patch.txt',
            "diff --git a/a.py b/a.py\n",
            "submission_protocol_error",
        ),
        (EXACT_SUBMISSION_COMMAND, "", "submission_patch_error"),
    ],
)
def test_rejected_submission_returns_recoverable_tool_observation(
    monkeypatch, command, patch, expected_error_tag
):
    from recipe.swe.loop_schema import AgentState
    from recipe.swe.swe_agent_loop import SWEAgentLoop

    raw_output = f"COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n{patch}"
    step_response = SimpleNamespace(
        output={
            "output": raw_output,
            "returncode": 0,
            "exception_info": "",
            "extra": {},
        },
        env_raw_output_str=raw_output,
        duration=0.0,
        time_info="0 s",
        timeout=False,
        msg="",
        success=True,
        execute_success=True,
    )

    class FakeEnv:
        async def execute_action(self, **kwargs):
            return step_response

        def check_finished_and_extract_predict_patch(self, output, log_prefix=""):
            return True, patch

    agent_data = SimpleNamespace(
        action_parse_response=SimpleNamespace(
            success=True,
            actions=[{"command": command, "tool_call_id": "bash_0"}],
            observation="",
            msg="",
        ),
        actions=[],
        cwd="/testbed",
        server_duration=0.0,
        env_timeout_cnt=0,
        user_turns=0,
        messages=[],
        prompt_ids=[],
        response_mask=[],
        response_logprobs=[],
        predict_patch="",
        assistant_turns=1,
        error_msg="",
        request_id="native_gate_test",
        group_id=0,
        container_id="container",
    )

    loop = SWEAgentLoop.__new__(SWEAgentLoop)
    loop.env = FakeEnv()
    loop.agent_config = SimpleNamespace(
        use_tool_call=True,
        enforce_exact_submission_command=True,
        enforce_valid_submission_patch=True,
        observation_template="{{ output.output }}",
    )
    loop.tool_call_parser_name = "qwen3_coder"
    loop.action_execute_timeout = 10
    loop.max_env_timeout_cnt = 3
    loop.enable_thinking = False
    loop.apply_chat_template_kwargs = {}
    loop.tokenizer = object()
    loop.response_length = 1024
    loop.max_assistant_turns = 80

    monkeypatch.setattr(
        "recipe.swe.swe_agent_loop.verl_apply_chat_template",
        lambda *args, **kwargs: [101, 102],
    )

    async def run_gate():
        loop.loop = asyncio.get_running_loop()
        return await loop._handle_processing_tools_state(agent_data)

    state = asyncio.run(run_gate())
    assert state == AgentState.GENERATING
    assert agent_data.predict_patch == ""
    assert len(agent_data.messages) == 1
    assert expected_error_tag in agent_data.messages[0]["content"]
    assert agent_data.prompt_ids == [101, 102]
    assert agent_data.response_mask == [0, 0]


def test_qwen_tool_response_incremental_encoding(tokenizer):
    ids = apply_chat_template(
        tokenizer,
        [{"role": "tool", "tool_call_id": "bash_0", "content": "command output"}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    assert decoded.startswith("<|im_start|>user\n<tool_response>\ncommand output")
    assert decoded.count("<tool_response>") == 1
    assert "# Tools" not in decoded
    assert decoded.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_qwen_thinking_tool_response_incremental_encoding(tokenizer):
    ids = apply_chat_template(
        tokenizer,
        [{"role": "tool", "tool_call_id": "bash_0", "content": "command output"}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    assert decoded.startswith("<|im_start|>user\n<tool_response>\ncommand output")
    assert decoded.count("<tool_response>") == 1
    assert "# Tools" not in decoded
    assert decoded.endswith("<|im_start|>assistant\n<think>\n")


def test_qwen35_native_parser_and_template_compatibility(qwen35_tokenizer):
    messages = [
        {"role": "system", "content": "Use the bash tool exactly once."},
        {"role": "user", "content": "Inspect the repository."},
    ]
    prompt_ids = qwen35_tokenizer.apply_chat_template(
        messages,
        tools=[BASH_TOOL],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    decoded_prompt = qwen35_tokenizer.decode(prompt_ids, skip_special_tokens=False)
    assert decoded_prompt.count("# Tools") == 1
    assert decoded_prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")

    text = (
        "<tool_call>\n<function=bash>\n<parameter=command>pwd</parameter>\n"
        "</function>\n</tool_call>"
    )
    parser = ToolParser.get_tool_parser("qwen3_coder", qwen35_tokenizer)
    schema = [OpenAIFunctionToolSchema.model_validate(BASH_TOOL)]
    _, calls = asyncio.run(
        parser.extract_tool_calls(
            qwen35_tokenizer.encode(text, add_special_tokens=False), schema
        )
    )
    actions = parse_toolcall_actions(
        calls,
        format_error_template=FORMAT_ERROR_TEMPLATE,
        toolcall_extract_error_msgs=[],
        enforce_single_tool_call=True,
    )
    assert actions[0]["command"] == "pwd"


def test_hermes_json_in_xml_regression():
    parser = CanopyHermesToolParser(None)
    text = '<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>'
    _, calls, errors = asyncio.run(parser.extract_tool_calls_from_text(text))
    assert errors == []
    assert calls == [FunctionCall(name="bash", arguments='{"command": "pwd"}')]


def test_mswea_text_xml_regression():
    content = "ACTION:\n<mswea_bash_command>pwd</mswea_bash_command>"
    assert re.findall(
        r"<mswea_bash_command>(.*?)</mswea_bash_command>", content, re.DOTALL
    ) == ["pwd"]
