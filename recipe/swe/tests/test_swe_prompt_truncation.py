import asyncio
from types import SimpleNamespace

# Import the verl package first to avoid the recipe-level registration cycle.
from verl.experimental.agent_loop.tool_parser import ToolParser  # noqa: F401

from recipe.swe.loop_schema import AgentData, AgentState
from recipe.swe.step_diagnostics import new_trajectory_diagnostics
from recipe.swe.swe_agent_loop import SWEAgentLoop


class CharacterTokenizer:
    """Small reversible tokenizer for prompt-budget unit tests."""

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, **kwargs):
        del kwargs
        parts = []
        if tools:
            parts.append("<tools>bash</tools>\n")
        for message in messages:
            parts.append(f"<{message['role']}>\n{message['content']}\n</{message['role']}>\n")
        if add_generation_prompt:
            parts.append("<assistant>\n")
        return self.encode("".join(parts), add_special_tokens=False)


def _build_loop(*, prompt_length, enabled, use_tool_call=True):
    loop = SWEAgentLoop.__new__(SWEAgentLoop)
    loop.tokenizer = CharacterTokenizer()
    loop.agent_config = SimpleNamespace(use_tool_call=use_tool_call)
    loop.tool_schemas = [{"type": "function", "function": {"name": "bash"}}]
    loop.enable_thinking = False
    loop.apply_chat_template_kwargs = {}
    loop.prompt_length = prompt_length
    loop.enable_message_aware_prompt_truncation = enabled
    return loop


def _messages(body):
    return [
        {"role": "system", "content": "SYSTEM RULES MUST SURVIVE"},
        {
            "role": "user",
            "content": (
                "<pr_description>\n"
                f"{body}\n"
                "</pr_description>\n"
                "<instructions>EXACT SUBMISSION RULE MUST SURVIVE</instructions>"
            ),
        },
    ]


def test_disabled_flag_keeps_legacy_whole_prompt_left_slice():
    messages = _messages("x" * 100)
    loop = _build_loop(prompt_length=1, enabled=False, use_tool_call=False)
    raw_prompt_ids = loop._tokenize_initial_messages(messages)
    loop.prompt_length = len(raw_prompt_ids) - 5
    agent_data = AgentData(messages=messages, metrics={}, request_id="legacy")

    async def run_pending():
        loop.loop = asyncio.get_running_loop()
        return await loop._handle_pending_state(agent_data, {})

    state = asyncio.run(run_pending())

    assert state == AgentState.GENERATING
    assert agent_data.messages == messages
    assert agent_data.prompt_ids == raw_prompt_ids[-loop.prompt_length :]


def test_enabled_flag_truncates_only_pr_body_and_records_diagnostics():
    body = "HEAD_START\n" + ("repetitive output\n" * 200) + "TAIL_END"
    messages = _messages(body)
    loop = _build_loop(prompt_length=650, enabled=True)
    agent_data = AgentData(messages=messages, metrics={}, request_id="structured")
    agent_data.swe_diagnostics = new_trajectory_diagnostics(True, "validation")

    async def run_pending():
        loop.loop = asyncio.get_running_loop()
        return await loop._handle_pending_state(agent_data, {})

    state = asyncio.run(run_pending())

    assert state == AgentState.GENERATING
    assert len(agent_data.prompt_ids) <= loop.prompt_length
    assert agent_data.messages[0] == messages[0]
    truncated_user = agent_data.messages[1]["content"]
    assert "HEAD_START" in truncated_user
    assert "TAIL_END" in truncated_user
    assert "<truncation_notice>" in truncated_user
    assert "EXACT SUBMISSION RULE MUST SURVIVE" in truncated_user
    assert agent_data.prompt_truncation["removed_task_tokens"] > 0
    assert agent_data.swe_diagnostics["prompt_truncation"] == agent_data.prompt_truncation


def test_structured_truncation_fails_closed_when_fixed_scaffold_does_not_fit():
    loop = _build_loop(prompt_length=1, enabled=True)
    result = loop._truncate_initial_messages_to_prompt_length(_messages("long task"))
    assert result is None


def test_structured_truncation_fails_closed_without_pr_description_boundary():
    loop = _build_loop(prompt_length=100, enabled=True)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "unstructured task"},
    ]
    assert loop._truncate_initial_messages_to_prompt_length(messages) is None
