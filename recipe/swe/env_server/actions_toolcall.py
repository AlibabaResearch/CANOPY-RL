"""Parse actions & format observations with toolcalls"""

import json
import time

from jinja2 import StrictUndefined, Template

from .exceptions import FormatError

# from minisweagent.exceptions import FormatError
# from minisweagent.models.utils.openai_multimodal import expand_multimodal_content

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}

SUBMISSION_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
EXACT_SUBMISSION_COMMAND = f"echo {SUBMISSION_MARKER} && cat patch.txt"


def is_exact_submission_command(command: object) -> bool:
    """Return whether a bash action is the one allowed final submission command."""

    return isinstance(command, str) and command.strip() == EXACT_SUBMISSION_COMMAND


def is_git_diff_patch(patch: object) -> bool:
    """Reject empty or obvious non-patch submission payloads before evaluation."""

    return isinstance(patch, str) and patch.lstrip().startswith("diff --git ")


def parse_toolcall_actions(
    tool_calls: list,
    *,
    format_error_template: str,
    toolcall_extract_error_msgs: list,
    enforce_single_tool_call: bool = False,
    enforce_exact_bash_arguments: bool = False,
    enforce_exact_submission_command: bool = False,
) -> list[dict]:
    """Parse tool calls from the response. Raises FormatError if unknown tool or invalid args."""
    if not tool_calls:
        error_msg = "No tool calls found in the response. Every response MUST include at least one tool call."
        if toolcall_extract_error_msgs:
            error_msg += " Additionally, there were errors in extracting tool calls: " + "; ".join(toolcall_extract_error_msgs)
        raise FormatError(
            {
                "role": "user",
                "content": Template(format_error_template, undefined=StrictUndefined).render(
                    error=error_msg,
                    actions=[],
                ),
                "extra": {"interrupt_type": "FormatError"},
            }
        )
    if enforce_single_tool_call and len(tool_calls) != 1:
        error_msg = f"Expected exactly 1 tool call, found {len(tool_calls)}."
        raise FormatError(
            {
                "role": "user",
                "content": Template(format_error_template, undefined=StrictUndefined).render(
                    error=error_msg,
                    actions=tool_calls,
                ),
                "extra": {"interrupt_type": "FormatError"},
            }
        )
    # print("=== TOOL CALLS ===")
    # print(tool_calls)
    actions = []
    for idx, tool_call in enumerate(tool_calls):
        error_msg = ""
        diagnostic_reason = "native_tool_protocol_error"
        args = {}
        # print(tool_call)
        try:
            args = json.loads(tool_call.arguments)
            # print(args)
        except Exception as e:
            error_msg = f"Error parsing tool call arguments: {e}."
        t_id = getattr(tool_call, 'id', None) or f"{tool_call.name}_{idx}"
        if tool_call.name != "bash":
            error_msg += f"Unknown tool '{tool_call.name}'."
        if not isinstance(args, dict) or "command" not in args:
            error_msg += "Missing 'command' argument in bash tool call."
        if enforce_exact_bash_arguments and isinstance(args, dict):
            unexpected_arguments = sorted(set(args) - {"command"})
            if unexpected_arguments:
                error_msg += (
                    "Unexpected bash tool call arguments: "
                    f"{unexpected_arguments}. Only 'command' is allowed."
                )
        command = args.get("command") if isinstance(args, dict) else None
        if isinstance(args, dict) and "command" in args:
            if not isinstance(command, str) or not command.strip():
                error_msg += (
                    "The 'command' argument in a bash tool call must be a "
                    "non-empty string."
                )
        if (
            enforce_exact_submission_command
            and isinstance(command, str)
            and SUBMISSION_MARKER in command
            and not is_exact_submission_command(command)
        ):
            diagnostic_reason = "submission_protocol_rejected"
            error_msg += (
                "The final submission command must be exactly: "
                f"{EXACT_SUBMISSION_COMMAND}"
            )
        if error_msg:
            raise FormatError(
                {
                    "role": "user",
                    "content": Template(format_error_template, undefined=StrictUndefined).render(
                        actions=[], error=error_msg.strip()
                    ),
                    "extra": {
                        "interrupt_type": "FormatError",
                        "diagnostic_reason": diagnostic_reason,
                    },
                }
            )

        actions.append({"command": args["command"], "tool_call_id": t_id})
    # print(json.dumps(actions, indent=2))
    return actions


def format_toolcall_observation_messages(
    *,
    actions: list[dict],
    outputs: list[dict],
    observation_template: str,
    template_vars: dict | None = None,
    multimodal_regex: str = "",
) -> list[dict]:
    """Format execution outputs into tool result messages."""
    not_executed = {"output": "", "returncode": -1, "exception_info": "action was not executed"}
    padded_outputs = outputs + [not_executed] * (len(actions) - len(outputs))
    results = []
    for action, output in zip(actions, padded_outputs):
        content = Template(observation_template, undefined=StrictUndefined).render(
            output=output, **(template_vars or {})
        )
        msg = {
            "content": content,
            "extra": {
                "raw_output": output.get("output", ""),
                "returncode": output.get("returncode"),
                "timestamp": time.time(),
                "exception_info": output.get("exception_info"),
                **output.get("extra", {}),
            },
        }
        tool_call_id = action.get("tool_call_id", "")
        if tool_call_id:
            msg["tool_call_id"] = action["tool_call_id"]
            msg["role"] = "tool"
        else:
            msg["role"] = "user"  # human issued commands
        # print(f"tool_call_id: {tool_call_id}, role: {msg['role']}", flush=True)
        # if multimodal_regex:
            # msg = expand_multimodal_content(msg, pattern=multimodal_regex)
        results.append(msg)
    return results
