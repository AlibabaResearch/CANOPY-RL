#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Validation helpers for Qwen3 native XML-like tool calls.

The generic verl Qwen parser intentionally accepts partial streaming output.
SWE executes shell commands, so its policy must reject truncated tool calls
instead of executing a partially generated command.
"""

import re


_COMPLETE_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_COMPLETE_FUNCTION_RE = re.compile(
    r"\s*<function=([^>\r\n]+)>(.*?)</function>\s*", re.DOTALL
)
_COMPLETE_PARAMETER_RE = re.compile(
    r"<parameter=([^>\r\n]+)>(.*?)</parameter>", re.DOTALL
)
_STOP_TOKEN_SUFFIX_RE = re.compile(
    r"(?:(?:<\|im_end\|>|<\|endoftext\|>)\s*)*\Z"
)


def validate_complete_qwen3_tool_calls(
    model_output: str, *, require_thinking_close: bool = False
) -> list[str]:
    """Return validation errors for incomplete Qwen3 native tool calls.

    This validates the native envelope and requires the final call to end the
    response. Tool names, argument names and argument types are validated later
    by the SWE action parser.
    """

    errors: list[str] = []
    tool_call_starts = model_output.count("<tool_call>")
    tool_call_ends = model_output.count("</tool_call>")
    if tool_call_starts == 0:
        return ["No native <tool_call> block was found."]

    if require_thinking_close:
        thinking_prefix = model_output.split("<tool_call>", 1)[0]
        if "<think>" in thinking_prefix:
            errors.append(
                "Invalid thinking response: the opening <think> tag is already "
                "present in the prompt and must not be generated again."
            )
        thinking_ends = thinking_prefix.count("</think>")
        if thinking_ends != 1:
            errors.append(
                "Incomplete thinking response: expected exactly one </think> before "
                f"the tool call, found {thinking_ends}."
            )
    if tool_call_starts != tool_call_ends:
        errors.append(
            "Incomplete native tool call: found "
            f"{tool_call_starts} opening and {tool_call_ends} closing tool_call tags."
        )

    complete_tool_calls = _COMPLETE_TOOL_CALL_RE.findall(model_output)
    if len(complete_tool_calls) != tool_call_starts:
        errors.append(
            "Incomplete native tool call: not every tool_call block is fully closed."
        )

    for index, tool_call_body in enumerate(complete_tool_calls):
        function_match = _COMPLETE_FUNCTION_RE.fullmatch(tool_call_body)
        if function_match is None:
            errors.append(
                f"Invalid native tool call {index}: expected one fully closed function block."
            )
            continue

        function_body = function_match.group(2)
        parameter_starts = function_body.count("<parameter=")
        parameter_ends = function_body.count("</parameter>")
        complete_parameters = _COMPLETE_PARAMETER_RE.findall(function_body)
        if parameter_starts != parameter_ends or len(complete_parameters) != parameter_starts:
            errors.append(
                f"Incomplete native tool call {index}: every parameter must be fully closed."
            )

    # Qwen's native contract allows ordinary assistant text before a call, but
    # the call must be the final response content. Stop tokens may still be
    # present because this validator sees the non-skip-special-tokens decode.
    if complete_tool_calls:
        trailing_content = model_output.rsplit("</tool_call>", 1)[-1].strip()
        if trailing_content and not _STOP_TOKEN_SUFFIX_RE.fullmatch(trailing_content):
            errors.append(
                "Invalid native tool call: no text is allowed after </tool_call>."
            )

    # Keep messages concise when several count checks describe the same truncation.
    return list(dict.fromkeys(errors))
