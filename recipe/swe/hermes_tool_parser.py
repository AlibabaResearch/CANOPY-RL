#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Modified by plm in 2026 for strict SWE tool-call diagnostics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hermes parser variant that returns structured parse diagnostics."""

from __future__ import annotations

import json
import logging
import os

import regex

from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.utils.ray_utils import get_event_loop
from verl.utils.rollout_trace import rollout_trace_op


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class CanopyHermesToolParser:
    """Parse complete Hermes blocks and preserve an error for every failure."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"
        self.tool_call_regex = regex.compile(
            r"<tool_call>(.*?)</tool_call>",
            regex.DOTALL,
        )

    @rollout_trace_op
    async def decode_text_from_response_ids(self, response_ids: list[int]) -> str:
        loop = get_event_loop()
        return await loop.run_in_executor(None, self.tokenizer.decode, response_ids)

    @rollout_trace_op
    async def regex_parse_tool_calls_from_text(
        self,
        text: str,
    ) -> tuple[str, list[FunctionCall], list[str]]:
        function_calls: list[FunctionCall] = []
        errors: list[str] = []
        for match in self.tool_call_regex.findall(text):
            try:
                decoded = json.loads(match, strict=False)
                name = decoded["name"]
                arguments = decoded["arguments"]
                function_calls.append(
                    FunctionCall(
                        name=name,
                        arguments=json.dumps(arguments, ensure_ascii=False),
                    )
                )
            except Exception as exc:
                errors.append(
                    "Failed to decode a Hermes tool call: "
                    f"{exc}. Raw block: [{match}]"
                )

        # SWE accepts only tool actions at this boundary. Any ordinary model
        # content is retained earlier in the trajectory and is not an action.
        return "", function_calls, errors

    @rollout_trace_op
    async def extract_tool_calls(
        self,
        response_ids: list[int],
    ) -> tuple[str, list[FunctionCall], list[str]]:
        model_output = await self.decode_text_from_response_ids(response_ids)
        if self.tool_call_start_token not in model_output:
            return model_output, [], [
                f"No tool call start token {self.tool_call_start_token} found."
            ]
        if self.tool_call_end_token not in model_output:
            return model_output, [], [
                f"No tool call end token {self.tool_call_end_token} found."
            ]
        return await self.regex_parse_tool_calls_from_text(model_output)

    @rollout_trace_op
    async def extract_tool_calls_from_text(
        self,
        text: str,
    ) -> tuple[str, list[FunctionCall], list[str]]:
        return await self.regex_parse_tool_calls_from_text(text)
