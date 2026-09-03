#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@desc agent loop schemas
@author: plm
@create: 2026-03-03
"""


from enum import Enum
from typing import Any, Optional


from recipe.swe.env_server.schemas import (ActionParseResponse,
                                           EnvEvaluateResponse,
                                           EnvInitResponse, EnvStepResponse,
                                           ServerStatusCodes)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    ERROR = "error"
    SKIPPED = "skipped"
    FINISHED = "finished"
    TIMEOUT = "timeout"


class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        metrics: dict[str, Any],
        request_id: str,
    ):
        self.messages = messages
        self.metrics = metrics
        self.request_id = request_id

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []
        self.actions: list[dict] = []  # Extracted actions from tool calls

        # env action
        self.action_parse_response: Optional[ActionParseResponse] = None

        self.reward_score = 0.0
        self.error_msg = ""
        self.env_timeout_cnt = 0  # 环境交互超时次数
        # Consecutive identical actions are tracked per trajectory so a model
        # cannot hold an entire validation batch behind an unbounded polling
        # loop.  The thresholds remain recipe-configurable.
        self.last_action_command: Optional[str] = None
        self.last_action_result_fingerprint: Optional[str] = None
        self.consecutive_identical_action_count = 0
        self.server_duration = 0.0 # 服务器执行时间累计
        self.llm_duration = 0.0 # llm 执行时间累计
        self.predict_patch: Optional[str] = ""  # 预测的补丁内容
        self.patch_is_valid = True  # 预测的补丁是否有效
        self.group_id = 0
        self.cwd = "/testbed"
        self.container_id = ""
        # Kept only while an evaluation environment exists so the outer
        # trajectory-timeout handler can clean it up after cancellation.
        self.active_eval_env = None
        self.is_validation = False
        # Optional, trajectory-local SWE diagnostics.  The dictionary is
        # created only when trainer.swe_step_diagnostics.enable=true and is
        # shared across the outer run/error handling and the inner agent loop.
        self.swe_diagnostics: Optional[dict[str, Any]] = None
