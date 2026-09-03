#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@desc SWE-Agent Loop
@author: plm
@create: 2026-01-30
"""


import asyncio
import copy
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from enum import Enum
from os.path import join
from pathlib import Path, PurePosixPath
from typing import Any, Optional
from uuid import uuid4

import ray
import yaml
from loguru import logger
from omegaconf.listconfig import ListConfig
from transformers import AutoProcessor, AutoTokenizer

from recipe.swe.env_server.actions_text import format_observation_messages
from recipe.swe.env_server.actions_toolcall import (
    BASH_TOOL,
    EXACT_SUBMISSION_COMMAND,
    format_toolcall_observation_messages,
    is_exact_submission_command,
    is_git_diff_patch,
)
from recipe.swe.env_server.config import AgentConfig, DockerEnvironmentConfig
from recipe.swe.env_server.dependency_mirrors import (
    resolve_dependency_mirror_policy,
)
from recipe.swe.env_server.exceptions import FormatError
from recipe.swe.env_server.schemas import (ActionParseResponse,
                                           EnvEvaluateResponse,
                                           EnvInitResponse, EnvStepResponse,
                                           ServerStatusCodes)
from recipe.swe.env_server.swe_env_client import SWEEnvClient
from recipe.swe.env_server.swe_utils import (Timer, get_instance_docker_image,
                                             load_yaml_config_from_file_path,
                                             resolve_swe_repo_dir)
from recipe.swe.env_server.test_spec import (SWEProTestSpec, TestSpec,
                                             make_swebench_pro_test_spec,
                                             make_test_spec)
from recipe.swe.hermes_tool_parser import CanopyHermesToolParser
from recipe.swe.loop_schema import AgentData, AgentState
from recipe.swe.qwen_native_toolcall import validate_complete_qwen3_tool_calls
from recipe.swe.step_diagnostics import (
    finalize_trajectory_diagnostics,
    new_trajectory_diagnostics,
    record_action_timing,
    record_event,
    record_timing,
    set_terminal_outcome,
)
from verl.experimental.agent_loop.agent_loop import (AgentLoopBase,
                                                     AgentLoopOutput,
                                                     DictConfigWrap, register)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.chat_template import apply_chat_template as verl_apply_chat_template
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.tokenizer import normalize_token_ids

# logger.add(sys.stderr, format="{time} | {level} | {message}")

logger.remove()  # 先移除默认的 handler
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


class _RolloutGlobalTimeoutError(Exception):
    """Internal marker that distinguishes the rollout deadline from eval errors."""


def get_agent_data_log_prefix(agent_data):
    group_id = 0
    if agent_data is None:
        request_id = "no_request_id"
        assistant_turns = "no_turn_info"
        container_id = ""
    else:
        request_id = agent_data.request_id
        assistant_turns = agent_data.assistant_turns
        group_id = agent_data.group_id
        container_id = agent_data.container_id
    if container_id:
        log_prefix = f"{request_id}, cid={container_id[:10]}, group_id={group_id}, turn={assistant_turns}"
    else:
        log_prefix = f"{request_id}, group_id={group_id}, turn={assistant_turns}"
    return log_prefix


def show_log(agent_data, msg):
    log_prefix = get_agent_data_log_prefix(agent_data)
    log_msg = f"{log_prefix}, {msg}"
    logger.info(log_msg)
    # print(log_msg, flush=True)
    return log_msg


def show_error(agent_data, msg):
    log_prefix = get_agent_data_log_prefix(agent_data)
    log_msg = f"{log_prefix} [ERROR] {msg}"
    logger.error(log_msg)
    return log_msg


def _append_tool_exception(output: dict[str, Any], notice: str) -> None:
    """Append a structured notice to the next tool observation in-place."""

    existing = str(output.get("exception_info", "") or "").strip()
    output["exception_info"] = "\n".join(part for part in (existing, notice.strip()) if part)


def _build_action_timeout_notice(
    timeout_seconds: int,
    count: int,
    limit: int,
    terminal_reward: float | None = None,
) -> str:
    remaining = max(limit - count, 0)
    reward_notice = (
        f" Terminal reward at the limit: {terminal_reward:g}."
        if terminal_reward is not None
        else ""
    )
    return (
        "<action_timeout>"
        f"The previous command exceeded the environment hard limit of {timeout_seconds} seconds "
        "and was killed. "
        f"Timeout count: {count}/{limit}; remaining before termination: {remaining}. "
        "Do not immediately repeat the unchanged command. Use a narrower test/build target, "
        "inspect existing progress, or choose a faster command. A shell timeout value cannot "
        "extend the environment hard limit. When the limit is reached, this trajectory is "
        f"terminated without evaluation.{reward_notice}"
        "</action_timeout>"
    )


def _build_repeated_action_notice(count: int, limit: int) -> str:
    termination = (
        f"If the next execution produces the same result, the trajectory terminates "
        f"when the count reaches {limit}. "
        if limit > 0
        else "The repeated-action termination guard is disabled, but this repetition is recorded. "
    )
    return (
        "<repeated_action_warning>"
        f"The same command produced the same result {count} consecutive times. "
        f"{termination}"
        "Do not repeat it unchanged; inspect the result, change strategy, or submit the patch."
        "</repeated_action_warning>"
    )


def _action_result_fingerprint(output: dict[str, Any]) -> str:
    """Hash the stable parts of an Action result for no-progress detection."""

    payload = "\0".join(
        (
            str(output.get("returncode", "")),
            str(output.get("exception_info", "") or "").strip(),
            str(output.get("output", "") or "").strip(),
        )
    )
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()


@register("swe_agent")
class SWEAgentLoop(AgentLoopBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.config
        self.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        self.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self.response_length = config.actor_rollout_ref.rollout.response_length
        self.max_env_timeout_cnt = config.actor_rollout_ref.swe_custom_config.max_env_timeout_cnt
        self.agent_config_path = config.actor_rollout_ref.swe_custom_config.agent_config_path
        self.env_config_path = config.actor_rollout_ref.swe_custom_config.env_config_path
        self.agent_config = AgentConfig(**load_yaml_config_from_file_path(self.agent_config_path)["agent"])
        self.env_config_dict = load_yaml_config_from_file_path(self.env_config_path)["environment"]
        self.env_config: Optional[DockerEnvironmentConfig] = None  # 延迟初始化
        self.env: Optional[SWEEnvClient] = None  # 延迟初始化
        self.test_spec: Optional[TestSpec | SWEProTestSpec] = None
        self.instance : Optional[dict] = None
        self.tool_schemas = [ BASH_TOOL ]
        self.tool_schema_models = [OpenAIFunctionToolSchema.model_validate(BASH_TOOL)]
        self.tool_call_parser_name = str(
            config.actor_rollout_ref.rollout.multi_turn.get("format", "hermes") or "hermes"
        )
        self.tool_parser = CanopyHermesToolParser(self.tokenizer)
        if self.agent_config.use_tool_call and self.tool_call_parser_name != "hermes":
            self.tool_parser = ToolParser.get_tool_parser(self.tool_call_parser_name, self.tokenizer)
        self.enable_thinking = config.actor_rollout_ref.swe_custom_config.enable_thinking
        self.env_start_timeout = config.actor_rollout_ref.swe_custom_config.env_start_timeout
        self.action_execute_timeout = config.actor_rollout_ref.swe_custom_config.action_execute_timeout
        self.eval_timeout = config.actor_rollout_ref.swe_custom_config.eval_timeout
        self.swe_bench_pro_eval_timeout = int(
            config.actor_rollout_ref.swe_custom_config.get(
                "swe_bench_pro_eval_timeout", 3600
            )
        )
        self.eval_env_start_timeout = config.actor_rollout_ref.swe_custom_config.eval_env_start_timeout
        self.max_env_start_sleep_seconds = config.actor_rollout_ref.swe_custom_config.max_env_start_sleep_seconds
        self.default_timeout_reward_score = config.actor_rollout_ref.swe_custom_config.default_timeout_reward_score
        self.default_no_patch_reward_score = config.actor_rollout_ref.swe_custom_config.default_no_patch_reward_score
        self.default_apply_patch_failed_reward_score = config.actor_rollout_ref.swe_custom_config.default_apply_patch_failed_reward_score
        self.default_terminated_reward_score = config.actor_rollout_ref.swe_custom_config.default_terminated_reward_score
        self.default_no_eval_reward_score = config.actor_rollout_ref.swe_custom_config.default_no_eval_reward_score
        self.min_valid_patch_length = 10
        self.env_resource_tokens = config.actor_rollout_ref.swe_custom_config.env_resource_tokens
        self.env_cpu_limit = str(config.actor_rollout_ref.swe_custom_config.env_cpu_limit)
        self.env_mem_limit = str(config.actor_rollout_ref.swe_custom_config.env_mem_limit)
        self.ray_env_actor_num_cpus = config.actor_rollout_ref.swe_custom_config.ray_env_actor_num_cpus
        self.enable_lxcfs_cpu_view = bool(
            config.actor_rollout_ref.swe_custom_config.get(
                "enable_lxcfs_cpu_view", True
            )
        )
        self.use_sparse_reward = config.actor_rollout_ref.swe_custom_config.use_sparse_reward
        swe_custom_config = config.actor_rollout_ref.swe_custom_config
        self.container_gc_enabled = bool(
            swe_custom_config.get("container_gc_enabled", False)
        )
        self.container_gc_workers_per_node = int(
            swe_custom_config.get("container_gc_workers_per_node", 1)
        )
        self.container_gc_queue_maxsize = int(
            swe_custom_config.get("container_gc_queue_maxsize", 4096)
        )
        self.container_gc_remove_timeout_seconds = float(
            swe_custom_config.get("container_gc_remove_timeout_seconds", 120.0)
        )
        self.container_gc_max_retries = int(
            swe_custom_config.get("container_gc_max_retries", 3)
        )
        self.container_gc_retry_backoff_seconds = float(
            swe_custom_config.get("container_gc_retry_backoff_seconds", 2.0)
        )
        self.container_gc_enqueue_timeout_seconds = float(
            swe_custom_config.get("container_gc_enqueue_timeout_seconds", 5.0)
        )
        self.container_gc_drain_timeout_seconds = float(
            swe_custom_config.get("container_gc_drain_timeout_seconds", 180.0)
        )
        self.dependency_mirror_policy = resolve_dependency_mirror_policy(
            swe_custom_config
        )
        self.max_rollout_trajectory_timeout = int(
            swe_custom_config.max_rollout_trajectory_timeout
        )
        self.repeated_action_warning_threshold = int(
            swe_custom_config.get("repeated_action_warning_threshold", 0)
        )
        self.repeated_action_termination_threshold = int(
            swe_custom_config.get("repeated_action_termination_threshold", 0)
        )
        if self.action_execute_timeout <= 0:
            raise ValueError("action_execute_timeout must be positive")
        if self.eval_timeout <= 0 or self.swe_bench_pro_eval_timeout <= 0:
            raise ValueError("evaluation timeouts must be positive")
        if self.max_env_timeout_cnt <= 0:
            raise ValueError("max_env_timeout_cnt must be positive")
        if self.max_rollout_trajectory_timeout <= 0:
            raise ValueError("max_rollout_trajectory_timeout must be positive")
        if (
            self.repeated_action_warning_threshold < 0
            or self.repeated_action_termination_threshold < 0
        ):
            raise ValueError("repeated-action thresholds must be non-negative")
        if (
            self.repeated_action_termination_threshold > 0
            and self.repeated_action_warning_threshold
            >= self.repeated_action_termination_threshold
        ):
            raise ValueError(
                "repeated-action warning threshold must be smaller than "
                "the termination threshold"
            )
        self.reset_git_log = config.actor_rollout_ref.swe_custom_config.reset_git_log
        self.map_testbed_to_tmpfs = config.actor_rollout_ref.swe_custom_config.map_testbed_to_tmpfs
        self.disable_manual_write_patch_cmd = bool(
            config.actor_rollout_ref.swe_custom_config.get(
                "disable_manual_write_patch_cmd", True
            )
        )
        self.drop_action_parse_failed_assistant_message = bool(
            config.actor_rollout_ref.swe_custom_config.get(
                "drop_action_parse_failed_assistant_message", False
            )
        )
        self.enable_message_aware_prompt_truncation = bool(
            config.actor_rollout_ref.swe_custom_config.get(
                "enable_message_aware_prompt_truncation", False
            )
        )
        if (
            self.agent_config.use_tool_call
            and self.tool_call_parser_name == "qwen3_coder"
            and self.drop_action_parse_failed_assistant_message
        ):
            raise ValueError(
                "qwen3_coder requires "
                "actor_rollout_ref.swe_custom_config."
                "drop_action_parse_failed_assistant_message=False; rolling back the "
                "assistant output would leave an invalid incremental chat boundary."
            )
        self.strip_thinking_for_action_parse = bool(
            config.actor_rollout_ref.swe_custom_config.get(
                "strip_thinking_for_action_parse", True
            )
        )
        diagnostics_config = config.trainer.get("swe_step_diagnostics", {})
        self.enable_step_diagnostics = bool(diagnostics_config.get("enable", False))
        return

    async def _extract_tool_calls(
        self, response_ids: list[int]
    ) -> tuple[str, list[FunctionCall], list[str]]:
        """Adapt legacy Hermes and native verl parsers to the SWE parser contract."""

        if self.tool_call_parser_name == "hermes":
            return await self.tool_parser.extract_tool_calls(response_ids)

        if self.tool_call_parser_name == "qwen3_coder":
            model_output = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(response_ids, skip_special_tokens=False),
            )
            validation_errors = validate_complete_qwen3_tool_calls(
                model_output, require_thinking_close=self.enable_thinking
            )
            if validation_errors:
                return model_output, [], validation_errors

        try:
            content, tool_calls = await self.tool_parser.extract_tool_calls(
                response_ids, self.tool_schema_models
            )
        except Exception as exc:
            return "", [], [
                f"Tool parser '{self.tool_call_parser_name}' failed: {type(exc).__name__}: {exc}"
            ]

        errors = []
        if not tool_calls:
            errors.append(
                f"Tool parser '{self.tool_call_parser_name}' found no valid tool calls."
            )
        return content, tool_calls, errors

    def build_fake_agent_data(self, request_id=None, reward_score=0.0) -> AgentData:
        """Build a fake AgentData for testing purposes."""
        agent_data = AgentData(
                messages=[],
                metrics={},
                request_id=request_id,
            )
        # prompt ids 是整体的
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        agent_data.prompt_ids = [pad_token_id]*800
        agent_data.response_ids = [pad_token_id]*400
        # agent_data.response_mask = [pad_token_id]*200
        agent_data.response_mask = [1] + [0]*399
        # agent_data.response_logprobs = [0] * 200
        agent_data.assistant_turns = 1
        agent_data.user_turns = 1
        agent_data.reward_score = reward_score
        agent_data.turn_scores = []
        agent_data.tool_rewards = []
        return agent_data

    def check_need_eval(self, agent_data: AgentData, state: AgentState) -> tuple[bool, str]:
        predict_patch = agent_data.predict_patch
        patch_length = 0
        if predict_patch:
            patch_length = len(predict_patch)
        if state == AgentState.FINISHED:
            if patch_length < self.min_valid_patch_length:
                return False, f"Agent finished but predict patch length: {patch_length}."
            return True, "Agent finished with a predict patch"
        if state == AgentState.TERMINATED:
            return False, "Agent terminated without finishing"
        if state == AgentState.SKIPPED:
            return False, f"Agent in state {state}"
        if state == AgentState.ERROR:
            return False, f"Agent in state {state}"
        if state == AgentState.TIMEOUT:
            return False, f"Agent in state {state}"
        if predict_patch:
            return True, "predict patch is not empty"
        return False, "predict patch is empty"

    async def _cleanup_eval_env(
        self,
        agent_data: AgentData,
        eval_env: SWEEnvClient,
        *,
        high_priority: bool = False,
        reason: str = "eval_completed",
    ) -> None:
        """Best-effort cleanup shared by normal, error, and cancellation paths."""

        try:
            await eval_env.kill_ray_actor(
                high_priority=high_priority,
                reason=reason,
            )
        except asyncio.CancelledError:
            await eval_env.kill_ray_actor(
                high_priority=True,
                reason="eval_cancelled",
                wait_for_gc_ack=False,
            )
            if getattr(agent_data, "active_eval_env", None) is eval_env:
                agent_data.active_eval_env = None
            raise
        except Exception:
            logger.warning("Failed to clean up eval environment", exc_info=True)
        if getattr(agent_data, "active_eval_env", None) is eval_env:
            agent_data.active_eval_env = None

    async def go_eval(self, agent_data: AgentData, state: AgentState, eval_response: EnvEvaluateResponse, validate=False, use_sparse_reward=True) -> EnvEvaluateResponse:
        """直接进入评估流程，跳过环境交互"""
        if not hasattr(agent_data, "swe_diagnostics"):
            agent_data.swe_diagnostics = None
        need_eval, reason = self.check_need_eval(agent_data, state)
        default_no_eval_reward_score = 0.0
        default_no_patch_reward_score = self.default_no_patch_reward_score
        default_apply_patch_failed_reward_score = self.default_apply_patch_failed_reward_score
        default_timeout_reward_score = self.default_timeout_reward_score
        default_terminated_reward_score = self.default_terminated_reward_score
        default_no_eval_reward_score = self.default_no_eval_reward_score
        patch_length = len(agent_data.predict_patch)
        if state == AgentState.FINISHED:
            if patch_length < self.min_valid_patch_length:
                set_terminal_outcome(
                    agent_data.swe_diagnostics,
                    "patch_empty",
                    stage="submission",
                    owner="model",
                    detail=f"patch length {patch_length} < {self.min_valid_patch_length}",
                )
                eval_response.reward_score = default_no_patch_reward_score if not validate else default_no_eval_reward_score
                eval_response.success = True
                msg = f"patch empty: set reward_score={eval_response.reward_score}, {reason}, error_msg: {agent_data.error_msg}"
                eval_response.msg = msg
                eval_response.report["msg"] = msg
                eval_response.skipped = False
                eval_response.report["model_patch"] = agent_data.predict_patch
                eval_response.time_info = "0 s"
                eval_response.did_real_eval = False
                # show_log(agent_data, f"no evaluation: {reason}, error_msg: {agent_data.error_msg}")
                return eval_response

            # 只有正常完成+有predict patch，才会去执行环境评估
            agent_data._final_eval_started_monotonic = time.monotonic()
            eval_env = SWEEnvClient(
                instance=self.instance,
                agent_config=self.agent_config,
                env_config=self.env_config,
                test_spec=self.test_spec,
                container_role="eval",
                request_id=agent_data.request_id,
            )
            agent_data.active_eval_env = eval_env
            try:
                init_response = await eval_env.initialize(
                    timeout=self.eval_env_start_timeout,
                    max_start_sleep_seconds=self.max_env_start_sleep_seconds,
                    log_prefix=f"{agent_data.request_id}, group_id={agent_data.group_id}, eval_init",
                )
            except asyncio.CancelledError:
                await eval_env.kill_ray_actor(
                    high_priority=True,
                    reason="eval_init_cancelled",
                    wait_for_gc_ack=False,
                )
                agent_data.active_eval_env = None
                raise
            except Exception as exc:
                set_terminal_outcome(
                    agent_data.swe_diagnostics,
                    "eval_env_init_failed",
                    stage="eval_env_init",
                    owner="infra",
                    detail=f"{type(exc).__name__}: {exc}",
                )
                await self._cleanup_eval_env(
                    agent_data,
                    eval_env,
                    high_priority=True,
                    reason="eval_init_exception",
                )
                raise

            if not init_response.success:
                # 环境启动失败，跳过该数据，评估失败
                eval_init_outcome = "eval_env_init_timeout" if init_response.timeout else "eval_env_init_failed"
                set_terminal_outcome(
                    agent_data.swe_diagnostics,
                    eval_init_outcome,
                    stage="eval_env_init",
                    owner="infra",
                    detail=init_response.msg,
                )
                eval_response.success = False
                eval_response.reward_score = default_no_eval_reward_score
                eval_response.msg = init_response.msg
                eval_response.skipped = True
                eval_response.report["msg"] = init_response.msg
                eval_response.report["model_patch"] = agent_data.predict_patch
                eval_response.time_info = init_response.time_info
                eval_response.msg = f"eval env: {init_response.msg}"
                # show_msg = f"Eval env init failed, time: {init_response.time_info}, msg: {init_response.msg}"
                # show_error(agent_data, show_msg)
                await self._cleanup_eval_env(
                    agent_data,
                    eval_env,
                    high_priority=True,
                    reason=eval_init_outcome,
                )
                return eval_response

            # show_error(agent_data, show_msg)
            start_eval_timer = Timer(can_print=False)
            show_msg = f"Eval env init success, time: {init_response.time_info}. Starting Eval, patch length: {len(agent_data.predict_patch)}, content preview: {agent_data.predict_patch[:10]}"
            # show_log(agent_data, show_msg)
            log_prefix = get_agent_data_log_prefix(agent_data)
            try:
                eval_timeout = self.eval_timeout
                if self.instance.get("evaluator_type") == "swe_bench_pro":
                    eval_timeout = self.swe_bench_pro_eval_timeout
                eval_response = await eval_env.evaluate(
                    predict_patch=agent_data.predict_patch,
                    timeout=eval_timeout,
                    log_prefix=log_prefix,
                    use_sparse_reward=use_sparse_reward,
                )
            except asyncio.CancelledError:
                await eval_env.kill_ray_actor(
                    high_priority=True,
                    reason="eval_cancelled",
                    wait_for_gc_ack=False,
                )
                agent_data.active_eval_env = None
                raise
            except Exception as exc:
                set_terminal_outcome(
                    agent_data.swe_diagnostics,
                    "eval_execution_failed",
                    stage="evaluation",
                    owner="infra",
                    detail=f"{type(exc).__name__}: {exc}",
                )
                await self._cleanup_eval_env(
                    agent_data,
                    eval_env,
                    high_priority=True,
                    reason="eval_exception",
                )
                raise
            eval_response.time_info = "env init: {}, eval: {}".format(init_response.time_info, eval_response.duration)
            await self._cleanup_eval_env(
                agent_data,
                eval_env,
                reason=str(
                    eval_response.report.get("eval_failure_code", "eval_completed")
                    or "eval_completed"
                ),
            )
            if eval_response.apply_patch_failed:
                # apply patch失败，需要惩罚
                apply_failure_code = str(eval_response.report.get("eval_failure_code", ""))
                set_terminal_outcome(
                    agent_data.swe_diagnostics,
                    "patch_apply_timeout" if apply_failure_code == "patch_apply_timeout" else "patch_apply_failed",
                    stage="apply_patch",
                    owner="evaluator" if apply_failure_code == "patch_apply_timeout" else "model",
                    detail=eval_response.msg,
                )
                eval_response.success = True
                eval_response.reward_score = default_apply_patch_failed_reward_score if not validate else default_no_eval_reward_score
                if agent_data.error_msg:
                    eval_response.msg = f"apply patch error, set reward_score={eval_response.reward_score}, msg: {eval_response.msg}, agent_error_msg: {agent_data.error_msg}"
                else:
                    eval_response.msg = f"apply patch error, set reward_score={eval_response.reward_score}, msg: {eval_response.msg}"
                eval_response.report["msg"] = eval_response.msg
                # show_error(agent_data, eval_response.msg)
                # eval_response.report["msg"] += eval_response.msg
                eval_response.did_real_eval = False
            elif not eval_response.success:
                eval_failure_code = str(eval_response.report.get("eval_failure_code", ""))
                eval_outcome = {
                    "eval_timeout": "eval_timeout",
                    "eval_output_limit": "eval_output_limit",
                    "eval_execution_failed": "eval_execution_failed",
                    "eval_report_missing": "eval_report_missing",
                }.get(eval_failure_code, "eval_execution_failed")
                set_terminal_outcome(
                    agent_data.swe_diagnostics,
                    eval_outcome,
                    stage="evaluation",
                    owner="evaluator",
                    detail=eval_response.msg,
                )
                msg = f"agent_error_msg: {agent_data.error_msg}"
                eval_response.msg += msg
                eval_response.report["msg"] += msg
                eval_response.reward_score = default_no_eval_reward_score
                eval_response.success = False
                # show_log(agent_data, f"Evaluation success, reward_score: {eval_response.reward_score}, f2p: {eval_response.report['f2p_rate']}, p2p: {eval_response.report['p2p_rate']}")
            else:
                eval_response.did_real_eval = not bool(
                    eval_response.report.get("eval_failure_code", "")
                )
            # used_seconds, total_seconds, used_info, total_info = start_eval_timer.tok("Evaluation")
            # eval_time_used = start_eval_timer.get_total_used_seconds()
            # eval_time_used_info = start_eval_timer.get_print_info_by_seconds(eval_time_used)
            # agent_data.server_duration += total_seconds
        elif state == AgentState.SKIPPED:
            # 跳过，需要做fake处理
            eval_response.success = False
            skip_reason = f"data skipped, because: {reason}, agent_error_msg: {agent_data.error_msg}"
            eval_response.report["msg"] = skip_reason
            eval_response.msg = skip_reason
            eval_response.report["model_patch"] = agent_data.predict_patch
            # show_log(agent_data, f"no evaluation: {skip_reason}")
            eval_response.skipped = True
            eval_response.skip_reason = skip_reason
            eval_response.time_info = "0 s"
            eval_response.reward_score = default_no_eval_reward_score
            set_terminal_outcome(
                agent_data.swe_diagnostics,
                "skipped_other",
                stage="rollout",
                owner="infra",
                detail=skip_reason,
            )
        elif state == AgentState.ERROR:
            # 跳过，需要做fake处理
            eval_response.success = False
            skip_reason = f"data skipped, because: {reason}, agent_error_msg: {agent_data.error_msg}"
            eval_response.report["msg"] = skip_reason
            eval_response.msg = skip_reason
            eval_response.report["model_patch"] = agent_data.predict_patch
            # show_log(agent_data, f"no evaluation: {skip_reason}")
            eval_response.skipped = True
            eval_response.skip_reason = skip_reason
            eval_response.time_info = "0 s"
            eval_response.reward_score = default_no_eval_reward_score
            set_terminal_outcome(
                agent_data.swe_diagnostics,
                "rollout_step_infra_failed",
                stage="rollout",
                owner="infra",
                detail=skip_reason,
            )
        elif state == AgentState.TERMINATED:
            eval_response.success = True
            eval_response.reward_score = default_terminated_reward_score if not validate else default_no_eval_reward_score
            msg = f"agent terminated, set reward_score={eval_response.reward_score}, do not run eval,  because: {reason}, agent_error_msg: {agent_data.error_msg}"
            # show_log(agent_data, f"no evaluation: {error_reason}")
            eval_response.report["msg"] = msg
            eval_response.msg = msg
            eval_response.report["model_patch"] = agent_data.predict_patch
            eval_response.skipped = False
            eval_response.time_info = "0 s"
            set_terminal_outcome(
                agent_data.swe_diagnostics,
                "agent_terminated_other",
                stage="rollout",
                owner="model",
                detail=msg,
            )
        elif state == AgentState.TIMEOUT:
            eval_response.success = True
            eval_response.reward_score = default_timeout_reward_score if not validate else default_no_eval_reward_score
            msg = f"agent timeout, set reward_score={eval_response.reward_score}, do not run eval,  because: {reason}, agent_error_msg: {agent_data.error_msg}"
            eval_response.report["msg"] = msg
            eval_response.msg = msg
            eval_response.report["model_patch"] = agent_data.predict_patch
            eval_response.skipped = False
            eval_response.time_info = "0 s"
            set_terminal_outcome(
                agent_data.swe_diagnostics,
                "action_timeout_limit",
                stage="action",
                owner="model",
                detail=msg,
            )
        return eval_response

    def _build_live_agent_data(
        self,
        env: SWEEnvClient,
        request_id: str,
        group_id: int,
        swe_diagnostics=None,
        validate: bool = False,
    ) -> AgentData:
        msg_resp = env.get_init_messages()
        agent_data = AgentData(
            messages=msg_resp.messages,
            metrics={},
            request_id=request_id,
        )
        agent_data.group_id = group_id
        agent_data.cwd = self.env_config.cwd
        agent_data.container_id = env.container_id or ""
        agent_data.swe_diagnostics = swe_diagnostics
        agent_data.is_validation = bool(validate)
        return agent_data

    async def run_core_success_loop(
        self,
        env,
        step_response,
        eval_response,
        sampling_params,
        validate,
        request_id,
        group_id,
        swe_diagnostics=None,
        agent_data=None,
    ):
        if agent_data is None:
            agent_data = self._build_live_agent_data(
                env,
                request_id,
                group_id,
                swe_diagnostics,
                validate,
            )

        # show_log(agent_data, f"SWEEnv initialized successfully, container_id: {env.container_id}, duration: {init_response.duration}, group={self.env_config.group_id}")
        # agent_data.server_duration += init_response.duration
        state = AgentState.PENDING
        was_cancelled = False
        try:
            while state != AgentState.TERMINATED:
                if state == AgentState.PENDING:
                    state = await self._handle_pending_state(agent_data, sampling_params)
                elif state == AgentState.GENERATING:
                    state = await self._handle_generating_state(agent_data, sampling_params)
                elif state == AgentState.PROCESSING_TOOLS:
                    state = await self._handle_processing_tools_state(agent_data)
                elif state == AgentState.ERROR:
                    # 环境执行错误，直接终止
                    step_response.msg = agent_data.error_msg
                    step_response.success = True
                    break
                elif state == AgentState.SKIPPED:
                    # 不走go eval，直接skipp
                    step_response.msg = agent_data.error_msg
                    step_response.success = False
                    break
                elif state == AgentState.FINISHED:
                    # go_eval，去赋值reward
                    step_response.msg = "Task completed successfully"
                    step_response.success = True
                    break
                elif state == AgentState.TIMEOUT:
                    # 超时也走go_eval，去赋值reward
                    step_response.msg = agent_data.error_msg
                    step_response.success = True
                    break
                else:
                    show_error(agent_data, f"Invalid state: {state}")
                    state = AgentState.TERMINATED
                    break

        except asyncio.CancelledError:
            was_cancelled = True
            try:
                await env.kill_ray_actor(
                    high_priority=True,
                    reason="rollout_cancelled",
                    wait_for_gc_ack=False,
                )
            except Exception:
                logger.warning(
                    "Failed to force-kill cancelled rollout environment",
                    exc_info=True,
                )
            raise
        except Exception as e:
            error_msg = "Error during agent loop execution, e: {}".format(e)
            set_terminal_outcome(
                agent_data.swe_diagnostics,
                "rollout_internal_error",
                stage="rollout",
                owner="infra",
                detail=error_msg,
            )
            state = AgentState.ERROR
            step_response.success = False
            step_response.msg = error_msg
            # show_error(agent_data, error_msg)
            # logger.trace()
            if random.random() < 0.1:
                logger.exception(error_msg)
        finally:
            # Always release the interaction environment. Evaluation uses a
            # separate environment and runs outside this rollout-only timeout.
            if not was_cancelled:
                try:
                    abnormal_state = state in {
                        AgentState.ERROR,
                        AgentState.SKIPPED,
                        AgentState.TIMEOUT,
                    }
                    await env.kill_ray_actor(
                        high_priority=abnormal_state,
                        reason=f"rollout_{state.value}",
                    )
                except asyncio.CancelledError:
                    await env.kill_ray_actor(
                        high_priority=True,
                        reason="rollout_cancelled",
                        wait_for_gc_ack=False,
                    )
                    raise
                except Exception:
                    logger.warning("Failed to clean up rollout environment", exc_info=True)
        return agent_data, state, step_response, eval_response

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # messages = list(kwargs["raw_prompt"])
        instance_id = list(kwargs["raw_prompt"])[0]["content"]
        instance = kwargs.get("extra_info", {})
        data_source = kwargs.get("data_source", "swe_unknown")
        data_docker_source = instance.get("data_docker_source", "SWE-bench")
        resolved_repo_dir = resolve_swe_repo_dir(instance)
        # Keep one source of truth for rollout commands, patch application, and
        # the generated evaluation script.  Existing 0815 V2 parquet files
        # contain the legacy /testbed placeholder, so overwrite it at runtime.
        instance["repo_dir"] = resolved_repo_dir
        if instance.get("evaluator_type") == "swe_bench_pro":
            test_spec = make_swebench_pro_test_spec(instance)
        else:
            test_spec = make_test_spec(instance)
        self.test_spec = test_spec
        self.instance = instance
        validate = kwargs.get("validate", False)
        image_name = get_instance_docker_image(instance_id=instance_id, data_docker_source=data_docker_source, instance=instance)
        # show_log(None, f"Received request for instance_id: {instance_id}, data_source: {data_source}, validate: {validate}, image_name: {image_name}, data_docker_source: {instance.get('data_docker_source', 'SWE-bench')}")


        self.env_config_dict["image"] = image_name
        self.env_config_dict["data_source"] = data_source
        self.env_config = DockerEnvironmentConfig(**self.env_config_dict)
        self.env_config.cwd = resolved_repo_dir
        self.env_config.env_resource_tokens = self.env_resource_tokens
        self.env_config.env_cpu_limit = self.env_cpu_limit
        self.env_config.env_mem_limit = self.env_mem_limit
        self.env_config.ray_env_actor_num_cpus = self.ray_env_actor_num_cpus
        self.env_config.container_gc_enabled = self.container_gc_enabled
        self.env_config.container_gc_workers_per_node = self.container_gc_workers_per_node
        self.env_config.container_gc_queue_maxsize = self.container_gc_queue_maxsize
        self.env_config.container_gc_remove_timeout_seconds = self.container_gc_remove_timeout_seconds
        self.env_config.container_gc_max_retries = self.container_gc_max_retries
        self.env_config.container_gc_retry_backoff_seconds = self.container_gc_retry_backoff_seconds
        self.env_config.container_gc_enqueue_timeout_seconds = self.container_gc_enqueue_timeout_seconds
        self.env_config.container_gc_drain_timeout_seconds = self.container_gc_drain_timeout_seconds
        for name, enabled in self.dependency_mirror_policy.items():
            setattr(self.env_config, name, enabled)
        self.env_config.repo_language = str(
            instance.get("repo_language")
            or instance.get("language")
            or getattr(test_spec, "language", "")
            or ""
        )
        self.env_config.enable_lxcfs_cpu_view = self.enable_lxcfs_cpu_view
        self.env_config.map_testbed_to_tmpfs = self.map_testbed_to_tmpfs
        self.env_config.disable_manual_write_patch_cmd = self.disable_manual_write_patch_cmd
        # print(f"env config: { self.env_config}, {self.env_config_dict}")
        group_id = self.instance["group_id"]
        self.env_config.group_id = group_id
        timer = Timer(can_print=False)
        metrics = {}
        request_id = uuid4().hex
        request_id = f"{data_source}_{instance_id}_{request_id}"
        swe_diagnostics = new_trajectory_diagnostics(
            self.enable_step_diagnostics,
            "validation" if validate else "train",
        )

        # fake
        fake_reward_score = 0.0
        fake_agent_data = self.build_fake_agent_data(request_id=request_id, reward_score=fake_reward_score)
        fake_agent_data.swe_diagnostics = swe_diagnostics
        agent_data = copy.deepcopy(fake_agent_data)
        agent_data.swe_diagnostics = swe_diagnostics
        agent_data.group_id = group_id

        error_msg = ""
        # eval_time_used_info = ""

        env = SWEEnvClient(
            instance,
            agent_config=self.agent_config,
            env_config=self.env_config,
            test_spec=self.test_spec,
            container_role="rollout",
            request_id=request_id,
        )
        self.env = env

        init_log_prefix = f"{request_id}, group_id={group_id}, init"
        rollout_env_init_started = time.monotonic()
        try:
            try:
                init_response = await env.initialize(
                    timeout=self.env_start_timeout,
                    max_start_sleep_seconds=self.max_env_start_sleep_seconds,
                    log_prefix=init_log_prefix,
                )
            except asyncio.CancelledError:
                await env.kill_ray_actor(
                    high_priority=True,
                    reason="rollout_init_cancelled",
                    wait_for_gc_ack=False,
                )
                raise
            except Exception:
                await env.kill_ray_actor(
                    high_priority=True,
                    reason="rollout_init_exception",
                )
                raise
        finally:
            record_timing(
                swe_diagnostics,
                "rollout_env_init_seconds",
                time.monotonic() - rollout_env_init_started,
            )
        # show_log(agent_data, f"env init success={init_response.success}, group={self.env_config.group_id}, container_id: {env.container_id}, duration={init_response.duration}, msg={init_response.msg}")


        step_response = EnvStepResponse(success=True, msg="")
        eval_response = EnvEvaluateResponse(success=True, msg="")
        state = AgentState.ERROR
        is_fake_output = False

        if not init_response.success:
            rollout_init_outcome = (
                "rollout_env_init_timeout" if init_response.timeout else "rollout_env_init_failed"
            )
            set_terminal_outcome(
                swe_diagnostics,
                rollout_init_outcome,
                stage="rollout_env_init",
                owner="infra",
                detail=init_response.msg,
            )
            await env.kill_ray_actor(
                high_priority=True,
                reason=rollout_init_outcome,
            )

        if init_response.success:
            # if "SWE-bench_Verified" not in data_source:
            if self.reset_git_log:
                # 去掉git log，防止reward hacking，抄作业
                try:
                    reset_res = await env.reset_git_log(
                        cwd=self.env_config.cwd,
                        log_prefix=init_log_prefix,
                    )
                except asyncio.CancelledError:
                    await env.kill_ray_actor(
                        high_priority=True,
                        reason="reset_git_log_cancelled",
                        wait_for_gc_ack=False,
                    )
                    raise
                except Exception as exc:
                    init_response.success = False
                    init_response.msg = (
                        f"reset log raised {type(exc).__name__}: {exc}"
                    )
                    set_terminal_outcome(
                        swe_diagnostics,
                        "reset_git_log_failed",
                        stage="rollout_env_init",
                        owner="infra",
                        detail=init_response.msg,
                    )
                    await env.kill_ray_actor(
                        high_priority=True,
                        reason="reset_git_log_exception",
                    )
                else:
                    if not reset_res.success:
                        # Treat reset failure as an environment-init failure.
                        init_response.success = False
                        init_response.msg = f"reset log failed: {reset_res.msg}"
                        set_terminal_outcome(
                            swe_diagnostics,
                            "reset_git_log_failed",
                            stage="rollout_env_init",
                            owner="infra",
                            detail=init_response.msg,
                        )
                        await env.kill_ray_actor(
                            high_priority=True,
                            reason="reset_git_log_failed",
                        )
            else:
                pass

        if init_response.success:
            # show_log(agent_data, f"env init success, time: {init_response.time_info}")
            # 核心loop
            # --- 核心改进：全局硬超时保护 ---
            timeout = self.max_rollout_trajectory_timeout
            try:
                agent_data = self._build_live_agent_data(
                    env,
                    request_id,
                    group_id,
                    swe_diagnostics,
                    validate,
                )
                # 使用 wait_for 包裹核心业务逻辑
                interaction_started = time.monotonic()
                try:
                    try:
                        agent_data, state, step_response, eval_response = await asyncio.wait_for(
                            self.run_core_success_loop(
                                env,
                                step_response,
                                eval_response,
                                sampling_params,
                                validate,
                                request_id,
                                group_id,
                                swe_diagnostics,
                                agent_data,
                            ),
                            timeout=timeout,
                        )
                    except asyncio.TimeoutError as exc:
                        raise _RolloutGlobalTimeoutError from exc
                finally:
                    record_timing(
                        swe_diagnostics,
                        "interaction_seconds",
                        time.monotonic() - interaction_started,
                    )

                use_sparse_reward = True if validate else self.use_sparse_reward
                try:
                    eval_response = await self.go_eval(
                        agent_data,
                        state,
                        eval_response,
                        validate=validate,
                        use_sparse_reward=use_sparse_reward,
                    )
                finally:
                    final_eval_started = getattr(
                        agent_data, "_final_eval_started_monotonic", None
                    )
                    if final_eval_started is not None:
                        record_timing(
                            swe_diagnostics,
                            "final_eval_seconds",
                            time.monotonic() - final_eval_started,
                        )
            except _RolloutGlobalTimeoutError:
                show_error(agent_data, f"CRITICAL: Global timeout triggered after {timeout}s")
                step_response.success = False
                step_response.msg = f"Global Timeout after {timeout}s"
                agent_data.error_msg = step_response.msg
                state = AgentState.SKIPPED
                set_terminal_outcome(
                    swe_diagnostics,
                    "rollout_global_timeout",
                    stage="rollout",
                    owner="model",
                    detail=step_response.msg,
                )
                if swe_diagnostics is not None:
                    swe_diagnostics["rollout_progress"] = {
                        "assistant_turns": int(agent_data.assistant_turns),
                        "user_turns": int(agent_data.user_turns),
                        "context_tokens": int(len(agent_data.prompt_ids)),
                        "response_tokens": int(len(agent_data.response_mask)),
                        "server_duration_seconds": float(agent_data.server_duration),
                        "llm_duration_seconds": float(agent_data.llm_duration),
                        "container_id": str(agent_data.container_id),
                    }
                try:
                    await env.kill_ray_actor(
                        high_priority=True,
                        reason="rollout_global_timeout",
                    )
                    show_log(agent_data, "Rollout environment queued for urgent cleanup.")
                except Exception:
                    logger.warning(
                        "Failed to release rollout environment after timeout",
                        exc_info=True,
                    )
            except Exception as e:
                show_error(agent_data, f"Core loop uncaught error: {e}")
                step_response.success = False
                step_response.msg = f"Core loop uncaught error: {e}"
                set_terminal_outcome(
                    swe_diagnostics,
                    "rollout_internal_error",
                    stage="rollout",
                    owner="infra",
                    detail=step_response.msg,
                )
                active_eval_env = getattr(agent_data, "active_eval_env", None)
                if active_eval_env is not None:
                    await self._cleanup_eval_env(agent_data, active_eval_env)
                try:
                    await env.kill_ray_actor(
                        high_priority=True,
                        reason="rollout_internal_error",
                    )
                except Exception:
                    logger.warning("Failed to clean up rollout environment", exc_info=True)
            # agent_data, state, step_response, eval_response = self.run_core_success_loop(env, step_response, eval_response, sampling_params, validate)
        used_seconds, total_seconds, used_info, total_info = timer.tok("finished")
        record_timing(
            swe_diagnostics,
            "trajectory_wall_seconds",
            total_seconds,
        )
        agent_data.server_duration = round(agent_data.server_duration, 2)
        agent_data.llm_duration = round(agent_data.llm_duration, 2)
        env_time_info = timer.get_print_info_by_seconds(agent_data.server_duration)
        llm_time_info = timer.get_print_info_by_seconds(agent_data.llm_duration)
        env_start_time_info = init_response.time_info if init_response else "0 s"
        total_time_info = f"total_time: {total_info}, init_time: {env_start_time_info}, env_step_time: {env_time_info}, llm_time: {llm_time_info}, eval_time: {eval_response.time_info}"

        if init_response.success:
            if step_response.success:
                if eval_response.success:
                    agent_data.metrics = eval_response.report
                    agent_data.reward_score = eval_response.reward_score
                    # show_log(agent_data, msg=f"metrics: {agent_data.metrics}")
                    basic_reward_msg = f"reward_score: {eval_response.reward_score}, f2p: {eval_response.report['f2p_rate']}, p2p: {eval_response.report['p2p_rate']}, turns: {agent_data.assistant_turns}, length: {len(agent_data.prompt_ids)}, time: {total_time_info}"
                    if eval_response.did_real_eval:
                        show_log(agent_data, msg=f"evaluate success, real_eval, {basic_reward_msg}")
                    else:
                        show_log(agent_data, msg=f"evaluate success, set_eval, {eval_response.msg}, {basic_reward_msg}")

                    # show_log(agent_data, msg=f"metrics: {agent_data.metrics}")
                else:
                    show_error(agent_data, f"fake data, evaluate failed: {eval_response.msg}, time: {total_time_info}")
                    agent_data = fake_agent_data
                    is_fake_output = True
            else:
                show_error(agent_data, f"fake data, step failed: {step_response.msg}, time: {total_time_info}")
                eval_response.report["msg"] = step_response.msg
                agent_data = fake_agent_data
                is_fake_output = True
        else:
            error_msg = f"fake data, rollout env: {init_response.msg}, time: {init_response.time_info}"
            eval_response.report["msg"] = init_response.msg
            show_error(agent_data, error_msg)
            # await self.env.close()
            agent_data = fake_agent_data
            is_fake_output = True

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        eval_response.report["request_id"] = request_id
        prompt_truncation = getattr(agent_data, "prompt_truncation", None)
        if prompt_truncation is not None:
            eval_response.report["prompt_truncation"] = prompt_truncation
        diagnostic_patch = eval_response.report.get("model_patch", agent_data.predict_patch or "")
        finalized_diagnostics = finalize_trajectory_diagnostics(
            swe_diagnostics,
            report=eval_response.report,
            agent_state=state.value,
            did_real_eval=eval_response.did_real_eval,
            is_fake=is_fake_output,
            request_id=request_id,
            task_id=instance_id,
            patch_length=len(diagnostic_patch or ""),
            reward_score=agent_data.reward_score,
        )
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            reward_score=agent_data.reward_score,
            multi_modal_data={},
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics={},
            extra_fields={"report": eval_response.report},
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        if finalized_diagnostics is not None:
            output.extra_fields["swe_diagnostics"] = finalized_diagnostics

        return output

    def _tokenize_initial_messages(self, messages: list[dict[str, Any]]) -> list[int]:
        """Render an initial chat prompt exactly as the legacy pending path does."""
        template_kwargs = {
            "add_generation_prompt": True,
            "tokenize": True,
            "enable_thinking": self.enable_thinking,
            "return_dict": False,
            **self.apply_chat_template_kwargs,
        }
        if self.agent_config.use_tool_call:
            template_kwargs["tools"] = self.tool_schemas
        return self.tokenizer.apply_chat_template(messages, **template_kwargs)

    @staticmethod
    def _find_pr_description_span(
        messages: list[dict[str, Any]],
    ) -> Optional[tuple[int, int, int]]:
        """Locate the task body while leaving the system prompt and instructions intact."""
        open_tag = "<pr_description>"
        close_tag = "</pr_description>"
        for message_index, message in enumerate(messages):
            if message.get("role") != "user" or not isinstance(message.get("content"), str):
                continue
            content = message["content"]
            body_start = content.find(open_tag)
            if body_start < 0:
                continue
            body_start += len(open_tag)
            body_end = content.rfind(close_tag, body_start)
            if body_end >= body_start:
                return message_index, body_start, body_end
        return None

    def _build_middle_truncated_text(
        self,
        token_ids: list[int],
        retained_tokens: int,
    ) -> str:
        """Keep the beginning and end of a task body and mark the omitted middle."""
        marker = (
            "\n\n<truncation_notice>Middle of the PR description was omitted "
            "to fit the initial prompt token budget.</truncation_notice>\n\n"
        )
        retained_tokens = max(0, min(int(retained_tokens), len(token_ids)))
        head_tokens = int(retained_tokens * 0.6)
        tail_tokens = retained_tokens - head_tokens
        if retained_tokens and head_tokens == 0:
            head_tokens, tail_tokens = 1, retained_tokens - 1

        head = self.tokenizer.decode(
            token_ids[:head_tokens],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        tail = self.tokenizer.decode(
            token_ids[-tail_tokens:] if tail_tokens else [],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return f"{head}{marker}{tail}"

    def _truncate_initial_messages_to_prompt_length(
        self,
        messages: list[dict[str, Any]],
    ) -> Optional[tuple[list[dict[str, Any]], list[int], dict[str, Any]]]:
        """Truncate only the PR body, then re-render a complete chat template.

        The fixed prompt scaffold (system message, tool schema, chat boundaries, and
        task instructions) is never sliced.  ``None`` means even that fixed scaffold
        cannot fit, or the expected SWE task boundary is unavailable.
        """
        span = self._find_pr_description_span(messages)
        if span is None:
            return None

        message_index, body_start, body_end = span
        truncated_messages = copy.deepcopy(messages)
        content = truncated_messages[message_index]["content"]
        body = content[body_start:body_end]
        body_token_ids = self.tokenizer.encode(body, add_special_tokens=False)

        def render(retained_tokens: int) -> tuple[list[dict[str, Any]], list[int]]:
            candidate_messages = copy.deepcopy(truncated_messages)
            candidate_content = candidate_messages[message_index]["content"]
            candidate_body = self._build_middle_truncated_text(
                body_token_ids, retained_tokens
            )
            candidate_messages[message_index]["content"] = (
                candidate_content[:body_start]
                + candidate_body
                + candidate_content[body_end:]
            )
            return candidate_messages, self._tokenize_initial_messages(candidate_messages)

        # Measure the immutable scaffold first. This also proves that a safe
        # message-level truncation is possible before we attempt to fill the budget.
        minimal_messages, minimal_prompt_ids = render(0)
        if len(minimal_prompt_ids) > self.prompt_length:
            return None

        retained_tokens = min(
            len(body_token_ids),
            max(self.prompt_length - len(minimal_prompt_ids), 0),
        )
        candidate_messages, candidate_prompt_ids = render(retained_tokens)
        while len(candidate_prompt_ids) > self.prompt_length and retained_tokens > 0:
            overflow = len(candidate_prompt_ids) - self.prompt_length
            retained_tokens = max(retained_tokens - max(overflow, 1), 0)
            candidate_messages, candidate_prompt_ids = render(retained_tokens)

        if len(candidate_prompt_ids) > self.prompt_length:
            # ``minimal_prompt_ids`` already fit, so this is only a defensive
            # fallback for a tokenizer with unusual non-monotonic boundaries.
            candidate_messages, candidate_prompt_ids = minimal_messages, minimal_prompt_ids
            retained_tokens = 0

        truncation_info = {
            "strategy": "pr_description_head_tail",
            "head_fraction": 0.6,
            "original_task_tokens": len(body_token_ids),
            "retained_task_tokens": retained_tokens,
            "removed_task_tokens": len(body_token_ids) - retained_tokens,
            "final_prompt_tokens": len(candidate_prompt_ids),
        }
        return candidate_messages, candidate_prompt_ids, truncation_info

    async def _handle_pending_state_with_message_aware_prompt_truncation(
        self,
        agent_data: AgentData,
    ) -> AgentState:
        """Prepare an overlong initial prompt without breaking message boundaries."""
        raw_prompt_ids = await self.loop.run_in_executor(
            None, self._tokenize_initial_messages, agent_data.messages
        )
        if len(raw_prompt_ids) <= self.prompt_length:
            agent_data.prompt_ids = raw_prompt_ids
            return AgentState.GENERATING

        truncation_result = await self.loop.run_in_executor(
            None,
            self._truncate_initial_messages_to_prompt_length,
            agent_data.messages,
        )
        if truncation_result is None:
            msg = (
                "task skipped, message-aware prompt truncation could not fit the "
                f"fixed prompt scaffold, raw/max: {len(raw_prompt_ids)}/{self.prompt_length}"
            )
            agent_data.error_msg = msg
            set_terminal_outcome(
                agent_data.swe_diagnostics,
                "prompt_too_long",
                stage="prompt",
                owner="infra",
                detail=msg,
            )
            return AgentState.SKIPPED

        truncated_messages, prompt_ids, truncation_info = truncation_result
        truncation_info["original_prompt_tokens"] = len(raw_prompt_ids)
        agent_data.messages = truncated_messages
        agent_data.prompt_ids = prompt_ids
        agent_data.prompt_truncation = truncation_info
        if agent_data.swe_diagnostics is not None:
            agent_data.swe_diagnostics["prompt_truncation"] = truncation_info
        show_log(
            agent_data,
            "Initial prompt message-aware truncation, raw/now/max/task_removed: "
            f"{len(raw_prompt_ids)}/{len(prompt_ids)}/{self.prompt_length}/"
            f"{truncation_info['removed_task_tokens']}.",
        )
        return AgentState.GENERATING

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        if not hasattr(agent_data, "swe_diagnostics"):
            agent_data.swe_diagnostics = None
        if self.enable_message_aware_prompt_truncation:
            return await self._handle_pending_state_with_message_aware_prompt_truncation(
                agent_data
            )
        # show_log(agent_data, f"use_tool_call: {self.agent_config.use_tool_call}")
        if self.agent_config.use_tool_call:
            agent_data.prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    agent_data.messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=True,
                    enable_thinking=self.enable_thinking,
                    return_dict=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
        else:
            agent_data.prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    agent_data.messages,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                    tokenize=True,
                    return_dict=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
        buffer_size = 1024
        if len(agent_data.prompt_ids) > self.prompt_length + buffer_size:
            msg = f"task skipped, prompt length {len(agent_data.prompt_ids)} exceeds max {self.prompt_length}, skip this task"
            # show_log(agent_data, msg)
            # show_log(agent_data, agent_data.messages[1])
            agent_data.error_msg = msg
            set_terminal_outcome(
                agent_data.swe_diagnostics,
                "prompt_too_long",
                stage="prompt",
                owner="model",
                detail=msg,
            )
            return AgentState.SKIPPED
        elif len(agent_data.prompt_ids) > self.prompt_length:
            raw_prompt_length = len(agent_data.prompt_ids)
            agent_data.prompt_ids = agent_data.prompt_ids[-self.prompt_length :]
            show_log(agent_data, f"Initial prompt length, task prompt truncated from left, raw/now/max: {raw_prompt_length}/{len(agent_data.prompt_ids)}/{self.prompt_length}.")
        else:
            # show_log(agent_data, f"Initial prompt length: {len(agent_data.prompt_ids)}, assistant_turn={agent_data.assistant_turns},")
            pass
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False,
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        if not hasattr(agent_data, "swe_diagnostics"):
            agent_data.swe_diagnostics = None
        # 检查限制，但实际上执行时，应该是在llm生成action，执行环境以后，去终止的。这里一般不会生效。
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            msg = f"AgentLoop-Terminated-By-ResponseLength-LLMResp[loop_response>response_length] {len(agent_data.response_mask)} >= {self.response_length}"
            set_terminal_outcome(
                agent_data.swe_diagnostics,
                "response_length_limit",
                stage="rollout",
                owner="model",
                detail=msg,
            )
            # show_log(agent_data, msg)
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            msg = f"AgentLoop-Terminated-By-AssistantTurns {agent_data.assistant_turns} >= {self.max_assistant_turns}"
            set_terminal_outcome(
                agent_data.swe_diagnostics,
                "assistant_turn_limit",
                stage="rollout",
                owner="model",
                detail=msg,
            )
            # show_log(agent_data, msg)
            return AgentState.TERMINATED

        add_messages: list[dict[str, Any]] = []
        timer = Timer(can_print=False)
        llm_started = time.monotonic()
        try:
            with simple_timer("generate_sequences", agent_data.metrics):
                output = await self.server_manager.generate(
                    request_id=agent_data.request_id,
                    prompt_ids=agent_data.prompt_ids,
                    sampling_params=sampling_params,
                )
        finally:
            time_used = time.monotonic() - llm_started
            agent_data.llm_duration += time_used
            record_timing(
                agent_data.swe_diagnostics,
                "llm_seconds",
                time_used,
                accumulate=True,
            )
        time_used_info = timer.get_print_info_by_seconds(time_used)
        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        prev_prompt_len = len(agent_data.prompt_ids)
        prev_response_mask_len = len(agent_data.response_mask)
        prev_response_logprobs_len = len(agent_data.response_logprobs)
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        # Append the assistant message after decoding with skip_special_tokens=True.
        llm_response = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
        # show_log(agent_data, f"LLM generation completed, duration: {time_used_info}s, tokens: {len(output.token_ids)}")
        # show_log(agent_data, f"LLM generation completed, duration: {time_used_info}s, tokens: {len(output.token_ids)}, message={llm_response[:300]}")
        # show_log(agent_data, f"LLM generation completed, duration: {time_used_info}s, tokens: {len(output.token_ids)}, message={llm_response}")

        llm_output = await self.loop.run_in_executor(None, self.tokenizer.decode, output.token_ids)
        # Extract tool calls
        # show_log(agent_data, f"LLM generation completed, Parsed action: success={agent_data.action_parse_response.success}, action={agent_data.action_parse_response.action}, observation={agent_data.action_parse_response.observation}")
        if self.agent_config.use_tool_call:
            # add_messages.append({"role": "assistant", "content": llm_response})
            # agent_data.messages.extend(add_messages)
            _, agent_data.tool_calls, toolcall_extract_error_msgs = await self._extract_tool_calls(
                agent_data.response_ids
            )
            agent_data.action_parse_response = self.env.parse_actions_by_tool_calls(
                tool_calls=agent_data.tool_calls,
                toolcall_extract_error_msgs=toolcall_extract_error_msgs
            )
        else:
            parse_content = llm_response
            if self.enable_thinking and self.strip_thinking_for_action_parse and "</think>" in parse_content:
                parse_content = parse_content.rsplit("</think>", 1)[-1]
            agent_data.action_parse_response = self.env.parse_actions_by_text(
                content=parse_content,
                action_regex=self.agent_config.action_regex,
                format_error_template=self.agent_config.format_error_template,
            )

        if agent_data.action_parse_response.success:
            # action解析成功，自然添加保留llm response
            add_messages.append({"role": "assistant", "content": llm_response})
            agent_data.messages.extend(add_messages)
        else:
            # action 解析失败，默认不保留llm response
            if self.drop_action_parse_failed_assistant_message:
                agent_data.prompt_ids = agent_data.prompt_ids[:prev_prompt_len]
                agent_data.response_mask = agent_data.response_mask[:prev_response_mask_len]
                if agent_data.response_logprobs:
                    agent_data.response_logprobs = agent_data.response_logprobs[:prev_response_logprobs_len]
            else:
                add_messages.append({"role": "assistant", "content": llm_response})
                agent_data.messages.extend(add_messages)

        # show_log(agent_data, f"LLM Generation, Parsed action={agent_data.action_parse_response.success}: duration: {time_used_info}, actions={len(agent_data.action_parse_response.actions)}...")
        # agent_data.env_action = await parse_action_merge_all(llm_output)
        # show_log(agent_data, f"env_action: {agent_data.env_action}")
        return AgentState.PROCESSING_TOOLS


    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        if not hasattr(agent_data, "swe_diagnostics"):
            agent_data.swe_diagnostics = None

        action_parse_response = agent_data.action_parse_response
        agent_data.actions = action_parse_response.actions
        timer = Timer(can_print=False)
        add_messages = []
        if not action_parse_response.success:
            # llm的输出格式错误，直接返回错误信息，然后算一轮交互
            raw_action_count = len(agent_data.tool_calls) if self.agent_config.use_tool_call else 0
            record_event(agent_data.swe_diagnostics, "action_attempt", max(raw_action_count, 1))
            record_event(agent_data.swe_diagnostics, "action_parse_failed")
            if self.tool_call_parser_name == "qwen3_coder":
                record_event(agent_data.swe_diagnostics, "native_tool_protocol_error")
            if action_parse_response.diagnostic_reason == "submission_protocol_rejected":
                record_event(agent_data.swe_diagnostics, "submission_attempt")
                record_event(agent_data.swe_diagnostics, "submission_protocol_rejected")
            observation = action_parse_response.observation
            agent_data.user_turns += 1
            time_used = timer.get_total_used_seconds()
            time_used_info = timer.get_print_info_by_seconds(time_used)
            show_error(agent_data, f"action parse failed: {action_parse_response.msg}")
            add_messages = [{"role": "user", "content": observation}]
            agent_data.last_action_command = None
            agent_data.last_action_result_fingerprint = None
            agent_data.consecutive_identical_action_count = 0
        else:
            # 执行环境交互
            parsed_action_count = len(agent_data.actions)
            # A parser invocation with no action is still one attempted Action,
            # but it is not counted as successfully parsed.
            record_event(agent_data.swe_diagnostics, "action_attempt", max(parsed_action_count, 1))
            record_event(agent_data.swe_diagnostics, "action_parse_success", parsed_action_count)
            log_prefix = get_agent_data_log_prefix(agent_data)
            outputs = []
            is_finished, predict_patch = False, ""
            for action in agent_data.actions:
                command = action["command"]
                normalized_command = command.strip()
                last_action_command = getattr(agent_data, "last_action_command", None)
                last_result_fingerprint = getattr(
                    agent_data, "last_action_result_fingerprint", None
                )
                previous_identical_count = int(
                    getattr(agent_data, "consecutive_identical_action_count", 0)
                )
                same_command = normalized_command == last_action_command
                warning_threshold = int(
                    getattr(self, "repeated_action_warning_threshold", 0)
                )
                repeated_action_limit = int(
                    getattr(self, "repeated_action_termination_threshold", 0)
                )
                submission_intent = is_exact_submission_command(command)
                record_event(
                    agent_data.swe_diagnostics, "action_execute_attempt"
                )
                action_started = time.monotonic()
                try:
                    step_response = await self.env.execute_action(
                        action=action,
                        cwd=agent_data.cwd,
                        log_prefix=log_prefix,
                        timeout=self.action_execute_timeout,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    record_event(
                        agent_data.swe_diagnostics, "action_system_error"
                    )
                    raise
                finally:
                    action_elapsed = time.monotonic() - action_started
                    agent_data.server_duration += action_elapsed
                    record_action_timing(
                        agent_data.swe_diagnostics, action_elapsed
                    )
                outputs.append(step_response.output)
                is_finished, predict_patch = self.env.check_finished_and_extract_predict_patch(step_response.output, log_prefix=log_prefix)
                env_raw_output_str = step_response.env_raw_output_str

                output_extra = step_response.output.get("extra", {}) or {}
                is_output_limit = output_extra.get("killed_by") == "size"
                killed_by = str(output_extra.get("killed_by", "") or "")
                is_wall_timeout = killed_by == "timeout" or bool(
                    step_response.timeout and not killed_by and not is_output_limit
                )
                result_fingerprint = _action_result_fingerprint(step_response.output)
                if is_wall_timeout:
                    # Wall-timeout retries are governed by max_env_timeout_cnt;
                    # do not let the repeated-result guard preempt the explicit
                    # 1/3, 2/3, 3/3 feedback contract.
                    repeated_action_count = 0
                    agent_data.last_action_command = None
                    agent_data.last_action_result_fingerprint = None
                elif same_command and result_fingerprint == last_result_fingerprint:
                    repeated_action_count = previous_identical_count + 1
                    agent_data.last_action_command = normalized_command
                    agent_data.last_action_result_fingerprint = result_fingerprint
                else:
                    repeated_action_count = 1
                    agent_data.last_action_command = normalized_command
                    agent_data.last_action_result_fingerprint = result_fingerprint
                agent_data.consecutive_identical_action_count = repeated_action_count
                if repeated_action_count > 1:
                    record_event(agent_data.swe_diagnostics, "repeated_action")

                if not step_response.success and not is_wall_timeout and not is_output_limit:
                    record_event(agent_data.swe_diagnostics, "action_system_error")
                elif (
                    not step_response.execute_success
                    and not is_wall_timeout
                    and not is_output_limit
                ):
                    record_event(agent_data.swe_diagnostics, "action_nonzero_exit")
                if (
                    step_response.success
                    and step_response.execute_success
                    and not step_response.timeout
                    and output_extra.get("killed_by") != "size"
                    and output_extra.get("reason") != "manual_patch_writing"
                ):
                    record_event(agent_data.swe_diagnostics, "action_execute_success")
                if output_extra.get("killed_by") == "size":
                    record_event(agent_data.swe_diagnostics, "action_output_limit")
                if output_extra.get("reason") == "manual_patch_writing":
                    record_event(agent_data.swe_diagnostics, "manual_patch_write_blocked")
                if (
                    repeated_action_limit > 0
                    and repeated_action_count >= repeated_action_limit
                ):
                    error_msg = (
                        "repeated action terminated after the same command produced "
                        f"the same result {repeated_action_count} consecutive times "
                        f"(limit={repeated_action_limit}), action: [{command[:100]}]..."
                    )
                    agent_data.error_msg = error_msg
                    set_terminal_outcome(
                        agent_data.swe_diagnostics,
                        "repeated_action_limit",
                        stage="action",
                        owner="model",
                        detail=error_msg,
                    )
                    show_error(agent_data, error_msg)
                    return AgentState.TIMEOUT
                if (
                    warning_threshold > 0
                    and repeated_action_count >= warning_threshold
                    and (repeated_action_limit <= 0 or repeated_action_count < repeated_action_limit)
                ):
                    _append_tool_exception(
                        step_response.output,
                        _build_repeated_action_notice(
                            repeated_action_count,
                            repeated_action_limit,
                        ),
                    )
                if submission_intent or is_finished:
                    record_event(agent_data.swe_diagnostics, "submission_attempt")

                # The parser gives a friendly early rejection when the marker is
                # literal. This post-execution gate also covers shell quoting,
                # variables, escapes, or any other command that constructs the
                # marker dynamically.
                if (
                    is_finished
                    and self.agent_config.enforce_exact_submission_command
                    and not is_exact_submission_command(command)
                ):
                    record_event(agent_data.swe_diagnostics, "submission_protocol_rejected")
                    is_finished, predict_patch = False, ""
                    protocol_error = (
                        "<submission_protocol_error>Submission output was ignored. "
                        "Invoke bash with this exact command and no additions: "
                        f"{EXACT_SUBMISSION_COMMAND}</submission_protocol_error>"
                    )
                    raw_output = step_response.output.get("output", "")
                    step_response.output["output"] = (
                        f"{raw_output.rstrip()}\n\n{protocol_error}\n"
                    )
                elif (
                    is_finished
                    and self.agent_config.enforce_valid_submission_patch
                    and not is_git_diff_patch(predict_patch)
                ):
                    if not str(predict_patch or "").strip():
                        record_event(agent_data.swe_diagnostics, "submission_patch_empty_rejected")
                    else:
                        record_event(agent_data.swe_diagnostics, "submission_patch_invalid_rejected")
                    is_finished, predict_patch = False, ""
                    patch_error = (
                        "<submission_patch_error>Submission output was ignored "
                        "because patch.txt was empty or was not a git-format diff. "
                        "Recreate patch.txt with the instructed `git diff -- ... > "
                        "patch.txt`, inspect it, then submit again.</submission_patch_error>"
                    )
                    raw_output = step_response.output.get("output", "")
                    step_response.output["output"] = (
                        f"{raw_output.rstrip()}\n\n{patch_error}\n"
                    )

                # 处理打印的message
                basic_step_msg = ""
                timeout_msg = ""
                if is_wall_timeout:
                    agent_data.env_timeout_cnt += 1
                    record_event(agent_data.swe_diagnostics, "action_timeout")
                    if submission_intent:
                        record_event(agent_data.swe_diagnostics, "submission_timeout")
                    timeout_msg = f"timeout: {step_response.timeout}, timeoutcnt: {agent_data.env_timeout_cnt}/{self.max_env_timeout_cnt}"
                    _append_tool_exception(
                        step_response.output,
                        _build_action_timeout_notice(
                            int(self.action_execute_timeout),
                            int(agent_data.env_timeout_cnt),
                            int(self.max_env_timeout_cnt),
                            float(
                                self.default_no_eval_reward_score
                                if bool(getattr(agent_data, "is_validation", False))
                                else self.default_timeout_reward_score
                            ),
                        ),
                    )
                finish_msg = ""
                if is_finished:
                    finish_msg = f"is_finished: {is_finished}, predict_patch: {len(predict_patch)} chars"
                duration_msg = f"duration: {step_response.time_info}"
                env_detail_msg = ""
                if step_response.msg:
                    env_detail_msg = f"step_msg: {step_response.msg[:100]}"
                basic_step_msg = ", ".join(filter(None, [duration_msg, timeout_msg, finish_msg, env_detail_msg]))
                show_msg = ""
                if step_response.success:
                    cmd_msg = f"action: [{command[:20]}]...,  output: [{env_raw_output_str[:20]}]..."
                    # if random.random() < 0:
                    #     show_msg = f"action success: real_execute_success: {step_response.execute_success}, {basic_step_msg}, {cmd_msg}"
                    #     show_log(agent_data, show_msg)
                else:
                    cmd_msg = f"action: [{command[:100]}]..."
                    show_msg = f"action failed: {basic_step_msg}, {cmd_msg}"
                    show_error(agent_data, show_msg)

                if is_wall_timeout:
                    if agent_data.env_timeout_cnt >= self.max_env_timeout_cnt:
                        error_msg = (
                            "agent timeout terminated: "
                            f"{agent_data.env_timeout_cnt}/{self.max_env_timeout_cnt} commands "
                            f"exceeded the {self.action_execute_timeout}s hard limit. {cmd_msg}"
                        )
                        agent_data.error_msg = error_msg
                        set_terminal_outcome(
                            agent_data.swe_diagnostics,
                            "action_timeout_limit",
                            stage="action",
                            owner="model",
                            detail=error_msg,
                        )
                        return AgentState.TIMEOUT
                else:
                    pass

                if is_finished:
                    record_event(agent_data.swe_diagnostics, "accepted_submission")
                    agent_data.predict_patch = predict_patch
                    if len(predict_patch) < self.min_valid_patch_length:
                        agent_data.error_msg = f"Patch Exception: patch length {len(predict_patch)}, raw_output: {step_response.env_raw_output_str}, action: {command}"
                        # agent_data.patch_is_valid = False
                        # show_error(agent_data, agent_data.error_msg)
                        pass
                    else:
                        # agent_data.patch_is_valid = True
                        # show_log(agent_data, f"Task Patch Success, patch length: {len(predict_patch)}, action: {command}")
                        pass
                    return AgentState.FINISHED
            if self.agent_config.use_tool_call:
                add_messages = format_toolcall_observation_messages(
                    actions=agent_data.actions,
                    outputs=outputs,
                    observation_template=self.agent_config.observation_template
                )
                # show_log(agent_data, f"add message role: {add_messages[0]['role']}")
            else:
                add_messages = format_observation_messages(
                    outputs=outputs,
                    observation_template=self.agent_config.observation_template
                )
            time_used = timer.get_total_used_seconds()
            time_used_info = timer.get_print_info_by_seconds(time_used)
            agent_data.user_turns += 1

        agent_data.messages.extend(add_messages)
        if self.agent_config.use_tool_call and self.tool_call_parser_name == "qwen3_coder":
            # Qwen3.5/3.6 reject a tool-only message list. verl's wrapper inserts
            # and removes a dummy user message, producing only the incremental
            # <tool_response> turn plus the next assistant generation prefix.
            response_ids = await self.loop.run_in_executor(
                None,
                lambda: verl_apply_chat_template(
                    self.tokenizer,
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    enable_thinking=self.enable_thinking,
                    return_dict=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            response_ids = normalize_token_ids(response_ids)
        else:
            response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    enable_thinking=self.enable_thinking,
                    return_dict=False,
                    **self.apply_chat_template_kwargs
                ),
            )

        if not (
            self.agent_config.use_tool_call
            and self.tool_call_parser_name == "qwen3_coder"
        ):
            response_ids = response_ids[len(self.system_prompt) :]
        # LLM生成以后，必定执行环境，所以在这里检查长度和循环数量，超出则直接终止，去做评估，但是如果没有停止，就没有拿到git patch，也就无需做评估，除非直接基于原环境做评估，但这样可能会有问题，模型修改了单元测试。
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            msg = f"AgentLoop-Terminated-By-ResponseLength-EnvResp[loop_response+env_response>response_length]: {len(agent_data.response_mask)} + {len(response_ids)} >= {self.response_length}"
            # show_log(agent_data, msg)
            agent_data.error_msg = msg
            set_terminal_outcome(
                agent_data.swe_diagnostics,
                "response_length_limit",
                stage="rollout",
                owner="model",
                detail=msg,
            )
            return AgentState.TERMINATED
        if self.max_assistant_turns >= 1 and agent_data.assistant_turns >= self.max_assistant_turns:
            msg = f"AgentLoop-Terminated-By-AssistantTurns {agent_data.assistant_turns} >= {self.max_assistant_turns}"
            # show_log(agent_data, msg)
            agent_data.error_msg = msg
            set_terminal_outcome(
                agent_data.swe_diagnostics,
                "assistant_turn_limit",
                stage="rollout",
                owner="model",
                detail=msg,
            )
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        return AgentState.GENERATING
