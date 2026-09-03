#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""AppWorld agent loop for the current verl async rollout interface."""


import asyncio
import logging
import os
import random
from enum import Enum
from typing import Any
from uuid import uuid4

from recipe.appworld.appworld_utils import parse_action_only_first
from recipe.appworld.env_server.client import (
    AppWorldEnvClient,
    load_global_server_urls,
)
from recipe.appworld.env_server.data_utils import Timer
from recipe.appworld.env_server.schemas import (
    EnvEvaluateResponse,
    EnvStepResponse,
    ServerStatusCodes,
)
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    ERROR = "error"


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

        self.extra_fields: dict[str, Any] = {}

        # env action
        self.env_action = ""
        self.reward_score = 0.0
        self.error_msg = ""
        self.env_time_out_turn_count = 0
        self.server_duration = 0.0
        self.llm_duration = 0.0


def show_log(agent_data, msg):
    request_id = agent_data.request_id
    log_msg = f"{request_id}, turn={agent_data.assistant_turns}: {msg}"
    print(log_msg)
    return log_msg


def action_log_summary(action: str) -> str:
    """Avoid leaking credentials from generated code into logs by default."""
    if os.getenv("APPWORLD_LOG_ACTIONS") == "1":
        return f"action:\n{action}"
    return f"action=<redacted; {len(action)} chars>"


@register("appworld_env_agent")
class AppworldAgentLoop(AgentLoopBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.config

        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.max_tool_response_length = self.rollout_config.multi_turn.max_tool_response_length
        self.response_length = self.rollout_config.response_length

        custom_config = config.actor_rollout_ref.appworld_custom_config
        self.sparse_reward = custom_config.sparse_reward
        self.server_url_config_folder = custom_config.server_url_config_folder
        self.max_completion_tokens = int(
            custom_config.max_completion_tokens
        )
        if self.max_completion_tokens <= 0:
            raise ValueError("appworld_custom_config.max_completion_tokens must be positive")

        self.server_urls = load_global_server_urls(self.server_url_config_folder)
        # 环境最多超时次数，如果超时2次，则放弃该条数据，直接作为error skip掉；如果小于等于2次，则重试，把错误信息返回给LLM，让LLM调整后继续执行
        self.max_env_timeout_turn = int(
            custom_config.get("max_env_timeout_turn", 2)
        )
        # 实验结果的输出路径，训练不用设置，仅在专有评测提交bench时使用
        self.experiments_outputs_directory = custom_config.get(
            "experiments_outputs_directory", None
        )
        # 是否需要删除输出文件夹，训练时都需要删除，仅在提交评测时保留
        self.rm_outdir_after_finished = custom_config.get(
            "rm_outdir_after_finished", True
        )

    async def _truncate_environment_output(
        self, environment_output: str, max_tokens: int
    ) -> str:
        """Limit output-content tokens and append a truncation notice.

        A non-positive limit disables truncation.
        """

        if max_tokens <= 0:
            return environment_output

        def truncate() -> str:
            token_ids = self.tokenizer.encode(
                environment_output,
                add_special_tokens=True,
            )
            if len(token_ids) <= max_tokens:
                return environment_output

            notice = (
                "\n... [Output Truncated by System. "
                f"Total length: {len(token_ids)} tokens > {max_tokens} tokens.]"
            )
            try:
                prefix = self.tokenizer.decode(
                    token_ids[:max_tokens],
                    skip_special_tokens=False,
                )
            except Exception:
                try:
                    prefix = self.tokenizer.decode(
                        token_ids[:max_tokens],
                        skip_special_tokens=True,
                    )
                except Exception:
                    logger.warning(
                        "Tokenizer could not decode truncated environment output; "
                        "falling back to a proportional character prefix",
                        exc_info=True,
                    )
                    ratio = max_tokens / len(token_ids)
                    prefix_length = max(1, int(len(environment_output) * ratio))
                    prefix = environment_output[:prefix_length]
            return prefix + notice

        return await self.loop.run_in_executor(None, truncate)

    def build_fake_agent_data(self, request_id=None, reward_score=0.0) -> AgentData:
        """Build a fake AgentData for testing purposes."""
        agent_data = AgentData(
                messages=[],
                metrics={},
                request_id=request_id,
            )

        # prompt ids 是整体的
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        agent_data.prompt_ids = [pad_token_id]*400
        agent_data.response_ids = [pad_token_id]*200
        agent_data.response_mask = [0] * 200
        # agent_data.response_logprobs = [0] * 200
        agent_data.assistant_turns = 1
        agent_data.user_turns = 1
        agent_data.reward_score = reward_score
        agent_data.turn_scores = []
        agent_data.tool_rewards = []
        return agent_data
    
    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # messages = list(kwargs["raw_prompt"])
        task_id = list(kwargs["raw_prompt"])[0]["content"]
        data_source = kwargs.get("data_source", "appworld_unknown")
        validate = kwargs.get("validate", False)
        timer = Timer(can_print=False)
        metrics = {}
        request_id = uuid4().hex
        request_id = f"{data_source}_{task_id}_{request_id}"
        agent_data = self.build_fake_agent_data(request_id=request_id)
        error_msg = ""
        server_url = random.choice(self.server_urls)
        show_log(agent_data, f"server_url: {server_url}")

        # 目前, experiment_name 就用 request_id，二者等价，AppWorld际上需要的是experiment_name
        experiment_name = request_id
        env = AppWorldEnvClient(task_id=task_id, request_id=request_id, experiment_name=experiment_name, remote_environment_url=server_url, 
                                rm_outdir_after_finished=self.rm_outdir_after_finished, 
                                experiments_outputs_directory=self.experiments_outputs_directory)

        fake_reward_score = 0.0

        # 初始化环境、环境交互执行、环境评估，三个判断指标
        init_response = await env.initialize()
        step_response = EnvStepResponse(success=True, msg="")
        eval_response = EnvEvaluateResponse(success=True, msg="")
        init_duration = init_response.duration

        if init_response.success:
            # 环境初始化成功
            show_log(agent_data, f"AppWorldEnv initialized successfully, time: {init_response.duration}s")
            msg_resp = await env.get_init_messages()
            messages = msg_resp.messages
            agent_data = AgentData(
                messages=messages,
                metrics=metrics,
                request_id=request_id,
            )
            agent_data.server_duration += init_duration
            agent_data.server_duration += msg_resp.duration
            state = AgentState.PENDING
            try:
                if not msg_resp.success or not messages:
                    raise RuntimeError(
                        f"failed to load initial messages: {msg_resp.msg}"
                    )
                while state != AgentState.TERMINATED:
                    if state == AgentState.PENDING:
                        state = await self._handle_pending_state(agent_data, sampling_params)
                    elif state == AgentState.GENERATING:
                        state = await self._handle_generating_state(agent_data, sampling_params)
                    elif state == AgentState.PROCESSING_TOOLS:
                        state = await self._handle_processing_tools_state(agent_data, env)
                    elif state == AgentState.ERROR:
                        # 环境执行错误，直接终止
                        step_response.msg = agent_data.error_msg
                        step_response.success = False
                        break
                    else:
                        logger.error(f"Invalid state: {state}")
                        state = AgentState.TERMINATED
                use_sparse_eval = self.sparse_reward
                if validate:
                    use_sparse_eval = True
                eval_response = await env.evaluate(sparse=use_sparse_eval)
                agent_data.server_duration += eval_response.duration
            except Exception as e:
                error_msg = "Error during agent loop execution, request_id={}, e: {}".format(request_id, e)
                step_response.success = False
                step_response.msg = error_msg
                show_log(agent_data, error_msg)
                # logger.trace()
                if random.random() < 0.1:
                    logger.exception(error_msg)
                # agent_data = self.build_fake_agent_data(request_id=request_id, reward_score=fake_reward_score)
            
            if step_response.success:
                # 环境交互成功，进行评估
                if eval_response.success:
                    # 评估成功
                    agent_data.reward_score = eval_response.reward_score
                else:
                    # 评估
                    error_msg = f"Env evaluate failed: {eval_response.msg}"
                    # show_log(agent_data, error_msg)
                    # agent_data = self.build_fake_agent_data(request_id=request_id, reward_score=fake_reward_score)
            
        else:
            # 环境初始化失败
            error_msg = init_response.msg
            # show_log(agent_data, f"AppWorldEnv initialized failed: {error_msg}")
            # agent_data = self.build_fake_agent_data(request_id=request_id, reward_score=fake_reward_score)

        if init_response.success:
            # 删除本地文件夹等内容
            env_close_response = await env.close()
            if env_close_response.success:
                # show_log(agent_data, f"AppWorldEnv closed successfully")
                pass
            else:
                show_log(agent_data, f"AppWorldEnv close failed: {env_close_response.msg}")
        time_used, time_total = timer.tok("交互完成")
        agent_data.server_duration = round(agent_data.server_duration, 2)
        agent_data.llm_duration = round(agent_data.llm_duration, 2)
        env_time_info = timer.get_print_info_by_seconds(agent_data.server_duration)
        llm_time_info = timer.get_print_info_by_seconds(agent_data.llm_duration)
        time_info = f" total_time: {time_total}, env_time: {env_time_info}, llm_time: {llm_time_info}"
        if init_response.success:
            if step_response.success:
                if eval_response.success:
                    show_log(agent_data, f"execute success, reward_score: {agent_data.reward_score}, time: {time_info}")
                else:
                    show_log(agent_data, f"evaluate failed: {eval_response.msg}, time: {time_info}")
                    agent_data = self.build_fake_agent_data(request_id=request_id, reward_score=fake_reward_score)
            else:
                show_log(agent_data, f"step failed: {step_response.msg}, time: {time_info}")
                agent_data = self.build_fake_agent_data(request_id=request_id, reward_score=fake_reward_score)
        else:
            show_log(agent_data, f"init failed: {error_msg}, time: {time_info}")
            agent_data = self.build_fake_agent_data(request_id=request_id, reward_score=fake_reward_score)

        # Finalize output
        response_start = len(agent_data.prompt_ids) - len(agent_data.response_mask)
        response_ids = agent_data.prompt_ids[response_start:]
        prompt_ids = agent_data.prompt_ids[:response_start]
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
            metrics=agent_data.metrics,
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        agent_data.prompt_ids = await self.apply_chat_template(agent_data.messages)
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        # 检查限制，但实际上执行时，应该是在llm生成action，执行环境以后，去终止的。这里一般不会生效。
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            show_log(agent_data, f"AgentLoop-Terminated-By-ResponseLength-LLMResp[loop_response>response_length] {len(agent_data.response_mask)} >= {self.response_length}")
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            show_log(agent_data, f"AgentLoop-Terminated-By-AssistantTurns {agent_data.assistant_turns} >= {self.max_assistant_turns}")
            return AgentState.TERMINATED

        remaining_tokens = self.response_length - len(agent_data.response_mask)
        if remaining_tokens <= 0:
            return AgentState.TERMINATED
        generation_params = dict(sampling_params)
        requested_limit = generation_params.pop("max_tokens", None)
        requested_limit = generation_params.pop(
            "max_new_tokens", requested_limit
        )
        per_turn_limit = min(self.max_completion_tokens, remaining_tokens)
        if requested_limit is not None:
            per_turn_limit = min(per_turn_limit, int(requested_limit))
        if per_turn_limit <= 0:
            return AgentState.TERMINATED
        generation_params["max_new_tokens"] = per_turn_limit

        add_messages: list[dict[str, Any]] = []
        timer = Timer(can_print=False)
        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=generation_params,
            )
        time_used = timer.get_total_used_seconds()
        agent_data.llm_duration += time_used
        time_used_info = timer.get_print_info_by_seconds(time_used)
        show_log(agent_data, f"LLM generation completed, duration: {time_used_info}s, tokens: {len(output.token_ids)}")
        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs
        output_extra_fields = getattr(output, "extra_fields", None) or {}
        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output_extra_fields)
        else:
            max_global_steps = output_extra_fields.get("max_global_steps")
            if max_global_steps is not None:
                agent_data.extra_fields["max_global_steps"] = max_global_steps

        # Append the assistant message after decoding with skip_special_tokens=True.
        assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
        add_messages.append({"role": "assistant", "content": assistant_message})
        agent_data.messages.extend(add_messages)
        # Extract tool calls
        agent_data.env_action = await parse_action_only_first(assistant_message)
        return AgentState.PROCESSING_TOOLS
       

    async def _handle_processing_tools_state(self, agent_data: AgentData, env: AppWorldEnvClient) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""

        # 访问环境，返回环境结果
        # env_output = await session.execute({"step": {'action': agent_data.env_action}})
        step_response = await env.step(agent_data.env_action)
        agent_data.server_duration += round(step_response.duration, 2)
        if step_response.success:
            show_log(agent_data, f"Env step response success: {step_response.success}, msg: {step_response.msg}, duration: {step_response.duration}s")
        else:
            show_log(
                agent_data,
                "Env step failed: "
                f"msg={step_response.msg}, duration={step_response.duration}s, "
                f"{action_log_summary(agent_data.env_action)}",
            )
        agent_data.user_turns += 1
        if not step_response.success and step_response.code != ServerStatusCodes.EXEC_TIMEOUT:
            error_msg = (
                f"Env step failed: {step_response.msg}, "
                f"{action_log_summary(agent_data.env_action)}"
            )
            error_msg = show_log(agent_data, error_msg)
            agent_data.error_msg = error_msg
            return AgentState.ERROR
        elif not step_response.success and step_response.code == ServerStatusCodes.EXEC_TIMEOUT:
            # 环境交互超时，记录次数
            agent_data.env_time_out_turn_count += 1
            if agent_data.env_time_out_turn_count > self.max_env_timeout_turn:
                error_msg = (
                    "Env step timeout exceeded max retries "
                    f"({self.max_env_timeout_turn}), "
                    f"{action_log_summary(agent_data.env_action)}"
                )
                error_msg = show_log(agent_data, error_msg)
                agent_data.error_msg = error_msg
                return AgentState.ERROR
            else:
                # 把超时信息作为环境观察
                step_response.observation = step_response.msg
                # 系统稍微等待一下再继续    
                await asyncio.sleep(3)
        env_output = step_response.observation
        # print(f"env_output: {env_output}")
        # 检查是否结束
        finish_response = await env.task_completed()
        agent_data.server_duration += finish_response.duration
        if not finish_response.success:
            error_msg = f"Env complete check failed: {finish_response.msg}"
            error_msg = show_log(agent_data, error_msg)
            agent_data.error_msg = error_msg
            return AgentState.ERROR
        else:
            if finish_response.finished:
                # 任务完成，终止交互
                return AgentState.TERMINATED

        # 检查截断超长环境回复
        if self.max_tool_response_length > 0:
            env_output = await self._truncate_environment_output(
                env_output,
                self.max_tool_response_length,
            )

        env_output = "\n\nOutput:\n```\n" + env_output + "\n```"
        add_messages = [{"role": "user", "content": env_output}]
        agent_data.messages.extend(add_messages)
        # test 
        # raw_response_ids = self.tokenizer(env_output, return_tensors="pt", padding=False)
        # print(agent_data.request_id, agent_data.user_turns, len(raw_response_ids.input_ids[0]), raw_response_ids.input_ids[0])
        response_ids = await self.apply_chat_template(
            add_messages,
            remove_system_prompt=True,
        )
        # print(agent_data.request_id, agent_data.user_turns, len(response_ids), response_ids)
        # 超出总长度
        # LLM生成以后，必定执行环境，所以在这里检查长度和循环数量，超出则直接终止，去做评估。
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            show_log(agent_data, f"AgentLoop-Terminated-By-ResponseLength-EnvResp[loop_response+env_response>response_length]: {len(agent_data.response_mask)} + {len(response_ids)} >= {self.response_length}")
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            show_log(agent_data, f"AgentLoop-Terminated-By-AssistantTurns {agent_data.assistant_turns} >= {self.max_assistant_turns}")
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        return AgentState.GENERATING
