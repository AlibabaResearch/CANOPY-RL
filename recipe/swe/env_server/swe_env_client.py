#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@desc SWE-Env Client
@author: plm
@create: 2026-02-04
"""


import asyncio
import copy
import json
import os
import re
import shlex
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

import ray
from jinja2 import StrictUndefined, Template
from loguru import logger

from .actions_toolcall import BASH_TOOL, parse_toolcall_actions
from .config import AgentConfig, DockerEnvironmentConfig
from .container_gc import (
    GC_PRIORITY_NORMAL,
    GC_PRIORITY_URGENT,
    LABEL_GROUP_ID,
    LABEL_MANAGED,
    LABEL_NODE_ID,
    LABEL_NONCE,
    LABEL_OWNER,
    LABEL_REQUEST,
    LABEL_ROLE,
    ContainerGCRequest,
    ContainerGCSettings,
    current_ray_job_id,
    get_gc_actor,
    request_hash,
    resolve_group_node_id,
)
from .dependency_mirrors import configure_dependency_mirror_environment
from .environments import (DEFAULT_MAX_OUTPUT_BYTES, EVAL_MAX_OUTPUT_BYTES,
                           INFRA_MAX_OUTPUT_BYTES, DockerEnvironment)
from .evaluate import get_eval_report
from .exceptions import FormatError
from .ray_docker_env import RayDockerEnviroment
from .schemas import (ActionParseResponse, EnvEvaluateResponse,
                      EnvInitResponse, EnvStepResponse, GetInitMsgResponse,
                      ServerStatusCodes)
from .swe_env_constants import (RESET_GIT_LOG_COMMAND,
                                RESET_GIT_LOG_COMMAND_SIMPLE_FORCE)
from .swe_utils import Timer, load_yaml_config_from_file_path, render_template
from .test_spec import SWEProTestSpec, TestSpec

GIT_APPLY_CMDS = [
    "patch --batch --fuzz=5 -p1 -i",
    "git apply --verbose",
    "git apply --verbose --reject",
]


class SWEEnvClient:

    def __init__(
        self,
        instance,
        agent_config: AgentConfig,
        env_config: DockerEnvironmentConfig,
        test_spec: TestSpec | SWEProTestSpec,
        container_role: str = "standalone",
        request_id: str = "",
    ):
        self.instance = instance
        self.agent_config = agent_config
        # Each rollout/evaluator needs its own run_args and immutable GC
        # identity.  The caller intentionally reuses one base config object.
        self.env_config = copy.deepcopy(env_config)
        self.test_spec = test_spec

        if not self.env_config.repo_language:
            self.env_config.repo_language = str(
                instance.get("repo_language")
                or instance.get("language")
                or getattr(test_spec, "language", "")
                or ""
            )

        self.container_id = ""
        self.container_name = ""
        self.instance_id = instance["instance_id"]
        self.env = None
        self._gc_urgent = False
        self._gc_enqueued = False
        self._ray_release_done = False
        config_values = (
            self.env_config.model_dump()
            if hasattr(self.env_config, "model_dump")
            else self.env_config.dict()
        )
        self._gc_settings = ContainerGCSettings(
            enabled=bool(config_values.get("container_gc_enabled", False)),
            workers_per_node=int(config_values.get("container_gc_workers_per_node", 1)),
            queue_maxsize=int(config_values.get("container_gc_queue_maxsize", 4096)),
            remove_timeout_seconds=float(
                config_values.get("container_gc_remove_timeout_seconds", 120.0)
            ),
            max_retries=int(config_values.get("container_gc_max_retries", 3)),
            retry_backoff_seconds=float(
                config_values.get("container_gc_retry_backoff_seconds", 2.0)
            ),
            enqueue_timeout_seconds=float(
                config_values.get("container_gc_enqueue_timeout_seconds", 5.0)
            ),
            drain_timeout_seconds=float(
                config_values.get("container_gc_drain_timeout_seconds", 180.0)
            ),
        )
        if self._gc_settings.enabled:
            self._configure_container_gc_identity(
                container_role=container_role,
                request_id=request_id or self.instance_id,
            )
        return

    def _configure_container_gc_identity(
        self, container_role: str, request_id: str
    ) -> None:
        if container_role not in {"rollout", "eval", "standalone"}:
            raise ValueError(f"Unsupported SWE container role: {container_role!r}")
        owner = current_ray_job_id()
        node_id = resolve_group_node_id(int(self.env_config.group_id))
        nonce = uuid4().hex
        container_name = f"minisweagent-{uuid4().hex[:8]}"
        req_hash = request_hash(request_id)
        labels = {
            LABEL_MANAGED: "true",
            LABEL_OWNER: owner,
            LABEL_NONCE: nonce,
            LABEL_NODE_ID: node_id,
            LABEL_GROUP_ID: str(self.env_config.group_id),
            LABEL_ROLE: container_role,
            LABEL_REQUEST: req_hash,
        }

        # Identity labels may never be inherited from a YAML run_args list or
        # another client.  A conflict is safer to reject than to overwrite.
        identity_keys = set(labels)
        args = list(self.env_config.run_args)
        for index, token in enumerate(args[:-1]):
            if token != "--label":
                continue
            existing_key = str(args[index + 1]).split("=", 1)[0]
            if existing_key in identity_keys:
                raise ValueError(
                    f"GC identity label {existing_key!r} must not be preconfigured"
                )
        for key, value in labels.items():
            args.extend(["--label", f"{key}={value}"])

        self.env_config.run_args = args
        self.env_config.container_gc_owner = owner
        self.env_config.container_gc_nonce = nonce
        self.env_config.container_gc_node_id = node_id
        self.env_config.container_gc_role = container_role
        self.env_config.container_gc_request_hash = req_hash
        self.env_config.container_name = container_name
        self.container_name = container_name

    def _mark_gc_urgent_from_output(self, output: Any) -> None:
        if not self._gc_settings.enabled:
            return
        if isinstance(output, EnvStepResponse):
            raw = output.output
            timed_out = bool(output.timeout)
            detail = str(output.msg or "")
        elif isinstance(output, dict):
            raw = output
            timed_out = bool(output.get("timeout", False))
            detail = str(output.get("exception_info", "") or output.get("output", ""))
        else:
            return
        extra = raw.get("extra", {}) if isinstance(raw, dict) else {}
        if (
            timed_out
            or str(extra.get("killed_by", "")) == "timeout"
            or "timeout" in detail.lower()
            or "timed out" in detail.lower()
        ):
            self._gc_urgent = True

    def __init_ray_docker_env(self, resource_tokens=5.0, cpu_limit="4", mem_limit="8g", actor_num_cpus=1.0):
        """
        Args:
            resource_tokens:
            num_cpus:
            cpu_limit:
            mem_limit:
        """
        # 1. 核心减负：8G 内存和 4 核 CPU 对绝大多数 SWE-bench 评估代码绝对够用
        # mem_limit = "8g"
        # cpu_limit = "4"

        # 构造标准键值对参数
        # --- 本次新增的防卡死核心参数 --
        resource_settings = {
            "--memory": str(mem_limit),
            "--cpus": str(cpu_limit),
            "--cpu-shares": "512",      # Prefer model inference when CPUs are contended.
            "--pids-limit": "4096",     # Bound process proliferation in untrusted containers.
            "--shm-size": "64m",        # Bound per-container tmpfs use for retained evaluations.
                                        # Large concurrent runs magnify tmpfs metadata pressure.
            "--log-driver": "none",     # Container stdout is collected through exec; avoid duplicate disk logs.
            # Benchmark code is untrusted. The public default is an isolated
            # network; operators must explicitly opt in to bridge/host mode.
            "--network": self.env_config.network_mode,
        }

        # 批量注入标准参数
        for arg, value in resource_settings.items():
            if arg not in self.env_config.run_args:
                self.env_config.run_args.extend([arg, value])

        # 显式关闭禁止 OOM Kill，让超出内存的容器被内核精准干掉，而不是拖死整个 Ray Worker
        if "--oom-kill-disable" not in self.env_config.run_args:
            self.env_config.run_args.append("--oom-kill-disable=false")

        # Package mirrors are separately gated and do not opt the container in
        # to network access. Existing environment values always win.
        configure_dependency_mirror_environment(self.env_config)

        # 2. 致命挤兑点修复：降低不可回收的 tmpfs 大小
        # if "--tmpfs" not in self.env_config.run_args:
        #     # 【极致压缩】：将 /tmp 的 tmpfs 限制到 512m (注意是兆，不是 GB)
        #     self.env_config.run_args.extend(["--tmpfs", "/tmp:rw,size=512m"])
            # 【防 sparse file 膨胀】：将 /testbed 挂载到内存 tmpfs
            # 核心原理：Agent 在 /testbed 的所有写入（git操作、pip install、测试输出）
            # 原本写入 podman overlay 可写层 → ext4(loop) → sparse file → virtiofs
            # virtiofs 不支持 PUNCH_HOLE，sparse file 一旦膨胀永远不缩小 → 磁盘爆满
            # 改为 tmpfs 后，写入在内存中，不经过 sparse file，彻底解决膨胀问题
            # 4g 对 SWE-bench 的 git repo + 测试输出足够（通常 <2g）
            # 加 exec 标志允许在 /testbed 中执行脚本
            # self.env_config.run_args.extend(["--tmpfs", "/testbed:rw,size=4g,exec"])
        # 【选项 A：挂载到内存盘 (tmpfs)】- 推荐！速度最快，不占任何磁盘
        if self.env_config.map_testbed_to_tmpfs:
            self.env_config.run_args.extend(["--tmpfs", "/ram_testbed:rw,size=1g,exec"])

        # Never pull during rollout. Images must be downloaded, archived and
        # preloaded before a training/evaluation job starts.
        if "--pull" not in self.env_config.run_args:
            self.env_config.run_args.extend(["--pull", "never"])

        # 4. Ray 资源调度
        resource_key = f"group_{self.env_config.group_id}"
        # resource_tokens = 2

        self.env = RayDockerEnviroment.options(
            resources={resource_key: resource_tokens},
            num_cpus=actor_num_cpus,
        ).remote(self.env_config)
        return

    async def _await_env_remote(
        self,
        object_ref,
        *,
        log_prefix: str = "",
        cancellation_grace_seconds: float = 15.0,
    ):
        """Await one Ray env call and propagate local cancellation remotely.

        Cancelling the local asyncio waiter does not by itself cancel a Ray
        actor task. Explicitly cancel the ObjectRef, then allow the remote
        cancellation-safe Podman runner to kill/reap its process group before
        the outer lifecycle force-kills the actor.
        """

        remote_future = asyncio.ensure_future(object_ref)
        try:
            return await asyncio.shield(remote_future)
        except asyncio.CancelledError:
            try:
                ray.cancel(object_ref, force=False)
            except Exception as exc:
                logger.warning(
                    f"{log_prefix} Failed to cancel remote env task cleanly: {exc}"
                )
            try:
                await asyncio.wait_for(
                    asyncio.shield(remote_future),
                    timeout=cancellation_grace_seconds,
                )
            except BaseException:
                pass
            if not remote_future.done():
                remote_future.cancel()
            raise

    async def initialize(self, timeout=60, max_start_sleep_seconds=30, reset_git_log=False, log_prefix="") -> EnvInitResponse:
        env_config = self.env_config
        self.__init_ray_docker_env(
            resource_tokens=env_config.env_resource_tokens,
            cpu_limit=env_config.env_cpu_limit,
            mem_limit=env_config.env_mem_limit,
            actor_num_cpus=env_config.ray_env_actor_num_cpus
        )
        timer = Timer(can_print=False)
        try:
            output = await self._await_env_remote(
                self.env.initialize.remote(
                    timeout=timeout,
                    max_start_sleep_seconds=max_start_sleep_seconds,
                    reset_git_log=reset_git_log,
                    log_prefix=log_prefix,
                ),
                log_prefix=log_prefix,
            )
        except ray.exceptions.RayActorError as e:
            output = {
                "returncode": -1,
                "exception_info": f"env init failed, ray execption: {e}",
                "timeout": False
            }
        except Exception as e:
            output = {
                "returncode": -1,
                "exception_info": f"env init failed: {e}",
                "timeout": False
            }
        used_seconds, total_seconds, used_info, total_info = timer.tok("")
        init_resp = EnvInitResponse.from_docker_output(output, duration=used_seconds, time_info=total_info)
        self.container_name = str(
            (output.get("extra", {}) or {}).get("container_name", self.container_name)
        )
        output_container_id = str(output.get("output", "") or "").strip()
        if re.fullmatch(r"[0-9a-f]{64}", output_container_id):
            self.container_id = output_container_id
        self._mark_gc_urgent_from_output(output)
        # Cache identity before the Ray actor is released. This also recovers a
        # full ID from partial setup failures where podman run itself succeeded.
        if self.env is not None:
            try:
                metadata = await asyncio.wait_for(
                    self.env.get_container_metadata.remote(), timeout=5
                )
                # Never discard a full ID already captured from ``podman run``
                # merely because a late metadata RPC returned an empty value.
                self.container_id = str(
                    metadata.get("container_id", "") or self.container_id
                )
                self.container_name = str(
                    metadata.get("container_name", "") or self.container_name
                )
            except Exception:
                pass
        if init_resp.success:
            if self.env_config.map_testbed_to_tmpfs:
                # =====================================================================
                # 核心修改点：代码搬家 + 原地创建软链接 (狸猫换太子)
                # 步骤：
                # 1. 拷贝镜像原生的 /testbed 到中转盘 /ram_testbed
                # 2. 删除原生的 /testbed 目录 (只会打上 overlay whiteout 标记，秒删)
                # 3. 创建同名软链接 /testbed -> 指向 /ram_testbed
                # =====================================================================
                # /testbed
                raw_cwd = self.env_config.cwd
                ram_cwd = "/ram_testbed"
                setup_cmd = (
                    f"cp -a {raw_cwd}/. {ram_cwd}/ && "
                    f"rm -rf {raw_cwd} && "
                    f"ln -s {ram_cwd} {raw_cwd}"
                )

                setup_resp = await self.execute_command(
                    command=setup_cmd,
                    cwd="/",
                    timeout=120,
                    log_prefix=log_prefix + "[Testbed_Setup]"
                )

                if not setup_resp.success:
                    return EnvInitResponse(
                        success=False,
                        msg=f"Testbed 挂载与软链接重定向失败: {setup_resp.msg}",
                        code=ServerStatusCodes.INTERNAL_ERROR
                    )
        return init_resp

    async def kill_ray_actor(
        self,
        log_prefix="",
        *,
        high_priority: bool = False,
        reason: str = "completed",
        wait_for_gc_ack: bool = True,
    ) -> bool:
        """
        Submit this exact container to its node-local GC queue, then release the
        Ray environment actor immediately. If GC is disabled/unavailable, the
        historical container TTL plus ``--rm`` remains the fallback.
        """
        if self._ray_release_done and (
            self._gc_enqueued or not self._gc_settings.enabled
        ):
            return True

        gc_ref = None
        if self._gc_settings.enabled and not self._gc_enqueued:
            priority = (
                GC_PRIORITY_URGENT
                if high_priority or self._gc_urgent
                else GC_PRIORITY_NORMAL
            )
            try:
                request = ContainerGCRequest(
                    container_id=str(self.container_id),
                    container_name=str(self.container_name),
                    owner=str(self.env_config.container_gc_owner),
                    nonce=str(self.env_config.container_gc_nonce),
                    node_id=str(self.env_config.container_gc_node_id),
                    group_id=int(self.env_config.group_id),
                    role=str(self.env_config.container_gc_role),
                    request_hash=str(self.env_config.container_gc_request_hash),
                    reason=str(reason),
                    priority=priority,
                    enqueued_at=time.time(),
                )
                gc_actor = get_gc_actor(request.owner, request.node_id)
                # Submit synchronously before any await/ray.kill. In a cancelled
                # task, the actor RPC is still queued even though we skip the ack.
                gc_ref = gc_actor.enqueue.remote(asdict(request))
                self._gc_enqueued = True
            except Exception as exc:
                logger.warning(
                    f"{log_prefix} container GC enqueue degraded to TTL cleanup: {exc}"
                )

        if self.env is None:
            self._ray_release_done = True
        elif not self._ray_release_done:
            # A transient Ray control-plane error must not leave the actor (and
            # its group resource token) alive just because every lifecycle
            # caller invokes this cleanup helper only once. Keep this bounded
            # and synchronous so it is also effective inside cancellation paths.
            for attempt in range(1, 3):
                try:
                    ray.kill(self.env, no_restart=True)
                except Exception as e:
                    logger.error(
                        f"{log_prefix} Failed to kill ray actor "
                        f"(attempt {attempt}/2): {e}"
                    )
                else:
                    self._ray_release_done = True
                    break
        if self._ray_release_done:
            self.env = None

        if gc_ref is not None and wait_for_gc_ack:
            try:
                ack = await asyncio.wait_for(
                    asyncio.shield(gc_ref),
                    timeout=self._gc_settings.enqueue_timeout_seconds + 1.0,
                )
                if not ack.get("accepted", False):
                    logger.warning(
                        f"{log_prefix} container GC rejected request; TTL fallback: {ack}"
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    f"{log_prefix} container GC ack unavailable; TTL fallback remains: {exc}"
                )
        return self._ray_release_done


    async def copy_to_container(self, local_path, remote_path, log_prefix=""):
        if isinstance(remote_path, str):
            remote_path = Path(remote_path)
        if isinstance(local_path, str):
            local_path = Path(local_path)
        output = await self._await_env_remote(
            self.env.copy_to_container.remote(
                local_path, remote_path, log_prefix=log_prefix
            ),
            log_prefix=log_prefix,
        )
        self._mark_gc_urgent_from_output(output)
        return output

    async def write_content_to_container(self, content, remote_path, log_prefix=""):
        if isinstance(remote_path, str):
            remote_path = Path(remote_path)
        output = await self._await_env_remote(
            self.env.write_content_to_container.remote(
                content, remote_path, log_prefix=log_prefix
            ),
            log_prefix=log_prefix,
        )
        self._mark_gc_urgent_from_output(output)
        return output

    def get_init_messages(self) -> GetInitMsgResponse:
        """
        Get the initial messages for the agent based on the instance and templates.
        """
        system_template = self.agent_config.system_template
        instance_template = self.agent_config.instance_template
        system_prompt = render_template(template=system_template)
        task = (
            self.instance.get("agent_problem_statement")
            or self.instance["problem_statement"]
        )
        instance_prompt = render_template(instance_template, task=task)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt},
        ]

        data = {
            "success": True,
            "messages": messages,
            "code": ServerStatusCodes.SUCCESS
        }

        return GetInitMsgResponse(**data)

    def parse_actions_by_tool_calls(self, tool_calls: list, toolcall_extract_error_msgs: list) -> ActionParseResponse:
        """Parse the action from the message. Returns the action."""
        try:
            actions = parse_toolcall_actions(
                tool_calls,
                format_error_template=self.agent_config.format_error_template,
                toolcall_extract_error_msgs=toolcall_extract_error_msgs,
                enforce_single_tool_call=self.agent_config.enforce_single_tool_call,
                enforce_exact_bash_arguments=(
                    self.agent_config.enforce_exact_bash_arguments
                ),
                enforce_exact_submission_command=(
                    self.agent_config.enforce_exact_submission_command
                ),
            )
            return ActionParseResponse(success=True, actions=actions, code=ServerStatusCodes.SUCCESS)
        except FormatError as e:
            observation = e.messages[0]["content"]
            diagnostic_reason = str(
                e.messages[0].get("extra", {}).get(
                    "diagnostic_reason", "native_tool_protocol_error"
                )
            )
            toolcall_extract_error_msgs_str = "; ".join(toolcall_extract_error_msgs) if toolcall_extract_error_msgs else "No errors in extracting tool calls."
            format_error_detail = re.sub(r"\s+", " ", observation).strip()[:500]
            error_msg = (
                "parse action failed, format error. extraction details: "
                f"{toolcall_extract_error_msgs_str[:200]}; format details: "
                f"{format_error_detail}"
            )
            return ActionParseResponse(
                success=False,
                actions=[],
                msg=error_msg,
                observation=observation,
                code=ServerStatusCodes.FORMAT_ERROR,
                diagnostic_reason=diagnostic_reason,
            )

    def parse_actions_by_text(self, content: str, action_regex: str = "", format_error_template: str = "") -> ActionParseResponse:
        """Parse the action from the message. Returns the action."""
        if not action_regex:
            action_regex = self.agent_config.action_regex
        if not format_error_template:
            format_error_template = self.agent_config.format_error_template
        commands = [a.strip() for a in re.findall(action_regex, content, re.DOTALL)]
        if len(commands) == 1:
            actions =[{"command": command} for command in commands]
            return ActionParseResponse(success=True, actions=actions, code=ServerStatusCodes.SUCCESS)
        else:
            showlen = 500
            if len(content) <= showlen:
                error_msg = f"Expected exactly 1 action, found {len(commands)}. content: {content[:showlen]}"
            else:
                error_msg = f"Expected exactly 1 action, found {len(commands)}. content: {content[:showlen]}. ...打印省略... {content[-showlen:]}"
            observation = Template(format_error_template, undefined=StrictUndefined).render(
                        actions=commands, error=error_msg
                    )
            return ActionParseResponse(
                success=False,
                actions=[],
                msg=error_msg,
                observation=observation,
                code=ServerStatusCodes.FORMAT_ERROR
            )

    async def execute_command(self, command: str, cwd: str = "", timeout: int = 240,
                              max_output_bytes: int | None = None,
                              log_prefix="") -> EnvStepResponse:
        """全异步执行命令；可选 max_output_bytes 字节上限（超即 killpg）。"""
        timer = Timer(can_print=False)
        output = await self._await_env_remote(
            self.env.execute_command.remote(
                command=command,
                cwd=cwd,
                timeout=timeout,
                max_output_bytes=max_output_bytes,
                log_prefix=log_prefix,
            ),
            log_prefix=log_prefix,
        )
        used_seconds, total_seconds, used_info, total_info = timer.tok("Execute command")
        response = EnvStepResponse.from_docker_output(output, duration=total_seconds, time_info=total_info)
        self._mark_gc_urgent_from_output(response)
        return response

    async def execute_action(self, action: dict, cwd: str = "", timeout: int = 240,
                             max_output_bytes: int | None = None,
                             log_prefix="") -> EnvStepResponse:
        """全异步执行命令；可选 max_output_bytes 字节上限（超即 killpg）。"""
        timer = Timer(can_print=False)
        output = await self._await_env_remote(
            self.env.execute_action.remote(
                action=action,
                cwd=cwd,
                timeout=timeout,
                max_output_bytes=max_output_bytes,
                log_prefix=log_prefix,
            ),
            log_prefix=log_prefix,
        )
        used_seconds, total_seconds, used_info, total_info = timer.tok("Execute action")
        response = EnvStepResponse.from_docker_output(output, duration=total_seconds, time_info=total_info)
        self._mark_gc_urgent_from_output(response)
        return response

    def check_finished_and_extract_predict_patch(self, output: dict, log_prefix=""):
        """Raises Submitted exception with final output if the agent has finished its task."""
        env_raw_output_str = output.get("output", "")
        lines = env_raw_output_str.lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            predict_patch = "".join(lines[1:])
            # update
            predict_patch = predict_patch.lstrip()
            return True, predict_patch
        return False, ""

    async def apply_patch(self, predict_patch: str, cwd="", timeout=240, log_prefix="") -> EnvStepResponse:
        remote_path = "/tmp/patch.diff"
        write_res = await self.write_content_to_container(predict_patch, remote_path, log_prefix=log_prefix)
        # print(log_prefix, f"write patch to container result: {write_res}", flush=True)
        resp = EnvStepResponse(success=False, observation="", env_raw_output_str="", msg="", duration=0.0, code=ServerStatusCodes.SUCCESS)
        timer = Timer(can_print=False)
        failed_msgs = []
        time_infos = []
        total_duration = 0.0
        for i, git_apply_cmd in enumerate(GIT_APPLY_CMDS):
            # cmd = f"{git_apply_cmd} {remote_path} && rm {remote_path}"
            cmd = f"{git_apply_cmd} {remote_path}"
            resp = await self.execute_command(cmd, cwd=cwd, timeout=timeout, log_prefix=log_prefix)
            total_duration += resp.duration
            time_infos.append(f"try-{i} {resp.time_info}")
            if resp.execute_success:
                msg = "{} apply patch success cmd={}, patch_length: {}, returncode: {}, output_preview: {}".format(
                    log_prefix, cmd, len(predict_patch), resp.returncode, resp.env_raw_output_str[:10]
                    )
                # logger.info(msg)
                if len(failed_msgs) > 0:
                    summ_failed_msg = "; ".join(failed_msgs)
                    msg = f"{log_prefix} patch_length: {len(predict_patch)}, {git_apply_cmd} success but previous failed : {summ_failed_msg}"
                    resp.msg = msg
                resp.duration = total_duration
                resp.time_info = "; ".join(time_infos)
                return resp
            cur_fail_msg = f"apply patch failed cmd={cmd}, returncode: {resp.returncode}, output_preview: {resp.env_raw_output_str[:50]}"
            failed_msgs.append(cur_fail_msg)
            # 回滚所有变动，确保下一次 try 是在干净的 testbed 上
            await self.execute_command("git checkout . && git clean -fd && find . -name '*.rej' -delete", cwd=cwd)

        if len(failed_msgs) > 0:
            summ_failed_msg = f"patch_length: {len(predict_patch)}, apply patch all failed " + "; ".join(failed_msgs)
            # logger.error(f"{log_prefix} : {summ_failed_msg}")
            resp.msg = summ_failed_msg
        # 如果所有命令都失败了，返回最后一个命令的错误信息
        resp.success = False
        resp.execute_success = False
        resp.time_info = "; ".join(time_infos)
        resp.duration = total_duration
        return resp

    async def run_eval_script(self, eval_script: str, cwd="", timeout=360, log_prefix="") -> EnvStepResponse:
        remote_path = "/tmp/eval.sh"
        await self.write_content_to_container(eval_script, remote_path, log_prefix=log_prefix)
        cmd = f"/bin/bash {remote_path}"
        # 评估走单独的较大上限 (256MB)，保留完整 PASS/FAIL 解析能力，
        # 但仍然防止某条 instance 的失控 stdout 撑爆 host 内存。
        resp = await self.execute_command(
            cmd, cwd=cwd, timeout=timeout,
            max_output_bytes=EVAL_MAX_OUTPUT_BYTES,
            log_prefix=log_prefix,
        )
        return resp

    async def reset_git_log(self, cwd="/testbed", timeout=120, log_prefix="") -> EnvStepResponse:
        # 1. 将 Bash 脚本内容写入容器的 /tmp 目录
        remote_path = "/tmp/clean_git_log.sh"
        reset_git_command = RESET_GIT_LOG_COMMAND
        # reset_git_command = RESET_GIT_LOG_COMMAND_SIMPLE_FORCE
        await self.write_content_to_container(reset_git_command, remote_path, log_prefix=log_prefix)

        # 2. 根据 repo 动态决定是否保留 Git Tags
        # 注意：传给 Bash 脚本的 true/false 必须全小写
        if "pytest" in self.test_spec.repo:
            cmd = f"/bin/bash {remote_path} HEAD --remove_tag false"
        else:
            cmd = f"/bin/bash {remote_path} HEAD --remove_tag true"

        # 3. 在目标仓库目录 (默认 /testbed) 执行清理脚本
        resp = await self.execute_command(cmd, cwd=cwd, timeout=timeout, log_prefix=log_prefix)

        # 4. 阅后即焚，清理战场，不给 Agent 留痕迹
        rm_cmd = f"rm -rf {remote_path}"
        await self.execute_command(rm_cmd, cwd=cwd, timeout=timeout, log_prefix=log_prefix)
        return resp

    @staticmethod
    def _strip_pro_binary_hunks(patch: str) -> str:
        """Match the official Pro evaluator's handling of binary diffs."""

        if not patch:
            return patch
        sections = re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE)
        kept = []
        for section in sections:
            if not section.strip():
                continue
            if re.search(r"^Binary files .* differ$", section, re.MULTILINE):
                continue
            if re.search(r"^GIT binary patch$", section, re.MULTILINE):
                continue
            kept.append(section)
        return "".join(kept)

    @staticmethod
    def _pro_failure_response(
        *,
        message: str,
        predict_patch: str,
        failure_code: str,
        apply_patch_failed: bool = False,
        patch_successfully_applied: bool = False,
    ) -> EnvEvaluateResponse:
        return EnvEvaluateResponse(
            success=False,
            reward_score=0.0,
            num_passes=0,
            num_failures=0,
            msg=message,
            code=ServerStatusCodes.INTERNAL_ERROR,
            report={
                "patch_is_None": predict_patch is None,
                "patch_exists": bool(predict_patch),
                "patch_successfully_applied": patch_successfully_applied,
                "resolved": False,
                "f2p_rate": 0.0,
                "p2p_rate": 0.0,
                "resolve_status": "RESOLVED_NO",
                "reward_score": 0.0,
                "msg": message,
                "model_patch": predict_patch,
                "eval_failure_code": failure_code,
            },
            did_real_eval=False,
            apply_patch_failed=apply_patch_failed,
        )

    def _compute_pro_eval_response(
        self,
        parsed_output: dict,
        predict_patch: str,
    ) -> EnvEvaluateResponse:
        spec = self.test_spec
        if not isinstance(spec, SWEProTestSpec):
            raise TypeError(
                "SWE-bench Pro evaluation requires SWEProTestSpec, "
                f"got {type(spec).__name__}"
            )

        tests = parsed_output.get("tests")
        if not isinstance(tests, list):
            raise ValueError("SWE-bench Pro parser output is missing a tests list")
        passed_tests = {
            item["name"]
            for item in tests
            if isinstance(item, dict)
            and isinstance(item.get("name"), str)
            and item.get("status") == "PASSED"
        }
        f2p = set(spec.FAIL_TO_PASS)
        p2p = set(spec.PASS_TO_PASS)
        required = f2p | p2p
        resolved = required <= passed_tests
        f2p_success = sorted(f2p & passed_tests)
        f2p_failure = sorted(f2p - passed_tests)
        p2p_success = sorted(p2p & passed_tests)
        p2p_failure = sorted(p2p - passed_tests)
        f2p_rate = len(f2p_success) / len(f2p) if f2p else 1.0
        p2p_rate = len(p2p_success) / len(p2p) if p2p else 1.0
        reward_score = float(resolved)
        report = {
            "patch_is_None": predict_patch is None,
            "patch_exists": bool(predict_patch),
            "patch_successfully_applied": True,
            "resolved": resolved,
            "f2p_rate": f2p_rate,
            "p2p_rate": p2p_rate,
            "resolve_status": "RESOLVED_FULL" if resolved else "RESOLVED_NO",
            "reward_score": reward_score,
            "eval_failure_code": "",
            "tests_status": {
                "FAIL_TO_PASS": {
                    "success": f2p_success,
                    "failure": f2p_failure,
                },
                "PASS_TO_PASS": {
                    "success": p2p_success,
                    "failure": p2p_failure,
                },
                "FAIL_TO_FAIL": {"success": [], "failure": []},
                "PASS_TO_FAIL": {"success": [], "failure": []},
            },
            "parser_total_tests": len(tests),
            "test_log": "",
            "model_patch": predict_patch,
            "msg": "evaluated.",
        }
        return EnvEvaluateResponse(
            success=True,
            reward_score=reward_score,
            num_passes=len(required & passed_tests),
            num_failures=len(required - passed_tests),
            msg="Evaluation completed",
            code=ServerStatusCodes.SUCCESS,
            report=report,
            did_real_eval=True,
        )

    async def evaluate_swebench_pro(
        self,
        predict_patch: str,
        timeout=480,
        cwd="",
        log_prefix="",
    ) -> EnvEvaluateResponse:
        """Run the official per-instance Pro scripts inside our eval container."""

        spec = self.test_spec
        if not isinstance(spec, SWEProTestSpec):
            raise TypeError(
                "SWE-bench Pro evaluation requires SWEProTestSpec, "
                f"got {type(spec).__name__}"
            )
        cwd = (
            cwd
            or getattr(getattr(self, "env_config", None), "cwd", None)
            or spec.repo_directory
        )

        try:
            run_script = Path(spec.run_script_path).read_text(encoding="utf-8")
            parser_script = Path(spec.parser_path).read_text(encoding="utf-8")
        except Exception as exc:
            return self._pro_failure_response(
                message=f"failed to load SWE-bench Pro scripts: {exc}",
                predict_patch=predict_patch,
                failure_code="eval_script_missing",
            )

        workspace_files = {
            "/workspace/patch.diff": self._strip_pro_binary_hunks(predict_patch),
            "/workspace/run_script.sh": run_script,
            "/workspace/parser.py": parser_script,
        }
        for remote_path, content in workspace_files.items():
            write_result = await self.write_content_to_container(
                content,
                remote_path,
                log_prefix=log_prefix,
            )
            if write_result.get("returncode", -1) != 0:
                return self._pro_failure_response(
                    message=(
                        f"failed to write {remote_path}: "
                        f"{write_result.get('exception_info', write_result)}"
                    ),
                    predict_patch=predict_patch,
                    failure_code="eval_script_write_failed",
                )

        quoted_cwd = shlex.quote(cwd)
        quoted_commit = shlex.quote(spec.base_commit)
        prepare_command = (
            f"git reset --hard {quoted_commit} && "
            f"git checkout {quoted_commit}"
        )
        prepare_response = await self.execute_command(
            prepare_command,
            cwd=cwd,
            timeout=timeout,
            log_prefix=log_prefix,
        )
        if not prepare_response.execute_success:
            failure_code = (
                "eval_repo_prepare_timeout"
                if prepare_response.timeout
                else "eval_repo_prepare_failed"
            )
            return self._pro_failure_response(
                message=(
                    prepare_response.msg
                    or prepare_response.env_raw_output_str
                ),
                predict_patch=predict_patch,
                failure_code=failure_code,
            )

        if predict_patch.strip():
            apply_response = await self.execute_command(
                "git apply -v /workspace/patch.diff",
                cwd=cwd,
                timeout=timeout,
                log_prefix=log_prefix,
            )
            if not apply_response.execute_success:
                failure_code = (
                    "patch_apply_timeout"
                    if apply_response.timeout
                    else "patch_apply_failed"
                )
                return self._pro_failure_response(
                    message=(
                        apply_response.msg
                        or apply_response.env_raw_output_str
                    ),
                    predict_patch=predict_patch,
                    failure_code=failure_code,
                    apply_patch_failed=True,
                )

        selected_tests = ",".join(spec.selected_test_files_to_run)
        run_tests_command = "bash /workspace/run_script.sh"
        if selected_tests:
            run_tests_command += f" {shlex.quote(selected_tests)}"
        script_lines = ["#!/bin/bash", f"cd {quoted_cwd}"]
        if spec.before_repo_set_cmd:
            script_lines.append(spec.before_repo_set_cmd)
        script_lines.extend(
            [
                f"{run_tests_command} > /workspace/stdout.log 2> /workspace/stderr.log",
                (
                    "python /workspace/parser.py /workspace/stdout.log "
                    "/workspace/stderr.log /workspace/output.json"
                ),
            ]
        )
        run_response = await self.run_eval_script(
            "\n".join(script_lines) + "\n",
            cwd=cwd,
            timeout=timeout,
            log_prefix=log_prefix,
        )
        if not run_response.execute_success:
            killed_by = run_response.output.get("extra", {}).get("killed_by")
            if killed_by == "size":
                failure_code = "eval_output_limit"
            elif run_response.timeout:
                failure_code = "eval_timeout"
            else:
                failure_code = "eval_execution_failed"
            return self._pro_failure_response(
                message=run_response.msg or run_response.env_raw_output_str,
                predict_patch=predict_patch,
                failure_code=failure_code,
                patch_successfully_applied=True,
            )

        output_response = await self.execute_command(
            "cat /workspace/output.json",
            cwd=cwd,
            timeout=60,
            max_output_bytes=EVAL_MAX_OUTPUT_BYTES,
            log_prefix=log_prefix,
        )
        if not output_response.execute_success:
            return self._pro_failure_response(
                message=output_response.msg or "SWE-bench Pro output.json is missing",
                predict_patch=predict_patch,
                failure_code="eval_report_missing",
                patch_successfully_applied=True,
            )
        try:
            parsed_output = json.loads(output_response.env_raw_output_str)
            return self._compute_pro_eval_response(parsed_output, predict_patch)
        except Exception as exc:
            return self._pro_failure_response(
                message=f"invalid SWE-bench Pro parser output: {exc}",
                predict_patch=predict_patch,
                failure_code="eval_report_missing",
                patch_successfully_applied=True,
            )


    async def evaluate_v2(self, predict_patch: str, timeout=480, cwd="", log_prefix="", use_sparse_reward=True) -> EnvEvaluateResponse:
        cwd = (
            cwd
            or getattr(getattr(self, "env_config", None), "cwd", None)
            or "/testbed"
        )
        apply_patch_resp: EnvStepResponse = await self.apply_patch(predict_patch, cwd=cwd, timeout=timeout, log_prefix=log_prefix)
        if not apply_patch_resp.success:
            apply_failure_code = "patch_apply_timeout" if apply_patch_resp.timeout else "patch_apply_failed"
            eval_resp = EnvEvaluateResponse(
                success=False,
                reward_score=0.0,
                num_passes=0,
                num_failures=0,
                msg=f"{apply_patch_resp.msg}",
                code=ServerStatusCodes.INTERNAL_ERROR,
            )
            eval_resp.report["msg"] = eval_resp.msg
            eval_resp.report["model_patch"] = predict_patch
            eval_resp.report["eval_failure_code"] = apply_failure_code
            eval_resp.apply_patch_failed = True
            return eval_resp

        run_eval_step_resp: EnvStepResponse = await self.run_eval_script(self.test_spec.eval_script, cwd=cwd, timeout=timeout, log_prefix=log_prefix)
        if not run_eval_step_resp.success:
            killed_by = run_eval_step_resp.output.get("extra", {}).get("killed_by")
            if killed_by == "size":
                eval_failure_code = "eval_output_limit"
            elif run_eval_step_resp.timeout:
                eval_failure_code = "eval_timeout"
            else:
                eval_failure_code = "eval_execution_failed"
            eval_resp = EnvEvaluateResponse(
                success=False,
                reward_score=0.0,
                num_passes=0,
                num_failures=0,
                msg=f"{run_eval_step_resp.msg}",
                code=ServerStatusCodes.INTERNAL_ERROR
            )
            eval_resp.report["msg"] = run_eval_step_resp.msg
            eval_resp.report["model_patch"] = predict_patch
            eval_resp.report["eval_failure_code"] = eval_failure_code
            return eval_resp
        # print("evaluate reponse: {}".format(response))
        env_eval_resp = self.compute_env_eval_response(run_eval_step_resp, predict_patch=predict_patch, use_sparse_reward=use_sparse_reward)
        env_eval_resp.duration = apply_patch_resp.duration + run_eval_step_resp.duration
        env_eval_resp.time_info = f"apply_patch: {apply_patch_resp.time_info}, run_eval: {run_eval_step_resp.time_info}"
        return env_eval_resp

    async def evaluate(self, predict_patch: str, timeout=480, cwd="", log_prefix="", use_sparse_reward=True) -> EnvEvaluateResponse:
        # Measure the whole evaluate call so early-return failures (apply
        # failure, timeout, output limit, report error) never appear as 0 s.
        started = time.monotonic()
        cwd = (
            cwd
            or getattr(getattr(self, "env_config", None), "cwd", None)
            or "/testbed"
        )
        if (
            getattr(getattr(self, "test_spec", None), "evaluator_type", "")
            == "swe_bench_pro"
        ):
            resp = await self.evaluate_swebench_pro(
                predict_patch,
                timeout,
                cwd,
                log_prefix,
            )
        else:
            resp = await self.evaluate_v2(
                predict_patch,
                timeout,
                cwd,
                log_prefix,
                use_sparse_reward=use_sparse_reward,
            )
        resp.duration = time.monotonic() - started
        report = getattr(resp, "report", None) or {}
        if "timeout" in str(report.get("eval_failure_code", "")).lower():
            self._gc_urgent = True
        return resp


    def compute_env_eval_response(self, run_eval_step_resp: EnvStepResponse, predict_patch: str, use_sparse_reward=True) -> EnvEvaluateResponse:
        """通过run_eval_step_resp来计算EnvEvaluateResponse。使用临时文件处理日志。"""

        test_log = run_eval_step_resp.env_raw_output_str
        instance_id = self.test_spec.instance_id
        prediction = {
            "instance_id": instance_id,
            "model_patch": predict_patch,
        }

        # 使用 NamedTemporaryFile 创建临时文件
        # delete=delete_log_file 参数控制超出 context 后是否自动删除
        with tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', suffix='.log', delete=True) as tmp:
            tmp.write(test_log)
            tmp.flush()  # 确保内容写入磁盘

            test_log_path = tmp.name  # 获取临时文件路径

            report_map = get_eval_report(
                test_spec=self.test_spec,
                prediction=prediction,
                test_log_path=test_log_path,
                include_tests_status=True,
                use_sparse_reward=use_sparse_reward
            )

        # 退出 with 语句块后，如果 delete=True，文件已在磁盘上被物理删除

        report = report_map.get(self.instance_id) if isinstance(report_map, dict) else None
        if not isinstance(report, dict):
            detail = f"evaluation report missing for instance_id={self.instance_id}"
            return EnvEvaluateResponse(
                success=False,
                reward_score=0.0,
                num_passes=0,
                num_failures=0,
                msg=detail,
                code=ServerStatusCodes.INTERNAL_ERROR,
                report={
                    "msg": detail,
                    "model_patch": predict_patch,
                    "eval_failure_code": "eval_report_missing",
                },
                did_real_eval=False,
            )
        resolved = report["resolved"]
        reward_score = report["reward_score"]
        # report["test_log"] = test_log
        report["test_log"] = ""
        report["model_patch"] = predict_patch
        report["msg"] = "evaluated."

        if report["f2p_rate"] <= 0:
            if len(predict_patch) <= 10:
                report["msg"] += f" but f2p<=0, predict_patch length: {len(predict_patch)}, maybe no change."

        data = {
            "success": True,
            "reward_score": reward_score,
            "num_passes": 0,
            "num_failures": 0,
            "msg": "Evaluation completed",
            "duration": 0.0,
            "code": ServerStatusCodes.SUCCESS,
            "report": report,
            "did_real_eval": True,
        }

        return EnvEvaluateResponse(**data)

    def compute_env_eval_response_log_path(self, run_eval_step_resp: EnvStepResponse, predict_patch: str, test_log_path: str, delete_log_file=True) -> EnvEvaluateResponse:
        """通过run_eval_step_resp来计算EnvEvaluateResponse,包括f2p, p2p，reward_score等。

        Args:
            run_eval_step_resp (EnvStepResponse): run step resp
            predict_patch (str): model patch
            test_log_path (str): log path
            delete_log_file (bool, optional): 是否删除Log文件. Defaults to True.

        Returns:
            EnvEvaluateResponse: _description_
        """
        test_log = run_eval_step_resp.env_raw_output_str
        test_log_file = Path(test_log_path)
        test_log_file.write_text(test_log, encoding="utf-8")
        instance_id = self.test_spec.instance_id
        prediction = {
            "instance_id": instance_id,
            "model_patch": predict_patch,
        }

        report_map = get_eval_report(
            test_spec=self.test_spec,
            prediction=prediction,
            test_log_path=test_log_path,
            include_tests_status=True
        )
        report = report_map[self.instance_id]
        resolved = report["resolved"]
        reward_score = report["reward_score"]
        report["test_log"] = test_log
        report["model_patch"] = predict_patch
        report["msg"] = "evaluated."
        if report["f2p_rate"] <= 0:
            if len(predict_patch) <= 10:
                report["msg"] += f"but f2p<=0, predict_patch length: {len(predict_patch)}, maybe no change."
        data = {
            "success": True,
            "reward_score": reward_score,
            "num_passes": 0,
            "num_failures": 0,
            "msg": "Evaluation completed",
            "duration": 0.0,
            "code": ServerStatusCodes.SUCCESS,
            "report": report,
            "did_real_eval": True,
        }
        response = EnvEvaluateResponse(**data)
        if delete_log_file:
            os.remove(test_log_path)
        return response

    async def close(self, log_prefix=""):
        """非常重要：显式释放资源"""
        await self.kill_ray_actor(log_prefix=log_prefix, reason="client_close")


def load_env_config(
    image: str,
    config_path: str | Path | None = None,
) -> DockerEnvironmentConfig:
    """Load a local SWE environment config without embedding a host path."""

    env_path = Path(config_path) if config_path else (
        Path(__file__).resolve().parents[1] / "config" / "swe_env_config.yaml"
    )
    env_config_dict = load_yaml_config_from_file_path(str(env_path))["environment"]
    env_config_dict["image"] = image
    env_config = DockerEnvironmentConfig(**env_config_dict)
    return env_config
