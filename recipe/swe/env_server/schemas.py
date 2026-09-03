#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@desc env post schema
@author: plm
@create: 2025-12-04
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

# --- Request Schemas ---

class InitRequest(BaseModel):
    task_id: str
    request_id: str
    experiment_name: str
    remote_environment_url: Optional[str] = None
    rm_outdir_after_finished: bool = True
    experiments_outputs_directory: Optional[str] = None

class StepRequest(BaseModel):
    action: str
    request_id: str
    experiment_name: str

class EvaluateRequest(BaseModel):
    request_id: str
    experiment_name: str
    task_id: str
    sparse: bool = False

class CloseRequest(BaseModel):
    request_id: str
    experiment_name: str

class GetInitMsgRequest(BaseModel):
    request_id: str
    experiment_name: str

class CheckCompleteRequest(BaseModel):
    request_id: str
    experiment_name: str


# --- Status Codes Definition ---

class ServerStatusCodes:
    # 成功
    SUCCESS = "OK"

    # 客户端/Agent 错误 (通常不需要重试，给负分)
    EXECUTION_ERROR = "EXECUTION_ERROR"     # Agent 代码运行报错 (SyntaxError 等)
    BAD_REQUEST = "BAD_REQUEST"             # 请求参数错误

    # 资源/系统 错误 (环境崩溃，需要重置)
    OOM_KILLED = "OOM_KILLED"               # 【关键】因内存超标被 Server 强杀
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND" # Session 不存在 (从未创建或已清理)
    SESSION_EXPIRED = "SESSION_EXPIRED"     # 因空闲太久被清理

    # 超时 (根据情况决定是否重试)
    INIT_TIMEOUT = "INIT_TIMEOUT"           # 初始化超时
    EXEC_TIMEOUT = "EXEC_TIMEOUT"           # 单步执行超时 (死循环)

    # 未知/内部错误
    INTERNAL_ERROR = "INTERNAL_ERROR"
    ENV_INIT_FAILED = "ENV_INIT_FAILED"     # AppWorld 启动失败 (代码或环境问题)

    # FormatError
    FORMAT_ERROR = "FORMAT_ERROR"           # LM 输出格式错误


# --- Response Schemas ---

class BaseResponse(BaseModel):
    success: bool = False
    msg: str = ""
    duration: float = 0.0
    code: str = ServerStatusCodes.SUCCESS  # 默认 OK
    time_info: str = ""  # 可选的时间信息字符串

class EnvInitResponse(BaseResponse):
    env_raw_output_str: str = ""
    returncode: int = 0
    output: Dict[str, Any] = Field(default_factory=dict)
    timeout: bool = False

    @classmethod
    def from_docker_output(cls, docker_output: Dict[str, Any], duration: float = 0.0, time_info: str = ""):
        """
        通过 execute() 返回的 dict 直接构造 Response 对象
        """
        # 提取关键字段，如果没有则提供默认值
        if docker_output is not None and len(docker_output) > 0:
            raw_stdout = docker_output.get("output", "")
            ret_code = docker_output.get("returncode", 0)
            timeout = docker_output.get("timeout", False)
        else:
            raw_stdout = ""
            ret_code = -1  # 表示没有输出，视为失败
            timeout = False

        success = (ret_code == 0)
        if "time_info" in docker_output:
            time_info += ", real docker cost:" + docker_output["time_info"]

        return cls(
            success=success,
            returncode=ret_code,
            env_raw_output_str=raw_stdout,
            output=docker_output,
            duration=duration,
            time_info=time_info,
            timeout=timeout,
            msg=docker_output.get("exception_info", "")
        )

class EnvStepResponse(BaseResponse):
    observation: str = ""
    env_raw_output_str: str = ""
    returncode: int = 0
    output: Dict[str, Any] = Field(default_factory=dict)
    execute_success: bool = True
    timeout: bool = False

    @classmethod
    def from_docker_output(cls, docker_output: Dict[str, Any], duration: float = 0.0, time_info: str = ""):
        """
        通过 execute() 返回的 dict 直接构造 Response 对象
        """
        # 提取关键字段，如果没有则提供默认值
        raw_stdout = docker_output.get("output", "")
        ret_code = docker_output.get("returncode", 0)
        timeout = docker_output.get("timeout", False)

        # 逻辑：
        # 1. success: 只有当系统级别报错(ret_code == -1)时才为 False
        # 2. execute_success: 只有当命令执行成功(ret_code == 0)时才为 True
        success = (ret_code != -1)
        execute_success = (ret_code == 0)
        if "time_info" in docker_output:
            time_info += ", real docker exec cost:" + docker_output["time_info"]

        return cls(
            success=success,
            execute_success=execute_success,
            returncode=ret_code,
            observation=raw_stdout,
            env_raw_output_str=raw_stdout,
            output=docker_output,
            duration=duration,
            time_info=time_info,
            timeout=timeout,
            msg=docker_output.get("exception_info", "")
        )

    # @model_validator(mode='after')
    # def sync_from_output(self) -> 'EnvStepResponse':
    #     # 如果初始化时传入了 output 且其他字段为空，则自动同步
    #     if self.output and not self.observation:
    #         self.env_raw_output_str = self.output.get("output", "")
    #         self.observation = self.env_raw_output_str
    #         self.returncode = self.output.get("returncode", 0)
    #         self.success = (self.returncode == 0)
    #         if not self.msg:
    #             self.msg = self.output.get("exception_info", "")
    #     return self

class EnvCompleteResponse(BaseResponse):
    finished: bool = False

class EnvEvaluateResponse(BaseResponse):
    reward_score: float = 0.0
    num_passes: int = 0
    num_failures: int = 0
    report: dict = {
        "patch_is_None": False,
        "patch_exists": False,
        "patch_successfully_applied": False,
        "resolved": False,
        "f2p_rate": 0,
        "p2p_rate": 0,
        "resolve_status": "RESOLVED_NO",
        "reward_score": 0,
        "msg": "",
        "model_patch": "",
    }
    skipped: bool = False
    skip_reason: str = ""
    # eval_time_used_info: str = ""
    did_real_eval: bool = False
    apply_patch_failed: bool = False

class EnvCloseResponse(BaseResponse):
    pass

class ActionParseResponse(BaseResponse):
    action: str = ""
    actions: list[dict] = Field(default_factory=list)  # 支持多个工具调用的情况
    observation: str = ""
    # Stable, low-cardinality producer code used by optional step diagnostics.
    diagnostic_reason: str = ""

class GetInitMsgResponse(BaseResponse):
    messages: List[Dict[str, Any]] = Field(default_factory=list)



class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class FormatError(NonTerminatingException):
    """Raised when the LM's output is not in the expected format."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""    """Raised when the agent has reached its cost or step limit."""
