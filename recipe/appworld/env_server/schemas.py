"""Validated request and response models for the AppWorld service."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


_SAFE_ID = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"


class InitRequest(BaseModel):
    task_id: str = Field(min_length=1, max_length=128, pattern=_SAFE_ID)
    request_id: str = Field(min_length=1, max_length=256, pattern=_SAFE_ID)
    experiment_name: str = Field(min_length=1, max_length=256, pattern=_SAFE_ID)
    remote_environment_url: Optional[str] = Field(default=None, max_length=2048)
    rm_outdir_after_finished: bool = True
    experiments_outputs_directory: Optional[str] = Field(default=None, max_length=4096)


class StepRequest(BaseModel):
    action: str = Field(max_length=1_000_000)
    request_id: str = Field(min_length=1, max_length=256, pattern=_SAFE_ID)
    experiment_name: str = Field(min_length=1, max_length=256, pattern=_SAFE_ID)


class EvaluateRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=256, pattern=_SAFE_ID)
    experiment_name: str = Field(min_length=1, max_length=256, pattern=_SAFE_ID)
    task_id: str = Field(min_length=1, max_length=128, pattern=_SAFE_ID)
    sparse: bool = False


class CloseRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=256, pattern=_SAFE_ID)
    experiment_name: str = Field(min_length=1, max_length=256, pattern=_SAFE_ID)


class GetInitMsgRequest(CloseRequest):
    pass


class CheckCompleteRequest(CloseRequest):
    pass


class ServerStatusCodes:
    SUCCESS = "OK"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    BAD_REQUEST = "BAD_REQUEST"
    OOM_KILLED = "OOM_KILLED"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    SESSION_EXPIRED = "SESSION_EXPIRED"
    INIT_TIMEOUT = "INIT_TIMEOUT"
    EXEC_TIMEOUT = "EXEC_TIMEOUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    ENV_INIT_FAILED = "ENV_INIT_FAILED"


class BaseResponse(BaseModel):
    success: bool = False
    msg: str = ""
    duration: float = 0.0
    code: str = ServerStatusCodes.SUCCESS


class EnvInitResponse(BaseResponse):
    pass


class EnvStepResponse(BaseResponse):
    observation: str = ""


class EnvCompleteResponse(BaseResponse):
    finished: bool = False


class EnvEvaluateResponse(BaseResponse):
    reward_score: float = 0.0
    num_passes: int = 0
    num_failures: int = 0


class EnvCloseResponse(BaseResponse):
    pass


class GetInitMsgResponse(BaseResponse):
    messages: list[dict[str, Any]] = Field(default_factory=list)
