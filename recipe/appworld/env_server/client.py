#!/usr/bin/env python3
"""AppWorld client using an isolated HTTP session for every request.

AppWorld trajectories can be idle while the model generates the next action.
Keeping those HTTP/1.1 connections pooled for several minutes made stale
connections accumulate under evaluation concurrency, so request isolation is
intentional here.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

import aiohttp

try:
    from .config import ClientConfig
    from .schemas import (
        EnvCloseResponse,
        EnvCompleteResponse,
        EnvEvaluateResponse,
        EnvInitResponse,
        EnvStepResponse,
        GetInitMsgResponse,
        ServerStatusCodes,
    )
except ImportError:
    from config import ClientConfig
    from schemas import (
        EnvCloseResponse,
        EnvCompleteResponse,
        EnvEvaluateResponse,
        EnvInitResponse,
        EnvStepResponse,
        GetInitMsgResponse,
        ServerStatusCodes,
    )


def normalize_server_url(raw_url: str) -> str:
    """Validate an AppWorld endpoint before sending benchmark data to it."""
    parsed = urlsplit(raw_url.strip().rstrip("/"))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"invalid AppWorld server URL: {raw_url!r}")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(f"AppWorld server URL must not contain credentials/query: {raw_url!r}")
    if parsed.path not in {"", "/"}:
        raise ValueError(f"AppWorld server URL must not contain a path: {raw_url!r}")
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def load_global_server_urls(folder: Optional[str] = None) -> list[str]:
    configured = folder or os.getenv("APPWORLD_SERVER_URL_DIR")
    url_dir = (
        Path(configured).expanduser().resolve()
        if configured
        else Path(__file__).resolve().parents[3] / "runtime" / "appworld_urls"
    )
    if not url_dir.is_dir():
        raise FileNotFoundError(f"AppWorld URL directory not found: {url_dir}")
    urls: list[str] = []
    for path in sorted(url_dir.glob("*.txt")):
        with open(path, encoding="utf-8") as handle:
            loaded = [
                normalize_server_url(line)
                for line in handle
                if line.strip() and not line.lstrip().startswith("#")
            ]
        urls.extend(loaded)
        print(f"loaded {len(loaded)} AppWorld endpoints from {path.name}")
    # Preserve stable ordering while rejecting duplicate endpoints.
    unique_urls = list(dict.fromkeys(urls))
    if not unique_urls:
        raise ValueError(f"no AppWorld endpoints found in {url_dir}")
    return unique_urls


class AppWorldEnvClient:
    def __init__(
        self,
        task_id: str,
        experiment_name: Optional[str] = None,
        request_id: Optional[str] = None,
        remote_environment_url: str = "",
        init_timeout: int = ClientConfig.INIT_TIMEOUT,
        exec_timeout: int = ClientConfig.EXEC_TIMEOUT,
        eval_timeout: int = ClientConfig.EVAL_TIMEOUT,
        experiments_outputs_directory: Optional[str] = None,
        rm_outdir_after_finished: bool = True,
    ):
        request_id = request_id or str(uuid.uuid4())
        experiment_name = experiment_name or request_id
        self.request_id = request_id
        self.task_id = task_id
        self.experiment_name = experiment_name
        self.remote_environment_url = normalize_server_url(remote_environment_url)
        self.init_timeout = init_timeout
        self.exec_timeout = exec_timeout
        self.eval_timeout = eval_timeout
        self.experiments_outputs_directory = experiments_outputs_directory
        self.rm_outdir_after_finished = rm_outdir_after_finished

    async def aclose(self) -> None:
        """Compatibility no-op: request-scoped sessions close in ``_post``."""
        return None

    async def _post(self, endpoint: str, payload: dict, timeout_sec: int) -> dict:
        url = f"{self.remote_environment_url}{endpoint}"
        request_timeout = aiohttp.ClientTimeout(
            total=timeout_sec,
            connect=min(15, timeout_sec),
            sock_connect=min(15, timeout_sec),
            sock_read=timeout_sec,
        )
        try:
            # Deliberately match the previously stable behavior: do not reuse a
            # connection across init / model generation / step boundaries.
            async with aiohttp.ClientSession(timeout=request_timeout) as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        detail = (await response.text())[:500]
                        return {
                            "success": False,
                            "msg": f"HTTP Error {response.status}: {detail}",
                            "code": ServerStatusCodes.INTERNAL_ERROR,
                        }
                    return await response.json()
        except asyncio.TimeoutError:
            return {
                "success": False,
                "msg": f"Client Timeout after {timeout_sec}s",
                "code": (
                    ServerStatusCodes.INIT_TIMEOUT
                    if endpoint == "/init"
                    else ServerStatusCodes.EXEC_TIMEOUT
                ),
            }
        except (aiohttp.ClientError, OSError) as exc:
            return {
                "success": False,
                "msg": f"Connection Error: {type(exc).__name__}: {exc}",
                "code": ServerStatusCodes.INTERNAL_ERROR,
            }

    async def initialize(self) -> EnvInitResponse:
        payload = {
            "task_id": self.task_id,
            "request_id": self.request_id,
            "experiment_name": self.experiment_name,
            "remote_environment_url": self.remote_environment_url,
            "rm_outdir_after_finished": self.rm_outdir_after_finished,
        }
        if self.experiments_outputs_directory is not None:
            payload["experiments_outputs_directory"] = (
                self.experiments_outputs_directory
            )
        data = await self._post("/init", payload, self.init_timeout)
        response = EnvInitResponse(**data)
        if not response.success:
            await self.aclose()
        return response

    async def get_init_messages(self) -> GetInitMsgResponse:
        data = await self._post(
            "/get_init_messages",
            {
                "experiment_name": self.experiment_name,
                "request_id": self.request_id,
            },
            self.exec_timeout,
        )
        return GetInitMsgResponse(**data)

    async def step(self, action: str) -> EnvStepResponse:
        max_action_chars = int(os.getenv("APPWORLD_MAX_ACTION_CHARS", "100000"))
        if len(action) > max_action_chars:
            return EnvStepResponse(
                success=False,
                msg=f"action exceeds APPWORLD_MAX_ACTION_CHARS={max_action_chars}",
                code=ServerStatusCodes.BAD_REQUEST,
            )
        data = await self._post(
            "/step",
            {
                "experiment_name": self.experiment_name,
                "request_id": self.request_id,
                "action": action,
            },
            self.exec_timeout,
        )
        return EnvStepResponse(**data)

    async def task_completed(self) -> EnvCompleteResponse:
        data = await self._post(
            "/completed",
            {
                "experiment_name": self.experiment_name,
                "request_id": self.request_id,
            },
            self.exec_timeout,
        )
        return EnvCompleteResponse(**data)

    async def evaluate(self, sparse: bool = False) -> EnvEvaluateResponse:
        data = await self._post(
            "/evaluate",
            {
                "task_id": self.task_id,
                "experiment_name": self.experiment_name,
                "request_id": self.request_id,
                "sparse": sparse,
            },
            self.eval_timeout,
        )
        return EnvEvaluateResponse(**data)

    async def close(self) -> EnvCloseResponse:
        try:
            data = await self._post(
                "/close",
                {
                    "experiment_name": self.experiment_name,
                    "request_id": self.request_id,
                },
                30,
            )
            return EnvCloseResponse(**data)
        finally:
            await self.aclose()

    async def __aenter__(self) -> "AppWorldEnvClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
