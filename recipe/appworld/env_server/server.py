#!/usr/bin/env python3
"""Reliable AppWorld environment server.

This module keeps the existing HTTP API while fixing session/IPC races, stale
worker replies, incomplete process cleanup, and missing operational telemetry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import os
import resource
import socket
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from ipaddress import ip_address
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Optional

import psutil
from fastapi import FastAPI
from jinja2 import Template

try:
    from .config import ServerConfig
    from .prompts import SYSTEM_PROMPT_TEMPLATE
    from .schemas import (
        CheckCompleteRequest,
        CloseRequest,
        EnvCloseResponse,
        EnvCompleteResponse,
        EnvEvaluateResponse,
        EnvInitResponse,
        EnvStepResponse,
        EvaluateRequest,
        GetInitMsgRequest,
        GetInitMsgResponse,
        InitRequest,
        ServerStatusCodes,
        StepRequest,
    )
except ImportError:
    from config import ServerConfig
    from prompts import SYSTEM_PROMPT_TEMPLATE
    from schemas import (
        CheckCompleteRequest,
        CloseRequest,
        EnvCloseResponse,
        EnvCompleteResponse,
        EnvEvaluateResponse,
        EnvInitResponse,
        EnvStepResponse,
        EvaluateRequest,
        GetInitMsgRequest,
        GetInitMsgResponse,
        InitRequest,
        ServerStatusCodes,
        StepRequest,
    )


SERVER_VERSION = "canopy-appworld-v1"
SERVER_STARTED_AT = time.time()
WORKER_MEMORY_LIMIT_MB = getattr(ServerConfig, "WORKER_MEMORY_LIMIT_MB", 6144)
MAX_GRAVEYARD_ENTRIES = int(os.getenv("APPWORLD_MAX_GRAVEYARD_ENTRIES", "4096"))
MP_START_METHOD = os.getenv("APPWORLD_MP_START_METHOD", "fork")
MP_CONTEXT = multiprocessing.get_context(MP_START_METHOD)
INIT_TIMEOUT = int(os.getenv("APPWORLD_INIT_TIMEOUT", str(ServerConfig.INIT_TIMEOUT)))
EXEC_TIMEOUT = int(os.getenv("APPWORLD_EXEC_TIMEOUT", str(ServerConfig.EXEC_TIMEOUT)))
EVAL_TIMEOUT = int(os.getenv("APPWORLD_EVAL_TIMEOUT", str(ServerConfig.EVAL_TIMEOUT)))
ENV_INTERNAL_TIMEOUT = int(
    os.getenv("APPWORLD_ENV_INTERNAL_TIMEOUT", str(ServerConfig.ENV_INTERNAL_TIMEOUT))
)
WORKER_IDLE_TIMEOUT = int(
    os.getenv("APPWORLD_WORKER_IDLE_TIMEOUT", str(ServerConfig.WORKER_IDLE_TIMEOUT))
)
MAX_SESSIONS = ServerConfig.MAX_SESSIONS
MAX_ACTION_CHARS = ServerConfig.MAX_ACTION_CHARS
MAX_OBSERVATION_CHARS = ServerConfig.MAX_OBSERVATION_CHARS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("AppWorldServerResolved")


@dataclass
class Session:
    process: multiprocessing.Process
    connection: Connection
    created_at: float
    last_used_at: float
    request_id: str


@dataclass
class Failure:
    code: str
    recorded_at: float


PROCESS_REGISTRY: dict[str, Session] = {}
PROCESS_GRAVEYARD: OrderedDict[str, Failure] = OrderedDict()
SESSION_LOCKS: dict[str, asyncio.Lock] = {}
PENDING_SESSION_NAMES: set[str] = set()


def resolve_output_directory(raw_path: Optional[str]) -> Optional[str]:
    """Limit AppWorld output writes/deletions to an explicitly allowed root."""
    if raw_path is None:
        return None
    candidate = Path(raw_path).expanduser().resolve()
    allowed_root = Path(ServerConfig.ALLOWED_OUTPUT_ROOT).resolve()
    if candidate != allowed_root and allowed_root not in candidate.parents:
        raise ValueError(
            "experiments_outputs_directory must be under "
            "APPWORLD_ALLOWED_OUTPUT_ROOT"
        )
    return str(candidate)


def session_lock(experiment_name: str) -> asyncio.Lock:
    lock = SESSION_LOCKS.get(experiment_name)
    if lock is None:
        lock = asyncio.Lock()
        SESSION_LOCKS[experiment_name] = lock
    return lock


def schedule_lock_prune(experiment_name: str, lock: asyncio.Lock) -> None:
    """Drop inactive per-session locks after pending waiters have drained."""

    def prune() -> None:
        waiters = getattr(lock, "_waiters", None)
        has_waiters = bool(waiters)
        if (
            PROCESS_REGISTRY.get(experiment_name) is None
            and SESSION_LOCKS.get(experiment_name) is lock
            and not lock.locked()
            and not has_waiters
        ):
            SESSION_LOCKS.pop(experiment_name, None)

    asyncio.get_running_loop().call_soon(prune)


def remember_failure(experiment_name: str, code: str) -> None:
    PROCESS_GRAVEYARD.pop(experiment_name, None)
    PROCESS_GRAVEYARD[experiment_name] = Failure(code=code, recorded_at=time.time())
    while len(PROCESS_GRAVEYARD) > MAX_GRAVEYARD_ENTRIES:
        PROCESS_GRAVEYARD.popitem(last=False)


def safe_send(connection: Connection, payload: dict[str, Any]) -> bool:
    try:
        connection.send(payload)
        return True
    except (BrokenPipeError, EOFError, OSError):
        return False


def appworld_worker(
    connection: Connection,
    task_id: str,
    experiment_name: str,
    experiments_outputs_directory: Optional[str] = None,
    rm_outdir_after_finished: bool = True,
) -> None:
    """Own exactly one AppWorld instance and one end of its IPC pipe."""
    env = None
    env_closed = False
    os.environ["APPWORLD_ROOT"] = ServerConfig.APPWORLD_ROOT

    try:
        try:
            from appworld import AppWorld
            from appworld.evaluator import evaluate_task
        except Exception as exc:
            safe_send(
                connection,
                {
                    "status": "error",
                    "msg": f"Import Error: {exc}",
                    "code": ServerStatusCodes.ENV_INIT_FAILED,
                },
            )
            return

        started = time.time()
        logger.info("[%s] Worker initializing", experiment_name)
        try:
            env = AppWorld(
                task_id=task_id,
                experiment_name=experiment_name,
                remote_environment_url=None,
                timeout_seconds=ENV_INTERNAL_TIMEOUT,
                experiments_outputs_directory=experiments_outputs_directory,
                rm_outdir_after_finished=rm_outdir_after_finished,
            )
        except Exception as exc:
            safe_send(
                connection,
                {
                    "status": "error",
                    "msg": f"Init Failed: {exc}",
                    "code": ServerStatusCodes.ENV_INIT_FAILED,
                },
            )
            return

        if not safe_send(
            connection,
            {
                "status": "success",
                "msg": "Loaded",
                "duration": round(time.time() - started, 2),
                "code": ServerStatusCodes.SUCCESS,
            },
        ):
            return

        while True:
            if not connection.poll(timeout=WORKER_IDLE_TIMEOUT):
                logger.warning("[%s] Worker idle timeout", experiment_name)
                break

            try:
                command = connection.recv()
            except (EOFError, OSError):
                break

            command_type = command.get("type")
            command_started = time.time()
            try:
                if command_type == "get_init_messages":
                    app_descriptions = json.dumps(
                        [
                            {"name": name, "description": description}
                            for name, description in env.task.app_descriptions.items()
                        ],
                        indent=1,
                    )
                    system = Template(SYSTEM_PROMPT_TEMPLATE.lstrip()).render(
                        {
                            "main_user": env.task.supervisor,
                            "app_descriptions": app_descriptions,
                        }
                    )
                    response = {
                        "status": "success",
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": env.task.instruction},
                        ],
                        "code": ServerStatusCodes.SUCCESS,
                    }
                elif command_type == "step":
                    observation = env.execute(command.get("action"))
                    if not isinstance(observation, str):
                        observation = str(observation)
                    if len(observation) > MAX_OBSERVATION_CHARS:
                        observation = (
                            observation[:MAX_OBSERVATION_CHARS]
                            + "\n... [Output Truncated. "
                            + f"Total: {len(observation)} chars]"
                        )
                    response = {
                        "status": "success",
                        "observation": observation,
                        "code": ServerStatusCodes.SUCCESS,
                    }
                elif command_type == "completed":
                    response = {
                        "status": "success",
                        "finished": env.task_completed(),
                        "code": ServerStatusCodes.SUCCESS,
                    }
                elif command_type == "evaluate":
                    tracker = evaluate_task(
                        task_id=task_id,
                        experiment_name=experiment_name,
                        suppress_errors=True,
                        save_report=False,
                        experiments_outputs_directory=experiments_outputs_directory,
                    )
                    response = {
                        "status": "success",
                        "num_passes": len(tracker.passes),
                        "num_failures": len(tracker.failures),
                        "code": ServerStatusCodes.SUCCESS,
                    }
                elif command_type == "close":
                    env.close()
                    env_closed = True
                    response = {
                        "status": "success",
                        "code": ServerStatusCodes.SUCCESS,
                    }
                    response["duration"] = round(time.time() - command_started, 2)
                    safe_send(connection, response)
                    break
                else:
                    response = {
                        "status": "error",
                        "msg": f"Unknown command: {command_type}",
                        "code": ServerStatusCodes.BAD_REQUEST,
                    }

                response["duration"] = round(time.time() - command_started, 2)
                if not safe_send(connection, response):
                    break
            except Exception as exc:
                if not safe_send(
                    connection,
                    {
                        "status": "error",
                        "msg": f"Exec Error: {exc}",
                        "duration": round(time.time() - command_started, 2),
                        "code": ServerStatusCodes.EXECUTION_ERROR,
                    },
                ):
                    break
    except BaseException as exc:
        logger.exception("[%s] Worker crashed: %s", experiment_name, exc)
        safe_send(
            connection,
            {
                "status": "error",
                "msg": f"Worker Crash: {exc}",
                "code": ServerStatusCodes.INTERNAL_ERROR,
            },
        )
    finally:
        if env is not None and not env_closed:
            try:
                env.close()
            except Exception:
                logger.exception("[%s] Failed to close AppWorld", experiment_name)
        try:
            connection.close()
        except Exception:
            pass


def process_tree_rss_mb(pid: int) -> float:
    try:
        process = psutil.Process(pid)
        processes = [process, *process.children(recursive=True)]
        return sum(item.memory_info().rss for item in processes if item.is_running()) / 2**20
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


def stop_session(session: Session, graceful: bool, timeout: float = 5.0) -> None:
    """Close IPC and reap the child; never leave a zombie or live stale worker."""
    process = session.process
    connection = session.connection
    try:
        if graceful and process.is_alive():
            try:
                connection.send({"type": "close"})
                if connection.poll(timeout):
                    connection.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
            process.join(timeout)

        if process.is_alive():
            process.terminate()
            process.join(2)
        if process.is_alive():
            process.kill()
            process.join(2)
        else:
            process.join(timeout=0)
    finally:
        try:
            connection.close()
        except Exception:
            pass
        try:
            process.close()
        except (ValueError, AttributeError):
            pass


def communicate(session: Session, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    try:
        session.connection.send(payload)
        if session.connection.poll(timeout):
            return session.connection.recv()
        return {
            "status": "error",
            "msg": f"Worker Timeout ({timeout}s); session invalidated",
            "code": ServerStatusCodes.EXEC_TIMEOUT,
        }
    except (BrokenPipeError, EOFError, OSError) as exc:
        return {
            "status": "error",
            "msg": f"IPC Error: {exc}",
            "code": ServerStatusCodes.INTERNAL_ERROR,
        }


async def send_command_to_worker(
    experiment_name: str,
    request_id: str,
    payload: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    lock = session_lock(experiment_name)
    async with lock:
        session = PROCESS_REGISTRY.get(experiment_name)
        if session is None:
            failure = PROCESS_GRAVEYARD.get(experiment_name)
            if failure is None:
                return {
                    "status": "error",
                    "msg": "Session not found",
                    "code": ServerStatusCodes.SESSION_NOT_FOUND,
                }
            messages = {
                ServerStatusCodes.OOM_KILLED: (
                    f"Session killed by OOM guard (>{WORKER_MEMORY_LIMIT_MB} MB)."
                ),
                ServerStatusCodes.SESSION_EXPIRED: "Session expired due to inactivity.",
            }
            return {
                "status": "error",
                "msg": messages.get(failure.code, "Session terminated."),
                "code": failure.code,
            }

        if session.request_id != request_id:
            return {
                "status": "error",
                "msg": "request_id does not own this session",
                "code": ServerStatusCodes.BAD_REQUEST,
            }

        if not session.process.is_alive():
            PROCESS_REGISTRY.pop(experiment_name, None)
            remember_failure(experiment_name, ServerStatusCodes.SESSION_EXPIRED)
            await asyncio.to_thread(stop_session, session, False)
            schedule_lock_prune(experiment_name, lock)
            return {
                "status": "error",
                "msg": "Worker process died unexpectedly",
                "code": ServerStatusCodes.SESSION_EXPIRED,
            }

        session.last_used_at = time.time()
        response = await asyncio.to_thread(communicate, session, payload, timeout)
        if response.get("code") in {
            ServerStatusCodes.EXEC_TIMEOUT,
            ServerStatusCodes.INTERNAL_ERROR,
        }:
            # A timed-out command may reply later. Keeping the pipe would let that
            # stale reply be consumed by the next command and corrupt the protocol.
            PROCESS_REGISTRY.pop(experiment_name, None)
            remember_failure(experiment_name, response["code"])
            await asyncio.to_thread(stop_session, session, False)
            schedule_lock_prune(experiment_name, lock)
        return response


async def registry_monitor_task() -> None:
    logger.info(
        "Monitor started: memory_limit=%sMB, mp_start_method=%s",
        WORKER_MEMORY_LIMIT_MB,
        MP_START_METHOD,
    )
    while True:
        try:
            await asyncio.sleep(5)
            for experiment_name in list(PROCESS_REGISTRY):
                lock = session_lock(experiment_name)
                if lock.locked():
                    continue
                async with lock:
                    session = PROCESS_REGISTRY.get(experiment_name)
                    if session is None:
                        continue
                    if not session.process.is_alive():
                        PROCESS_REGISTRY.pop(experiment_name, None)
                        remember_failure(
                            experiment_name, ServerStatusCodes.SESSION_EXPIRED
                        )
                        await asyncio.to_thread(stop_session, session, False)
                        schedule_lock_prune(experiment_name, lock)
                        continue

                    rss_mb = process_tree_rss_mb(session.process.pid)
                    if rss_mb > WORKER_MEMORY_LIMIT_MB:
                        logger.error(
                            "[%s] OOM guard: %.0fMB > %sMB, pid=%s",
                            experiment_name,
                            rss_mb,
                            WORKER_MEMORY_LIMIT_MB,
                            session.process.pid,
                        )
                        PROCESS_REGISTRY.pop(experiment_name, None)
                        remember_failure(
                            experiment_name, ServerStatusCodes.OOM_KILLED
                        )
                        await asyncio.to_thread(stop_session, session, False)
                        schedule_lock_prune(experiment_name, lock)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("Monitor iteration failed")


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info(
        "Starting %s: init_timeout=%ss exec_timeout=%ss eval_timeout=%ss",
        SERVER_VERSION,
        INIT_TIMEOUT,
        EXEC_TIMEOUT,
        EVAL_TIMEOUT,
    )
    monitor = asyncio.create_task(registry_monitor_task())
    try:
        yield
    finally:
        monitor.cancel()
        try:
            await monitor
        except asyncio.CancelledError:
            pass
        sessions = list(PROCESS_REGISTRY.values())
        PROCESS_REGISTRY.clear()
        await asyncio.gather(
            *(asyncio.to_thread(stop_session, session, False) for session in sessions)
        )


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    process = psutil.Process()
    children = process.children(recursive=False)
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        fd_count = process.num_fds()
    except (AttributeError, psutil.AccessDenied):
        fd_count = len(os.listdir("/proc/self/fd"))
    dead_sessions = sum(
        not session.process.is_alive() for session in PROCESS_REGISTRY.values()
    )
    return {
        "status": "ok" if dead_sessions == 0 else "degraded",
        "version": SERVER_VERSION,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "uptime_seconds": round(time.time() - SERVER_STARTED_AT, 2),
        "active_sessions": len(PROCESS_REGISTRY),
        "initializing_sessions": len(PENDING_SESSION_NAMES),
        "max_sessions": MAX_SESSIONS,
        "dead_sessions": dead_sessions,
        "graveyard_entries": len(PROCESS_GRAVEYARD),
        "session_lock_entries": len(SESSION_LOCKS),
        "fd_count": fd_count,
        "nofile_soft_limit": soft_limit,
        "nofile_hard_limit": hard_limit,
        "rss_mb": round(process.memory_info().rss / 2**20, 2),
        "child_processes": len(children),
        "zombie_children": sum(
            child.status() == psutil.STATUS_ZOMBIE
            for child in children
            if child.is_running()
        ),
        "init_timeout_seconds": INIT_TIMEOUT,
        "exec_timeout_seconds": EXEC_TIMEOUT,
        "eval_timeout_seconds": EVAL_TIMEOUT,
        "worker_idle_timeout_seconds": WORKER_IDLE_TIMEOUT,
    }


@app.post("/init", response_model=EnvInitResponse)
async def init_env(request: InitRequest) -> EnvInitResponse:
    started = time.time()
    experiment_name = request.experiment_name
    lock = session_lock(experiment_name)

    async with lock:
        try:
            output_directory = resolve_output_directory(
                request.experiments_outputs_directory
            )
        except ValueError as exc:
            return EnvInitResponse(
                success=False,
                msg=str(exc),
                duration=round(time.time() - started, 2),
                code=ServerStatusCodes.BAD_REQUEST,
            )

        existing = PROCESS_REGISTRY.get(experiment_name)
        if existing is not None and existing.process.is_alive():
            if existing.request_id != request.request_id:
                return EnvInitResponse(
                    success=False,
                    msg="experiment_name is already owned by another request_id",
                    duration=round(time.time() - started, 2),
                    code=ServerStatusCodes.BAD_REQUEST,
                )
            return EnvInitResponse(
                success=True,
                msg="Already initialized",
                duration=round(time.time() - started, 2),
                code=ServerStatusCodes.SUCCESS,
            )
        if existing is not None:
            PROCESS_REGISTRY.pop(experiment_name, None)
            await asyncio.to_thread(stop_session, existing, False)

        if len(PROCESS_REGISTRY) + len(PENDING_SESSION_NAMES) >= MAX_SESSIONS:
            return EnvInitResponse(
                success=False,
                msg=f"server session capacity reached ({MAX_SESSIONS})",
                duration=round(time.time() - started, 2),
                code=ServerStatusCodes.BAD_REQUEST,
            )

        PROCESS_GRAVEYARD.pop(experiment_name, None)
        parent_connection, child_connection = MP_CONTEXT.Pipe()
        process = MP_CONTEXT.Process(
            target=appworld_worker,
            args=(
                child_connection,
                request.task_id,
                experiment_name,
                output_directory,
                request.rm_outdir_after_finished,
            ),
            daemon=True,
        )
        PENDING_SESSION_NAMES.add(experiment_name)

        try:
            process.start()
            # The parent must not retain the child's pipe end across sessions.
            child_connection.close()

            def wait_for_init() -> Optional[dict[str, Any]]:
                deadline = time.monotonic() + INIT_TIMEOUT
                while time.monotonic() < deadline:
                    try:
                        if parent_connection.poll(1):
                            return parent_connection.recv()
                    except (EOFError, OSError) as exc:
                        return {
                            "status": "error",
                            "msg": f"IPC init error: {exc}",
                            "code": ServerStatusCodes.ENV_INIT_FAILED,
                        }
                    if not process.is_alive():
                        return {
                            "status": "error",
                            "msg": "Worker died during init",
                            "code": ServerStatusCodes.ENV_INIT_FAILED,
                        }
                return None

            response = await asyncio.to_thread(wait_for_init)
            duration = round(time.time() - started, 2)
            if response and response.get("status") == "success":
                PENDING_SESSION_NAMES.discard(experiment_name)
                PROCESS_REGISTRY[experiment_name] = Session(
                    process=process,
                    connection=parent_connection,
                    created_at=time.time(),
                    last_used_at=time.time(),
                    request_id=request.request_id,
                )
                return EnvInitResponse(
                    success=True,
                    duration=duration,
                    code=ServerStatusCodes.SUCCESS,
                )

            code = (
                response.get("code", ServerStatusCodes.INIT_TIMEOUT)
                if response
                else ServerStatusCodes.INIT_TIMEOUT
            )
            message = response.get("msg", "Init Timeout") if response else "Init Timeout"
            PENDING_SESSION_NAMES.discard(experiment_name)
            remember_failure(experiment_name, code)
            if process.pid is not None:
                await asyncio.to_thread(
                    stop_session,
                    Session(
                        process,
                        parent_connection,
                        time.time(),
                        time.time(),
                        request.request_id,
                    ),
                    False,
                )
            else:
                parent_connection.close()
                process.close()
            logger.error("[%s] init failed: %s", experiment_name, message)
            schedule_lock_prune(experiment_name, lock)
            return EnvInitResponse(
                success=False,
                msg=message,
                duration=duration,
                code=code,
            )
        except BaseException as exc:
            PENDING_SESSION_NAMES.discard(experiment_name)
            try:
                child_connection.close()
            except Exception:
                pass
            if process.pid is not None:
                await asyncio.to_thread(
                    stop_session,
                    Session(
                        process,
                        parent_connection,
                        time.time(),
                        time.time(),
                        request.request_id,
                    ),
                    False,
                )
            else:
                parent_connection.close()
                process.close()
            remember_failure(experiment_name, ServerStatusCodes.INTERNAL_ERROR)
            schedule_lock_prune(experiment_name, lock)
            if not isinstance(exc, Exception):
                raise
            return EnvInitResponse(
                success=False,
                msg=str(exc),
                duration=round(time.time() - started, 2),
                code=ServerStatusCodes.INTERNAL_ERROR,
            )


@app.post("/get_init_messages", response_model=GetInitMsgResponse)
async def get_init_messages(request: GetInitMsgRequest) -> GetInitMsgResponse:
    started = time.time()
    response = await send_command_to_worker(
        request.experiment_name,
        request.request_id,
        {"type": "get_init_messages"},
        EXEC_TIMEOUT,
    )
    return GetInitMsgResponse(
        success=response["status"] == "success",
        messages=response.get("messages", []),
        msg=response.get("msg", ""),
        code=response.get("code", ServerStatusCodes.INTERNAL_ERROR),
        duration=response.get("duration", time.time() - started),
    )


@app.post("/step", response_model=EnvStepResponse)
async def step_env(request: StepRequest) -> EnvStepResponse:
    started = time.time()
    if len(request.action) > MAX_ACTION_CHARS:
        return EnvStepResponse(
            success=False,
            msg=f"action exceeds APPWORLD_MAX_ACTION_CHARS={MAX_ACTION_CHARS}",
            code=ServerStatusCodes.BAD_REQUEST,
            duration=round(time.time() - started, 2),
        )
    response = await send_command_to_worker(
        request.experiment_name,
        request.request_id,
        {"type": "step", "action": request.action},
        EXEC_TIMEOUT,
    )
    return EnvStepResponse(
        success=response["status"] == "success",
        observation=response.get("observation", ""),
        msg=response.get("msg", ""),
        code=response.get("code", ServerStatusCodes.INTERNAL_ERROR),
        duration=response.get("duration", time.time() - started),
    )


@app.post("/completed", response_model=EnvCompleteResponse)
async def check_completed(request: CheckCompleteRequest) -> EnvCompleteResponse:
    started = time.time()
    response = await send_command_to_worker(
        request.experiment_name,
        request.request_id,
        {"type": "completed"},
        EXEC_TIMEOUT,
    )
    return EnvCompleteResponse(
        success=response["status"] == "success",
        finished=response.get("finished", False),
        msg=response.get("msg", ""),
        code=response.get("code", ServerStatusCodes.INTERNAL_ERROR),
        duration=response.get("duration", time.time() - started),
    )


@app.post("/evaluate", response_model=EnvEvaluateResponse)
async def evaluate_env(request: EvaluateRequest) -> EnvEvaluateResponse:
    started = time.time()
    response = await send_command_to_worker(
        request.experiment_name,
        request.request_id,
        {"type": "evaluate"},
        EVAL_TIMEOUT,
    )
    if response["status"] != "success":
        return EnvEvaluateResponse(
            success=False,
            msg=response.get("msg", ""),
            duration=response.get("duration", time.time() - started),
            code=response.get("code", ServerStatusCodes.INTERNAL_ERROR),
        )

    passes = response.get("num_passes", 0)
    failures = response.get("num_failures", 0)
    reward = (
        (1.0 if failures == 0 else 0.0)
        if request.sparse
        else (passes / (passes + failures) if passes + failures else 0.0)
    )
    return EnvEvaluateResponse(
        success=True,
        reward_score=reward,
        num_passes=passes,
        num_failures=failures,
        duration=response.get("duration", time.time() - started),
        code=ServerStatusCodes.SUCCESS,
    )


@app.post("/close", response_model=EnvCloseResponse)
async def close_env(request: CloseRequest) -> EnvCloseResponse:
    started = time.time()
    experiment_name = request.experiment_name
    lock = session_lock(experiment_name)
    async with lock:
        session = PROCESS_REGISTRY.get(experiment_name)
        if session is not None and session.request_id != request.request_id:
            return EnvCloseResponse(
                success=False,
                msg="request_id does not own this session",
                duration=round(time.time() - started, 2),
                code=ServerStatusCodes.BAD_REQUEST,
            )
        session = PROCESS_REGISTRY.pop(experiment_name, None)
        PROCESS_GRAVEYARD.pop(experiment_name, None)
        if session is not None:
            await asyncio.to_thread(stop_session, session, True)
    schedule_lock_prune(experiment_name, lock)
    return EnvCloseResponse(
        success=True,
        msg="Closed",
        duration=time.time() - started,
        code=ServerStatusCodes.SUCCESS,
    )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("APPWORLD_HOST", "127.0.0.1")
    try:
        loopback = host.lower() == "localhost" or ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if not loopback and os.getenv("APPWORLD_ALLOW_REMOTE_BIND") != "1":
        raise RuntimeError(
            "Refusing non-loopback APPWORLD_HOST without "
            "APPWORLD_ALLOW_REMOTE_BIND=1"
        )
    if not loopback:
        logger.warning(
            "AppWorld executes model-generated Python; expose this service only "
            "on an isolated trusted network."
        )
    uvicorn.run(
        app,
        host=host,
        port=int(os.getenv("APPWORLD_START_PORT", "32000")),
    )
