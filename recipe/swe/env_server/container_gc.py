#!/usr/bin/env python3
# Copyright 2026 Alibaba Group Holding Limited
# SPDX-License-Identifier: Apache-2.0

"""Per-node, bounded Podman container garbage collection for SWE environments.

The online collector is deliberately narrow: it resolves only an exact managed
name or full ID, verifies immutable ownership labels, and removes the resulting
full ID. It never enumerates containers, removes images, prunes storage, or
manipulates mounts.
The existing ``sleep <container_timeout>`` plus ``--rm`` lifecycle remains the
fallback when this optional collector is disabled or unavailable.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import logging
import os
import random
import re
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


logger = logging.getLogger(__name__)

GC_PROTOCOL_VERSION = 1
GC_PRIORITY_URGENT = 0
GC_PRIORITY_NORMAL = 10

CONTAINER_NAME_RE = re.compile(r"^minisweagent-[0-9a-f]{8}$")
FULL_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")

LABEL_MANAGED = "io.verl.swe.gc.managed"
LABEL_OWNER = "io.verl.swe.gc.owner"
LABEL_NONCE = "io.verl.swe.gc.nonce"
LABEL_NODE_ID = "io.verl.swe.gc.node_id"
LABEL_GROUP_ID = "io.verl.swe.gc.group_id"
LABEL_ROLE = "io.verl.swe.gc.role"
LABEL_REQUEST = "io.verl.swe.gc.request"


def _mapping_get(mapping: Mapping[str, Any], key: str, default: Any) -> Any:
    getter = getattr(mapping, "get", None)
    if callable(getter):
        return getter(key, default)
    return default


@dataclass(frozen=True)
class ContainerGCSettings:
    """Validated runtime settings shared by TaskRunner, clients, and actors."""

    enabled: bool = False
    workers_per_node: int = 1
    queue_maxsize: int = 4096
    remove_timeout_seconds: float = 120.0
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    enqueue_timeout_seconds: float = 5.0
    drain_timeout_seconds: float = 180.0

    def __post_init__(self) -> None:
        if self.workers_per_node not in (1, 2):
            raise ValueError("container_gc_workers_per_node must be 1 or 2")
        if self.queue_maxsize <= 0:
            raise ValueError("container_gc_queue_maxsize must be positive")
        if self.remove_timeout_seconds <= 0:
            raise ValueError("container_gc_remove_timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("container_gc_max_retries must be non-negative")
        if self.retry_backoff_seconds < 0:
            raise ValueError("container_gc_retry_backoff_seconds must be non-negative")
        if self.enqueue_timeout_seconds <= 0:
            raise ValueError("container_gc_enqueue_timeout_seconds must be positive")
        if self.drain_timeout_seconds <= 0:
            raise ValueError("container_gc_drain_timeout_seconds must be positive")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ContainerGCSettings":
        return cls(
            enabled=bool(_mapping_get(mapping, "container_gc_enabled", False)),
            workers_per_node=int(
                _mapping_get(mapping, "container_gc_workers_per_node", 1)
            ),
            queue_maxsize=int(
                _mapping_get(mapping, "container_gc_queue_maxsize", 4096)
            ),
            remove_timeout_seconds=float(
                _mapping_get(mapping, "container_gc_remove_timeout_seconds", 120.0)
            ),
            max_retries=int(
                _mapping_get(mapping, "container_gc_max_retries", 3)
            ),
            retry_backoff_seconds=float(
                _mapping_get(mapping, "container_gc_retry_backoff_seconds", 2.0)
            ),
            enqueue_timeout_seconds=float(
                _mapping_get(mapping, "container_gc_enqueue_timeout_seconds", 5.0)
            ),
            drain_timeout_seconds=float(
                _mapping_get(mapping, "container_gc_drain_timeout_seconds", 180.0)
            ),
        )


@dataclass(frozen=True)
class ContainerGCRequest:
    container_id: str
    container_name: str
    owner: str
    nonce: str
    node_id: str
    group_id: int
    role: str
    request_hash: str
    reason: str
    priority: int = GC_PRIORITY_NORMAL
    enqueued_at: float = 0.0

    def __post_init__(self) -> None:
        if self.container_id and not FULL_CONTAINER_ID_RE.fullmatch(self.container_id):
            raise ValueError(
                "container_id must be empty or a full 64-character lowercase hex ID"
            )
        if not CONTAINER_NAME_RE.fullmatch(self.container_name):
            raise ValueError("container_name does not match the managed MiniSWE pattern")
        if not self.owner or not self.nonce or not self.node_id:
            raise ValueError("owner, nonce, and node_id must be non-empty")
        if not re.fullmatch(r"[0-9a-f]{32}", self.nonce):
            raise ValueError("nonce must be 32 lowercase hex characters")
        if not re.fullmatch(r"[0-9a-f]+", self.owner):
            raise ValueError("owner must be a lowercase hex Ray job ID")
        if not re.fullmatch(r"[0-9a-f]+", self.node_id):
            raise ValueError("node_id must be lowercase hex")
        if self.group_id < 0:
            raise ValueError("group_id must be non-negative")
        if self.role not in {"rollout", "eval", "standalone"}:
            raise ValueError(f"unsupported container role: {self.role!r}")
        if self.priority not in {GC_PRIORITY_URGENT, GC_PRIORITY_NORMAL}:
            raise ValueError(f"unsupported GC priority: {self.priority}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ContainerGCRequest":
        return cls(
            container_id=str(value["container_id"]),
            container_name=str(value["container_name"]),
            owner=str(value["owner"]),
            nonce=str(value["nonce"]),
            node_id=str(value["node_id"]),
            group_id=int(value["group_id"]),
            role=str(value["role"]),
            request_hash=str(value.get("request_hash", "")),
            reason=str(value.get("reason", "completed")),
            priority=int(value.get("priority", GC_PRIORITY_NORMAL)),
            enqueued_at=float(value.get("enqueued_at", time.time())),
        )

    def fingerprint(self) -> str:
        stable = (
            self.container_id,
            self.container_name,
            self.owner,
            self.nonce,
            self.node_id,
            str(self.group_id),
            self.role,
            self.request_hash,
        )
        return hashlib.sha256("\0".join(stable).encode()).hexdigest()

    def queue_key(self) -> str:
        # The exact generated name exists before ``podman run`` starts, while
        # the full ID may be unavailable when that command or the Ray actor is
        # cancelled. Labels and nonce still make name-only recovery fail-closed.
        return self.container_name

    def expected_labels(self) -> dict[str, str]:
        return {
            LABEL_MANAGED: "true",
            LABEL_OWNER: self.owner,
            LABEL_NONCE: self.nonce,
            LABEL_NODE_ID: self.node_id,
            LABEL_GROUP_ID: str(self.group_id),
            LABEL_ROLE: self.role,
            LABEL_REQUEST: self.request_hash,
        }


@dataclass(frozen=True)
class _CommandOutcome:
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    exception: str = ""


class _RetryableGCError(RuntimeError):
    pass


class _ManagedNameNotVisible(_RetryableGCError):
    """A name-only registration may still be racing with container startup."""


class _PermanentGCRefusal(RuntimeError):
    pass


def _run_command(argv: list[str], timeout: float) -> _CommandOutcome:
    """Run one Podman CLI command with bounded process-group cleanup."""

    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            close_fds=True,
            start_new_session=True,
        )
        stdout, stderr = proc.communicate(timeout=timeout)
        return _CommandOutcome(proc.returncode, stdout, stderr)
    except subprocess.TimeoutExpired as exc:
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except Exception:
                stdout, stderr = "", ""
        else:
            stdout, stderr = "", ""
        return _CommandOutcome(
            None,
            stdout or str(getattr(exc, "output", "") or ""),
            stderr or str(getattr(exc, "stderr", "") or ""),
            timed_out=True,
            exception=f"command timed out after {timeout}s",
        )
    except Exception as exc:
        return _CommandOutcome(
            None,
            timed_out=False,
            exception=f"{type(exc).__name__}: {exc}",
        )


async def _run_command_async(
    argv: list[str], timeout: float
) -> _CommandOutcome:
    """Cancellation-safe production command runner.

    The child owns a separate process group. Both timeout and task
    cancellation kill that group and reap the process before returning, so a
    GC actor shutdown cannot orphan a Podman CLI command.
    """

    proc: asyncio.subprocess.Process | None = None
    communicate_task: asyncio.Task[tuple[bytes, bytes]] | None = None

    async def terminate_and_reap() -> tuple[bytes, bytes]:
        if proc is None:
            return b"", b""
        if proc.returncode is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
        try:
            if communicate_task is None:
                return b"", b""
            return await asyncio.wait_for(
                asyncio.shield(communicate_task), timeout=5
            )
        except BaseException:
            return b"", b""

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        communicate_task = asyncio.create_task(proc.communicate())
        stdout, stderr = await asyncio.wait_for(
            asyncio.shield(communicate_task), timeout=timeout
        )
        return _CommandOutcome(
            proc.returncode,
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
        )
    except asyncio.TimeoutError:
        stdout, stderr = await terminate_and_reap()
        return _CommandOutcome(
            None,
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
            timed_out=True,
            exception=f"command timed out after {timeout}s",
        )
    except asyncio.CancelledError:
        await terminate_and_reap()
        raise
    except Exception as exc:
        await terminate_and_reap()
        return _CommandOutcome(
            None,
            exception=f"{type(exc).__name__}: {exc}",
        )


class _NodeContainerGCCore:
    """Pure-Python queue core; the Ray actor below is intentionally thin."""

    def __init__(
        self,
        settings: ContainerGCSettings,
        owner: str,
        node_id: str,
        command_runner: Callable[[list[str], float], _CommandOutcome] | None = None,
        sleep_fn: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        self.settings = settings
        self.owner = owner
        self.node_id = node_id
        self._command_runner = command_runner
        self._sleep_fn = sleep_fn
        self._queue: asyncio.PriorityQueue[
            tuple[int, int, str, ContainerGCRequest]
        ] = asyncio.PriorityQueue(maxsize=settings.queue_maxsize)
        self._sequence = itertools.count()
        self._state_lock = asyncio.Lock()
        self._states: dict[str, str] = {}
        self._fingerprints: dict[str, str] = {}
        self._queued_at: dict[str, float] = {}
        self._workers: list[asyncio.Task[Any]] = []
        self._closing = False
        self._stats: dict[str, int | float] = {
            "enqueued": 0,
            "deduplicated": 0,
            "enqueue_rejected": 0,
            "remove_attempts": 0,
            "removed": 0,
            "already_absent": 0,
            "retries": 0,
            "permanent_refused": 0,
            "failed": 0,
            "remove_timeouts": 0,
            "inflight": 0,
            "max_inflight": 0,
            "last_success_at": 0.0,
        }

    def start(self) -> None:
        if self._workers:
            return
        self._workers = [
            asyncio.create_task(self._worker_loop(index), name=f"swe-container-gc-{index}")
            for index in range(self.settings.workers_per_node)
        ]

    async def enqueue(self, raw_request: Mapping[str, Any]) -> dict[str, Any]:
        try:
            request = ContainerGCRequest.from_mapping(raw_request)
        except Exception as exc:
            self._stats["enqueue_rejected"] += 1
            return {"accepted": False, "reason": f"invalid_request: {exc}"}
        if request.owner != self.owner or request.node_id != self.node_id:
            self._stats["enqueue_rejected"] += 1
            return {"accepted": False, "reason": "owner_or_node_mismatch"}

        key = request.queue_key()
        fingerprint = request.fingerprint()
        async with self._state_lock:
            if self._closing:
                self._stats["enqueue_rejected"] += 1
                return {"accepted": False, "reason": "actor_closing"}
            prior_fingerprint = self._fingerprints.get(key)
            if prior_fingerprint is not None:
                if prior_fingerprint == fingerprint:
                    self._stats["deduplicated"] += 1
                    return {
                        "accepted": True,
                        "deduplicated": True,
                        "state": self._states.get(key, "unknown"),
                    }
                self._stats["enqueue_rejected"] += 1
                return {"accepted": False, "reason": "conflicting_metadata"}
            item = (request.priority, next(self._sequence), key, request)
            try:
                self._queue.put_nowait(item)
            except asyncio.QueueFull:
                self._stats["enqueue_rejected"] += 1
                return {"accepted": False, "reason": "queue_full"}
            self._fingerprints[key] = fingerprint
            self._states[key] = "queued"
            self._queued_at[key] = time.time()
            self._stats["enqueued"] += 1
        return {
            "accepted": True,
            "deduplicated": False,
            "queue_depth": self._queue.qsize(),
        }

    async def _worker_loop(self, worker_index: int) -> None:
        del worker_index
        while True:
            try:
                _, _, key, request = await self._queue.get()
            except asyncio.CancelledError:
                return
            try:
                async with self._state_lock:
                    self._states[key] = "inflight"
                    self._queued_at.pop(key, None)
                self._stats["inflight"] += 1
                self._stats["max_inflight"] = max(
                    int(self._stats["max_inflight"]), int(self._stats["inflight"])
                )
                outcome = await self._remove_with_retry(request)
                self._states[key] = outcome
            except asyncio.CancelledError:
                self._states[key] = "cancelled"
                raise
            except Exception as exc:
                self._states[key] = "failed"
                self._stats["failed"] += 1
                logger.error(
                    "SWE container GC unexpected error on node=%s cid=%s: %s",
                    self.node_id[:12],
                    (request.container_id or request.container_name)[:20],
                    exc,
                )
            finally:
                self._stats["inflight"] = max(0, int(self._stats["inflight"]) - 1)
                self._queue.task_done()

    async def _run(self, argv: list[str]) -> _CommandOutcome:
        if self._command_runner is None:
            return await _run_command_async(
                argv,
                self.settings.remove_timeout_seconds,
            )
        return await asyncio.to_thread(
            self._command_runner,
            argv,
            self.settings.remove_timeout_seconds,
        )

    async def _exists(self, container_id: str) -> bool:
        outcome = await self._run(["podman", "container", "exists", container_id])
        if outcome.timed_out:
            self._stats["remove_timeouts"] += 1
            raise _RetryableGCError(outcome.exception or "podman exists timed out")
        if outcome.returncode == 0:
            return True
        if outcome.returncode == 1:
            return False
        raise _RetryableGCError(
            f"podman exists rc={outcome.returncode}: "
            f"{(outcome.stderr or outcome.exception)[-500:]}"
        )

    async def _inspect(
        self, request: ContainerGCRequest, identifier: str
    ) -> str:
        outcome = await self._run(["podman", "inspect", identifier])
        if outcome.timed_out:
            self._stats["remove_timeouts"] += 1
            raise _RetryableGCError(outcome.exception or "podman inspect timed out")
        if outcome.returncode != 0:
            if not await self._exists(identifier):
                raise FileNotFoundError(identifier)
            raise _RetryableGCError(
                f"podman inspect rc={outcome.returncode}: "
                f"{(outcome.stderr or outcome.exception)[-500:]}"
            )
        try:
            payload = json.loads(outcome.stdout)
            if not isinstance(payload, list) or len(payload) != 1:
                raise ValueError("inspect response is not a one-item list")
            inspected = payload[0]
        except Exception as exc:
            raise _RetryableGCError(f"invalid podman inspect JSON: {exc}") from exc

        actual_id = str(inspected.get("Id", ""))
        actual_name = str(inspected.get("Name", "")).lstrip("/")
        labels = (inspected.get("Config") or {}).get("Labels") or {}
        if not FULL_CONTAINER_ID_RE.fullmatch(actual_id):
            raise _PermanentGCRefusal("inspect returned a non-full container ID")
        if request.container_id and actual_id != request.container_id:
            raise _PermanentGCRefusal("full container ID mismatch")
        if actual_name != request.container_name or not CONTAINER_NAME_RE.fullmatch(actual_name):
            raise _PermanentGCRefusal("container name mismatch")
        for label, expected in request.expected_labels().items():
            if str(labels.get(label, "")) != expected:
                raise _PermanentGCRefusal(f"container label mismatch: {label}")
        return actual_id

    async def _remove_once(self, request: ContainerGCRequest) -> str:
        identifier = request.container_id or request.container_name
        if not await self._exists(identifier):
            if not request.container_id:
                # ``podman run`` may still be registering the pre-generated
                # name when an init task is cancelled. Retry for the normal
                # bounded retry window instead of permanently declaring it
                # absent on the first lookup.
                raise _ManagedNameNotVisible(
                    f"managed container name not visible yet: {request.container_name}"
                )
            return "already_absent"
        try:
            full_container_id = await self._inspect(request, identifier)
        except FileNotFoundError:
            if not request.container_id:
                raise _ManagedNameNotVisible(
                    f"managed container disappeared during name lookup: "
                    f"{request.container_name}"
                )
            return "already_absent"

        self._stats["remove_attempts"] += 1
        outcome = await self._run(
            [
                "podman",
                "rm",
                "--force",
                "--time",
                "0",
                "--ignore",
                full_container_id,
            ]
        )
        if outcome.timed_out:
            self._stats["remove_timeouts"] += 1
            raise _RetryableGCError(outcome.exception or "podman rm timed out")
        if outcome.returncode == 0:
            return "removed"
        if not await self._exists(full_container_id):
            return "already_absent"
        raise _RetryableGCError(
            f"podman rm rc={outcome.returncode}: "
            f"{(outcome.stderr or outcome.exception)[-500:]}"
        )

    async def _remove_with_retry(self, request: ContainerGCRequest) -> str:
        last_error: _RetryableGCError | None = None
        attempts = self.settings.max_retries + 1
        for attempt in range(attempts):
            try:
                outcome = await self._remove_once(request)
                self._stats[outcome] += 1
                self._stats["last_success_at"] = time.time()
                return outcome
            except _PermanentGCRefusal as exc:
                self._stats["permanent_refused"] += 1
                logger.error(
                    "SWE container GC refused node=%s cid=%s reason=%s",
                    self.node_id[:12],
                    (request.container_id or request.container_name)[:20],
                    exc,
                )
                return "permanent_refused"
            except _RetryableGCError as exc:
                last_error = exc
                if attempt >= self.settings.max_retries:
                    break
                self._stats["retries"] += 1
                backoff = min(
                    self.settings.retry_backoff_seconds * (2**attempt), 60.0
                )
                if backoff:
                    await self._sleep_fn(backoff + random.uniform(0.0, min(0.25, backoff / 10)))
        if isinstance(last_error, _ManagedNameNotVisible):
            self._stats["already_absent"] += 1
            self._stats["last_success_at"] = time.time()
            logger.info(
                "SWE container GC confirmed absent after registration wait node=%s name=%s",
                self.node_id[:12],
                request.container_name,
            )
            return "already_absent"
        self._stats["failed"] += 1
        logger.error(
            "SWE container GC exhausted retries node=%s cid=%s error=%s",
            self.node_id[:12],
            (request.container_id or request.container_name)[:20],
            str(last_error or "unknown GC error"),
        )
        return "failed"

    async def drain(self, timeout_seconds: float) -> dict[str, Any]:
        # Stop admission before observing queue.join(); otherwise a late
        # enqueue can race between join returning and worker shutdown.
        async with self._state_lock:
            self._closing = True
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout_seconds)
            drained = True
        except asyncio.TimeoutError:
            drained = False
        status = self.status()
        status["drained"] = drained
        return status

    async def close(self) -> dict[str, Any]:
        self._closing = True
        for task in self._workers:
            task.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        return self.status()

    def status(self) -> dict[str, Any]:
        now = time.time()
        oldest = max((now - value for value in self._queued_at.values()), default=0.0)
        return {
            "protocol_version": GC_PROTOCOL_VERSION,
            "owner": self.owner,
            "node_id": self.node_id,
            "workers_per_node": self.settings.workers_per_node,
            "queue_depth": self._queue.qsize(),
            "oldest_queued_seconds": oldest,
            **self._stats,
        }


@ray.remote(max_restarts=0)
class NodeContainerGC:
    """Thin Ray wrapper around one node-local GC queue."""

    def __init__(self, settings_dict: Mapping[str, Any], owner: str, node_id: str):
        actual_node_id = str(ray.get_runtime_context().get_node_id())
        if actual_node_id != node_id:
            raise RuntimeError(
                f"GC actor node mismatch: expected={node_id}, actual={actual_node_id}"
            )
        settings = ContainerGCSettings(**dict(settings_dict))
        self._core = _NodeContainerGCCore(settings, owner, actual_node_id)

    async def ready(self) -> dict[str, Any]:
        self._core.start()
        return self._core.status()

    async def enqueue(self, request: Mapping[str, Any]) -> dict[str, Any]:
        self._core.start()
        return await self._core.enqueue(request)

    async def drain(self, timeout_seconds: float) -> dict[str, Any]:
        self._core.start()
        return await self._core.drain(timeout_seconds)

    async def close(self) -> dict[str, Any]:
        return await self._core.close()

    async def status(self) -> dict[str, Any]:
        return self._core.status()


def current_ray_job_id() -> str:
    job_id = ray.get_runtime_context().get_job_id()
    hex_method = getattr(job_id, "hex", None)
    raw = hex_method() if callable(hex_method) else str(job_id)
    value = re.sub(r"[^0-9a-f]", "", str(raw).lower())
    if not value:
        raise RuntimeError(f"Unable to derive a stable Ray job ID from {raw!r}")
    return value


def request_hash(request_id: str) -> str:
    return hashlib.sha256(str(request_id).encode()).hexdigest()[:24]


def gc_actor_name(owner: str, node_id: str) -> str:
    return f"swe-container-gc-v{GC_PROTOCOL_VERSION}-{owner}-{node_id}"


def discover_group_nodes(expected_groups: int | None = None) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        group_keys = [
            key
            for key, value in (node.get("Resources") or {}).items()
            if value > 0 and re.fullmatch(r"group_[0-9]+", key)
        ]
        if not group_keys:
            continue
        if len(group_keys) != 1:
            raise RuntimeError(
                f"Node {node.get('NodeID')} has ambiguous group resources: {group_keys}"
            )
        group_id = int(group_keys[0].split("_", 1)[1])
        node_id = str(node.get("NodeID", ""))
        if not node_id or group_id in mapping:
            raise RuntimeError(f"Duplicate or invalid Ray group mapping for group_{group_id}")
        mapping[group_id] = node_id
    if expected_groups is not None:
        expected = set(range(expected_groups))
        missing = expected - set(mapping)
        if missing:
            raise RuntimeError(
                f"Missing expected group nodes {sorted(missing)}; "
                f"discovered {sorted(mapping)}"
            )
        mapping = {group_id: mapping[group_id] for group_id in sorted(expected)}
    return mapping


_ACTOR_HANDLE_CACHE: dict[tuple[str, str], Any] = {}
_ACTOR_HANDLE_LOCK = threading.Lock()
_GROUP_NODE_CACHE: dict[int, str] = {}


def get_gc_actor(owner: str, node_id: str) -> Any:
    key = (owner, node_id)
    with _ACTOR_HANDLE_LOCK:
        handle = _ACTOR_HANDLE_CACHE.get(key)
        if handle is None:
            handle = ray.get_actor(gc_actor_name(owner, node_id))
            _ACTOR_HANDLE_CACHE[key] = handle
        return handle


def resolve_group_node_id(group_id: int) -> str:
    """Resolve one custom ``group_N`` resource to its hard Ray node ID."""

    with _ACTOR_HANDLE_LOCK:
        cached = _GROUP_NODE_CACHE.get(group_id)
    if cached:
        return cached
    mapping = discover_group_nodes()
    if group_id not in mapping:
        raise RuntimeError(f"No alive Ray node advertises group_{group_id}")
    with _ACTOR_HANDLE_LOCK:
        _GROUP_NODE_CACHE.update(mapping)
        return _GROUP_NODE_CACHE[group_id]


class ContainerGCManager:
    """Long-lived owner for all non-detached node-local GC actors in one job."""

    def __init__(
        self,
        settings: ContainerGCSettings,
        owner: str,
        group_nodes: Mapping[int, str],
        actor_handles: Mapping[int, Any],
    ) -> None:
        self.settings = settings
        self.owner = owner
        self.group_nodes = dict(group_nodes)
        self.actor_handles = dict(actor_handles)

    @classmethod
    def start(
        cls, settings: ContainerGCSettings, expected_groups: int
    ) -> "ContainerGCManager | None":
        if not settings.enabled:
            return None
        owner = current_ray_job_id()
        group_nodes = discover_group_nodes(expected_groups)
        handles: dict[int, Any] = {}
        settings_dict = asdict(settings)
        try:
            for group_id, node_id in group_nodes.items():
                name = gc_actor_name(owner, node_id)
                handles[group_id] = NodeContainerGC.options(
                    name=name,
                    get_if_exists=True,
                    num_cpus=0,
                    max_concurrency=max(32, settings.workers_per_node * 8),
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node_id,
                        soft=False,
                    ),
                ).remote(settings_dict, owner, node_id)

            ready = ray.get(
                [handle.ready.remote() for handle in handles.values()], timeout=120
            )
            for group_id, status in zip(handles, ready, strict=True):
                expected_node = group_nodes[group_id]
                if (
                    status.get("protocol_version") != GC_PROTOCOL_VERSION
                    or status.get("owner") != owner
                    or status.get("node_id") != expected_node
                    or status.get("workers_per_node") != settings.workers_per_node
                ):
                    raise RuntimeError(
                        f"Container GC handshake failed for group_{group_id}: {status}"
                    )
        except Exception:
            for handle in handles.values():
                try:
                    ray.kill(handle, no_restart=True)
                except Exception:
                    pass
            raise

        with _ACTOR_HANDLE_LOCK:
            for group_id, expected_node in group_nodes.items():
                _ACTOR_HANDLE_CACHE[(owner, expected_node)] = handles[group_id]
                _GROUP_NODE_CACHE[group_id] = expected_node
        logger.info(
            "Started %d per-node SWE container GC actors (workers_per_node=%d)",
            len(handles),
            settings.workers_per_node,
        )
        return cls(settings, owner, group_nodes, handles)

    def drain_and_shutdown(self) -> dict[int, dict[str, Any]]:
        reports: dict[int, dict[str, Any]] = {}
        if not self.actor_handles:
            return reports
        group_ids = list(self.actor_handles)
        drain_refs = [
            self.actor_handles[group_id].drain.remote(
                self.settings.drain_timeout_seconds
            )
            for group_id in group_ids
        ]
        try:
            values = ray.get(
                drain_refs,
                timeout=self.settings.drain_timeout_seconds + 15,
            )
            reports.update(zip(group_ids, values, strict=True))
        except Exception:
            logger.exception("Timed out while draining SWE container GC actors")
        close_refs = []
        for handle in self.actor_handles.values():
            try:
                close_refs.append(handle.close.remote())
            except Exception:
                pass
        if close_refs:
            try:
                ray.get(close_refs, timeout=30)
            except Exception:
                logger.warning("Some SWE container GC actors did not close cleanly")
        for handle in self.actor_handles.values():
            try:
                ray.kill(handle, no_restart=True)
            except Exception:
                pass
        return reports
