"""Tests for the optional per-node SWE container collector."""

# Copyright 2026 Alibaba Group Holding Limited
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import threading
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from recipe.swe.env_server.container_gc import (
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
    _CommandOutcome,
    _NodeContainerGCCore,
)
from recipe.swe.env_server.environments import (
    _run_capture_async,
    _run_exec_streaming_async,
)


OWNER = "01aabbcc"
NODE_ID = "ab" * 28


def _request(index: int, *, priority: int = GC_PRIORITY_NORMAL) -> ContainerGCRequest:
    return ContainerGCRequest(
        container_id=f"{index:064x}",
        container_name=f"minisweagent-{index:08x}",
        owner=OWNER,
        nonce=f"{index + 100:032x}",
        node_id=NODE_ID,
        group_id=7,
        role="rollout",
        request_hash=f"request-{index}",
        reason="timeout" if priority == GC_PRIORITY_URGENT else "completed",
        priority=priority,
    )


def _inspect_payload(request: ContainerGCRequest, **label_overrides: str) -> str:
    labels = request.expected_labels()
    labels.update(label_overrides)
    return json.dumps(
        [
            {
                "Id": request.container_id,
                "Name": request.container_name,
                "Config": {"Labels": labels},
                "State": {"Running": True},
            }
        ]
    )


class _HappyRunner:
    def __init__(self, requests: list[ContainerGCRequest]):
        self.requests = {request.container_id: request for request in requests}
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], timeout: float) -> _CommandOutcome:
        del timeout
        self.calls.append(list(argv))
        container_id = argv[-1]
        if argv[:3] == ["podman", "container", "exists"]:
            return _CommandOutcome(0)
        if argv[:2] == ["podman", "inspect"]:
            return _CommandOutcome(0, _inspect_payload(self.requests[container_id]))
        if argv[:2] == ["podman", "rm"]:
            return _CommandOutcome(0)
        raise AssertionError(f"unexpected command: {argv}")


def _run(coro):
    return asyncio.run(coro)


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def _wait_for_processes_gone(pids: list[int]) -> None:
    for _ in range(100):
        if all(not _process_exists(pid) for pid in pids):
            return
        await asyncio.sleep(0.02)


@pytest.mark.skipif(os.name != "posix", reason="process-group cleanup is POSIX-only")
@pytest.mark.parametrize("mode", ["timeout", "cancel"])
def test_capture_runner_reaps_the_whole_process_group(tmp_path, mode):
    pid_file = tmp_path / f"{mode}.pids"
    command = [
        "/bin/sh",
        "-c",
        'sleep 30 & printf "%s %s\\n" "$$" "$!" > "$1"; wait',
        "capture-test",
        str(pid_file),
    ]

    async def scenario():
        pids: list[int] = []
        task = asyncio.create_task(
            _run_capture_async(command, 2.0 if mode == "timeout" else 30)
        )
        try:
            for _ in range(250):
                if pid_file.exists() and pid_file.read_text().strip():
                    break
                await asyncio.sleep(0.02)
            assert pid_file.exists()
            pids = [int(value) for value in pid_file.read_text().split()]
            assert len(pids) == 2

            if mode == "timeout":
                result = await task
                assert result["timeout"] is True
            else:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

            await _wait_for_processes_gone(pids)
            assert all(not _process_exists(pid) for pid in pids)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            for pid in pids:
                if _process_exists(pid):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    _run(scenario())


def test_capture_runner_preserves_output_and_return_code():
    result = _run(
        _run_capture_async(
            [
                sys.executable,
                "-c",
                "import sys; print('stdout-ok'); print('stderr-ok', file=sys.stderr); sys.exit(7)",
            ],
            5,
        )
    )

    assert result["stdout"].strip() == "stdout-ok"
    assert result["stderr"].strip() == "stderr-ok"
    assert result["returncode"] == 7
    assert result["timeout"] is False
    assert result["exception"] == ""


@pytest.mark.skipif(os.name != "posix", reason="process-group cleanup is POSIX-only")
def test_streaming_runner_cancellation_stops_thread_and_process_group(tmp_path):
    pid_file = tmp_path / "streaming-cancel.pids"
    command = [
        "/bin/sh",
        "-c",
        'sleep 30 & printf "%s %s\\n" "$$" "$!" > "$1"; wait',
        "streaming-test",
        str(pid_file),
    ]

    async def scenario():
        pids: list[int] = []
        task = asyncio.create_task(
            _run_exec_streaming_async(command, 30, 1024 * 1024)
        )
        try:
            for _ in range(100):
                if pid_file.exists() and pid_file.read_text().strip():
                    break
                await asyncio.sleep(0.02)
            assert pid_file.exists()
            pids = [int(value) for value in pid_file.read_text().split()]
            assert len(pids) == 2

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            await _wait_for_processes_gone(pids)
            assert all(not _process_exists(pid) for pid in pids)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            for pid in pids:
                if _process_exists(pid):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    _run(scenario())


def test_client_propagates_local_cancellation_to_remote_ref(monkeypatch):
    from recipe.swe.env_server import swe_env_client as client_module

    cancel_calls: list[tuple[object, bool]] = []

    def fake_ray_cancel(object_ref, force):
        cancel_calls.append((object_ref, force))
        object_ref.cancel()

    monkeypatch.setattr(client_module.ray, "cancel", fake_ray_cancel)
    client = object.__new__(client_module.SWEEnvClient)

    async def scenario():
        remote_ref = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(
            client._await_env_remote(
                remote_ref,
                cancellation_grace_seconds=0.1,
            )
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return remote_ref

    remote_ref = _run(scenario())
    assert cancel_calls == [(remote_ref, False)]
    assert remote_ref.cancelled()


def test_settings_validate_worker_boundaries():
    assert ContainerGCSettings().workers_per_node == 1
    assert ContainerGCSettings(workers_per_node=2).workers_per_node == 2
    with pytest.raises(ValueError):
        ContainerGCSettings(workers_per_node=0)
    with pytest.raises(ValueError):
        ContainerGCSettings(workers_per_node=3)
    with pytest.raises(ValueError):
        ContainerGCSettings(queue_maxsize=0)


def test_request_requires_exact_managed_identity():
    request = _request(1)
    assert request.expected_labels() == {
        LABEL_MANAGED: "true",
        LABEL_OWNER: OWNER,
        LABEL_NONCE: request.nonce,
        LABEL_NODE_ID: NODE_ID,
        LABEL_GROUP_ID: "7",
        LABEL_ROLE: "rollout",
        LABEL_REQUEST: "request-1",
    }
    with pytest.raises(ValueError):
        ContainerGCRequest(**{**asdict(request), "container_id": "abc"})
    assert ContainerGCRequest(**{**asdict(request), "container_id": ""}).container_id == ""
    with pytest.raises(ValueError):
        ContainerGCRequest(**{**asdict(request), "container_name": "other"})


def test_disabled_environment_command_has_no_gc_identity_labels():
    from recipe.swe.env_server.config import DockerEnvironmentConfig
    from recipe.swe.env_server.environments import DockerEnvironment

    config = DockerEnvironmentConfig(
        image="example.invalid/swe:test",
        enable_lxcfs_cpu_view=False,
        container_gc_enabled=False,
    )
    original_run_args = list(config.run_args)
    environment = DockerEnvironment(config)
    name, command, _, _ = environment.prepare_container_start_cmd()
    rendered = "\n".join(command)
    assert name.startswith("minisweagent-")
    assert "io.verl.swe.gc." not in rendered
    assert config.run_args == original_run_args


def test_already_absent_never_invokes_remove():
    calls: list[list[str]] = []

    def runner(argv: list[str], timeout: float) -> _CommandOutcome:
        del timeout
        calls.append(list(argv))
        return _CommandOutcome(1)

    async def scenario():
        core = _NodeContainerGCCore(ContainerGCSettings(), OWNER, NODE_ID, runner)
        core.start()
        assert (await core.enqueue(asdict(_request(1))))["accepted"]
        report = await core.drain(2)
        await core.close()
        return report

    report = _run(scenario())
    assert report["already_absent"] == 1
    assert calls == [["podman", "container", "exists", _request(1).container_id]]


def test_exact_labels_and_full_id_are_required_before_remove():
    good = _request(2)
    bad = _request(3)
    calls: list[list[str]] = []

    def runner(argv: list[str], timeout: float) -> _CommandOutcome:
        del timeout
        calls.append(list(argv))
        container_id = argv[-1]
        if argv[:3] == ["podman", "container", "exists"]:
            return _CommandOutcome(0)
        if argv[:2] == ["podman", "inspect"]:
            request = good if container_id == good.container_id else bad
            overrides = {} if request is good else {LABEL_OWNER: "deadbeef"}
            return _CommandOutcome(0, _inspect_payload(request, **overrides))
        if argv[:2] == ["podman", "rm"]:
            return _CommandOutcome(0)
        raise AssertionError(argv)

    async def scenario():
        core = _NodeContainerGCCore(ContainerGCSettings(), OWNER, NODE_ID, runner)
        core.start()
        await core.enqueue(asdict(good))
        await core.enqueue(asdict(bad))
        report = await core.drain(2)
        await core.close()
        return report

    report = _run(scenario())
    rm_calls = [argv for argv in calls if argv[:2] == ["podman", "rm"]]
    assert rm_calls == [
        ["podman", "rm", "--force", "--time", "0", "--ignore", good.container_id]
    ]
    assert report["removed"] == 1
    assert report["permanent_refused"] == 1
    flattened = {token for argv in rm_calls for token in argv}
    assert not ({"--all", "--latest", "--filter", "rmi", "prune"} & flattened)


def test_retryable_exists_error_is_retried_not_treated_as_absent():
    request = _request(4)
    happy = _HappyRunner([request])
    exists_attempts = 0
    sleeps: list[float] = []

    def runner(argv: list[str], timeout: float) -> _CommandOutcome:
        nonlocal exists_attempts
        if argv[:3] == ["podman", "container", "exists"]:
            exists_attempts += 1
            if exists_attempts == 1:
                return _CommandOutcome(125, stderr="database is locked")
        return happy(argv, timeout)

    async def sleep_fn(seconds: float):
        sleeps.append(seconds)

    async def scenario():
        settings = ContainerGCSettings(max_retries=1, retry_backoff_seconds=0.01)
        core = _NodeContainerGCCore(settings, OWNER, NODE_ID, runner, sleep_fn)
        core.start()
        await core.enqueue(asdict(request))
        report = await core.drain(2)
        await core.close()
        return report

    report = _run(scenario())
    assert exists_attempts == 2
    assert report["removed"] == 1
    assert report["retries"] == 1
    assert len(sleeps) == 1


def test_name_only_request_retries_registration_then_removes_resolved_full_id():
    full_request = _request(12, priority=GC_PRIORITY_URGENT)
    name_only = ContainerGCRequest(
        **{**asdict(full_request), "container_id": ""}
    )
    calls: list[list[str]] = []
    name_exists_attempts = 0

    def runner(argv: list[str], timeout: float) -> _CommandOutcome:
        nonlocal name_exists_attempts
        del timeout
        calls.append(list(argv))
        if argv[:3] == ["podman", "container", "exists"]:
            if argv[-1] == name_only.container_name:
                name_exists_attempts += 1
                return _CommandOutcome(1 if name_exists_attempts == 1 else 0)
            return _CommandOutcome(0)
        if argv[:2] == ["podman", "inspect"]:
            assert argv[-1] == name_only.container_name
            return _CommandOutcome(0, _inspect_payload(full_request))
        if argv[:2] == ["podman", "rm"]:
            assert argv[-1] == full_request.container_id
            return _CommandOutcome(0)
        raise AssertionError(argv)

    async def no_sleep(seconds: float):
        del seconds

    async def scenario():
        core = _NodeContainerGCCore(
            ContainerGCSettings(max_retries=1),
            OWNER,
            NODE_ID,
            runner,
            no_sleep,
        )
        core.start()
        await core.enqueue(asdict(name_only))
        report = await core.drain(2)
        await core.close()
        return report

    report = _run(scenario())
    assert name_exists_attempts == 2
    assert report["retries"] == 1
    assert report["removed"] == 1
    assert report["already_absent"] == 0
    assert report["failed"] == 0
    assert [argv for argv in calls if argv[:2] == ["podman", "rm"]] == [
        [
            "podman",
            "rm",
            "--force",
            "--time",
            "0",
            "--ignore",
            full_request.container_id,
        ]
    ]


def test_name_only_request_never_visible_is_absent_after_registration_wait():
    full_request = _request(13, priority=GC_PRIORITY_URGENT)
    name_only = ContainerGCRequest(
        **{**asdict(full_request), "container_id": ""}
    )
    calls: list[list[str]] = []

    def runner(argv: list[str], timeout: float) -> _CommandOutcome:
        del timeout
        calls.append(list(argv))
        if argv[:3] == ["podman", "container", "exists"]:
            assert argv[-1] == name_only.container_name
            return _CommandOutcome(1)
        raise AssertionError(argv)

    async def no_sleep(seconds: float):
        del seconds

    async def scenario():
        core = _NodeContainerGCCore(
            ContainerGCSettings(max_retries=2),
            OWNER,
            NODE_ID,
            runner,
            no_sleep,
        )
        core.start()
        await core.enqueue(asdict(name_only))
        report = await core.drain(2)
        await core.close()
        return report

    report = _run(scenario())
    assert len(calls) == 3
    assert report["retries"] == 2
    assert report["already_absent"] == 1
    assert report["removed"] == 0
    assert report["failed"] == 0


def test_urgent_items_overtake_queued_normal_items():
    normal_one = _request(5)
    normal_two = _request(6)
    urgent = _request(7, priority=GC_PRIORITY_URGENT)
    runner = _HappyRunner([normal_one, normal_two, urgent])

    async def scenario():
        core = _NodeContainerGCCore(ContainerGCSettings(), OWNER, NODE_ID, runner)
        # Fill the queue before starting the worker to make ordering deterministic.
        await core.enqueue(asdict(normal_one))
        await core.enqueue(asdict(normal_two))
        await core.enqueue(asdict(urgent))
        core.start()
        await core.drain(2)
        await core.close()

    _run(scenario())
    removed_ids = [argv[-1] for argv in runner.calls if argv[:2] == ["podman", "rm"]]
    assert removed_ids == [urgent.container_id, normal_one.container_id, normal_two.container_id]


def test_duplicate_descriptor_removes_only_once():
    request = _request(8)
    runner = _HappyRunner([request])

    async def scenario():
        core = _NodeContainerGCCore(ContainerGCSettings(), OWNER, NODE_ID, runner)
        first = await core.enqueue(asdict(request))
        second = await core.enqueue(asdict(request))
        core.start()
        report = await core.drain(2)
        await core.close()
        return first, second, report

    first, second, report = _run(scenario())
    assert first["accepted"] and not first["deduplicated"]
    assert second["accepted"] and second["deduplicated"]
    assert report["deduplicated"] == 1
    assert len([argv for argv in runner.calls if argv[:2] == ["podman", "rm"]]) == 1


def test_two_workers_never_exceed_configured_remove_concurrency():
    requests = [_request(9), _request(10)]
    request_map = {request.container_id: request for request in requests}
    gate = threading.Event()
    lock = threading.Lock()
    active = 0
    max_active = 0

    def runner(argv: list[str], timeout: float) -> _CommandOutcome:
        nonlocal active, max_active
        container_id = argv[-1]
        if argv[:3] == ["podman", "container", "exists"]:
            return _CommandOutcome(0)
        if argv[:2] == ["podman", "inspect"]:
            return _CommandOutcome(0, _inspect_payload(request_map[container_id]))
        if argv[:2] == ["podman", "rm"]:
            with lock:
                active += 1
                max_active = max(max_active, active)
                if active == 2:
                    gate.set()
            gate.wait(timeout=min(timeout, 10))
            with lock:
                active -= 1
            return _CommandOutcome(0)
        raise AssertionError(argv)

    async def scenario():
        core = _NodeContainerGCCore(
            ContainerGCSettings(workers_per_node=2), OWNER, NODE_ID, runner
        )
        core.start()
        for request in requests:
            await core.enqueue(asdict(request))
        report = await core.drain(3)
        await core.close()
        return report

    report = _run(scenario())
    assert max_active == 2
    assert report["max_inflight"] == 2


def test_client_release_submits_before_killing_ray_actor(monkeypatch):
    from recipe.swe.env_server import swe_env_client as client_module

    events: list[str] = []

    class FakeMethod:
        def remote(self, request):
            events.append(f"enqueue:{request['container_id']}")

            async def ack():
                return {"accepted": True}

            return ack()

    fake_actor = SimpleNamespace(enqueue=FakeMethod())
    monkeypatch.setattr(client_module, "get_gc_actor", lambda owner, node: fake_actor)
    monkeypatch.setattr(client_module.ray, "kill", lambda actor, no_restart: events.append("kill"))

    client = object.__new__(client_module.SWEEnvClient)
    client._ray_release_done = False
    client._gc_enqueued = False
    client._gc_urgent = False
    client._gc_settings = ContainerGCSettings(enabled=True)
    request = _request(11)
    client.container_id = request.container_id
    client.container_name = request.container_name
    client.env_config = SimpleNamespace(
        container_gc_owner=request.owner,
        container_gc_nonce=request.nonce,
        container_gc_node_id=request.node_id,
        container_gc_role=request.role,
        container_gc_request_hash=request.request_hash,
        group_id=request.group_id,
    )
    client.env = object()

    _run(client.kill_ray_actor(reason="completed"))
    assert events == [f"enqueue:{request.container_id}", "kill"]
    assert client.env is None


def test_client_release_retries_ray_kill_after_transient_failure(monkeypatch):
    from recipe.swe.env_server import swe_env_client as client_module

    attempts = 0

    def flaky_kill(actor, no_restart):
        nonlocal attempts
        del actor, no_restart
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient ray control-plane error")

    monkeypatch.setattr(client_module.ray, "kill", flaky_kill)
    client = object.__new__(client_module.SWEEnvClient)
    client._ray_release_done = False
    client._gc_enqueued = False
    client._gc_urgent = False
    client._gc_settings = ContainerGCSettings(enabled=False)
    client.env = object()

    assert _run(client.kill_ray_actor()) is True
    assert client.env is None
    assert attempts == 2


def test_client_release_preserves_actor_after_two_ray_kill_failures(monkeypatch):
    from recipe.swe.env_server import swe_env_client as client_module

    attempts = 0

    def failing_kill(actor, no_restart):
        nonlocal attempts
        del actor, no_restart
        attempts += 1
        raise RuntimeError("ray control plane unavailable")

    monkeypatch.setattr(client_module.ray, "kill", failing_kill)
    client = object.__new__(client_module.SWEEnvClient)
    client._ray_release_done = False
    client._gc_enqueued = False
    client._gc_urgent = False
    client._gc_settings = ContainerGCSettings(enabled=False)
    original_actor = object()
    client.env = original_actor

    assert _run(client.kill_ray_actor()) is False
    assert client.env is original_actor
    assert attempts == 2


def test_initialize_keeps_podman_stdout_id_when_metadata_rpc_fails(monkeypatch):
    from recipe.swe.env_server import swe_env_client as client_module

    request = _request(13)

    class FakeRemoteMethod:
        def __init__(self, value=None, error: Exception | None = None):
            self.value = value
            self.error = error

        def remote(self, *args, **kwargs):
            del args, kwargs

            async def result():
                if self.error is not None:
                    raise self.error
                return self.value

            return result()

    output = {
        "output": request.container_id,
        "returncode": 0,
        "exception_info": "",
        "extra": {"container_name": request.container_name},
        "timeout": False,
    }
    fake_env = SimpleNamespace(
        initialize=FakeRemoteMethod(output),
        get_container_metadata=FakeRemoteMethod(
            error=RuntimeError("metadata actor unavailable")
        ),
    )

    def attach_fake_env(self, **kwargs):
        del kwargs
        self.env = fake_env

    monkeypatch.setattr(
        client_module.SWEEnvClient,
        "_SWEEnvClient__init_ray_docker_env",
        attach_fake_env,
    )
    client = object.__new__(client_module.SWEEnvClient)
    client.env_config = SimpleNamespace(
        env_resource_tokens=10,
        env_cpu_limit="2",
        env_mem_limit="12g",
        ray_env_actor_num_cpus=0,
        map_testbed_to_tmpfs=False,
    )
    client.env = None
    client.container_id = ""
    client.container_name = request.container_name
    client._gc_settings = ContainerGCSettings(enabled=False)

    response = _run(client.initialize(max_start_sleep_seconds=0))
    assert response.success
    assert client.container_id == request.container_id
    assert client.container_name == request.container_name
