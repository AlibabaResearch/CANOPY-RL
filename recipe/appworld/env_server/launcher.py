#!/usr/bin/env python3
"""Supervise a configurable bank of resolved AppWorld Uvicorn servers."""

from __future__ import annotations

import os
import resource
import signal
import subprocess
import sys
import time
import urllib.request
from ipaddress import ip_address
from pathlib import Path


START_PORT = int(os.getenv("APPWORLD_START_PORT", "32000"))
NUM_SERVERS = int(os.getenv("APPWORLD_NUM_SERVERS", "8"))
MAX_SERVERS = int(os.getenv("APPWORLD_MAX_SERVERS", "256"))
HOST = os.getenv("APPWORLD_HOST", "127.0.0.1")
CONFIG_DIR = Path(
    os.getenv("APPWORLD_CONFIG_DIR")
    or os.getenv("APPWORLD_SERVER_URL_DIR")
    or str(Path(__file__).resolve().parents[3] / "runtime" / "appworld_urls")
)
LOG_LEVEL = os.getenv("APPWORLD_LOG_LEVEL", "warning")
SERVER_MODULE = os.getenv(
    "APPWORLD_SERVER_MODULE", "recipe.appworld.env_server.server:app"
)
PID_FILE = Path(
    os.getenv("APPWORLD_LAUNCHER_PID_FILE", str(CONFIG_DIR / "launcher.pid"))
)
TARGET_NOFILE = int(os.getenv("APPWORLD_NOFILE_LIMIT", "65535"))
processes: list[subprocess.Popen] = []
stopping = False


def set_nofile_limit() -> int:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(TARGET_NOFILE, hard)
    if soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    effective = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    print(f"RLIMIT_NOFILE soft={effective}, hard={hard}", flush=True)
    if effective < min(TARGET_NOFILE, 8192):
        raise RuntimeError(
            f"RLIMIT_NOFILE={effective} is unsafe; expected at least "
            f"{min(TARGET_NOFILE, 8192)}"
        )
    return effective


def is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def advertised_ip() -> str:
    """Return a validated address and enforce explicit remote exposure."""
    configured = os.getenv("APPWORLD_ADVERTISE_IP")
    if is_loopback_host(HOST):
        return configured or "127.0.0.1"
    if os.getenv("APPWORLD_ALLOW_REMOTE_BIND") != "1":
        raise RuntimeError(
            "Refusing non-loopback APPWORLD_HOST. Set "
            "APPWORLD_ALLOW_REMOTE_BIND=1 only on an isolated trusted network."
        )
    if HOST in {"0.0.0.0", "::"} and not configured:
        raise RuntimeError(
            "APPWORLD_ADVERTISE_IP is required when binding a wildcard address"
        )
    print(
        "WARNING: AppWorld executes model-generated Python; remote bind is "
        "safe only on an isolated trusted network.",
        flush=True,
    )
    return configured or HOST


def url_host(host: str) -> str:
    """Bracket IPv6 literals when constructing endpoint URLs."""
    try:
        return f"[{host}]" if ip_address(host).version == 6 else host
    except ValueError:
        if any(character in host for character in "/?#@"):
            raise ValueError(f"invalid APPWORLD_ADVERTISE_IP: {host!r}")
        return host


def command_for(port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "uvicorn",
        SERVER_MODULE,
        "--host",
        HOST,
        "--port",
        str(port),
        "--workers",
        "1",
        "--log-level",
        LOG_LEVEL,
        "--no-access-log",
    ]


def start_one(index: int) -> subprocess.Popen:
    port = START_PORT + index
    process = subprocess.Popen(command_for(port), env=os.environ.copy())
    print(f"started index={index} pid={process.pid} port={port}", flush=True)
    return process


def stop_all(*_: object) -> None:
    global stopping
    stopping = True
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 15
    for process in processes:
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    try:
        PID_FILE.unlink()
    except FileNotFoundError:
        pass
    try:
        print("stopped resolved AppWorld servers", flush=True)
    except BrokenPipeError:
        pass


def wait_for_health(url: str, timeout: float = 90) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(0.25)
    raise RuntimeError(f"server did not become healthy: {url}: {last_error}")


def write_urls_atomically(ip: str) -> list[str]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    endpoint_host = url_host(ip)
    urls = [
        f"http://{endpoint_host}:{START_PORT + index}"
        for index in range(NUM_SERVERS)
    ]
    safe_name = ip.replace(":", "_")
    destination = CONFIG_DIR / f"{safe_name}_appworld_urls.txt"
    temporary = destination.with_suffix(".txt.tmp")
    temporary.write_text("\n".join(urls) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    print(f"wrote {len(urls)} URLs to {destination}", flush=True)
    return urls


def main() -> None:
    if not 1 <= NUM_SERVERS <= MAX_SERVERS <= 1024:
        raise ValueError(f"APPWORLD_NUM_SERVERS out of range: {NUM_SERVERS}")
    if START_PORT + NUM_SERVERS - 1 > 65535:
        raise ValueError("configured port range exceeds 65535")

    set_nofile_limit()
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(f"{os.getpid()}\n", encoding="utf-8")
    signal.signal(signal.SIGINT, stop_all)
    signal.signal(signal.SIGTERM, stop_all)
    signal.signal(signal.SIGHUP, stop_all)

    ip = advertised_ip()
    try:
        for index in range(NUM_SERVERS):
            processes.append(start_one(index))

        # Publish endpoints only after every process has started successfully.
        health_host = "127.0.0.1" if HOST in {"0.0.0.0", "::"} else HOST
        health_host = url_host(health_host)
        local_urls = [f"http://{health_host}:{START_PORT + i}" for i in range(NUM_SERVERS)]
        for url in local_urls:
            wait_for_health(url)
        write_urls_atomically(ip)

        while not stopping:
            time.sleep(2)
            if stopping:
                break
            for index, process in enumerate(list(processes)):
                if process.poll() is None:
                    continue
                print(
                    f"server index={index} exited rc={process.returncode}; restarting",
                    flush=True,
                )
                processes[index] = start_one(index)
                wait_for_health(f"http://{health_host}:{START_PORT + index}")
    finally:
        stop_all()


if __name__ == "__main__":
    main()
