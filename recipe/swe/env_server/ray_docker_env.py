#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@desc Ray Docker Env
@author: plm
@create: 2026-02-14
"""


import asyncio
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict

import ray

from recipe.swe.env_server.config import DockerEnvironmentConfig
from recipe.swe.env_server.environments import DockerEnvironment


# =====================================================================
# 单 Actor 进程 RSS 上限（用户态 watcher 线程实现）
#
# 为什么不用 RLIMIT_AS？
#   PyTorch/CUDA 一启动就预留几百 GB 虚拟地址空间（VMS），但实际 RSS
#   通常 < 1GB。RLIMIT_AS 限制的是 VMS，会让 actor 开机就 mmap fail。
#
# 这里用一个后台守护线程，每 N 秒读 psutil.Process.memory_info().rss，
# 超过阈值就 os._exit(137)，Ray 视为 actor 异常退出（RayActorError），
# 触发上游重试，但不会拖死节点上其他 actor 和训练 worker。
#
# 默认 32GB：正常 actor RSS < 1GB，64MB×并发 stdout 也 < 数 GB，
# 32GB 阈值给得很宽松，只兜住灾难场景。
# 设 SWE_RAY_DOCKER_ACTOR_RSS_GB=0 可禁用此 watcher。
# =====================================================================
_RSS_LIMIT_GB = int(os.environ.get("SWE_RAY_DOCKER_ACTOR_RSS_GB", "32"))
RAY_DOCKER_ACTOR_RSS_LIMIT = (
    _RSS_LIMIT_GB * 1024 * 1024 * 1024 if _RSS_LIMIT_GB > 0 else 0
)
_RSS_POLL_INTERVAL_SEC = float(os.environ.get("SWE_RAY_DOCKER_ACTOR_RSS_POLL_S", "5"))


def _start_rss_watchdog():
    """
    启动一个守护线程：每隔 _RSS_POLL_INTERVAL_SEC 检查本进程 RSS，
    超过 RAY_DOCKER_ACTOR_RSS_LIMIT 立即 os._exit(137)。
    """
    if RAY_DOCKER_ACTOR_RSS_LIMIT <= 0:
        print("[RayDockerEnviroment] RSS watchdog disabled (SWE_RAY_DOCKER_ACTOR_RSS_GB=0)", flush=True)
        return
    try:
        import psutil  # noqa: F401  (确认可用)
    except ImportError:
        print("[RayDockerEnviroment] WARN: psutil not available, RSS watchdog disabled", flush=True)
        return

    def _watch():
        import psutil
        proc = psutil.Process(os.getpid())
        limit = RAY_DOCKER_ACTOR_RSS_LIMIT
        last_log = 0.0
        while True:
            try:
                rss = proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return
            now = time.monotonic()
            # 每 5 分钟正常打一次 RSS（便于观察缓慢增长）
            if now - last_log > 300 and rss > 1 * 1024**3:
                print(
                    f"[RayDockerEnviroment] RSS watchdog: pid={os.getpid()} "
                    f"rss={rss / 1024**3:.2f}GB / cap={limit / 1024**3:.0f}GB",
                    flush=True,
                )
                last_log = now
            if rss > limit:
                # 立刻自杀；Ray 会捕获异常退出码 137
                print(
                    f"[RayDockerEnviroment] FATAL: RSS={rss / 1024**3:.2f}GB exceeds "
                    f"cap={limit / 1024**3:.0f}GB, killing self (pid={os.getpid()}). "
                    f"This is a defensive kill to protect the host from OOM.",
                    flush=True,
                )
                os._exit(137)
            time.sleep(_RSS_POLL_INTERVAL_SEC)

    t = threading.Thread(target=_watch, name="ray-docker-rss-watchdog", daemon=True)
    t.start()
    print(
        f"[RayDockerEnviroment] RSS watchdog started: cap={RAY_DOCKER_ACTOR_RSS_LIMIT / 1024**3:.0f}GB "
        f"poll={_RSS_POLL_INTERVAL_SEC}s pid={os.getpid()}",
        flush=True,
    )


@ray.remote
class RayDockerEnviroment:
    def __init__(self, config: DockerEnvironmentConfig):
        # 兜底防御：后台 watchdog 监控本进程 RSS，超阈值自杀。
        # 不会连累节点上其他 actor 和训练 worker，Ray 那侧会重试。
        _start_rss_watchdog()
        self.env = DockerEnvironment(config=config)
        self.logger = logging.getLogger("RayDockerEnv")
        return

    async def get_container_id(self):
        return self.env.container_id

    async def get_container_metadata(self):
        """Return the exact identity cached by this environment actor."""
        return {
            "container_id": self.env.container_id,
            "container_name": self.env.container_name,
        }

    async def initialize(self, timeout=240, max_start_sleep_seconds=30, reset_git_log=False, log_prefix="") -> Dict[str, Any]:
        """启动容器并激活监控"""
        return await self.env._start_container(
            timeout=timeout,
            max_start_sleep_seconds=max_start_sleep_seconds,
            reset_git_log=reset_git_log,
            log_prefix=log_prefix,
        )

    async def execute_command(self, command: str, cwd: str = "", timeout: int = 180,
                              max_output_bytes: int | None = None,
                              log_prefix="") -> Dict[str, Any]:
        """全异步执行命令；可选 max_output_bytes 字节上限（超即 killpg）。"""
        return await self.env.execute(
            command=command, cwd=cwd, timeout=timeout,
            max_output_bytes=max_output_bytes, log_prefix=log_prefix,
        )

    async def execute_action(self, action: dict, cwd: str = "", timeout: int = 180,
                             max_output_bytes: int | None = None,
                             log_prefix="") -> Dict[str, Any]:
        """全异步执行命令；可选 max_output_bytes 字节上限（超即 killpg）。"""
        return await self.env.execute(
            action=action, cwd=cwd, timeout=timeout,
            max_output_bytes=max_output_bytes, log_prefix=log_prefix,
        )

    async def copy_to_container(self, local_path: Path, remote_path: Path, log_prefix=""):
        return await self.env.copy_to_container(local_path, remote_path, log_prefix=log_prefix)

    async def write_content_to_container(self, content: str, remote_path: Path, log_prefix=""):
        return await self.env.write_content_to_container(content, remote_path, log_prefix=log_prefix)

    async def cleanup(self, log_prefix=""):
        # if self.env:
            # await self.env.cleanup(log_prefix=log_prefix)
        pass

    # async def release_ray_resource(self, log_prefix=""):
    #     """
    #     仅主动终止当前 Actor 以释放 Ray 资源 (resource_tokens)。
    #     绝对不清理底层 Docker 容器，避免高并发下 Docker Daemon 锁竞争卡死。
    #     """
    #     # if self.env:
    #     #     self.logger.info(f"{log_prefix} Exiting actor to release Ray resources. Docker container [{self.env.container_id}] is kept alive.")
    #     ray.actor.exit_actor()

    async def exit(self, log_prefix=""):
        ray.actor.exit_actor()
