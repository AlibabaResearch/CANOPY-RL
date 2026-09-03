#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@desc docker aysnc env
@author: plm
@create: 2026-02-24
"""

import asyncio
import logging
import os
import random
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict

import yaml
from loguru import logger

from .config import DockerEnvironmentConfig
from .dependency_mirrors import build_dependency_mirror_script
from .swe_utils import Timer, load_yaml_config_from_file_path

# from .exceptions import Submitted

# =====================================================================
# 输出字节上限：防止单条 podman exec 在 host 侧 Python 内存里堆 GB 级 stdout
# 超过上限会立刻 killpg 整个进程组（podman + conmon + 容器 exec session）
#
# - 普通 action：64 MB（正常 < 1MB；4MB 已经是异常；64MB 留充足边界）
# - 评估脚本：256 MB（保留完整 PASS/FAIL 解析能力，又防止失控）
# - container start / inspect / cp 这种命令：8 MB（基本不输出东西）
# =====================================================================
DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024 * 1024            # 64MB - action
EVAL_MAX_OUTPUT_BYTES = 256 * 1024 * 1024              # 256MB - eval
INFRA_MAX_OUTPUT_BYTES = 8 * 1024 * 1024               # 8MB  - inspect/cp/etc.
_READ_CHUNK = 64 * 1024                                # 64KB chunk


logger.remove()  # 先移除默认的 handler
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


def _killpg_safe(proc: subprocess.Popen) -> None:
    """尽力 killpg，整个进程组送走 SIGKILL；失败兜底 proc.kill()。"""
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        return
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.kill()
    except Exception:
        pass


def _run_capture_sync(
    cmd: list[str], timeout: float, cancel_event: threading.Event
) -> Dict[str, Any]:
    """Run a short Podman control-plane command from a worker thread.

    Ray async actors have previously deadlocked when ``podman run -d`` was
    driven through ``asyncio.create_subprocess_exec(...).communicate()``.
    Keep Podman on the proven synchronous ``Popen`` path while polling often
    enough to honour coroutine cancellation.  The child owns a process group
    so timeout/cancellation also removes any Podman CLI descendants.
    """

    proc: subprocess.Popen | None = None
    killed_by = ""
    stdout = b""
    stderr = b""
    deadline = time.monotonic() + max(0.1, float(timeout))

    def _force_kill_group() -> None:
        if proc is None:
            return
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            if proc.poll() is None:
                try:
                    proc.kill()
                except (ProcessLookupError, OSError):
                    pass

    try:
        if cancel_event.is_set():
            return {
                "stdout": "",
                "stderr": "",
                "returncode": -1,
                "timeout": False,
                "exception": "command cancelled before spawn",
            }
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            start_new_session=True,
        )

        while True:
            if cancel_event.is_set():
                killed_by = "cancelled"
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                killed_by = "timeout"
                break
            try:
                stdout, stderr = proc.communicate(timeout=min(0.25, remaining))
                return {
                    "stdout": stdout.decode(errors="replace"),
                    "stderr": stderr.decode(errors="replace"),
                    "returncode": int(proc.returncode or 0),
                    "timeout": False,
                    "exception": "",
                }
            except subprocess.TimeoutExpired:
                continue

        _force_kill_group()
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            _force_kill_group()
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                stdout, stderr = b"", b""

        return {
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
            "returncode": -1,
            "timeout": killed_by == "timeout",
            "exception": (
                f"command timed out after {timeout}s"
                if killed_by == "timeout"
                else "command cancelled"
            ),
        }
    except Exception as exc:
        _force_kill_group()
        if proc is not None:
            try:
                proc.communicate(timeout=5)
            except Exception:
                pass
        return {
            "stdout": "",
            "stderr": "",
            "returncode": -1,
            "timeout": False,
            "exception": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if proc is not None:
            if proc.poll() is None:
                _force_kill_group()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
            for stream in (proc.stdout, proc.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass


async def _run_capture_async(cmd: list[str], timeout: float) -> Dict[str, Any]:
    """Async facade for the thread-backed, cancellation-safe Podman runner."""

    cancel_event = threading.Event()
    worker = asyncio.create_task(
        asyncio.to_thread(_run_capture_sync, cmd, timeout, cancel_event)
    )
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        cancel_event.set()
        try:
            await asyncio.wait_for(asyncio.shield(worker), timeout=12)
        except BaseException:
            pass
        raise


def _run_exec_streaming(cmd, exec_timeout: float, max_output_bytes: int,
                       container_id: str = "", log_prefix: str = "",
                       cancel_event: threading.Event | None = None) -> Dict[str, Any]:
    """
    流式跑 podman exec：
      - stdout/stderr 合并到一条 PIPE
      - 每读 64KB 累计字节数；超 max_output_bytes 立即 killpg → 防 host RSS 爆炸
      - 超时也会 killpg
      - 任何路径下 Python 内存里的 stdout 永远 ≤ max_output_bytes + 一个 chunk

    返回的 dict 形状跟原 _run_exec_sync 一致：
        {output, returncode, exception_info, extra, timeout}
    被 size 杀掉时 timeout=True（让上游按 timeout 失败重试逻辑走）。
    """
    deadline = time.monotonic() + max(0.5, float(exec_timeout))

    # start_new_session=True ⇒ 子进程自成 process group，killpg 一发干净
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # 合并，跟原行为一致
            bufsize=0,
            close_fds=True,
            start_new_session=True,
        )
    except FileNotFoundError as e:
        return {
            "output": "",
            "returncode": -1,
            "exception_info": f"Executable not found: {e}",
            "extra": {"exception_type": "FileNotFoundError"},
            "timeout": False,
        }
    except Exception as e:
        return {
            "output": "",
            "returncode": -1,
            "exception_info": f"Failed to spawn subprocess: {e}",
            "extra": {"exception_type": type(e).__name__},
            "timeout": False,
        }

    buf = bytearray()
    killed_by = None  # None | "size" | "timeout" | "cancelled"

    try:
        # 把 fd 设为非阻塞，配合 select 实现「按时间片读 + 时间到就跳出」
        try:
            os.set_blocking(proc.stdout.fileno(), False)
        except (OSError, ValueError):
            pass  # 兜底：阻塞读也能跑，只是 deadline 精度差点

        import select

        while True:
            if cancel_event is not None and cancel_event.is_set():
                killed_by = "cancelled"
                break
            # 时间检查
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                killed_by = "timeout"
                break

            # 等 stdout 可读，最多 0.5s 让我们能定期检查 deadline
            wait_for = min(0.5, remaining)
            try:
                r, _, _ = select.select([proc.stdout], [], [], wait_for)
            except (ValueError, OSError):
                # fd 关了
                break
            if not r:
                # 没数据；如果子进程已退出 → 收尾退出
                if proc.poll() is not None:
                    # 可能还有残留：尽力再读一把
                    try:
                        tail = proc.stdout.read()
                        if tail:
                            buf.extend(tail)
                    except (BlockingIOError, ValueError, OSError):
                        pass
                    break
                continue

            try:
                chunk = proc.stdout.read(_READ_CHUNK)
            except BlockingIOError:
                continue
            except (ValueError, OSError):
                break

            if not chunk:
                # EOF
                break

            buf.extend(chunk)

            if len(buf) >= max_output_bytes:
                killed_by = "size"
                break

        if killed_by is not None:
            _killpg_safe(proc)

        # 等子进程真正回收；最多等 5 秒（被 SIGKILL 后通常瞬间）
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _killpg_safe(proc)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

        try:
            output_text = bytes(buf).decode("utf-8", errors="replace")
        except Exception:
            output_text = ""

        if killed_by == "size":
            cap_mb = max_output_bytes // (1024 * 1024)
            note = (
                f"\n\n[KILLED-BY-OUTPUT-SIZE] command produced more than {cap_mb} MB "
                f"of stdout/stderr; the process group was killed to protect the host. "
                f"Likely causes: tight loop with print, very verbose pytest output, "
                f"recursive listing (e.g. `find /`), or pip install of a huge package "
                f"with -v. Consider piping through `head -c <bytes>` or `head -n <lines>`."
            )
            try:
                print(
                    f"[BIG_STDOUT] {log_prefix} cid={container_id[:8] if container_id else '?'} "
                    f"bytes={len(buf):,} cap={cap_mb}MB cmd_tail={str(cmd[-1])[:100]!r}",
                    flush=True,
                )
            except Exception:
                pass
            return {
                "output": output_text + note,
                "returncode": -1,
                "exception_info": f"Output exceeded size cap ({cap_mb} MB); process killed.",
                "extra": {
                    "killed_by": "size",
                    "bytes_captured": len(buf),
                    "max_output_bytes": max_output_bytes,
                },
                "timeout": True,  # 让上游按超时/失败处理（计入 env_timeout_cnt）
            }

        if killed_by == "timeout":
            return {
                "output": output_text or "Command timed out with no output",
                "returncode": -1,
                "exception_info": f"Command Timed out after {exec_timeout} seconds",
                "extra": {"killed_by": "timeout", "bytes_captured": len(buf)},
                "timeout": True,
            }

        if killed_by == "cancelled":
            return {
                "output": output_text,
                "returncode": -1,
                "exception_info": "Command cancelled; process group killed.",
                "extra": {"killed_by": "cancelled", "bytes_captured": len(buf)},
                "timeout": True,
            }

        # 正常退出
        rc = proc.returncode if proc.returncode is not None else -1
        return {
            "output": output_text,
            "returncode": rc,
            "exception_info": "",
            "extra": {"bytes_captured": len(buf)},
            "timeout": False,
        }

    finally:
        # 兜底：还活着就再杀一次
        if proc.poll() is None:
            _killpg_safe(proc)
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass


async def _run_exec_streaming_async(
    cmd: list[str],
    exec_timeout: float,
    max_output_bytes: int,
    container_id: str = "",
    log_prefix: str = "",
) -> Dict[str, Any]:
    """Cancellation-safe bridge to the bounded streaming reader thread."""

    cancel_event = threading.Event()
    task = asyncio.create_task(
        asyncio.to_thread(
            _run_exec_streaming,
            cmd,
            exec_timeout,
            max_output_bytes,
            container_id,
            log_prefix,
            cancel_event,
        )
    )
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        cancel_event.set()
        # The reader checks this at least every 0.5s, kills the Podman process
        # group, and reaps it. Do not leave its worker thread behind.
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=10)
        except BaseException:
            pass
        raise


def show_command_log(self, command, cmd_list, cwd, timeout, log_prefix, result):
    if cmd_list[-1] == command:
        cmd_to_show = shlex.join(cmd_list[:-1])
    else:
        cmd_to_show = shlex.join(cmd_list)
    msg = "{}, prefix_cmd: [{}], command: [{}], returncode: {}, output preview: [{}]".format(
        log_prefix, cmd_to_show, command, result.returncode, result.stdout[:200]
    )
    print(msg, flush=True)
    return


def show_log(msg, log_prefix=""):
    logger.info(f"{log_prefix} {msg}")
    return

def show_error(msg, log_prefix=""):
    logger.error(f"{log_prefix} {msg}")
    return


class DockerEnvironment:
    def __init__(self, config: DockerEnvironmentConfig, logger: logging.Logger | None = None, container_id: str = "", test_mode: bool = False):
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.container_id = container_id
        self.container_name = ""
        self.config = config
        self.test_mode = test_mode
        self._cleaned_up = False # 防止重复清理

    @classmethod
    async def from_existing(cls, container_id: str, config: DockerEnvironmentConfig, logger: logging.Logger | None = None, test_mode: bool = False):
        instance = cls(config=config, logger=logger, container_id=container_id, test_mode=test_mode)
        try:
            check_cmd = [config.executable, "inspect", "-f", "{{.State.Running}}", container_id]
            result = await _run_capture_async(check_cmd, 5)
            if result["returncode"] != 0:
                raise RuntimeError(result["stderr"] or result["exception"])
            if "true" not in result["stdout"].lower():
                show_error(f"Container {container_id} is found but not running.")
        except Exception as e:
            raise RuntimeError(f"Container {container_id} does not exist or executable not found. Error: {e}")

        show_log(f"Successfully attached to existing container: {container_id}")
        return instance

    async def _check_container_alive(self, log_prefix="") -> dict[str, Any]:
        """
        通用执行前置检查，直接返回标准格式的字典。
        调用方判断 result.get("is_alive")，如果为 False，直接将此 result 返回，实现优雅阻断。
        """
        # 预设一个死亡状态的标准返回结构
        dead_result = {
            "is_alive": False,
            "output": "",
            "returncode": -1,
            "exception_info": "Execution aborted: Container is not running or engine is unresponsive.",
            "timeout": False,
            "extra": {"container_dead": True}
        }

        if not getattr(self, "container_id", None):
            dead_result["exception_info"] = "Container ID is empty. Container was not successfully started."
            return dead_result

        check_cmd = [self.config.executable, "inspect", "-f", "{{.State.Running}}", self.container_id]

        try:
            result = await _run_capture_async(check_cmd, 5)
            if result["returncode"] == 0 and "true" in result["stdout"].lower():
                return {"is_alive": True} # 存活时，仅返回通行证

            if result["timeout"]:
                dead_result["exception_info"] = f"Inspect timeout! Podman engine hung for container {self.container_id[:8]}."
                dead_result["timeout"] = True
                return dead_result

            # 容器已非 Running 状态 (Exited 或 Dead)
            dead_result["exception_info"] = f"Container {self.container_id[:8]} exited. Inspect returned: {result['stdout'].strip()}"
            return dead_result
        except Exception as e:
            dead_result["exception_info"] = f"Health check exception for container {self.container_id[:8]}: {e}"
            return dead_result

    async def _fix_apt_sources(self, log_prefix=""):
        mirror = self.config.apt_mirror
        if not mirror:
            return None
        remote_script = f"""
        set -e
        if [ -f /etc/apt/sources.list ]; then
            sed -i 's|http://.*.ubuntu.com|{mirror}|g' /etc/apt/sources.list
            sed -i 's|http://deb.debian.org|{mirror}|g' /etc/apt/sources.list
            sed -i 's|http://security.debian.org|{mirror}|g' /etc/apt/sources.list
        fi
        if [ -d /etc/apt/sources.list.d ]; then
            find /etc/apt/sources.list.d/ -type f -name "*.sources" -o -name "*.list" | xargs -r sed -i 's|http://.*.ubuntu.com|{mirror}|g'
            find /etc/apt/sources.list.d/ -type f -name "*.sources" -o -name "*.list" | xargs -r sed -i 's|http://deb.debian.org|{mirror}|g'
        fi
        rm -rf /var/lib/apt/lists/*
        """
        # show_log(f"Injecting optimized APT sources... {self.container_id}", log_prefix)
        return await self.execute(command=remote_script, log_prefix=log_prefix)

    async def _configure_dependency_mirrors(self, log_prefix=""):
        """Apply the optional, public-only package-manager configuration."""

        remote_script = build_dependency_mirror_script(self.config)
        if not remote_script:
            return None
        return await self.execute(command=remote_script, log_prefix=log_prefix)

    def prepare_container_start_cmd(self):
        container_name = self.config.container_name or f"minisweagent-{uuid.uuid4().hex[:8]}"
        if not re.fullmatch(r"minisweagent-[0-9a-f]{8}", container_name):
            raise ValueError(f"Invalid managed container name: {container_name!r}")
        run_args = list(self.config.run_args)
        if self.config.enable_lxcfs_cpu_view:
            lxcfs_root = self.config.lxcfs_root.rstrip("/")
            if not lxcfs_root:
                raise ValueError("lxcfs_root must not be empty when LXCFS CPU view is enabled")
            lxcfs_mounts = [
                f"{lxcfs_root}/proc/cpuinfo:/proc/cpuinfo:rw",
                f"{lxcfs_root}/proc/stat:/proc/stat:rw",
                f"{lxcfs_root}/sys/devices/system/cpu:/sys/devices/system/cpu:rw",
            ]
            for mount in lxcfs_mounts:
                if mount not in run_args:
                    run_args.extend(["--volume", mount])
        if "--pull" not in run_args:
            run_args.extend(["--pull", "never"])

        normalized_data_source = (
            str(self.config.data_source).strip().lower().replace("_", "-")
        )
        is_swebench_pro = normalized_data_source == "swe-bench-pro"
        if is_swebench_pro and "--entrypoint" not in run_args:
            # Most Pro images declare ENTRYPOINT ["/bin/bash"]. Without an
            # override, the generic trailing `sleep 4h` becomes
            # `/bin/bash sleep 4h` and the container exits immediately.
            run_args.extend(["--entrypoint", "/bin/bash"])
        if is_swebench_pro and self.config.go_proxy:
            run_args.extend(["--env", f"GOPROXY={self.config.go_proxy}"])
        container_command = ["sleep", str(self.config.container_timeout)]
        if is_swebench_pro:
            container_command = [
                "-c",
                f"exec sleep {self.config.container_timeout}",
            ]

        cmd = [
            self.config.executable, "run", "-d",
            "--name", container_name,
            "-w", self.config.cwd,
            *run_args,
            self.config.image,
            *container_command,
        ]
        cmd_str = shlex.join(cmd)
        output = {
            "output": "", "returncode": 0, "exception_info": "",
            "extra": {"container_name": container_name}, "timeout": False
        }
        return container_name, cmd, cmd_str, output

    async def _start_container(self, timeout=240, max_start_sleep_seconds=30, reset_git_log=False, log_prefix="") -> dict[str, Any]:
        # 大规模并发时稍微打散启动时间（0-1.5秒随机延迟），防止瞬间压垮底层 containerd/磁盘IO
        delay = random.uniform(0, max_start_sleep_seconds)
        await asyncio.sleep(delay)
        return await self._start_container_async(timeout=timeout, log_prefix=log_prefix)

    async def _start_container_async(self, timeout=240, log_prefix="") -> dict[str, Any]:
        """完美全异步启动方法，防 Ray 阻塞版"""
        container_name, cmd, cmd_str, output = self.prepare_container_start_cmd()
        timer = Timer(can_print=False)
        # 尽早记录 container_name，作为兜底凭证
        self.container_name = container_name
        # show_log(f"Starting container {container_name}, cmd: {cmd_str}", log_prefix)

        try:
            result = await _run_capture_async(cmd, timeout)
            output["output"] = result["stdout"].strip()
            output["returncode"] = result["returncode"]
            output["timeout"] = bool(result["timeout"])

            # Podman can print the full ID before a congested CLI is cancelled
            # or times out. Preserve it so the exact-ID GC path remains usable.
            if re.fullmatch(r"[0-9a-f]{64}", output["output"]):
                self.container_id = output["output"]

            if result["timeout"]:
                output["extra"]["exception_type"] = "TimeoutError"
                output["exception_info"] = (
                    f"Start Timeout out after {timeout}s. Output: "
                    f"{output['output']}. CMD: {cmd_str}"
                )
            elif result["returncode"] != 0:
                if result["exception"]:
                    output["extra"]["exception_type"] = result["exception"].split(
                        ":", 1
                    )[0]
                output["exception_info"] = f"Container start failed (Exit {result['returncode']}). Output: {output['output']}. Stderr: {result['stderr'].strip() or result['exception']}. CMD: {cmd_str}"
                show_error(output["exception_info"], log_prefix)
            else:
                self.container_id = output["output"]
                # show_log(f"Container start success {container_name}, {self.container_id[:12]}", log_prefix)

        except asyncio.CancelledError:
            # The subprocess helper has already killed and reaped the Podman
            # process group. Let the outer lifecycle enqueue urgent GC by the
            # pre-generated exact name.
            raise

        if (
            getattr(self, "container_id", "")
            and output["returncode"] == 0
            and self.config.apt_mirror
        ):
            await self._fix_apt_sources(log_prefix=log_prefix)
        if getattr(self, "container_id", "") and output["returncode"] == 0:
            await self._configure_dependency_mirrors(log_prefix=log_prefix)
        used_seconds, total_seconds, used_info, total_info = timer.tok("Container Started")
        output["time_info"] = total_info
        return output

    async def write_content_to_container(self, content: str, remote_path: Path, timeout: int = 60, log_prefix="") -> dict[str, Any]:
        """
        【内存级稳定版】：
        1. 在 /dev/shm (内存盘) 创建临时文件，彻底规避宿主机磁盘 IO。
        2. 自动管理生命周期，无需手动清理。
        3. 适当调高 timeout，应对 Podman 并发排队。
        """
        if not self.container_id:
            return {"output": "Container not started", "returncode": -1}

        # 确保目标路径的父目录存在
        mkdir_cmd = f"mkdir -p {remote_path.parent}"
        await self.execute(command=mkdir_cmd, log_prefix=log_prefix)

        # Use /dev/shm for a short-lived in-memory command file.
        # delete=True removes it automatically when the file is closed.
        try:
            # Pin the temporary file to the in-memory filesystem.
            with tempfile.NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                dir="/dev/shm",
                delete=True
            ) as temp_file:
                temp_file.write(content)
                temp_file.flush() # 确保内容刷入内核缓冲区

                temp_file_path = temp_file.name
                cmd = [self.config.executable, "cp", temp_file_path, f"{self.container_id}:{remote_path}"]

                result = await _run_capture_async(cmd, timeout)

                if result["timeout"]:
                    msg = f"Write timeout (podman cp) at: {remote_path}, after {timeout}s"
                    show_error(msg, log_prefix)
                    return {
                        "output": f"Timeout after {timeout}s",
                        "returncode": -1,
                        "exception_info": msg,
                        "timeout": True,
                    }
                if result["returncode"] != 0:
                    error_msg = result["stderr"].strip() or result["exception"]
                    show_error(f"Copy to container failed: {error_msg}", log_prefix)
                    return {
                        "output": error_msg,
                        "returncode": result["returncode"],
                        "exception_info": error_msg,
                    }

                return {"output": "Successfully written", "returncode": 0}

        except asyncio.CancelledError:
            raise
        except Exception as e:
            msg = f"Write content failed at: {remote_path}, {str(e)}"
            show_error(msg, log_prefix)
            return {"output": str(e), "returncode": -1, "exception_info": msg}

    async def copy_to_container(self, src: Path, dst: Path, log_prefix="", timeout: int = 60):
        """
        用于拷贝真实文件（如果是字符串，请务必使用上面的 write_content_to_container 避免磁盘IO）
        【修复】：捕获异常并返回标准字典格式，防止未捕获异常导致 Ray Worker 崩溃。
        """
        if os.path.dirname(dst) == "":
            error_msg = f"Destination path parent directory cannot be empty!, dst: {dst}"
            show_error(error_msg, log_prefix)
            return {"output": error_msg, "returncode": -1}

        # 先确保目录存在
        mkdir_cmd = f"mkdir -p {dst.parent}"
        await self.execute(command=mkdir_cmd, log_prefix=log_prefix)

        cmd = [self.config.executable, "cp", str(src), f"{self.container_id}:{str(dst)}"]

        try:
            result = await _run_capture_async(cmd, timeout)
            if result["timeout"]:
                msg = f"Copy timeout (podman cp) at: {dst}, after {timeout}s"
                show_error(msg, log_prefix)
                return {
                    "output": f"Timeout after {timeout}s",
                    "returncode": -1,
                    "exception_info": msg,
                    "timeout": True,
                }
            if result["returncode"] != 0:
                error = result["stderr"] or result["exception"]
                show_error(f"Copy failed: {error}", log_prefix)
                return {"output": error, "returncode": result["returncode"], "exception_info": error}
            return {"output": result["stdout"], "returncode": result["returncode"], "exception_info": ""}
        except asyncio.CancelledError:
            raise
        except Exception as e:
            msg = f"Copy exception: {e}"
            show_error(msg, log_prefix)
            return {"output": str(e), "returncode": -1, "exception_info": msg}

    def check_illegal_command(self, command: str, log_prefix: str = "") -> dict[str, Any] | None:
        """
        检查命令是否在手动伪造 patch/diff 文件。
        如果命中非法规则，直接返回标准化的执行结果字典；如果合法，返回 None。
        """
        if not command:
            return None

        cmd_lower = command.lower()
        error_msg = ""
        is_illegal = False
        default_error_msg = """
Manually write patch.txt command (eg. manually echo content to) is not allowed. Please create the patch file using Git.
If you created NEW files, use `git add -N <file>` first.
Run: `git diff -- path/to/file1 path/to/file2 > patch.txt` (list only the source files you modified).
Do NOT commit your changes.
"""

        # 拦截规则 1: 捕捉 `cat > patch.txt << EOF` 包含补丁特征的行为
        if re.search(r'cat\s+>.*patch', cmd_lower) and ('--- a/' in cmd_lower or '+++ b/' in cmd_lower):
            is_illegal = True
            error_msg = default_error_msg

        # 拦截规则 2: 捕捉 `echo "diff --git..." > patch.txt` 的行为
        elif re.search(r'echo\s+.*>.*patch', cmd_lower) and ('diff --git' in cmd_lower or '--- a/' in cmd_lower):
            is_illegal = True
            error_msg = default_error_msg

        # 拦截规则 3: 泛化拦截，只要试图用 echo/cat 硬编码 diff 内容就拒绝
        elif ('--- a/' in cmd_lower and '+++ b/' in cmd_lower) and ('cat ' in cmd_lower or 'echo ' in cmd_lower):
             is_illegal = True
             error_msg = default_error_msg

        # 如果命中任意一条规则，构造并返回完整的拦截结果
        if is_illegal:
            show_error(f"Blocked illegal manual patch writing command, command: {command[:30]}", log_prefix)
            return {
                "output": error_msg,
                "returncode": 1,
                "exception_info": "Command execution blocked due to illegal manual patch writing.",
                "extra": {"blocked": True, "reason": "manual_patch_writing"},
                "timeout": False
            }

        return None

    async def execute_async(self, command: str = "", action: dict = None, cwd: str = "", *,
                            timeout: int | None = None,
                            max_output_bytes: int | None = None,
                            log_prefix="") -> dict[str, Any]:
        """
        【极度健壮版 execute】：
        1. 在线程中流式读取输出；协程取消会通知线程 killpg 并等待回收。
        2. 自带超时强杀逻辑，防止失控命令长期占用 Podman 控制面。
        3. 【新增】流式读取 stdout，超过 max_output_bytes 立即 killpg，防止 host RSS 爆炸。
        """
        if action is not None and "command" in action:
            command = action["command"]
        command = command or ""
        if self.config.disable_manual_write_patch_cmd:
            illegal = self.check_illegal_command(command, log_prefix=log_prefix)
            if illegal:
                return illegal

        cwd = cwd or self.config.cwd
        exec_timeout = timeout or self.config.timeout
        cap = max_output_bytes if max_output_bytes is not None else DEFAULT_MAX_OUTPUT_BYTES

        if not self.container_id:
            return {"output": "", "returncode": -1, "exception_info": "Container not started", "timeout": False}

        # 构造命令
        cmd = [self.config.executable, "exec", "-w", cwd]
        for key, value in self.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self.container_id, *self.config.interpreter, command])

        try:
            return await _run_exec_streaming_async(
                cmd, exec_timeout, cap, self.container_id, log_prefix
            )
        except Exception as e:
            return {
                "output": "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
                "timeout": False,
            }

    async def execute_async_timeouthint(self, command: str = "", action: dict = None, cwd: str = "", *,
                                       timeout: int | None = None,
                                       max_output_bytes: int | None = None,
                                       log_prefix="") -> dict[str, Any]:
        """
        execute_async 的增强变体：超时时返回带服务器/长任务诊断提示的友好错误信息。
        其余行为（流式读 + 字节上限 + killpg）跟 execute_async 一致。
        """
        if action is not None and "command" in action:
            command = action["command"]
        command = command or ""
        if self.config.disable_manual_write_patch_cmd:
            illegal = self.check_illegal_command(command, log_prefix=log_prefix)
            if illegal:
                return illegal
        cwd = cwd or self.config.cwd
        exec_timeout = timeout or self.config.timeout
        cap = max_output_bytes if max_output_bytes is not None else DEFAULT_MAX_OUTPUT_BYTES

        if not self.container_id:
            return {"output": "", "returncode": -1, "exception_info": "Container not started", "timeout": False}

        # 构造命令
        cmd = [self.config.executable, "exec", "-w", cwd]
        for key, value in self.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self.container_id, *self.config.interpreter, command])

        try:
            res = await _run_exec_streaming_async(
                cmd, exec_timeout, cap, self.container_id, log_prefix
            )
        except Exception as e:
            return {
                "output": "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
                "timeout": False,
            }

        # 仅在「真超时」时套友好提示；被 size 杀掉的不改写 message，让 size 信息原样透出
        if res.get("timeout") and res.get("extra", {}).get("killed_by") != "size":
            blocking_keywords = ["uvicorn", "flask", "python", "node", "npm", "server", "run"]
            is_likely_server = any(kw in command.lower() for kw in blocking_keywords)
            if is_likely_server:
                diag_msg = (
                    f"Command timed out after {exec_timeout} seconds. "
                    "HumanFixedNote: This could be due to some reasons:\n"
                    "1. **Blocking Service**: If you are starting a server (e.g., uvicorn, flask), it will not exit on its own. "
                    "In this environment, please run it in the background: `nohup <cmd> > server.log 2>&1 &` and then check the log.\n"
                    "2. **Long-running Task**: If this is a heavy computation or test, it exceeded the time limit. "
                    "Consider optimizing the code or breaking the task into smaller steps.\n"
                )
            else:
                diag_msg = f"Command timed out after {exec_timeout} seconds. "
            res["exception_info"] = diag_msg
        return res

    async def execute(self, command: str = "", action: dict = {}, cwd: str = "", *,
                      timeout: int = 120,
                      max_output_bytes: int | None = None,
                      log_prefix="") -> dict[str, Any]:
        return_res = await self.execute_async(
            command=command, action=action, cwd=cwd, timeout=timeout,
            max_output_bytes=max_output_bytes, log_prefix=log_prefix,
        )
        return return_res

    async def cleanup(self, log_prefix=""):
        pass

    def _fire_and_forget_cleanup(self, target: str, log_prefix: str = ""):
        pass

    def __del__(self):
        """
        __del__ 阶段属于 Python 垃圾回收阶段，绝不允许有任何阻塞或 await。
        """
        pass
