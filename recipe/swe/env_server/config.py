#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@desc env config
@author: plm
@create: 2026-02-03
"""


from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Check the config files in minisweagent/config for example settings."""

    system_template: str
    """Template for the system message (the first message)."""
    instance_template: str
    """Template for the first user message specifying the task (the second message overall)."""
    step_limit: int = 0
    """Maximum number of steps the agent can take."""

    format_error_template: str
    observation_template: str
    use_tool_call: bool = False
    enforce_single_tool_call: bool = False
    enforce_exact_bash_arguments: bool = False
    enforce_exact_submission_command: bool = False
    enforce_valid_submission_patch: bool = False
    action_regex: str = ""



class DockerEnvironmentConfig(BaseModel):
    image: str
    data_source: str = ""
    repo_language: str = ""
    cwd: str = "/testbed"
    """Working directory in which to execute commands."""
    env: dict[str, str] = Field(default_factory=dict)
    """Environment variables to set in the container."""
    forward_env: list[str] = Field(default_factory=list)
    """Environment variables to forward to the container.
    Variables are only forwarded if they are set in the host environment.
    In case of conflict with `env`, the `env` variables take precedence.
    """
    timeout: int = 30
    """Timeout for executing commands in the container."""
    executable: str = "podman"
    """Path to the docker/container executable."""
    run_args: list[str] = Field(default_factory=lambda: ["--rm", "--init"])
    """Additional arguments to pass to the docker/container executable.
    Default is ["--rm"], which removes the container after it exits.
    """
    container_timeout: str = "4h"
    """Max duration to keep container running. Uses the same format as the sleep command."""
    network_mode: str = "none"
    """Podman network mode. Opt in to ``bridge`` or ``host`` only on a trusted cluster."""
    apt_mirror: str | None = None
    """Optional APT mirror URL. No package mirror is injected by default."""
    go_proxy: str | None = None
    """Optional GOPROXY value for SWE-bench Pro evaluation containers."""
    pull_timeout: int = 120
    """Timeout in seconds for pulling images."""
    interpreter: list[str] = ["bash", "-lc"]
    """Interpreter to use to execute commands. Default is ["bash", "-lc"].
    The actual command will be appended as argument to this. Override this to e.g., modify shell flags
    (e.g., to remove the `-l` flag to disable login shell) or to use python instead of bash to interpret commands.
    """
    group_id: int = 0
    env_resource_tokens: float = 4.0
    env_cpu_limit: str = "4"
    env_mem_limit: str = "8g"
    ray_env_actor_num_cpus: float = 1.0
    # Optional online cleanup.  The TaskRunner owns one bounded GC actor on
    # each Ray node; disabled preserves the historical sleep+--rm lifecycle.
    container_gc_enabled: bool = False
    container_gc_workers_per_node: int = 1
    container_gc_queue_maxsize: int = 4096
    container_gc_remove_timeout_seconds: float = 120.0
    container_gc_max_retries: int = 3
    container_gc_retry_backoff_seconds: float = 2.0
    container_gc_enqueue_timeout_seconds: float = 5.0
    container_gc_drain_timeout_seconds: float = 180.0
    # Optional public dependency mirrors. The global switch is a true no-op
    # when disabled; the per-ecosystem switches only apply after it is enabled.
    dependency_mirror_enabled: bool = False
    dependency_mirror_apt_enabled: bool = True
    dependency_mirror_python_enabled: bool = True
    dependency_mirror_go_enabled: bool = True
    dependency_mirror_node_enabled: bool = False
    dependency_mirror_php_enabled: bool = True
    dependency_mirror_r_enabled: bool = True
    dependency_mirror_ruby_enabled: bool = True
    dependency_mirror_jvm_enabled: bool = True
    dependency_mirror_cargo_enabled: bool = True
    dependency_mirror_rustup_enabled: bool = False
    # Per-container identity fields are populated by SWEEnvClient only when GC
    # is enabled.  They are never user-facing Hydra knobs.
    container_gc_owner: str = ""
    container_gc_nonce: str = ""
    container_gc_node_id: str = ""
    container_gc_role: str = "standalone"
    container_gc_request_hash: str = ""
    container_name: str = ""
    # 通过 LXCFS 让容器内的 /proc 与 /sys CPU 视图匹配 --cpus 配额。
    enable_lxcfs_cpu_view: bool = False
    lxcfs_root: str = "/var/lib/lxcfs"
    # 禁止手动写patch.txt
    disable_manual_write_patch_cmd: bool = True

    # 是否把testbed挂载到内存盘（tmpfs），以避免磁盘膨胀问题（尤其是 SWE-bench 的测试输出会不断增加，导致磁盘空间不足）
    map_testbed_to_tmpfs: bool = False
