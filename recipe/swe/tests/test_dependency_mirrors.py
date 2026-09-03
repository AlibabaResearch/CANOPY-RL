#!/usr/bin/env python3
# Copyright 2026 Alibaba Group Holding Limited
# SPDX-License-Identifier: Apache-2.0

"""Tests for the default-off public dependency-mirror integration."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from recipe.swe.env_server.config import DockerEnvironmentConfig
from recipe.swe.env_server.dependency_mirrors import (
    DEPENDENCY_MIRROR_POLICY_DEFAULTS,
    PUBLIC_CARGO_REGISTRY,
    PUBLIC_MAVEN_REPOSITORY,
    PUBLIC_PYPI_INDEX,
    build_dependency_mirror_script,
    configure_dependency_mirror_environment,
    resolve_dependency_mirror_policy,
)
from recipe.swe.env_server.environments import DockerEnvironment


def _config(language: str = "python", **overrides):
    values = {
        "repo_language": language,
        "env": {},
        "apt_mirror": None,
        "go_proxy": None,
        "dependency_mirror_enabled": True,
        "dependency_mirror_apt_enabled": False,
        "dependency_mirror_python_enabled": True,
        "dependency_mirror_go_enabled": True,
        "dependency_mirror_node_enabled": False,
        "dependency_mirror_php_enabled": True,
        "dependency_mirror_r_enabled": True,
        "dependency_mirror_ruby_enabled": True,
        "dependency_mirror_jvm_enabled": True,
        "dependency_mirror_cargo_enabled": True,
        "dependency_mirror_rustup_enabled": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_global_gate_is_a_byte_for_byte_noop():
    config = _config(
        dependency_mirror_enabled=False,
        env={"PIP_INDEX_URL": "https://packages.example.test/simple"},
    )
    original = dict(config.env)

    assert build_dependency_mirror_script(config) == ""
    configure_dependency_mirror_environment(config)
    assert config.env == original


def test_trainer_policy_defaults_are_fail_closed_and_all_gates_are_resolved():
    policy = resolve_dependency_mirror_policy({})

    assert policy == DEPENDENCY_MIRROR_POLICY_DEFAULTS
    assert policy["dependency_mirror_enabled"] is False
    assert policy["dependency_mirror_node_enabled"] is False
    assert policy["dependency_mirror_rustup_enabled"] is False

    enabled = resolve_dependency_mirror_policy(
        {"dependency_mirror_enabled": True, "dependency_mirror_node_enabled": True}
    )
    assert enabled["dependency_mirror_enabled"] is True
    assert enabled["dependency_mirror_node_enabled"] is True


@pytest.mark.parametrize(
    ("language", "gate"),
    [
        ("python", "dependency_mirror_python_enabled"),
        ("go", "dependency_mirror_go_enabled"),
        ("javascript", "dependency_mirror_node_enabled"),
        ("php", "dependency_mirror_php_enabled"),
        ("r", "dependency_mirror_r_enabled"),
        ("ruby", "dependency_mirror_ruby_enabled"),
        ("java", "dependency_mirror_jvm_enabled"),
        ("rust", "dependency_mirror_cargo_enabled"),
    ],
)
def test_per_ecosystem_gate_is_effective(language: str, gate: str):
    config = _config(language, **{gate: False})

    assert build_dependency_mirror_script(config) == ""
    configure_dependency_mirror_environment(config)
    assert config.env == {}


@pytest.mark.parametrize(
    "language",
    ["python", "go", "javascript", "php", "r", "ruby", "java", "rust"],
)
def test_generated_script_is_public_https_only_and_shell_valid(language: str):
    config = _config(
        language,
        dependency_mirror_node_enabled=True,
    )
    script = build_dependency_mirror_script(config)
    syntax = subprocess.run(
        ["bash", "-n"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )

    assert syntax.returncode == 0, syntax.stderr
    assert "https://" in script
    assert "http://mirrors" not in script
    assert "/etc/hosts" not in script
    assert "gradle-wrapper.properties" not in script
    assert "--network host" not in script
    assert "seccomp=unconfined" not in script


def test_environment_defaults_are_language_scoped_and_preserve_operator_values():
    python = _config(
        "python",
        env={"PIP_INDEX_URL": "https://packages.example.test/simple"},
    )
    configure_dependency_mirror_environment(python)
    assert python.env == {
        "PIP_INDEX_URL": "https://packages.example.test/simple"
    }

    java = _config("java")
    configure_dependency_mirror_environment(java)
    assert java.env == {"MVNW_REPOURL": PUBLIC_MAVEN_REPOSITORY}

    unknown = _config("unknown")
    configure_dependency_mirror_environment(unknown)
    assert unknown.env == {}


def test_python_setup_preserves_existing_files_and_never_touches_checkout(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    sentinel = project / "gradle-wrapper.properties"
    sentinel.write_text("keep-me\n", encoding="utf-8")
    home = tmp_path / "home"
    pip_config = home / ".pip" / "pip.conf"
    pip_config.parent.mkdir(parents=True)
    pip_config.write_text(
        "[global]\nindex-url = https://packages.example.test/simple\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.pop("PIP_INDEX_URL", None)
    environment["HOME"] = str(home)

    result = subprocess.run(
        ["bash", "-c", build_dependency_mirror_script(_config("python"))],
        cwd=project,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "packages.example.test" in pip_config.read_text(encoding="utf-8")
    assert sentinel.read_text(encoding="utf-8") == "keep-me\n"


def test_python_setup_creates_public_config_when_no_value_exists(tmp_path: Path):
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    environment = dict(os.environ)
    environment.pop("PIP_INDEX_URL", None)
    environment["HOME"] = str(home)

    result = subprocess.run(
        ["bash", "-c", build_dependency_mirror_script(_config("python"))],
        cwd=project,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert PUBLIC_PYPI_INDEX in (
        home / ".pip" / "pip.conf"
    ).read_text(encoding="utf-8")


def test_jvm_setup_stays_outside_checkout_and_writes_valid_public_urls(tmp_path: Path):
    project = tmp_path / "project"
    wrapper = project / "gradle" / "wrapper" / "gradle-wrapper.properties"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("distributionUrl=https://services.gradle.org/keep.zip\n")
    home = tmp_path / "home"
    environment = dict(os.environ, HOME=str(home))

    result = subprocess.run(
        ["bash", "-c", build_dependency_mirror_script(_config("java"))],
        cwd=project,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert PUBLIC_MAVEN_REPOSITORY in (
        home / ".m2" / "settings.xml"
    ).read_text(encoding="utf-8")
    assert PUBLIC_MAVEN_REPOSITORY in (
        home / ".gradle" / "init.d" / "canopy-public-mirrors.gradle"
    ).read_text(encoding="utf-8")
    assert PUBLIC_MAVEN_REPOSITORY in (
        home / ".sbt" / "repositories"
    ).read_text(encoding="utf-8")
    assert wrapper.read_text() == (
        "distributionUrl=https://services.gradle.org/keep.zip\n"
    )


def test_cargo_setup_writes_public_sparse_registry_without_shell_quotes(tmp_path: Path):
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()

    result = subprocess.run(
        ["bash", "-c", build_dependency_mirror_script(_config("rust"))],
        cwd=project,
        env=dict(os.environ, HOME=str(home)),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    cargo_config = (home / ".cargo" / "config.toml").read_text(encoding="utf-8")
    assert f'registry = "{PUBLIC_CARGO_REGISTRY}"' in cargo_config
    assert "'" not in cargo_config


def test_go_proxy_rejects_non_https_endpoint():
    with pytest.raises(ValueError, match="HTTPS URL"):
        build_dependency_mirror_script(
            _config("go", go_proxy="http://packages.example.test/go,direct")
        )


def test_docker_environment_runs_setup_only_when_enabled(monkeypatch):
    calls: list[str] = []

    async def fake_execute(self, command="", **kwargs):
        del self, kwargs
        calls.append(command)
        return {"returncode": 0}

    monkeypatch.setattr(DockerEnvironment, "execute", fake_execute)

    disabled = DockerEnvironmentConfig(
        image="example.invalid/swe:latest",
        dependency_mirror_enabled=False,
    )
    enabled = disabled.model_copy(
        update={
            "repo_language": "python",
            "dependency_mirror_enabled": True,
            "dependency_mirror_apt_enabled": False,
        }
    )

    import asyncio

    assert asyncio.run(
        DockerEnvironment(disabled)._configure_dependency_mirrors()
    ) is None
    assert calls == []
    asyncio.run(DockerEnvironment(enabled)._configure_dependency_mirrors())
    assert len(calls) == 1
    assert PUBLIC_PYPI_INDEX in calls[0]
