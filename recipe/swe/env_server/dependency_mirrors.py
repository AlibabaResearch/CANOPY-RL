#!/usr/bin/env python3
# Copyright 2026 Alibaba Group Holding Limited
# SPDX-License-Identifier: Apache-2.0

"""Optional public dependency mirrors for disposable SWE containers.

The feature is fail closed: no script or environment change is produced unless
``dependency_mirror_enabled`` is explicitly enabled.  Generated configuration
is language-scoped, preserves operator/image configuration, uses HTTPS public
endpoints, and never writes inside the checked-out task repository.
"""

from __future__ import annotations

import re
import shlex
from typing import Any
from urllib.parse import urlsplit


PUBLIC_ALPINE_MIRROR = "https://mirrors.aliyun.com/alpine"
PUBLIC_APT_MIRROR = "https://mirrors.aliyun.com"
PUBLIC_CARGO_REGISTRY = "sparse+https://mirrors.aliyun.com/crates.io-index/"
PUBLIC_COMPOSER_REGISTRY = "https://mirrors.aliyun.com/composer/"
PUBLIC_CRAN_REPOSITORY = "https://mirrors.aliyun.com/CRAN/"
PUBLIC_GO_PROXY = "https://mirrors.aliyun.com/goproxy/,direct"
PUBLIC_GRADLE_PLUGIN_REPOSITORY = (
    "https://maven.aliyun.com/repository/gradle-plugin"
)
PUBLIC_MAVEN_REPOSITORY = "https://maven.aliyun.com/repository/public"
PUBLIC_NPM_REGISTRY = "https://registry.npmmirror.com/"
PUBLIC_PYPI_INDEX = "https://mirrors.aliyun.com/pypi/simple/"
PUBLIC_RUBYGEMS_SOURCE = "https://mirrors.aliyun.com/rubygems/"
PUBLIC_RUSTUP_DIST_SERVER = "https://mirrors.aliyun.com/rustup"
PUBLIC_RUSTUP_UPDATE_ROOT = "https://mirrors.aliyun.com/rustup/rustup"
SBT_PLUGIN_REPOSITORY_PATTERN = (
    "https://repo.scala-sbt.org/scalasbt/sbt-plugin-releases/, "
    "[organization]/[module]/(scala_[scalaVersion]/)(sbt_[sbtVersion]/)"
    "[revision]/[type]s/[artifact](-[classifier]).[ext]"
)

_GO_LANGUAGES = {"go", "golang"}
_JVM_LANGUAGES = {"clojure", "groovy", "java", "kotlin", "scala"}
_NODE_LANGUAGES = {
    "javascript",
    "javascript/typescript",
    "js",
    "node",
    "nodejs",
    "ts",
    "typescript",
}
_PHP_LANGUAGES = {"php"}
_PYTHON_LANGUAGES = {"py", "python", "python3"}
_R_LANGUAGES = {"r"}
_RUBY_LANGUAGES = {"rb", "ruby"}
_RUST_LANGUAGES = {"rs", "rust"}

DEPENDENCY_MIRROR_POLICY_DEFAULTS = {
    "dependency_mirror_enabled": False,
    "dependency_mirror_apt_enabled": True,
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


def _enabled(config: Any, name: str, default: bool = False) -> bool:
    return bool(getattr(config, name, default))


def _language(config: Any) -> str:
    return str(getattr(config, "repo_language", "") or "").strip().lower()


def resolve_dependency_mirror_policy(config: Any) -> dict[str, bool]:
    """Read every public mirror gate from a mapping-like trainer config."""

    getter = getattr(config, "get", None)
    if not callable(getter):
        raise TypeError("dependency mirror policy must be mapping-like")
    return {
        name: bool(getter(name, default))
        for name, default in DEPENDENCY_MIRROR_POLICY_DEFAULTS.items()
    }


def _safe_https_url(value: str, *, label: str, sparse: bool = False) -> str:
    """Validate a URL before interpolating it into a generated shell script."""

    raw = str(value or "").strip()
    parsed_value = raw[len("sparse+") :] if sparse and raw.startswith("sparse+") else raw
    parsed = urlsplit(parsed_value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or re.search(r"[\x00-\x20\x7f]", raw)
    ):
        raise ValueError(f"{label} must be an HTTPS URL without credentials or fragments")
    return raw


def _quote_url(value: str, *, label: str, sparse: bool = False) -> str:
    return shlex.quote(_safe_https_url(value, label=label, sparse=sparse))


def _apt_script(config: Any) -> str:
    # ``apt_mirror`` predates the dependency-mirror bundle and is handled by
    # DockerEnvironment's legacy explicit opt-in path. Avoid rewriting twice.
    if getattr(config, "apt_mirror", None):
        return ""
    apt = _quote_url(PUBLIC_APT_MIRROR, label="APT mirror")
    alpine = _quote_url(PUBLIC_ALPINE_MIRROR, label="Alpine mirror")
    return f"""
swe_apt_mirror={apt}
if [ -f /etc/apt/sources.list ]; then
    sed -Ei \
        -e "s|https?://([[:alnum:]-]+\\.)?archive\\.ubuntu\\.com|$swe_apt_mirror|g" \
        -e "s|https?://security\\.ubuntu\\.com|$swe_apt_mirror|g" \
        -e "s|https?://deb\\.debian\\.org|$swe_apt_mirror|g" \
        -e "s|https?://security\\.debian\\.org|$swe_apt_mirror|g" \
        /etc/apt/sources.list || true
fi
if [ -d /etc/apt/sources.list.d ]; then
    find /etc/apt/sources.list.d -type f \\( -name '*.sources' -o -name '*.list' \\) -print0 2>/dev/null |
        xargs -0 -r sed -Ei \
            -e "s|https?://([[:alnum:]-]+\\.)?archive\\.ubuntu\\.com|$swe_apt_mirror|g" \
            -e "s|https?://security\\.ubuntu\\.com|$swe_apt_mirror|g" \
            -e "s|https?://deb\\.debian\\.org|$swe_apt_mirror|g" \
            -e "s|https?://security\\.debian\\.org|$swe_apt_mirror|g" || true
fi
swe_apk_repositories=/etc/apk/repositories
if [ -f "$swe_apk_repositories" ] && grep -Eq 'https?://dl-cdn\\.alpinelinux\\.org/alpine' "$swe_apk_repositories"; then
    sed -Ei 's|https?://dl-cdn\\.alpinelinux\\.org/alpine|{alpine}|g' "$swe_apk_repositories" || true
fi
"""


def _python_script() -> str:
    index = _quote_url(PUBLIC_PYPI_INDEX, label="PyPI mirror")
    return f"""
swe_home="${{HOME:-/root}}"
swe_pip_legacy="$swe_home/.pip/pip.conf"
swe_pip_xdg="${{XDG_CONFIG_HOME:-$swe_home/.config}}/pip/pip.conf"
if [ -n "${{PIP_INDEX_URL:-}}" ]; then
    :
elif [ -e "$swe_pip_legacy" ] || [ -e "$swe_pip_xdg" ]; then
    :
else
    mkdir -p "$(dirname "$swe_pip_legacy")"
    printf '[global]\nindex-url = %s\n' {index} > "$swe_pip_legacy" || true
fi
"""


def _go_script(config: Any) -> str:
    proxy_value = str(getattr(config, "go_proxy", "") or PUBLIC_GO_PROXY)
    # GOPROXY is a comma-separated list. Validate every URL element while
    # retaining the special ``direct`` and ``off`` tokens.
    validated: list[str] = []
    for item in (part.strip() for part in proxy_value.split(",")):
        if item in {"direct", "off"}:
            validated.append(item)
        elif item:
            validated.append(_safe_https_url(item, label="Go proxy"))
    if not validated:
        raise ValueError("Go proxy must contain an HTTPS URL, direct, or off")
    proxy = shlex.quote(",".join(validated))
    return f"""
swe_home="${{HOME:-/root}}"
swe_go_env="${{GOENV:-${{XDG_CONFIG_HOME:-$swe_home/.config}}/go/env}}"
if [ "${{GOENV:-}}" = off ] || [ -n "${{GOPROXY:-}}" ]; then
    :
elif [ -e "$swe_go_env" ] && grep -Eq '^[[:space:]]*GOPROXY=' "$swe_go_env"; then
    :
else
    mkdir -p "$(dirname "$swe_go_env")"
    printf 'GOPROXY=%s\n' {proxy} >> "$swe_go_env" || true
fi
"""


def _node_script() -> str:
    registry = _quote_url(PUBLIC_NPM_REGISTRY, label="npm mirror")
    return f"""
swe_home="${{HOME:-/root}}"
swe_npmrc="${{NPM_CONFIG_USERCONFIG:-$swe_home/.npmrc}}"
if [ -n "${{NPM_CONFIG_REGISTRY:-${{npm_config_registry:-}}}}" ] || [ -e "$swe_npmrc" ]; then
    :
else
    mkdir -p "$(dirname "$swe_npmrc")"
    printf 'registry=%s\n' {registry} > "$swe_npmrc" || true
fi
"""


def _composer_script() -> str:
    registry = _quote_url(PUBLIC_COMPOSER_REGISTRY, label="Composer mirror")
    return f"""
swe_home="${{HOME:-/root}}"
swe_composer_xdg="${{XDG_CONFIG_HOME:-$swe_home/.config}}/composer/config.json"
swe_composer_legacy="$swe_home/.composer/config.json"
if [ -n "${{COMPOSER_REPO_PACKAGIST:-}}" ] || [ -e "$swe_composer_xdg" ] || [ -e "$swe_composer_legacy" ]; then
    :
elif command -v composer >/dev/null 2>&1; then
    COMPOSER_ALLOW_SUPERUSER=1 composer config -g repo.packagist composer {registry} >/dev/null 2>&1 || true
fi
"""


def _r_script() -> str:
    registry = _quote_url(PUBLIC_CRAN_REPOSITORY, label="CRAN mirror")
    return f"""
swe_home="${{HOME:-/root}}"
swe_r_profile="$swe_home/.Rprofile"
if [ -z "${{R_PROFILE_USER:-}}" ] && [ ! -e "$swe_r_profile" ]; then
    printf 'options("repos" = c(CRAN="%s"))\n' {registry} > "$swe_r_profile" || true
fi
"""


def _ruby_script() -> str:
    registry = _quote_url(PUBLIC_RUBYGEMS_SOURCE, label="RubyGems mirror")
    return f"""
swe_home="${{HOME:-/root}}"
swe_gemrc="$swe_home/.gemrc"
if [ ! -e "$swe_gemrc" ]; then
    printf '%s\n' '---' ':sources:' '- {registry}' > "$swe_gemrc" || true
fi
if command -v bundle >/dev/null 2>&1 && [ ! -e "${{BUNDLE_USER_CONFIG:-$swe_home/.bundle/config}}" ]; then
    bundle config set --global mirror.https://rubygems.org {registry} >/dev/null 2>&1 || true
fi
"""


def _cargo_script() -> str:
    registry = _quote_url(
        PUBLIC_CARGO_REGISTRY,
        label="Cargo mirror",
        sparse=True,
    )
    return f"""
swe_home="${{HOME:-/root}}"
swe_cargo_home="${{CARGO_HOME:-$swe_home/.cargo}}"
swe_cargo_toml="$swe_cargo_home/config.toml"
swe_cargo_legacy="$swe_cargo_home/config"
if [ -e "$swe_cargo_toml" ]; then
    swe_cargo_target="$swe_cargo_toml"
elif [ -e "$swe_cargo_legacy" ]; then
    swe_cargo_target="$swe_cargo_legacy"
else
    swe_cargo_target="$swe_cargo_toml"
fi
if [ -e "$swe_cargo_target" ] && \
    grep -Eq '^[[:space:]]*\\[source\\.(crates-io|[^]]+)\\][[:space:]]*$' "$swe_cargo_target"; then
    :
else
    mkdir -p "$swe_cargo_home"
    [ ! -s "$swe_cargo_target" ] || printf '\n' >> "$swe_cargo_target"
    printf '%s\n' \
        '[source.crates-io]' 'replace-with = "canopy-public"' '' \
        '[source.canopy-public]' 'registry = "{registry}"' \
        >> "$swe_cargo_target" || true
fi
"""


def _jvm_script() -> str:
    # These values are written inside single-quoted here-documents rather than
    # evaluated by the shell, so retain the validated URL without shell quote
    # characters in the generated XML/Groovy text.
    maven = _safe_https_url(PUBLIC_MAVEN_REPOSITORY, label="Maven mirror")
    plugin = _safe_https_url(
        PUBLIC_GRADLE_PLUGIN_REPOSITORY,
        label="Gradle plugin mirror",
    )
    return f"""
swe_home="${{HOME:-/root}}"
swe_maven_home="${{MAVEN_CONFIG:-$swe_home/.m2}}"
swe_gradle_init="${{GRADLE_USER_HOME:-$swe_home/.gradle}}/init.d/canopy-public-mirrors.gradle"
swe_sbt_repositories="$swe_home/.sbt/repositories"
mkdir -p "$swe_maven_home" "$(dirname "$swe_gradle_init")" "$(dirname "$swe_sbt_repositories")"
if [ ! -e "$swe_maven_home/settings.xml" ]; then
    cat > "$swe_maven_home/settings.xml" <<'CANOPY_MAVEN_SETTINGS'
<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0">
  <mirrors>
    <mirror>
      <id>canopy-public</id>
      <mirrorOf>central</mirrorOf>
      <url>{maven}</url>
    </mirror>
  </mirrors>
</settings>
CANOPY_MAVEN_SETTINGS
fi
if [ ! -e "$swe_gradle_init" ]; then
    cat > "$swe_gradle_init" <<'CANOPY_GRADLE_INIT'
import org.gradle.api.artifacts.repositories.MavenArtifactRepository
def canopyCentral = '{maven}'
def canopyPlugins = '{plugin}'
def rewriteCanopyRepository = {{ repo ->
    if (!(repo instanceof MavenArtifactRepository)) return
    def current = repo.url.toString().replaceAll('/+$', '')
    if (current in [
        'https://repo.maven.apache.org/maven2',
        'https://repo1.maven.org/maven2'
    ]) repo.setUrl(canopyCentral)
    if (current == 'https://plugins.gradle.org/m2') repo.setUrl(canopyPlugins)
}}
gradle.beforeProject {{ project ->
    project.buildscript.repositories.withType(MavenArtifactRepository).all(rewriteCanopyRepository)
    project.repositories.withType(MavenArtifactRepository).all(rewriteCanopyRepository)
}}
CANOPY_GRADLE_INIT
fi
if [ ! -e "$swe_sbt_repositories" ]; then
    cat > "$swe_sbt_repositories" <<'CANOPY_SBT_REPOSITORIES'
[repositories]
local
canopy-public: {maven}
maven-central
sbt-plugin-releases: {SBT_PLUGIN_REPOSITORY_PATTERN}
CANOPY_SBT_REPOSITORIES
fi
"""


def configure_dependency_mirror_environment(config: Any) -> None:
    """Add language-scoped defaults without replacing operator values."""

    if not _enabled(config, "dependency_mirror_enabled"):
        return
    environment = getattr(config, "env", None)
    if not isinstance(environment, dict):
        raise TypeError("Docker environment configuration must provide an env mapping")
    language = _language(config)
    if language in _PYTHON_LANGUAGES and _enabled(
        config, "dependency_mirror_python_enabled", True
    ):
        environment.setdefault("PIP_INDEX_URL", PUBLIC_PYPI_INDEX)
    elif language in _GO_LANGUAGES and _enabled(
        config, "dependency_mirror_go_enabled", True
    ):
        environment.setdefault(
            "GOPROXY",
            str(getattr(config, "go_proxy", "") or PUBLIC_GO_PROXY),
        )
    elif language in _NODE_LANGUAGES and _enabled(
        config, "dependency_mirror_node_enabled", False
    ):
        environment.setdefault("NPM_CONFIG_REGISTRY", PUBLIC_NPM_REGISTRY)
        environment.setdefault("YARN_NPM_REGISTRY_SERVER", PUBLIC_NPM_REGISTRY)
    elif language in _JVM_LANGUAGES and _enabled(
        config, "dependency_mirror_jvm_enabled", True
    ):
        environment.setdefault("MVNW_REPOURL", PUBLIC_MAVEN_REPOSITORY)
    elif language in _RUST_LANGUAGES and _enabled(
        config, "dependency_mirror_rustup_enabled", False
    ):
        environment.setdefault("RUSTUP_DIST_SERVER", PUBLIC_RUSTUP_DIST_SERVER)
        environment.setdefault("RUSTUP_UPDATE_ROOT", PUBLIC_RUSTUP_UPDATE_ROOT)


def build_dependency_mirror_script(config: Any) -> str:
    """Build a language-scoped setup script, or ``""`` when disabled."""

    if not _enabled(config, "dependency_mirror_enabled"):
        return ""

    language = _language(config)
    scripts: list[str] = []
    if _enabled(config, "dependency_mirror_apt_enabled", True):
        scripts.append(_apt_script(config))
    if language in _PYTHON_LANGUAGES and _enabled(
        config, "dependency_mirror_python_enabled", True
    ):
        scripts.append(_python_script())
    elif language in _GO_LANGUAGES and _enabled(
        config, "dependency_mirror_go_enabled", True
    ):
        scripts.append(_go_script(config))
    elif language in _NODE_LANGUAGES and _enabled(
        config, "dependency_mirror_node_enabled", False
    ):
        scripts.append(_node_script())
    elif language in _PHP_LANGUAGES and _enabled(
        config, "dependency_mirror_php_enabled", True
    ):
        scripts.append(_composer_script())
    elif language in _R_LANGUAGES and _enabled(
        config, "dependency_mirror_r_enabled", True
    ):
        scripts.append(_r_script())
    elif language in _RUBY_LANGUAGES and _enabled(
        config, "dependency_mirror_ruby_enabled", True
    ):
        scripts.append(_ruby_script())
    elif language in _JVM_LANGUAGES and _enabled(
        config, "dependency_mirror_jvm_enabled", True
    ):
        scripts.append(_jvm_script())
    elif language in _RUST_LANGUAGES and _enabled(
        config, "dependency_mirror_cargo_enabled", True
    ):
        scripts.append(_cargo_script())

    body = "\n".join(script for script in scripts if script.strip())
    return f"set +e\n{body}\nexit 0\n" if body else ""
