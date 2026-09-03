"""Environment-driven settings for the AppWorld service and client."""

from __future__ import annotations

import os
from pathlib import Path


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


DEFAULT_APPWORLD_ROOT = str(
    Path(os.getenv("APPWORLD_ROOT", "/opt/appworld")).expanduser().resolve()
)
os.environ.setdefault("APPWORLD_ROOT", DEFAULT_APPWORLD_ROOT)


class ServerConfig:
    APPWORLD_ROOT = DEFAULT_APPWORLD_ROOT
    ALLOWED_OUTPUT_ROOT = str(
        Path(
            os.getenv(
                "APPWORLD_ALLOWED_OUTPUT_ROOT",
                str(Path(DEFAULT_APPWORLD_ROOT) / "experiments"),
            )
        )
        .expanduser()
        .resolve()
    )
    INIT_TIMEOUT = _positive_int("APPWORLD_INIT_TIMEOUT", 180)
    EXEC_TIMEOUT = _positive_int("APPWORLD_EXEC_TIMEOUT", 110)
    ENV_INTERNAL_TIMEOUT = _positive_int("APPWORLD_ENV_INTERNAL_TIMEOUT", 100)
    WORKER_IDLE_TIMEOUT = _positive_int("APPWORLD_WORKER_IDLE_TIMEOUT", 1200)
    EVAL_TIMEOUT = _positive_int("APPWORLD_EVAL_TIMEOUT", 240)
    WORKER_MEMORY_LIMIT_MB = _positive_int("APPWORLD_WORKER_MEMORY_LIMIT_MB", 6144)
    MAX_SESSIONS = _positive_int("APPWORLD_MAX_SESSIONS", 32)
    MAX_ACTION_CHARS = _positive_int("APPWORLD_MAX_ACTION_CHARS", 100_000)
    MAX_OBSERVATION_CHARS = _positive_int(
        "APPWORLD_MAX_OBSERVATION_CHARS", 20_000
    )


class ClientConfig:
    APPWORLD_ROOT = DEFAULT_APPWORLD_ROOT
    INIT_TIMEOUT = _positive_int("APPWORLD_CLIENT_INIT_TIMEOUT", 190)
    EXEC_TIMEOUT = _positive_int("APPWORLD_CLIENT_EXEC_TIMEOUT", 120)
    EVAL_TIMEOUT = _positive_int("APPWORLD_CLIENT_EVAL_TIMEOUT", 240)
