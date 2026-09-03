#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
canopy_root="$(cd "${script_dir}/../../.." && pwd)"
cd "${script_dir}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${canopy_root}:${PYTHONPATH:-}"
export APPWORLD_START_PORT="${APPWORLD_START_PORT:-32000}"
export APPWORLD_NUM_SERVERS="${APPWORLD_NUM_SERVERS:-8}"
export APPWORLD_MAX_SERVERS="${APPWORLD_MAX_SERVERS:-256}"
export APPWORLD_HOST="${APPWORLD_HOST:-127.0.0.1}"
export APPWORLD_CONFIG_DIR="${APPWORLD_CONFIG_DIR:-${APPWORLD_SERVER_URL_DIR:-${canopy_root}/runtime/appworld_urls}}"
export APPWORLD_SERVER_URL_DIR="${APPWORLD_SERVER_URL_DIR:-${APPWORLD_CONFIG_DIR}}"
export APPWORLD_LAUNCHER_PID_FILE="${APPWORLD_LAUNCHER_PID_FILE:-${APPWORLD_CONFIG_DIR}/launcher.pid}"
export APPWORLD_LOG_LEVEL="${APPWORLD_LOG_LEVEL:-warning}"
export APPWORLD_NOFILE_LIMIT="${APPWORLD_NOFILE_LIMIT:-65535}"
export APPWORLD_SERVER_MODULE="${APPWORLD_SERVER_MODULE:-recipe.appworld.env_server.server:app}"

exec python launcher.py
