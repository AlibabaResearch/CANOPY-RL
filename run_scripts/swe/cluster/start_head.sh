#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CANOPY_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)"

RAY_PORT=6379
GROUP_CAPACITY=1000
HEAD_IP_FILE="${HEAD_IP_FILE:-${CANOPY_ROOT}/runtime/ray/head_ip.txt}"
DASHBOARD_HOST="${DASHBOARD_HOST:-127.0.0.1}"
HEAD_IP="${HEAD_IP:-}"
export CONTAINERS_STORAGE_CONF="/run/canopy-podman/storage.conf"

if [[ -z "${HEAD_IP}" ]]; then
    HEAD_IP="$(hostname -I | awk '{print $1}')"
fi
if [[ -z "${HEAD_IP}" || ! "${HEAD_IP}" =~ ^[A-Za-z0-9.:_-]+$ ]]; then
    echo "Unable to determine a valid head-node address; set HEAD_IP explicitly." >&2
    exit 1
fi

bash "${SCRIPT_DIR}/podman.sh"
mkdir -p "$(dirname -- "${HEAD_IP_FILE}")"
temporary_ip_file=""
cleanup() {
    if [[ -n "${temporary_ip_file}" ]]; then
        rm -f -- "${temporary_ip_file}"
    fi
}
trap cleanup EXIT

# Atomically invalidate any previous address before restarting Ray. Workers
# require a non-empty file, so they cannot join against a stale head address.
temporary_ip_file="$(mktemp "${HEAD_IP_FILE}.tmp.XXXXXX")"
chmod 0644 "${temporary_ip_file}"
mv -f -- "${temporary_ip_file}" "${HEAD_IP_FILE}"
temporary_ip_file=""

ulimit -n 65535
ray stop --force

ray start \
    --head \
    --node-ip-address="${HEAD_IP}" \
    --port="${RAY_PORT}" \
    --dashboard-host="${DASHBOARD_HOST}" \
    --resources="{\"group_0\": ${GROUP_CAPACITY}}"

# Publish the address only after the head starts successfully.
temporary_ip_file="$(mktemp "${HEAD_IP_FILE}.tmp.XXXXXX")"
printf '%s\n' "${HEAD_IP}" >"${temporary_ip_file}"
chmod 0644 "${temporary_ip_file}"
mv -f -- "${temporary_ip_file}" "${HEAD_IP_FILE}"
temporary_ip_file=""
trap - EXIT

echo "Ray head started at ${HEAD_IP}:${RAY_PORT}; group_0 is registered."
echo "Worker address file: ${HEAD_IP_FILE}"
