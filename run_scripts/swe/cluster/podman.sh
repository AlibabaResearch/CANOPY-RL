#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
STORAGE_TEMPLATE="${SCRIPT_DIR}/storage.conf"
PODMAN_CONFIG_ROOT="/run/canopy-podman"
PODMAN_STORAGE_CONF="${PODMAN_CONFIG_ROOT}/storage.conf"
PODMAN_GRAPH_ROOT="/podman_data/storage"
PODMAN_RUN_ROOT="/run/containers/storage"

if [[ "${EUID}" -ne 0 ]]; then
    echo "Podman setup requires root privileges." >&2
    exit 1
fi
if ! command -v podman >/dev/null 2>&1; then
    echo "podman is not installed or not on PATH." >&2
    exit 1
fi
if [[ ! -f "${STORAGE_TEMPLATE}" || -L "${STORAGE_TEMPLATE}" ]]; then
    echo "Missing or unsafe Podman storage template: ${STORAGE_TEMPLATE}" >&2
    exit 1
fi

for path in "${PODMAN_CONFIG_ROOT}" "${PODMAN_GRAPH_ROOT}" "${PODMAN_RUN_ROOT}"; do
    if [[ -L "${path}" ]]; then
        echo "Refusing symlinked Podman path: ${path}" >&2
        exit 1
    fi
done

mkdir -p "${PODMAN_GRAPH_ROOT}" "${PODMAN_RUN_ROOT}"
install -d -m 0700 -o root -g root "${PODMAN_CONFIG_ROOT}"
if [[ "$(stat -c '%u:%a' "${PODMAN_CONFIG_ROOT}")" != "0:700" ]]; then
    echo "${PODMAN_CONFIG_ROOT} must be owned by root with mode 0700." >&2
    exit 1
fi
install -m 0644 -o root -g root "${STORAGE_TEMPLATE}" "${PODMAN_STORAGE_CONF}"

if [[ ! -e /dev/fuse ]]; then
    mknod /dev/fuse c 10 229
elif [[ ! -c /dev/fuse ]]; then
    echo "/dev/fuse exists but is not a character device." >&2
    exit 1
fi
chmod 0666 /dev/fuse

export CONTAINERS_STORAGE_CONF="${PODMAN_STORAGE_CONF}"
podman info >/dev/null
echo "Podman storage is ready: ${PODMAN_STORAGE_CONF}"
