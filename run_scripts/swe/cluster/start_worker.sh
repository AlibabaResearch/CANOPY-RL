#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CANOPY_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)"

RAY_PORT=6379
GROUP_CAPACITY=1000
MAX_GROUP_NUM=11
HEAD_IP_FILE="${HEAD_IP_FILE:-${CANOPY_ROOT}/runtime/ray/head_ip.txt}"
HEAD_IP="${HEAD_IP:-}"
GROUP_NUM="${GROUP_NUM:-}"
export CONTAINERS_STORAGE_CONF="/run/canopy-podman/storage.conf"

extract_group_num() {
    local candidate="$1"
    if [[ "${candidate}" =~ mf_dsw_[0-9]+_([0-9]+) ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
    fi
}

if [[ -z "${HEAD_IP}" && -s "${HEAD_IP_FILE}" ]]; then
    HEAD_IP="$(tr -d '[:space:]' <"${HEAD_IP_FILE}")"
fi
if [[ -z "${HEAD_IP}" || ! "${HEAD_IP}" =~ ^[A-Za-z0-9.:_-]+$ ]]; then
    echo "Set HEAD_IP or make the shared address file readable: ${HEAD_IP_FILE}" >&2
    exit 1
fi

if [[ -z "${GROUP_NUM}" ]]; then
    for candidate in \
        "${DSW_INSTANCE_NAME:-}" \
        "${INSTANCE_NAME:-}" \
        "${HOSTNAME:-}" \
        "$(hostname)"; do
        GROUP_NUM="$(extract_group_num "${candidate}")"
        if [[ -n "${GROUP_NUM}" ]]; then
            echo "Detected DSW worker group_${GROUP_NUM} from ${candidate}."
            break
        fi
    done
fi

if [[ -z "${GROUP_NUM}" ]]; then
    shopt -s nullglob
    dsw_logs=(/etc/dsw-logs/dsw-agent*.log*)
    shopt -u nullglob
    if ((${#dsw_logs[@]})); then
        mapfile -t instance_names < <(
            grep -hoE 'mf_dsw_[0-9]+_[0-9]+' "${dsw_logs[@]}" 2>/dev/null | sort -u || true
        )
        if ((${#instance_names[@]} == 1)); then
            GROUP_NUM="$(extract_group_num "${instance_names[0]}")"
            echo "Detected DSW worker group_${GROUP_NUM} from ${instance_names[0]}."
        elif ((${#instance_names[@]} > 1)); then
            echo "Multiple DSW instance names were found in rotated logs; refusing ambiguous auto-detection." >&2
        fi
    fi
fi

if [[ "${GROUP_NUM}" =~ ^[0-9]+$ ]]; then
    GROUP_NUM="$((10#${GROUP_NUM}))"
fi
if ! [[ "${GROUP_NUM}" =~ ^[0-9]+$ ]] || ((GROUP_NUM < 1 || GROUP_NUM > MAX_GROUP_NUM)); then
    cat >&2 <<EOF
Unable to infer a worker group in the range 1-${MAX_GROUP_NUM}.
Set it explicitly, for example:
  GROUP_NUM=3 HEAD_IP=${HEAD_IP} bash ${SCRIPT_DIR}/start_worker.sh
Equivalent core command:
  ray start --address="${HEAD_IP}:${RAY_PORT}" --resources='{"group_3": 1000}'
EOF
    exit 1
fi

bash "${SCRIPT_DIR}/podman.sh"
ulimit -n 65535
ray stop --force

ray start \
    --address="${HEAD_IP}:${RAY_PORT}" \
    --resources="{\"group_${GROUP_NUM}\": ${GROUP_CAPACITY}}"

echo "Ray worker joined ${HEAD_IP}:${RAY_PORT} as group_${GROUP_NUM}."
