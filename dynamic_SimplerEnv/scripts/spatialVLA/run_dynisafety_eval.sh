#!/bin/bash
# Use policy model spatialvla_server. Outputs: dynamic_SimplerEnv/outputs/spatialvla_server/{logs,videos}/
set -euo pipefail

TASK="pick_coke_can"
LEVEL="1"
CKPT="ignored"
SERVER_HOST="${SPATIALVLA_SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SPATIALVLA_SERVER_PORT:-8112}"
CONDA_ENV_NAME="openvla-oft"
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)        shift; TASK="$1"; shift ;;
        --level)       shift; LEVEL="$1"; shift ;;
        --server-host) shift; SERVER_HOST="$1"; shift ;;
        --server-port) shift; SERVER_PORT="$1"; shift ;;
        --env-name)    shift; CONDA_ENV_NAME="$1"; shift ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIMPLER_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export SPATIALVLA_SERVER_HOST="${SERVER_HOST}"
export SPATIALVLA_SERVER_PORT="${SERVER_PORT}"

set +u
# shellcheck disable=SC1091
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "${CONDA_PREFIX:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

cd "${SIMPLER_ROOT}"
exec bash scripts/run_dynisafety.sh \
    --model spatialvla_server \
    --ckpt "${CKPT}" \
    --task "${TASK}" \
    --level "${LEVEL}" \
    "${EXTRA[@]}"
