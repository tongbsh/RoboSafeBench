#!/bin/bash
# DyniSafety-SimplerEnv + StarVLA. Outputs: dynamic_SimplerEnv/outputs/starvla/{logs,videos}/
# Prerequisite: scripts/starvla/start_starvla_server.sh
set -euo pipefail

TASK="pick_coke_can"
LEVEL="1"
CKPT="${STARVLA_CHECKPOINT:-download_checkpoints/starVLA/Qwen-GR00T-Bridge/checkpoints/steps_45000_pytorch_model.pt}"
SERVER_HOST="${STARVLA_SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${STARVLA_SERVER_PORT:-10093}"
CONDA_ENV_NAME="openvla-oft"
MAX_STEPS="${BRIDGE_MAX_EPISODE_STEPS:-120}"
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)        shift; TASK="$1"; shift ;;
        --level)       shift; LEVEL="$1"; shift ;;
        --ckpt-path)   shift; CKPT="$1"; shift ;;
        --server-host) shift; SERVER_HOST="$1"; shift ;;
        --server-port) shift; SERVER_PORT="$1"; shift ;;
        --env-name)    shift; CONDA_ENV_NAME="$1"; shift ;;
        --max-steps)   shift; MAX_STEPS="$1"; shift ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIMPLER_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ "${CKPT}" != /* ]]; then
    CKPT="${WORKSPACE}/${CKPT}"
fi

export STARVLA_SERVER_HOST="${SERVER_HOST}"
export STARVLA_SERVER_PORT="${SERVER_PORT}"
# Match official StarVLA SimplerEnv Bridge eval unless explicitly overridden.
export BRIDGE_MAX_EPISODE_STEPS="${MAX_STEPS}"

set +u
# shellcheck disable=SC1091
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "${CONDA_PREFIX:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

cd "${SIMPLER_ROOT}"
exec bash scripts/run_dynisafety.sh \
    --model starvla \
    --ckpt "${CKPT}" \
    --task "${TASK}" \
    --level "${LEVEL}" \
    "${EXTRA[@]}"
