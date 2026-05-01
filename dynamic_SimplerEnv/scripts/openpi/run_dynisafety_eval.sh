#!/bin/bash
# DyniSafety-SimplerEnv + openpi client (--model pi0 or pi05).
# Set OPENPI_SERVER_HOST / OPENPI_SERVER_PORT (defaults 127.0.0.1:8000).
# Start server: scripts/openpi/start_pi0_server.sh
#
# Outputs: dynamic_SimplerEnv/outputs/pi0/ or outputs/pi05/
set -euo pipefail

MODEL_NAME="pi0"
TASK="pick_coke_can"
LEVEL="1"
CKPT="ignored"
CONDA_ENV_NAME="openvla-oft"
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)     shift; MODEL_NAME="$1"; shift ;;  # pi0 | pi05
        --task)      shift; TASK="$1"; shift ;;
        --level)     shift; LEVEL="$1"; shift ;;
        --ckpt)      shift; CKPT="$1"; shift ;;
        --env-name)  shift; CONDA_ENV_NAME="$1"; shift ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

if [[ "${MODEL_NAME}" != "pi0" && "${MODEL_NAME}" != "pi05" ]]; then
    echo "ERROR: --model must be pi0 or pi05"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIMPLER_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

set +u
# shellcheck disable=SC1091
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "${CONDA_PREFIX:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

cd "${SIMPLER_ROOT}"
exec bash scripts/run_dynisafety.sh \
    --model "${MODEL_NAME}" \
    --ckpt "${CKPT}" \
    --task "${TASK}" \
    --level "${LEVEL}" \
    "${EXTRA[@]}"
