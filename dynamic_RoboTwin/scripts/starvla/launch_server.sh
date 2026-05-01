#!/usr/bin/env bash
#
# Launch the StarVLA websocket inference server for DyniSafety-RoboTwin.
#
# Usage:
#   bash launch_server.sh [--ckpt-path PATH] [--port PORT] [--env-name NAME]
#                         [--use-bf16|--no-bf16] [--starvla-root DIR]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../_common.sh
source "${SCRIPT_DIR}/../_common.sh"

CKPT_PATH="${STARVLA_CHECKPOINT:-${WORKSPACE_ROOT}/download_checkpoints/starVLA/starvla-robotwin2-all/checkpoints/steps_140000_pytorch_model.pt}"
PORT="${PORT:-10093}"
CONDA_ENV="${CONDA_ENV:-starVLA}"
USE_BF16=1
IDLE_TIMEOUT="${IDLE_TIMEOUT:-1800}"
STARVLA_ROOT_OVERRIDE="${STARVLA_ROOT:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt-path)    CKPT_PATH="$2";            shift 2 ;;
        --port)         PORT="$2";                 shift 2 ;;
        --env-name)     CONDA_ENV="$2";            shift 2 ;;
        --use-bf16)     USE_BF16=1;                shift ;;
        --no-bf16)      USE_BF16=0;                shift ;;
        --starvla-root) STARVLA_ROOT_OVERRIDE="$2"; shift 2 ;;
        --idle-timeout) IDLE_TIMEOUT="$2";         shift 2 ;;
        -h|--help)
            sed -n '3,8p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

conda activate "${CONDA_ENV}"

STARVLA_ROOT="${STARVLA_ROOT_OVERRIDE:-${WORKSPACE_ROOT}/vlaModelHome/starVLA}"
SERVER_SCRIPT="${STARVLA_ROOT}/deployment/model_server/server_policy.py"

if [[ "${CKPT_PATH}" != /* ]]; then
    CKPT_PATH="${WORKSPACE_ROOT}/${CKPT_PATH}"
fi

if [[ ! -f "${SERVER_SCRIPT}" ]]; then
    echo "ERROR: server_policy.py not found at '${SERVER_SCRIPT}'." >&2
    exit 1
fi
if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "ERROR: checkpoint not found: '${CKPT_PATH}'" >&2
    exit 1
fi

CMD=(python "${SERVER_SCRIPT}" --ckpt_path "${CKPT_PATH}" --port "${PORT}" --idle_timeout "${IDLE_TIMEOUT}")
if [[ "${USE_BF16}" -eq 1 ]]; then
    CMD+=(--use_bf16)
fi

echo "============================================================"
echo "StarVLA Inference Server (WebSocket)  [DyniSafety-RoboTwin]"
echo "  Checkpoint : ${CKPT_PATH}"
echo "  Port       : ${PORT}"
echo "  Conda env  : ${CONDA_ENV}"
echo "  Work dir   : ${STARVLA_ROOT}"
echo "  Idle sec   : ${IDLE_TIMEOUT}"
if [[ "${USE_BF16}" -eq 1 ]]; then
    echo "  bfloat16   : on"
else
    echo "  bfloat16   : off"
fi
echo "============================================================"

cd "${STARVLA_ROOT}"
exec "${CMD[@]}"
