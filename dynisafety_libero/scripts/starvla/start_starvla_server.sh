#!/bin/bash
# ============================================================
# Start the StarVLA WebSocket inference server (server_policy.py).
#
# Run this BEFORE DyniSafety-LIBERO with --model_family starvla or
# run_dynisafety_starvla_eval.py.
#
# Usage:
#   bash dynisafety_libero/scripts/starvla/start_starvla_server.sh [OPTIONS]
#
# Options:
#   --ckpt-path   PATH   Path to checkpoints/*.pt or *.safetensors (not the run folder alone)
#   --port        PORT   Server port (default: 10093)
#   --env-name    NAME   Conda env with StarVLA deps (default: starvla)
#   --use-bf16          Pass --use_bf16 to server
#
# Example:
#   bash dynisafety_libero/scripts/starvla/start_starvla_server.sh \\
#       --ckpt-path /path/to/exp/checkpoints/steps_100000_pytorch_model.pt
# ============================================================
export CUDA_VISIBLE_DEVICES=1
set -euo pipefail

CKPT_PATH="download_checkpoints/starVLA/starvla-libero-4in1/checkpoints/steps_50000_pytorch_model.pt"
PORT=10093
CONDA_ENV_NAME="starVLA"
USE_BF16=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt-path)   CKPT_PATH="$2";    shift 2 ;;
        --port)        PORT="$2";         shift 2 ;;
        --env-name)    CONDA_ENV_NAME="$2"; shift 2 ;;
        --use-bf16)    USE_BF16=1;        shift ;;
        -h|--help)
            sed -n '3,28p' "$0"
            exit 0 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "${CKPT_PATH}" ]]; then
    echo "ERROR: --ckpt-path is required."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
STARVLA_DIR="${PROJ_ROOT}/vlaModelHome/starVLA"
SERVER_SCRIPT="${STARVLA_DIR}/deployment/model_server/server_policy.py"

if [[ ! -f "${SERVER_SCRIPT}" ]]; then
    echo "ERROR: server_policy.py not found at '${SERVER_SCRIPT}'."
    exit 1
fi

if [[ "${CKPT_PATH}" != /* ]]; then
    CKPT_PATH="${PROJ_ROOT}/${CKPT_PATH}"
fi

set +u
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "${CONDA_PREFIX:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

CMD=(python "${SERVER_SCRIPT}" --ckpt_path "${CKPT_PATH}" --port "${PORT}")
if [[ "${USE_BF16}" -eq 1 ]]; then
    CMD+=(--use_bf16)
fi

echo "============================================================"
echo "StarVLA Inference Server (WebSocket)"
echo "  Checkpoint : ${CKPT_PATH}"
echo "  Port       : ${PORT}"
echo "  Conda env  : ${CONDA_ENV_NAME}"
echo "  Work dir   : ${STARVLA_DIR}"
echo "============================================================"

cd "${STARVLA_DIR}"
exec "${CMD[@]}"
