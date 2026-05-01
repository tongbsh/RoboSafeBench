#!/bin/bash
# ============================================================
# Start the Isaac-GR00T ZeroMQ inference server for LIBERO (sim wrapper).
#
# Run this BEFORE DyniSafety-LIBERO with run_dynisafety_gr00t_eval.py.
#
# Usage:
#   bash dynisafety_libero/scripts/gr00t/start_gr00t_server.sh [OPTIONS]
#
# Options:
#   --model-path   PATH   Checkpoint directory (required)
#   --port         PORT   Server port (default: 5555)
#   --device       STR    e.g. cuda, cuda:0 (default: cuda)
#   --host         STR    Bind address (default: 0.0.0.0)
#   --env-name     NAME   Conda env with GR00T deps (default: gr00t)
#
# Example:
#   bash dynisafety_libero/scripts/gr00t/start_gr00t_server.sh \\
#       --model-path /path/to/checkpoint-20000
# ============================================================

set -euo pipefail

MODEL_PATH=""
PORT=5555
DEVICE="cuda"
HOST="0.0.0.0"
CONDA_ENV_NAME="gr00t"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)  MODEL_PATH="$2"; shift 2 ;;
        --port)        PORT="$2";         shift 2 ;;
        --device)      DEVICE="$2";       shift 2 ;;
        --host)        HOST="$2";         shift 2 ;;
        --env-name)    CONDA_ENV_NAME="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,26p' "$0"
            exit 0 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "${MODEL_PATH}" ]]; then
    echo "ERROR: --model-path is required."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
GR00T_DIR="${PROJ_ROOT}/vlaModelHome/Isaac-GR00T"
SERVER_SCRIPT="${GR00T_DIR}/gr00t/eval/run_gr00t_server.py"

if [[ ! -f "${SERVER_SCRIPT}" ]]; then
    echo "ERROR: run_gr00t_server.py not found at '${SERVER_SCRIPT}'."
    exit 1
fi

if [[ "${MODEL_PATH}" != /* ]]; then
    MODEL_PATH="${PROJ_ROOT}/${MODEL_PATH}"
fi

set +u
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "${CONDA_PREFIX:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

echo "============================================================"
echo "Isaac-GR00T Inference Server (ZeroMQ, LIBERO_PANDA + sim wrapper)"
echo "  Model path : ${MODEL_PATH}"
echo "  Bind       : ${HOST}:${PORT}"
echo "  Device     : ${DEVICE}"
echo "  Conda env  : ${CONDA_ENV_NAME}"
echo "  Work dir   : ${GR00T_DIR}"
echo "============================================================"

cd "${GR00T_DIR}"
exec python "${SERVER_SCRIPT}" \
    --model-path "${MODEL_PATH}" \
    --embodiment-tag LIBERO_PANDA \
    --use-sim-policy-wrapper \
    --device "${DEVICE}" \
    --host "${HOST}" \
    --port "${PORT}"
