#!/bin/bash
# ============================================================
# Start the X-VLA FastAPI inference server.
#
# This script must be run BEFORE launching the DyniSafety-LIBERO
# or DyniSafety-SimplerEnv benchmark with --model_family xvla.
#
# Usage:
#   bash dynisafety_libero/scripts/start_xvla_server.sh [OPTIONS]
#
# Options:
#   --model-path  PATH     HuggingFace repo ID or local model directory.
#                          (e.g. 2toINF/X-VLA-Libero)
#   --lora-path   PATH     Optional LoRA adapter path (default: none)
#   --port        PORT     Server port (default: 8010)
#   --env-name    NAME     Conda environment with X-VLA deps (default: XVLA)
#
# Examples:
#   # X-VLA-Libero (LIBERO fine-tuned, domain_id=3):
#   bash dynisafety_libero/scripts/start_xvla_server.sh \
#       --model-path 2toINF/X-VLA-Libero
#
#   # X-VLA-WidowX (Bridge fine-tuned, domain_id=0):
#   bash dynisafety_libero/scripts/start_xvla_server.sh \
#       --model-path 2toINF/X-VLA-WidowX --port 8010
#
# After the server prints "Application startup complete", open a second terminal:
#
#   conda activate openvla-oft
#   python dynisafety_libero/run_dynisafety_eval.py \
#       --model_family xvla \
#       --server_host localhost \
#       --server_port 8010 \
#       --task_suite_name safelibero_spatial
# ============================================================
export CUDA_VISIBLE_DEVICES=0

rm -r vlaModelHome/X-VLA/logs
set -euo pipefail

MODEL_PATH="download_checkpoints/xVLA/xvla-libero-full"
LORA_PATH=""
PORT=8010
CONDA_ENV_NAME="XVLA"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)  MODEL_PATH="$2";     shift 2 ;;
        --lora-path)   LORA_PATH="$2";      shift 2 ;;
        --port)        PORT="$2";            shift 2 ;;
        --env-name)    CONDA_ENV_NAME="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,40p' "$0"
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
DEPLOY_SCRIPT="${PROJ_ROOT}/vlaModelHome/X-VLA/deploy.py"

if [[ ! -f "${DEPLOY_SCRIPT}" ]]; then
    echo "ERROR: deploy.py not found at '${DEPLOY_SCRIPT}'."
    exit 1
fi

set +u
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "${CONDA_PREFIX:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

# 将相对路径转换为绝对路径，避免 cd 进 X-VLA 目录后路径解析错误
if [[ "${MODEL_PATH}" != /* ]]; then
    MODEL_PATH="${PROJ_ROOT}/${MODEL_PATH}"
fi

CMD=(python "${DEPLOY_SCRIPT}" --model_path "${MODEL_PATH}" --port "${PORT}")
if [[ -n "${LORA_PATH}" ]]; then
    if [[ "${LORA_PATH}" != /* ]]; then
        LORA_PATH="${PROJ_ROOT}/${LORA_PATH}"
    fi
    CMD+=(--LoRA_path "${LORA_PATH}")
fi

echo "============================================================"
echo "X-VLA Inference Server"
echo "  Model      : ${MODEL_PATH}"
echo "  LoRA       : ${LORA_PATH:-none}"
echo "  Port       : ${PORT}"
echo "  Conda env  : ${CONDA_ENV_NAME}"
echo "============================================================"

# Run from within X-VLA repo dir so relative imports resolve correctly
cd "${PROJ_ROOT}/vlaModelHome/X-VLA"
exec "${CMD[@]}"
