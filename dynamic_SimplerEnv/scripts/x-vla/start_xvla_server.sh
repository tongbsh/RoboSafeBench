#!/bin/bash
# ============================================================
# Start the X-VLA FastAPI inference server.
#
# Run this BEFORE launching DyniSafety-SimplerEnv with --model xvla.
#
# Layout:
#   SIMPLER_ROOT   …/dynamic_SimplerEnv  (directory that contains scripts/)
#   BENCH_ROOT     parent of SIMPLER_ROOT (workspace with vlaModelHome/ and download_checkpoints/)
#   Deploy script  ${BENCH_ROOT}/vlaModelHome/X-VLA/deploy.py
#
# Usage:
#   bash dynamic_SimplerEnv/scripts/x-vla/start_xvla_server.sh [OPTIONS]
#
# Options:
#   --model-path  PATH     HuggingFace repo ID or local model directory.
#                          Default: download_checkpoints/xVLA/xvla-widowx-full
#   --lora-path   PATH     Optional LoRA adapter path (default: none)
#   --port        PORT     Server port (default: 8010)
#   --env-name    NAME     Conda environment with X-VLA deps (default: XVLA)
#   --bench-root  DIR      Override workspace root (default: parent of dynamic_SimplerEnv)
#
# SimplerEnv domain: WidowX robot (domain_id=0).
# The client in simpler_env/policies/xvla/ sets XVLA_DOMAIN_ID accordingly.
#
# Examples:
#   # X-VLA-WidowX (Bridge fine-tuned, SimplerEnv default):
#   bash scripts/x-vla/start_xvla_server.sh \
#       --model-path 2toINF/X-VLA-WidowX
#
#   # Local checkpoint:
#   bash scripts/x-vla/start_xvla_server.sh \
#       --model-path /path/to/xvla-checkpoint
#
# After the server prints "Application startup complete", open a second terminal:
#   XVLA_SERVER_HOST=127.0.0.1 XVLA_SERVER_PORT=8010 XVLA_DOMAIN_ID=0 \
#   bash scripts/x-vla/run_dynisafety_eval.sh --task pick_coke_can --level 1
# ============================================================
rm -r vlaModelHome/X-VLA/logs
set -euo pipefail

MODEL_PATH="download_checkpoints/xVLA/xvla-simpler-widowx"
LORA_PATH=""
PORT=8010
CONDA_ENV_NAME="XVLA"
BENCH_ROOT_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)  MODEL_PATH="$2";           shift 2 ;;
        --lora-path)   LORA_PATH="$2";            shift 2 ;;
        --port)        PORT="$2";                 shift 2 ;;
        --env-name)    CONDA_ENV_NAME="$2";       shift 2 ;;
        --bench-root)  BENCH_ROOT_OVERRIDE="$2";  shift 2 ;;
        -h|--help)
            sed -n '3,48p' "$0"
            exit 0 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIMPLER_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"      # dynamic_SimplerEnv/
BENCH_ROOT="${BENCH_ROOT_OVERRIDE:-$(cd "${SIMPLER_ROOT}/.." && pwd)}"

DEPLOY_SCRIPT="${BENCH_ROOT}/vlaModelHome/X-VLA/deploy.py"

if [[ ! -f "${DEPLOY_SCRIPT}" ]]; then
    echo "ERROR: deploy.py not found at '${DEPLOY_SCRIPT}'."
    echo "  Clone X-VLA into: ${BENCH_ROOT}/vlaModelHome/X-VLA"
    echo "  Or override workspace root with: --bench-root /path/to/workspace"
    exit 1
fi

# Resolve relative paths to absolute paths
if [[ "${MODEL_PATH}" != /* ]]; then
    MODEL_PATH="${BENCH_ROOT}/${MODEL_PATH}"
fi

set +u
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "${CONDA_PREFIX:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

CMD=(python "${DEPLOY_SCRIPT}" --model_path "${MODEL_PATH}" --port "${PORT}")
if [[ -n "${LORA_PATH}" ]]; then
    if [[ "${LORA_PATH}" != /* ]]; then
        LORA_PATH="${BENCH_ROOT}/${LORA_PATH}"
    fi
    CMD+=(--LoRA_path "${LORA_PATH}")
fi

echo "============================================================"
echo "X-VLA Inference Server  [DyniSafety-SimplerEnv]"
echo "  Model      : ${MODEL_PATH}"
echo "  LoRA       : ${LORA_PATH:-none}"
echo "  Port       : ${PORT}"
echo "  Conda env  : ${CONDA_ENV_NAME}"
echo "============================================================"

# Run from within X-VLA repo dir so relative imports resolve correctly
cd "${BENCH_ROOT}/vlaModelHome/X-VLA"
exec "${CMD[@]}"
