#!/bin/bash
# ============================================================
# Start the SpatialVLA FastAPI inference server.
#
# Run this BEFORE launching DyniSafety-SimplerEnv with --model spatialvla_server.
#
# Layout:
#   SIMPLER_ROOT   …/dynamic_SimplerEnv  (directory that contains scripts/)
#   BENCH_ROOT     parent of SIMPLER_ROOT (workspace with vlaModelHome/ and download_checkpoints/)
#   Server script  ${BENCH_ROOT}/vlaModelHome/SpatialVLA/server.py
#   Base model     ${BENCH_ROOT}/download_checkpoints/spatialVLA/spatialvla-4b-224-pt
#
# Usage:
#   bash dynamic_SimplerEnv/scripts/spatialVLA/start_spatialvla_server.sh [OPTIONS]
#
# Options:
#   --model-path PATH      Base SpatialVLA model directory.
#                          Default: download_checkpoints/spatialVLA/spatialvla-4b-224-pt
#   --lora-path  PATH      Optional LoRA adapter directory.
#   --unnorm-key KEY       Action un-normalisation key from dataset_statistics.json.
#                          SimplerEnv (WidowX / Bridge domain): bridge_orig/1.0.0
#                          Default: bridge_orig/1.0.0
#   --port       PORT      Server port (default: 8112)
#   --env-name   NAME      Conda environment (default: spatialvla)
#   --bench-root DIR       Override workspace root (default: parent of dynamic_SimplerEnv)
#
# Quick-start examples:
#   # Zero-shot with bridge_orig unnorm key (WidowX / SimplerEnv domain):
#   bash scripts/spatialVLA/start_spatialvla_server.sh
#
#   # With a fine-tuned LoRA adapter:
#   bash scripts/spatialVLA/start_spatialvla_server.sh \
#       --lora-path /path/to/spatialvla-lora/checkpoint-XXXXX \
#       --unnorm-key bridge_orig/1.0.0
#
#   # Explicit paths:
#   bash scripts/spatialVLA/start_spatialvla_server.sh \
#       --model-path /path/to/spatialvla-4b-224-pt \
#       --unnorm-key bridge_orig/1.0.0
#
# After the server prints "SpatialVLA server ready", open a second terminal and run:
#   SPATIALVLA_SERVER_HOST=127.0.0.1 SPATIALVLA_SERVER_PORT=8112 \
#   bash scripts/spatialVLA/run_dynisafety_eval.sh --task pick_coke_can --level 1
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIMPLER_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"      # dynamic_SimplerEnv/

# ---------------------------------------------------------------------------
# Defaults (empty means "not set by user")
# ---------------------------------------------------------------------------
MODEL_PATH=""
LORA_PATH=""
UNNORM_KEY="bridge_orig/1.0.0"
PORT=8112
CONDA_ENV_NAME="spatialvla"
BENCH_ROOT_OVERRIDE=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)  MODEL_PATH="$2";           shift 2 ;;
        --lora-path)   LORA_PATH="$2";            shift 2 ;;
        --unnorm-key)  UNNORM_KEY="$2";           shift 2 ;;
        --port)        PORT="$2";                 shift 2 ;;
        --env-name)    CONDA_ENV_NAME="$2";       shift 2 ;;
        --bench-root)  BENCH_ROOT_OVERRIDE="$2";  shift 2 ;;
        -h|--help)
            sed -n '3,54p' "$0"
            exit 0 ;;
        *)
            echo "ERROR: Unknown option: $1"
            echo "Run with --help for usage."
            exit 1 ;;
    esac
done

BENCH_ROOT="${BENCH_ROOT_OVERRIDE:-$(cd "${SIMPLER_ROOT}/.." && pwd)}"
CKPT_BASE="${BENCH_ROOT}/download_checkpoints/spatialVLA"

# ---------------------------------------------------------------------------
# Apply defaults for paths not set by user
# ---------------------------------------------------------------------------
[[ -z "${MODEL_PATH}" ]] && MODEL_PATH="${CKPT_BASE}/spatialvla-4b-224-pt"

# Resolve relative paths relative to BENCH_ROOT
[[ "${MODEL_PATH}" != /* ]] && MODEL_PATH="${BENCH_ROOT}/${MODEL_PATH}"
[[ -n "${LORA_PATH}" && "${LORA_PATH}" != /* ]] && LORA_PATH="${BENCH_ROOT}/${LORA_PATH}"

# ---------------------------------------------------------------------------
# Validate paths
# ---------------------------------------------------------------------------
SERVER_SCRIPT="${BENCH_ROOT}/vlaModelHome/SpatialVLA/server.py"

if [[ ! -f "${SERVER_SCRIPT}" ]]; then
    echo "ERROR: Server script not found at '${SERVER_SCRIPT}'."
    echo "  Clone SpatialVLA into: ${BENCH_ROOT}/vlaModelHome/SpatialVLA"
    echo "  Or override workspace root with: --bench-root /path/to/workspace"
    exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: Base model path not found: '${MODEL_PATH}'"
    echo "  Download the base model to: ${CKPT_BASE}/spatialvla-4b-224-pt"
    exit 1
fi

if [[ -n "${LORA_PATH}" && ! -d "${LORA_PATH}" ]]; then
    echo "ERROR: LoRA path not found: '${LORA_PATH}'"
    exit 1
fi

# ---------------------------------------------------------------------------
# Activate conda environment
# ---------------------------------------------------------------------------
set +u
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "${CONDA_PREFIX:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

# ---------------------------------------------------------------------------
# Build & run server command
# ---------------------------------------------------------------------------
CMD=(python "${SERVER_SCRIPT}" --model-path "${MODEL_PATH}" --port "${PORT}")
[[ -n "${LORA_PATH}"  ]] && CMD+=(--lora-path  "${LORA_PATH}")
[[ -n "${UNNORM_KEY}" ]] && CMD+=(--unnorm-key "${UNNORM_KEY}")

echo "============================================================"
echo "SpatialVLA Inference Server  [DyniSafety-SimplerEnv]"
echo "  Base model : ${MODEL_PATH}"
echo "  LoRA path  : ${LORA_PATH:-none}"
echo "  Unnorm key : ${UNNORM_KEY}"
echo "  Port       : ${PORT}"
echo "  Conda env  : ${CONDA_ENV_NAME}"
echo "============================================================"

exec "${CMD[@]}"
