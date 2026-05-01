#!/bin/bash
# ============================================================
# Start UniVLA FastAPI inference server (DyniSafety-LIBERO client).
#
# Run in conda env ``univla`` (see vlaModelHome/a-start_help/conda_env_install).
# Install HTTP stack if needed: pip install fastapi uvicorn
#
# Usage:
#   bash dynisafety_libero/scripts/univla/start_univla_server.sh [OPTIONS]
#
# Options:
#   --pretrained-checkpoint PATH   Local HF checkpoint dir (UniVLA LIBERO finetune)
#   --action-decoder-path   PATH   Usually .../action_decoder.pt under checkpoint
#   --unnorm-key            KEY    dataset_statistics.json key (e.g. libero_object_no_noops)
#   --task-suite            NAME   safelibero_spatial|object|goal|long (default: safelibero_spatial)
#                                    Maps to download_checkpoints/univla/sft-libero/<checkpoint-dir>
#   --window-size           N      ActionDecoder window (default: 12)
#   --center-crop                 Enable if trained with random crop aug
#   --port                  PORT   Default: 8020
#   --env-name              NAME   Conda env (default: univla)
#
# Env:
#   UNIVLA_ATTN_IMPLEMENTATION=sdpa      If flash_attention_2 is unavailable
#   UNIVLA_LIBERO_WEIGHTS_BASE=DIR       Override default .../download_checkpoints/univla/sft-libero
#
# Then in benchmark env:
#   bash dynisafety_libero/scripts/univla/run_dynisafety_eval.sh ...
# ============================================================

set -euo pipefail

PORT=8020
CONDA_ENV_NAME="univla"
WINDOW_SIZE=12
CENTER_CROP=0
TASK_SUITE="safelibero_spatial"
PRETRAINED_CHECKPOINT=""
ACTION_DECODER_PATH=""
UNNORM_KEY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pretrained-checkpoint) PRETRAINED_CHECKPOINT="$2"; shift 2 ;;
        --action-decoder-path)   ACTION_DECODER_PATH="$2"; shift 2 ;;
        --unnorm-key)            UNNORM_KEY="$2"; shift 2 ;;
        --task-suite)            TASK_SUITE="$2"; shift 2 ;;
        --window-size)           WINDOW_SIZE="$2"; shift 2 ;;
        --center-crop)           CENTER_CROP=1; shift 1 ;;
        --port)                  PORT="$2"; shift 2 ;;
        --env-name)              CONDA_ENV_NAME="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,29p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# UniVLA LIBERO SFT weights (HF snapshot layout). Override with UNIVLA_LIBERO_WEIGHTS_BASE if needed.
WEIGHTS_BASE="${UNIVLA_LIBERO_WEIGHTS_BASE:-${PROJ_ROOT}/download_checkpoints/univla/sft-libero}"

if [[ -z "${PRETRAINED_CHECKPOINT}" && -n "${TASK_SUITE}" ]]; then
    case "${TASK_SUITE}" in
        safelibero_spatial)
            PRETRAINED_CHECKPOINT="${WEIGHTS_BASE}/univla-libero-spatial"
            UNNORM_KEY="${UNNORM_KEY:-libero_spatial_no_noops}"
            ;;
        safelibero_object)
            PRETRAINED_CHECKPOINT="${WEIGHTS_BASE}/univla-libero-object"
            UNNORM_KEY="${UNNORM_KEY:-libero_object_no_noops}"
            ;;
        safelibero_goal)
            PRETRAINED_CHECKPOINT="${WEIGHTS_BASE}/univla-libero-goal"
            UNNORM_KEY="${UNNORM_KEY:-libero_goal_no_noops}"
            ;;
        safelibero_long)
            PRETRAINED_CHECKPOINT="${WEIGHTS_BASE}/univla-libero-10"
            UNNORM_KEY="${UNNORM_KEY:-libero_10_no_noops}"
            ;;
        *)
            echo "ERROR: unknown --task-suite ${TASK_SUITE}"
            exit 1
            ;;
    esac
fi

if [[ -z "${PRETRAINED_CHECKPOINT}" || -z "${UNNORM_KEY}" ]]; then
    echo "ERROR: set --pretrained-checkpoint and --unnorm-key, or pass --task-suite for defaults."
    exit 1
fi

if [[ "${PRETRAINED_CHECKPOINT}" != /* ]]; then
    PRETRAINED_CHECKPOINT="${PROJ_ROOT}/${PRETRAINED_CHECKPOINT}"
fi

if [[ -z "${ACTION_DECODER_PATH}" ]]; then
    ACTION_DECODER_PATH="${PRETRAINED_CHECKPOINT}/action_decoder.pt"
fi
if [[ "${ACTION_DECODER_PATH}" != /* ]]; then
    ACTION_DECODER_PATH="${PROJ_ROOT}/${ACTION_DECODER_PATH}"
fi

DEPLOY_SCRIPT="${PROJ_ROOT}/vlaModelHome/UniVLA/experiments/robot/libero/deploy_univla_dynisafety_server.py"
if [[ ! -f "${DEPLOY_SCRIPT}" ]]; then
    echo "ERROR: deploy script not found: ${DEPLOY_SCRIPT}"
    exit 1
fi

set +u
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "${CONDA_PREFIX:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

EXTRA=(--inference_mode libero)
if [[ "${CENTER_CROP}" -eq 1 ]]; then
    EXTRA+=(--center_crop)
fi

echo "============================================================"
echo "UniVLA inference server (DyniSafety-LIBERO — inference_mode=libero)"
echo "  Checkpoint     : ${PRETRAINED_CHECKPOINT}"
echo "  Action decoder : ${ACTION_DECODER_PATH}"
echo "  unnorm_key     : ${UNNORM_KEY}"
echo "  window_size    : ${WINDOW_SIZE}"
echo "  Port           : ${PORT}"
echo "  Conda env      : ${CONDA_ENV_NAME}"
echo "============================================================"

cd "${PROJ_ROOT}/vlaModelHome/UniVLA"
exec python experiments/robot/libero/deploy_univla_dynisafety_server.py \
    --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
    --action_decoder_path "${ACTION_DECODER_PATH}" \
    --unnorm_key "${UNNORM_KEY}" \
    --window_size "${WINDOW_SIZE}" \
    --port "${PORT}" \
    "${EXTRA[@]}"
