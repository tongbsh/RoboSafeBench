#!/bin/bash
# ============================================================
# DyniSafety-SimplerEnv — UniVLA HTTP server (SimplerEnv / WidowX Bridge only)
#
# Always runs deploy_univla_dynisafety_server.py with --inference_mode bridge
# (simpler-bridge parity: cv2 resize, TF crop 0.9, no hist prompt, no LIBERO gripper postprocess).
#
# For DyniSafety-LIBERO UniVLA, use dynisafety_libero/scripts/univla/start_univla_server.sh instead.
#
# Layout (defaults):
#   SIMPLER_ROOT     …/dynamic_SimplerEnv
#   BENCH_ROOT       parent of SIMPLER_ROOT (workspace with vlaModelHome/)
#   Default ckpt     …/download_checkpoints/univla/sft-simpler-bridge (HF: qwbu/univla-7b-224-sft-simpler-bridge)
#   Override         UNIVLA_SIMPLERENV_CHECKPOINT
#   UniVLA checkout  ${UNIVLA_VLA_HOME:-${BENCH_ROOT}/vlaModelHome/UniVLA}
#
# Conda: ``univla``.  pip install fastapi uvicorn
#
# Usage:
#   bash dynamic_SimplerEnv/scripts/univla/start_univla_server.sh [OPTIONS]
#
# Options:
#   --pretrained-checkpoint PATH   HF-format checkpoint directory
#   --action-decoder-path   PATH   Usually <checkpoint>/action_decoder.pt
#   --unnorm-key            KEY    Default bridge_oxe with --task-suite simpler_bridge
#   --task-suite            NAME   simpler_bridge (default) | simpler-bridge | bridge
#   --window-size           N      ActionDecoder window (default: 10)
#   --port                  PORT   Default: 8020
#   --env-name              NAME   Conda env (default: univla)
#
# Env:
#   UNIVLA_ATTN_IMPLEMENTATION=sdpa
#   UNIVLA_SIMPLERENV_CHECKPOINT=DIR
#   UNIVLA_VLA_HOME=DIR
# ============================================================

set -euo pipefail

PORT=8020
CONDA_ENV_NAME="univla"
WINDOW_SIZE=10
TASK_SUITE="simpler_bridge"
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
        --port)                  PORT="$2"; shift 2 ;;
        --env-name)              CONDA_ENV_NAME="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,36p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIMPLER_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${SIMPLER_ROOT}/.." && pwd)"
UNIVLA_HOME="${UNIVLA_VLA_HOME:-${BENCH_ROOT}/vlaModelHome/UniVLA}"

if [[ -n "${UNIVLA_SIMPLERENV_CHECKPOINT:-}" ]]; then
    _DEFAULT_SIMPLER_CKPT="${UNIVLA_SIMPLERENV_CHECKPOINT}"
elif [[ -d "${BENCH_ROOT}/download_checkpoints/univla/sft-simpler-bridge" ]]; then
    _DEFAULT_SIMPLER_CKPT="${BENCH_ROOT}/download_checkpoints/univla/sft-simpler-bridge"
else
    _DEFAULT_SIMPLER_CKPT="download_checkpoints/univla/sft-simpler-bridge"
fi

if [[ -z "${PRETRAINED_CHECKPOINT}" && -n "${TASK_SUITE}" ]]; then
    case "${TASK_SUITE}" in
        simpler_bridge|simpler-bridge|bridge)
            PRETRAINED_CHECKPOINT="${_DEFAULT_SIMPLER_CKPT}"
            UNNORM_KEY="${UNNORM_KEY:-bridge_oxe}"
            ;;
        *)
            echo "ERROR: unknown --task-suite ${TASK_SUITE}"
            echo "  Use: simpler_bridge (or simpler-bridge | bridge), or set --pretrained-checkpoint and --unnorm-key."
            exit 1
            ;;
    esac
fi

if [[ -z "${PRETRAINED_CHECKPOINT}" || -z "${UNNORM_KEY}" ]]; then
    echo "ERROR: set --pretrained-checkpoint and --unnorm-key, or pass --task-suite for defaults."
    exit 1
fi

if [[ "${PRETRAINED_CHECKPOINT}" != /* ]]; then
    PRETRAINED_CHECKPOINT="${SIMPLER_ROOT}/${PRETRAINED_CHECKPOINT}"
fi

if [[ -z "${ACTION_DECODER_PATH}" ]]; then
    ACTION_DECODER_PATH="${PRETRAINED_CHECKPOINT}/action_decoder.pt"
fi
if [[ "${ACTION_DECODER_PATH}" != /* ]]; then
    ACTION_DECODER_PATH="${SIMPLER_ROOT}/${ACTION_DECODER_PATH}"
fi

DEPLOY_SCRIPT="${UNIVLA_HOME}/experiments/robot/libero/deploy_univla_dynisafety_server.py"
if [[ ! -f "${DEPLOY_SCRIPT}" ]]; then
    echo "ERROR: deploy script not found: ${DEPLOY_SCRIPT}"
    echo "  Set UNIVLA_VLA_HOME to your UniVLA repo root."
    exit 1
fi

set +u
# shellcheck disable=SC1091
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "${CONDA_PREFIX:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

echo "============================================================"
echo "UniVLA server (SimplerEnv Bridge — inference_mode=bridge)"
echo "  SIMPLER_ROOT   : ${SIMPLER_ROOT}"
echo "  UniVLA home    : ${UNIVLA_HOME}"
echo "  Checkpoint     : ${PRETRAINED_CHECKPOINT}"
echo "  Action decoder : ${ACTION_DECODER_PATH}"
echo "  unnorm_key     : ${UNNORM_KEY}"
echo "  window_size    : ${WINDOW_SIZE}"
echo "  Port           : ${PORT}"
echo "  Conda env      : ${CONDA_ENV_NAME}"
echo "============================================================"

cd "${UNIVLA_HOME}"
exec python experiments/robot/libero/deploy_univla_dynisafety_server.py \
    --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
    --action_decoder_path "${ACTION_DECODER_PATH}" \
    --unnorm_key "${UNNORM_KEY}" \
    --window_size "${WINDOW_SIZE}" \
    --port "${PORT}" \
    --inference_mode bridge
