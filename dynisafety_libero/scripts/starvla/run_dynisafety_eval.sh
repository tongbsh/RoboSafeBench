#!/bin/bash
# ============================================================
# Run DyniSafety-LIBERO benchmark with StarVLA (WebSocket client).
#
# Usage:
#   bash dynisafety_libero/scripts/starvla/run_dynisafety_eval.sh [OPTIONS]
#
# Options:
#   --task-suite      <NAME>   safelibero_spatial | safelibero_object | ...
#   --dynamic-level   <N>      0–4 (default: 2)
#   --server-host     <HOST>   Default: localhost
#   --server-port     <PORT>   Default: 10093
#   --ckpt-path       <PATH>   Same .pt path as the server (default: LIBERO-4in1 download).
#                              Client only reads RUN/config.yaml + dataset_statistics.json next
#                              to that file for denorm / chunk size — it does not load weights.
#   --unnorm-key      <KEY>    Optional dataset_statistics.json key
#   --num-trials      <N>      Episodes per task (default: 50)
#   --arm-speed       <SPEED>  slow | medium | fast (default: medium)
#   --task-id         <ID>     If set, only this task index
#   --env-name        <NAME>   Conda env for benchmark (default: openvla-oft)
#
# Prerequisites (terminal 1 — StarVLA env):
#   bash dynisafety_libero/scripts/starvla/start_starvla_server.sh --ckpt-path ...
#
# Outputs:
#   dynisafety_libero/outputs/starvla/logs/
#   dynisafety_libero/outputs/starvla/videos/
# ============================================================

set -euo pipefail

TASK_SUITE="safelibero_spatial"
DYNAMIC_LEVEL="2"
SERVER_HOST="localhost"
SERVER_PORT=10093
CKPT_PATH="download_checkpoints/starVLA/starvla-libero-4in1/checkpoints/steps_50000_pytorch_model.pt"
UNNORM_KEY=""
NUM_TRIALS=50
ARM_SPEED="medium"
TASK_ID=0
CONDA_ENV_NAME="openvla-oft"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task-suite)      TASK_SUITE="$2";      shift 2 ;;
        --dynamic-level)   DYNAMIC_LEVEL="$2";   shift 2 ;;
        --server-host)     SERVER_HOST="$2";     shift 2 ;;
        --server-port)     SERVER_PORT="$2";     shift 2 ;;
        --ckpt-path)       CKPT_PATH="$2";       shift 2 ;;
        --unnorm-key)      UNNORM_KEY="$2";      shift 2 ;;
        --num-trials)      NUM_TRIALS="$2";      shift 2 ;;
        --arm-speed)       ARM_SPEED="$2";       shift 2 ;;
        --task-id)         TASK_ID="$2";         shift 2 ;;
        --env-name)        CONDA_ENV_NAME="$2";  shift 2 ;;
        -h|--help)
            sed -n '3,40p' "$0"
            exit 0 ;;
        *)
            echo "Unknown option: $1"
            exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ "${CKPT_PATH}" != /* ]]; then
    CKPT_PATH="${PROJ_ROOT}/${CKPT_PATH}"
fi
if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "ERROR: checkpoint file not found: ${CKPT_PATH}"
    echo "Pass --ckpt-path to the same .pt the StarVLA server uses (for client-side config/stats only)."
    exit 1
fi

set +u
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "${CONDA_PREFIX:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

export STARVLA_SERVER_HOST="${SERVER_HOST}"
export STARVLA_SERVER_PORT="${SERVER_PORT}"

EXTRA=()
if [[ -n "${UNNORM_KEY}" ]]; then
    EXTRA+=(--unnorm_key "${UNNORM_KEY}")
fi

echo "============================================================"
echo "DyniSafety-LIBERO benchmark (StarVLA)"
echo "  Task suite    : ${TASK_SUITE}"
echo "  Dynamic level : ${DYNAMIC_LEVEL}"
echo "  Server        : ${SERVER_HOST}:${SERVER_PORT}"
echo "  Checkpoint    : ${CKPT_PATH}"
echo "  Trials/task   : ${NUM_TRIALS}"
echo "  Arm speed     : ${ARM_SPEED}"
echo "  Task ID       : ${TASK_ID}"
echo "  Conda env     : ${CONDA_ENV_NAME}"
echo "============================================================"

cd "${PROJ_ROOT}"
exec python dynisafety_libero/scripts/starvla/run_dynisafety_starvla_eval.py \
    --server_host         "${SERVER_HOST}" \
    --server_port         "${SERVER_PORT}" \
    --pretrained_checkpoint "${CKPT_PATH}" \
    --task_suite_name     "${TASK_SUITE}" \
    --dynamic_level       "${DYNAMIC_LEVEL}" \
    --num_trials_per_task "${NUM_TRIALS}" \
    --arm_speed           "${ARM_SPEED}" \
    --task_id_override    "${TASK_ID}" \
    "${EXTRA[@]}"
