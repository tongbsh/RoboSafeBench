#!/bin/bash
# ============================================================
# Run DyniSafety-LIBERO benchmark with UniVLA (HTTP server).
#
# Usage:
#   bash dynisafety_libero/scripts/univla/run_dynisafety_eval.sh [OPTIONS]
#
# Options:
#   --task-suite    <NAME>   safelibero_spatial | safelibero_object |
#                            safelibero_goal | safelibero_long
#                            Default: safelibero_object
#   --dynamic-level <N>      Disturbance level 0–4. Default: 2
#   --server-host   <HOST>   UniVLA server host. Default: localhost
#   --server-port   <PORT>   UniVLA server port. Default: 8020
#   --num-trials    <N>      Episodes per task. Default: 50
#   --arm-speed     <SPEED>  slow | medium | fast. Default: medium
#   --task-id       <ID>     Evaluate only this task id. Default: 0
#   --env-name      <NAME>   Conda env for benchmark. Default: openvla-oft
#   --pretrained-checkpoint PATH  Log-only hint (must match server checkpoint)
#   --action-decoder-path   PATH  Log-only hint (must match server)
#
# Prerequisites (Terminal 1 — conda activate univla):
#   bash dynisafety_libero/scripts/univla/start_univla_server.sh \\
#       --task-suite safelibero_object --port 8020
#
# Outputs:
#   dynisafety_libero/outputs/univla/logs/
#   dynisafety_libero/outputs/univla/videos/
# ============================================================

set -euo pipefail

TASK_SUITE="safelibero_spatial"
DYNAMIC_LEVEL="1"
SERVER_HOST="localhost"
SERVER_PORT=8020
NUM_TRIALS=50
ARM_SPEED="medium"
TASK_ID=0
CONDA_ENV_NAME="openvla-oft"
PRETRAINED_CHECKPOINT=""
ACTION_DECODER_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task-suite)            TASK_SUITE="$2"; shift 2 ;;
        --dynamic-level)         DYNAMIC_LEVEL="$2"; shift 2 ;;
        --server-host)           SERVER_HOST="$2"; shift 2 ;;
        --server-port)           SERVER_PORT="$2"; shift 2 ;;
        --num-trials)            NUM_TRIALS="$2"; shift 2 ;;
        --arm-speed)             ARM_SPEED="$2"; shift 2 ;;
        --task-id)               TASK_ID="$2"; shift 2 ;;
        --env-name)              CONDA_ENV_NAME="$2"; shift 2 ;;
        --pretrained-checkpoint)   PRETRAINED_CHECKPOINT="$2"; shift 2 ;;
        --action-decoder-path)   ACTION_DECODER_PATH="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,40p' "$0"
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

set +u
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "${CONDA_PREFIX:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

EXTRA_ARGS=()
if [[ -n "${PRETRAINED_CHECKPOINT}" ]]; then
    EXTRA_ARGS+=(--pretrained_checkpoint "${PRETRAINED_CHECKPOINT}")
fi
if [[ -n "${ACTION_DECODER_PATH}" ]]; then
    EXTRA_ARGS+=(--action_decoder_path "${ACTION_DECODER_PATH}")
fi

echo "============================================================"
echo "DyniSafety-LIBERO benchmark (UniVLA)"
echo "  Task suite    : ${TASK_SUITE}"
echo "  Dynamic level : ${DYNAMIC_LEVEL}"
echo "  Server        : ${SERVER_HOST}:${SERVER_PORT}"
echo "  Trials/task   : ${NUM_TRIALS}"
echo "  Arm speed     : ${ARM_SPEED}"
echo "  Task ID       : ${TASK_ID}"
echo "  Conda env     : ${CONDA_ENV_NAME}"
echo "============================================================"

cd "${PROJ_ROOT}"
exec python dynisafety_libero/scripts/univla/run_dynisafety_univla_eval.py \
    --server_host         "${SERVER_HOST}" \
    --server_port         "${SERVER_PORT}" \
    --task_suite_name     "${TASK_SUITE}" \
    --dynamic_level       "${DYNAMIC_LEVEL}" \
    --num_trials_per_task "${NUM_TRIALS}" \
    --arm_speed           "${ARM_SPEED}" \
    --task_id_override    "${TASK_ID}" \
    "${EXTRA_ARGS[@]}"
