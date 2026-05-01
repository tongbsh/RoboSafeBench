#!/bin/bash
# ============================================================
# Run DyniSafety-LIBERO benchmark with Isaac-GR00T (ZeroMQ client).
#
# Usage:
#   bash dynisafety_libero/scripts/gr00t/run_dynisafety_eval.sh [OPTIONS]
#
# Options:
#   --task-suite      <NAME>   safelibero_spatial | safelibero_object | ...
#   --dynamic-level   <N>      0–4 (default: 2)
#   --server-host     <HOST>   Default: localhost
#   --server-port     <PORT>   Default: 5555
#   --model-path      <PATH>   Logged on client only (optional); weights load on server
#   --num-trials      <N>      Episodes per task (default: 50)
#   --arm-speed       <SPEED>  slow | medium | fast (default: medium)
#   --task-id         <ID>     If set, only this task index
#   --env-name        <NAME>   Conda env for benchmark (default: openvla-oft)
#
# Prerequisites (terminal 1 — GR00T env):
#   bash dynisafety_libero/scripts/gr00t/start_gr00t_server.sh --model-path ...
#
# Outputs:
#   dynisafety_libero/outputs/gr00t/logs/
#   dynisafety_libero/outputs/gr00t/videos/
# ============================================================

set -euo pipefail

TASK_SUITE="safelibero_spatial"
DYNAMIC_LEVEL="2"
SERVER_HOST="localhost"
SERVER_PORT=5555
MODEL_PATH=""
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
        --model-path)      MODEL_PATH="$2";      shift 2 ;;
        --num-trials)      NUM_TRIALS="$2";      shift 2 ;;
        --arm-speed)       ARM_SPEED="$2";       shift 2 ;;
        --task-id)         TASK_ID="$2";         shift 2 ;;
        --env-name)        CONDA_ENV_NAME="$2";  shift 2 ;;
        -h|--help)
            sed -n '3,36p' "$0"
            exit 0 ;;
        *)
            echo "Unknown option: $1"
            exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

set +u
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "${CONDA_PREFIX:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

export GR00T_SERVER_HOST="${SERVER_HOST}"
export GR00T_SERVER_PORT="${SERVER_PORT}"

EXTRA=()
if [[ -n "${MODEL_PATH}" ]]; then
    if [[ "${MODEL_PATH}" != /* ]]; then
        MODEL_PATH="${PROJ_ROOT}/${MODEL_PATH}"
    fi
    EXTRA+=(--pretrained_checkpoint "${MODEL_PATH}")
fi

echo "============================================================"
echo "DyniSafety-LIBERO benchmark (Isaac-GR00T)"
echo "  Task suite    : ${TASK_SUITE}"
echo "  Dynamic level : ${DYNAMIC_LEVEL}"
echo "  Server        : ${SERVER_HOST}:${SERVER_PORT}"
echo "  Trials/task   : ${NUM_TRIALS}"
echo "  Arm speed     : ${ARM_SPEED}"
echo "  Task ID       : ${TASK_ID}"
echo "  Conda env     : ${CONDA_ENV_NAME}"
echo "============================================================"

cd "${PROJ_ROOT}"
exec python dynisafety_libero/scripts/gr00t/run_dynisafety_gr00t_eval.py \
    --server_host         "${SERVER_HOST}" \
    --server_port         "${SERVER_PORT}" \
    --task_suite_name     "${TASK_SUITE}" \
    --dynamic_level       "${DYNAMIC_LEVEL}" \
    --num_trials_per_task "${NUM_TRIALS}" \
    --arm_speed           "${ARM_SPEED}" \
    --task_id_override    "${TASK_ID}" \
    "${EXTRA[@]}"
