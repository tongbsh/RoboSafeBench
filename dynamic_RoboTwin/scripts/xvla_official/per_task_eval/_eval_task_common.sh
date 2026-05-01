#!/usr/bin/env bash
#
# Shared X-VLA official-protocol launcher for DyniSafety-RoboTwin.
#
# Expected to be called by per-task wrappers that export TASK_NAME/TASK_IDX.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../_common.sh
source "${SCRIPT_DIR}/../../_common.sh"
# shellcheck source=../../task_registry.sh
source "${SCRIPT_DIR}/../../task_registry.sh"

CONDA_ENV="${CONDA_ENV:-internvla_a1}"
conda activate "${CONDA_ENV}"

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

DYNAMIC_LEVEL="${1:-${DYNAMIC_LEVEL:-0}}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8010}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
TASK_NAME="${TASK_NAME:?TASK_NAME must be exported by the wrapper script}"
dynisafety_export_task_env "${TASK_NAME}"
TASK_IDX="${TASK_IDX:?TASK_IDX must be derivable from TASK_NAME or exported by the wrapper script}"
TEST_NUM="${TEST_NUM:-20}"
FPS="${FPS:-30}"
OBSTACLE_SCALE="${OBSTACLE_SCALE:-1.0}"
OBSTACLE_TARGET_MODE="${OBSTACLE_TARGET_MODE:?OBSTACLE_TARGET_MODE must be derivable from TASK_NAME or exported by the wrapper script}"
OBSTACLE_SELECTION_MODE="${OBSTACLE_SELECTION_MODE:-random}"
L2_START_MODE="${L2_START_MODE:-head_on}"
L2_FLIP_SIDE="${L2_FLIP_SIDE:-0}"
L2_START_MIN_DIST="${L2_START_MIN_DIST:-0.85}"

XVLA_MODEL_PATH="${XVLA_MODEL_PATH:-${WORKSPACE_ROOT}/download_checkpoints/xVLA/X-VLA-RoboTwin2}"
XVLA_DOMAIN_ID="${XVLA_DOMAIN_ID:-6}"
XVLA_STEPS="${XVLA_STEPS:-10}"
XVLA_REPLAN_LIMIT="${XVLA_REPLAN_LIMIT:-10}"
XVLA_REQUEST_TIMEOUT="${XVLA_REQUEST_TIMEOUT:-180}"
XVLA_SERVER_TIMEOUT="${XVLA_SERVER_TIMEOUT:-300}"
XVLA_GRIPPER_THRESHOLD="${XVLA_GRIPPER_THRESHOLD:-0.7}"
# Shared benchmark control for all four model runners. Keep the old
# XVLA-specific variable name as a fallback for compatibility.
EXPERT_CHECK_TIMEOUT_S="${EXPERT_CHECK_TIMEOUT_S:-${XVLA_EXPERT_CHECK_TIMEOUT_S:-3600}}"
XVLA_EXPERT_CHECK_MAX_ATTEMPTS="${XVLA_EXPERT_CHECK_MAX_ATTEMPTS:-400}"
XVLA_EXPERT_CHECK_LOG_EVERY="${XVLA_EXPERT_CHECK_LOG_EVERY:-10}"

case "${L2_FLIP_SIDE,,}" in
    1|true|yes|on)  L2_FLIP_SIDE_FLAG="--l2_flip_side" ;;
    *)              L2_FLIP_SIDE_FLAG="--no-l2_flip_side" ;;
esac

OUTPUT_PATH="${PROJ_ROOT}/outputs/dynisafety/${TASK_CONFIG}/${TASK_NAME}/xvla_official/level${DYNAMIC_LEVEL}"

POLICY_KWARGS="$(
MODEL_PATH="${XVLA_MODEL_PATH}" \
HOST="${HOST}" \
PORT="${PORT}" \
DOMAIN_ID="${XVLA_DOMAIN_ID}" \
STEPS="${XVLA_STEPS}" \
REPLAN_LIMIT="${XVLA_REPLAN_LIMIT}" \
REQUEST_TIMEOUT="${XVLA_REQUEST_TIMEOUT}" \
SERVER_TIMEOUT="${XVLA_SERVER_TIMEOUT}" \
GRIPPER_THRESHOLD="${XVLA_GRIPPER_THRESHOLD}" \
python - <<'PY'
import json
import os

payload = {
    "model_path": os.environ["MODEL_PATH"],
    "host": os.environ["HOST"],
    "port": int(os.environ["PORT"]),
    "domain_id": int(os.environ["DOMAIN_ID"]),
    "steps": int(os.environ["STEPS"]),
    "replan_limit": int(os.environ["REPLAN_LIMIT"]),
    "request_timeout": float(os.environ["REQUEST_TIMEOUT"]),
    "server_timeout": float(os.environ["SERVER_TIMEOUT"]),
    "gripper_threshold": float(os.environ["GRIPPER_THRESHOLD"]),
}
print(json.dumps(payload))
PY
)"

echo "HOST                 = ${HOST}:${PORT}"
echo "XVLA_MODEL_PATH      = ${XVLA_MODEL_PATH}"
echo "XVLA_DOMAIN_ID       = ${XVLA_DOMAIN_ID}"
echo "XVLA_REPLAN_LIMIT    = ${XVLA_REPLAN_LIMIT}"
echo "EC_TIMEOUT_S         = ${EXPERT_CHECK_TIMEOUT_S}"
echo "XVLA_EC_MAX_ATTEMPTS = ${XVLA_EXPERT_CHECK_MAX_ATTEMPTS}"
echo "TASK_NAME            = ${TASK_NAME}"
echo "TASK_IDX             = ${TASK_IDX}"
echo "DYNAMIC_LEVEL        = ${DYNAMIC_LEVEL}"
echo "TEST_NUM             = ${TEST_NUM}"
echo "OUTPUT_PATH          = ${OUTPUT_PATH}"

cd "${PROJ_ROOT}"
PYTHONWARNINGS=ignore::UserWarning \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python xvla_official_inference.py \
    --policy_kwargs "${POLICY_KWARGS}" \
    --task_config "${TASK_CONFIG}" \
    --task_idx "${TASK_IDX}" \
    --dynamic_level "${DYNAMIC_LEVEL}" \
    --test_num "${TEST_NUM}" \
    --expert_check_timeout_s "${EXPERT_CHECK_TIMEOUT_S}" \
    --expert_check_max_attempts "${XVLA_EXPERT_CHECK_MAX_ATTEMPTS}" \
    --expert_check_log_every "${XVLA_EXPERT_CHECK_LOG_EVERY}" \
    --obstacle_scale "${OBSTACLE_SCALE}" \
    --obstacle_target_mode "${OBSTACLE_TARGET_MODE}" \
    --obstacle_selection_mode "${OBSTACLE_SELECTION_MODE}" \
    --l2_start_mode "${L2_START_MODE}" \
    ${L2_FLIP_SIDE_FLAG} \
    --l2_start_min_distance "${L2_START_MIN_DIST}" \
    --fps "${FPS}" \
    --video_dir "${OUTPUT_PATH}"
