#!/usr/bin/env bash
#
# Shared InternVLA-A1 evaluation launcher for DyniSafety-RoboTwin.
#
# Expected to be called with TASK_NAME exported or provided by a wrapper.

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

echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

DYNAMIC_LEVEL="${1:-${DYNAMIC_LEVEL:-0}}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
TASK_NAME="${TASK_NAME:?TASK_NAME must be exported by the wrapper script}"
dynisafety_export_task_env "${TASK_NAME}"
TASK_IDX="${TASK_IDX:?TASK_IDX must be derivable from TASK_NAME or exported by the wrapper script}"
TEST_NUM="${TEST_NUM:-20}"
FPS="${FPS:-30}"
EXPERT_CHECK_TIMEOUT_S="${EXPERT_CHECK_TIMEOUT_S:-3600}"
OBSTACLE_SCALE="${OBSTACLE_SCALE:-1.0}"
OBSTACLE_TARGET_MODE="${OBSTACLE_TARGET_MODE:?OBSTACLE_TARGET_MODE must be derivable from TASK_NAME or exported by the wrapper script}"
OBSTACLE_SELECTION_MODE="${OBSTACLE_SELECTION_MODE:-trajectory_order}"   # random | trajectory_order
L2_START_MODE="${L2_START_MODE:-head_on}"              # head_on | perpendicular
L2_FLIP_SIDE="${L2_FLIP_SIDE:-0}"                      # 1/true to enable; only used when L2_START_MODE=perpendicular
L2_START_MIN_DIST="${L2_START_MIN_DIST:-0.85}"         # min xy travel distance, keeps start outside the table

DEFAULT_CKPT_REPO="InternRobotics/InternVLA-A1-3B-RoboTwin"
DEFAULT_CKPT_DIR="${WORKSPACE_ROOT}/download_checkpoints/InternVLA/InternVLA-A1-3B-RoboTwin"
LEGACY_CKPT_DIR="${WORKSPACE_ROOT}/InternVLA-A1/download_checkpoints/InternVLA/InternVLA-A1-3B-RoboTwin"

if [[ -z "${PRETRAINED_CKPT:-}" ]]; then
    if [[ -d "${DEFAULT_CKPT_DIR}" ]]; then
        PRETRAINED_CKPT="${DEFAULT_CKPT_DIR}"
    elif [[ -d "${LEGACY_CKPT_DIR}" ]]; then
        PRETRAINED_CKPT="${LEGACY_CKPT_DIR}"
    else
        echo "[WARN] Local checkpoint dir not found."
        echo "[INFO] Tried: ${DEFAULT_CKPT_DIR}"
        echo "[INFO] Tried: ${LEGACY_CKPT_DIR}"
        echo "[INFO] Falling back to HuggingFace repo id: ${DEFAULT_CKPT_REPO}"
        PRETRAINED_CKPT="${DEFAULT_CKPT_REPO}"
    fi
fi

case "${L2_FLIP_SIDE,,}" in
    1|true|yes|on)  L2_FLIP_SIDE_FLAG="--args.l2_flip_side" ;;
    *)              L2_FLIP_SIDE_FLAG="--args.no-l2_flip_side" ;;
esac

BASE_OUTPUT_PATH="${PROJ_ROOT}/outputs/dynisafety"
OUTPUT_PATH="${BASE_OUTPUT_PATH}/${TASK_CONFIG}/${TASK_NAME}/internvla_a1/level${DYNAMIC_LEVEL}"

echo "PRETRAINED_CKPT      = ${PRETRAINED_CKPT}"
echo "TASK_NAME            = ${TASK_NAME}"
echo "TASK_IDX             = ${TASK_IDX}"
echo "DYNAMIC_LEVEL        = ${DYNAMIC_LEVEL}"
echo "TEST_NUM             = ${TEST_NUM}"
echo "EC_TIMEOUT_S         = ${EXPERT_CHECK_TIMEOUT_S}"
echo "OBSTACLE_SCALE       = ${OBSTACLE_SCALE}"
echo "OBSTACLE_TARGET_MODE = ${OBSTACLE_TARGET_MODE}"
echo "OBSTACLE_SELECT_MODE = ${OBSTACLE_SELECTION_MODE}"
echo "L2_START_MODE        = ${L2_START_MODE}"
echo "L2_FLIP_SIDE         = ${L2_FLIP_SIDE}"
echo "L2_START_MIN_DIST    = ${L2_START_MIN_DIST}"
echo "OUTPUT_PATH          = ${OUTPUT_PATH}"

cd "${ROBOTWIN_ROOT}"
python ../internvla_a1_inference.py \
    --args.ckpt_path "${PRETRAINED_CKPT}" \
    --args.video_dir "${OUTPUT_PATH}" \
    --args.task_config "${TASK_CONFIG}" \
    --args.task_idx "${TASK_IDX}" \
    --args.prompt_mode official \
    --args.dynamic_level "${DYNAMIC_LEVEL}" \
    --args.test_num "${TEST_NUM}" \
    --args.expert_check_timeout_s "${EXPERT_CHECK_TIMEOUT_S}" \
    --args.obstacle_scale "${OBSTACLE_SCALE}" \
    --args.obstacle_target_mode "${OBSTACLE_TARGET_MODE}" \
    --args.obstacle_selection_mode "${OBSTACLE_SELECTION_MODE}" \
    --args.l2_start_mode "${L2_START_MODE}" \
    ${L2_FLIP_SIDE_FLAG} \
    --args.l2_start_min_distance "${L2_START_MIN_DIST}" \
    --args.fps "${FPS}"
