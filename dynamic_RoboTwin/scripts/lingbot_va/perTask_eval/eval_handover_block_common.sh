#!/usr/bin/env bash
#
# Common client launcher for LingBot-VA evaluation on DyniSafety-RoboTwin.
#
# The client side (this script) runs in the ``internvla_a1`` conda env where
# RoboTwin / SAPIEN / curobo live. It connects over websocket to the server
# launched separately via ``launch_server.sh`` (which lives in the
# ``lingbot_va`` env).
#
# Positional arg: dynamic_level (0|1|2|3); env var DYNAMIC_LEVEL is also honored.
#
# Override-able env vars:
#   CONDA_ENV=internvla_a1
#   HOST=0.0.0.0
#   PORT=29056
#   TASK_CONFIG=demo_clean
#   TASK_IDX=8
#   TASK_NAME=handover_block
#   TEST_NUM=20
#   FPS=30
#   VIDEO_GUIDANCE_SCALE=5.0
#   ACTION_GUIDANCE_SCALE=1.0
#   OBSTACLE_SCALE=1.0
#   OBSTACLE_TARGET_MODE=object
#   OBSTACLE_SELECTION_MODE=random
#   L2_START_MODE=head_on
#   L2_FLIP_SIDE=0
#   L2_START_MIN_DIST=0.85

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../_common.sh
source "${SCRIPT_DIR}/../_common.sh"

CONDA_ENV="${CONDA_ENV:-internvla_a1}"
conda activate "${CONDA_ENV}"

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

DYNAMIC_LEVEL="${1:-${DYNAMIC_LEVEL:-0}}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-29056}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
TASK_IDX="${TASK_IDX:-8}"
TASK_NAME="${TASK_NAME:-handover_block}"
TEST_NUM="${TEST_NUM:-20}"
FPS="${FPS:-30}"
VIDEO_GUIDANCE_SCALE="${VIDEO_GUIDANCE_SCALE:-5.0}"
ACTION_GUIDANCE_SCALE="${ACTION_GUIDANCE_SCALE:-1.0}"
OBSTACLE_SCALE="${OBSTACLE_SCALE:-1.0}"
OBSTACLE_TARGET_MODE="${OBSTACLE_TARGET_MODE:-source_object}"
OBSTACLE_SELECTION_MODE="${OBSTACLE_SELECTION_MODE:-random}"
L2_START_MODE="${L2_START_MODE:-head_on}"
L2_FLIP_SIDE="${L2_FLIP_SIDE:-0}"
L2_START_MIN_DIST="${L2_START_MIN_DIST:-0.85}"

case "${L2_FLIP_SIDE,,}" in
    1|true|yes|on)  L2_FLIP_SIDE_FLAG="--l2_flip_side" ;;
    *)              L2_FLIP_SIDE_FLAG="--no-l2_flip_side" ;;
esac

OUTPUT_PATH="${PROJ_ROOT}/outputs/dynisafety/${TASK_CONFIG}/${TASK_NAME}/lingbot_va/level${DYNAMIC_LEVEL}"

POLICY_KWARGS=$(printf '{"host": "%s", "port": %s, "video_guidance_scale": %s, "action_guidance_scale": %s}' \
    "${HOST}" "${PORT}" "${VIDEO_GUIDANCE_SCALE}" "${ACTION_GUIDANCE_SCALE}")

echo "HOST                 = ${HOST}:${PORT}"
echo "TASK_NAME            = ${TASK_NAME}"
echo "TASK_IDX             = ${TASK_IDX}"
echo "DYNAMIC_LEVEL        = ${DYNAMIC_LEVEL}"
echo "TEST_NUM             = ${TEST_NUM}"
echo "OUTPUT_PATH          = ${OUTPUT_PATH}"

cd "${PROJ_ROOT}"
PYTHONWARNINGS=ignore::UserWarning \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python main_inference.py \
    --policy_name lingbot_va \
    --policy_kwargs "${POLICY_KWARGS}" \
    --task_config "${TASK_CONFIG}" \
    --task_idx "${TASK_IDX}" \
    --dynamic_level "${DYNAMIC_LEVEL}" \
    --test_num "${TEST_NUM}" \
    --obstacle_scale "${OBSTACLE_SCALE}" \
    --obstacle_target_mode "${OBSTACLE_TARGET_MODE}" \
    --obstacle_selection_mode "${OBSTACLE_SELECTION_MODE}" \
    --l2_start_mode "${L2_START_MODE}" \
    ${L2_FLIP_SIDE_FLAG} \
    --l2_start_min_distance "${L2_START_MIN_DIST}" \
    --fps "${FPS}" \
    --video_dir "${OUTPUT_PATH}"
