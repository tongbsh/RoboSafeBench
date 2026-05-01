#!/usr/bin/env bash

set -euo pipefail

###############################################################################
################################# ENV config ##################################

if [[ -n "${_CONDA_ROOT:-}" ]]; then
    CONDA_ROOT="${_CONDA_ROOT}"
elif [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    CONDA_ROOT="$("${CONDA_EXE}" info --base)"
elif command -v conda >/dev/null 2>&1; then
    CONDA_ROOT="$(conda info --base)"
else
    echo "[ERROR] Cannot determine conda root."
    echo "[ERROR] Please ensure 'conda' is available in PATH or export _CONDA_ROOT."
    exit 1
fi

if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    echo "[ERROR] conda.sh not found under: ${CONDA_ROOT}"
    exit 1
fi

CONDA_ENV="${CONDA_ENV:-internvla_a1}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

###############################################################################

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-6379}"
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

###############################################################################
############################## EVAL config ####################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJ_ROOT}/.." && pwd)"
echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"
echo "WORKSPACE_ROOT = ${WORKSPACE_ROOT}"

cd "${PROJ_ROOT}"

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

DYNAMIC_LEVEL="${1:-${DYNAMIC_LEVEL:-0}}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
TASK_IDX="${TASK_IDX:-31}"
TASK_NAME="${TASK_NAME:-place_mouse_pad}"
TEST_NUM="${TEST_NUM:-20}"
FPS="${FPS:-30}"
OBSTACLE_SCALE="${OBSTACLE_SCALE:-1.0}"
OBSTACLE_TARGET_MODE="${OBSTACLE_TARGET_MODE:-source_object}"  # place | object | source_object
OBSTACLE_SELECTION_MODE="${OBSTACLE_SELECTION_MODE:-trajectory_order}"   # random | trajectory_order
L2_START_MODE="${L2_START_MODE:-head_on}"              # head_on | perpendicular
L2_FLIP_SIDE="${L2_FLIP_SIDE:-0}"                      # 1/true to enable; only used when L2_START_MODE=perpendicular
L2_START_MIN_DIST="${L2_START_MIN_DIST:-0.85}"         # min xy travel distance, keeps start outside the table

# tyro convention for bool flags: presence = True, --no-<name> = False.
case "${L2_FLIP_SIDE,,}" in
    1|true|yes|on)  L2_FLIP_SIDE_FLAG="--args.l2_flip_side" ;;
    *)             L2_FLIP_SIDE_FLAG="--args.no-l2_flip_side" ;;
esac

BASE_OUTPUT_PATH="${PROJ_ROOT}/outputs/dynisafety"
OUTPUT_PATH="${BASE_OUTPUT_PATH}/${TASK_CONFIG}/${TASK_NAME}/internvla_a1/level${DYNAMIC_LEVEL}"

echo "PRETRAINED_CKPT      = ${PRETRAINED_CKPT}"
echo "TASK_NAME            = ${TASK_NAME}"
echo "TASK_IDX             = ${TASK_IDX}"
echo "DYNAMIC_LEVEL        = ${DYNAMIC_LEVEL}"
echo "TEST_NUM             = ${TEST_NUM}"
echo "OBSTACLE_SCALE       = ${OBSTACLE_SCALE}"
echo "OBSTACLE_TARGET_MODE = ${OBSTACLE_TARGET_MODE}"
echo "OBSTACLE_SELECT_MODE = ${OBSTACLE_SELECTION_MODE}"
echo "L2_START_MODE        = ${L2_START_MODE}"
echo "L2_FLIP_SIDE         = ${L2_FLIP_SIDE}"
echo "L2_START_MIN_DIST    = ${L2_START_MIN_DIST}"
echo "OUTPUT_PATH          = ${OUTPUT_PATH}"

cd "${PROJ_ROOT}/RoboTwin"
python ../inference.py \
    --args.ckpt_path "${PRETRAINED_CKPT}" \
    --args.video_dir "${OUTPUT_PATH}" \
    --args.task_config "${TASK_CONFIG}" \
    --args.task_idx "${TASK_IDX}" \
    --args.dynamic_level "${DYNAMIC_LEVEL}" \
    --args.test_num "${TEST_NUM}" \
    --args.obstacle_scale "${OBSTACLE_SCALE}" \
    --args.obstacle_target_mode "${OBSTACLE_TARGET_MODE}" \
    --args.obstacle_selection_mode "${OBSTACLE_SELECTION_MODE}" \
    --args.l2_start_mode "${L2_START_MODE}" \
    ${L2_FLIP_SIDE_FLAG} \
    --args.l2_start_min_distance "${L2_START_MIN_DIST}" \
    --args.fps "${FPS}"
