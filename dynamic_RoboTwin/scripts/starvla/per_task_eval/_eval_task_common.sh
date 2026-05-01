#!/usr/bin/env bash
#
# Shared StarVLA evaluation launcher for DyniSafety-RoboTwin.
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
PORT="${PORT:-10093}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
TASK_NAME="${TASK_NAME:?TASK_NAME must be exported by the wrapper script}"
dynisafety_export_task_env "${TASK_NAME}"
TASK_IDX="${TASK_IDX:?TASK_IDX must be derivable from TASK_NAME or exported by the wrapper script}"
TEST_NUM="${TEST_NUM:-20}"
FPS="${FPS:-30}"
EXPERT_CHECK_TIMEOUT_S="${EXPERT_CHECK_TIMEOUT_S:-3600}"
OBSTACLE_SCALE="${OBSTACLE_SCALE:-1.0}"
OBSTACLE_TARGET_MODE="${OBSTACLE_TARGET_MODE:?OBSTACLE_TARGET_MODE must be derivable from TASK_NAME or exported by the wrapper script}"
OBSTACLE_SELECTION_MODE="${OBSTACLE_SELECTION_MODE:-random}"
L2_START_MODE="${L2_START_MODE:-head_on}"
L2_FLIP_SIDE="${L2_FLIP_SIDE:-0}"
L2_START_MIN_DIST="${L2_START_MIN_DIST:-0.85}"

STARVLA_CKPT="${STARVLA_CKPT:-${WORKSPACE_ROOT}/download_checkpoints/starVLA/starvla-robotwin2-all/checkpoints/steps_140000_pytorch_model.pt}"
STARVLA_POLICY_SETUP="${STARVLA_POLICY_SETUP:-robotwin}"
STARVLA_UNNORM_KEY="${STARVLA_UNNORM_KEY:-}"
STARVLA_ACTION_MODE="${STARVLA_ACTION_MODE:-abs}"
STARVLA_NORMALIZATION_MODE="${STARVLA_NORMALIZATION_MODE:-min_max}"
STARVLA_USE_DDIM="${STARVLA_USE_DDIM:-true}"
STARVLA_NUM_DDIM_STEPS="${STARVLA_NUM_DDIM_STEPS:-10}"
STARVLA_ACTION_ENSEMBLE="${STARVLA_ACTION_ENSEMBLE:-false}"
STARVLA_ACTION_ENSEMBLE_HORIZON="${STARVLA_ACTION_ENSEMBLE_HORIZON:-3}"
STARVLA_ADAPTIVE_ENSEMBLE_ALPHA="${STARVLA_ADAPTIVE_ENSEMBLE_ALPHA:-0.1}"

case "${L2_FLIP_SIDE,,}" in
    1|true|yes|on)  L2_FLIP_SIDE_FLAG="--l2_flip_side" ;;
    *)              L2_FLIP_SIDE_FLAG="--no-l2_flip_side" ;;
esac

OUTPUT_PATH="${PROJ_ROOT}/outputs/dynisafety/${TASK_CONFIG}/${TASK_NAME}/starvla/level${DYNAMIC_LEVEL}"

POLICY_KWARGS="$(
CKPT_PATH="${STARVLA_CKPT}" \
HOST="${HOST}" \
PORT="${PORT}" \
POLICY_SETUP="${STARVLA_POLICY_SETUP}" \
UNNORM_KEY="${STARVLA_UNNORM_KEY}" \
ACTION_MODE="${STARVLA_ACTION_MODE}" \
NORMALIZATION_MODE="${STARVLA_NORMALIZATION_MODE}" \
USE_DDIM="${STARVLA_USE_DDIM}" \
NUM_DDIM_STEPS="${STARVLA_NUM_DDIM_STEPS}" \
ACTION_ENSEMBLE="${STARVLA_ACTION_ENSEMBLE}" \
ACTION_ENSEMBLE_HORIZON="${STARVLA_ACTION_ENSEMBLE_HORIZON}" \
ADAPTIVE_ENSEMBLE_ALPHA="${STARVLA_ADAPTIVE_ENSEMBLE_ALPHA}" \
python - <<'PY'
import json
import os

def parse_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

payload = {
    "ckpt_path": os.environ["CKPT_PATH"],
    "host": os.environ["HOST"],
    "port": int(os.environ["PORT"]),
    "policy_setup": os.environ["POLICY_SETUP"],
    "action_mode": os.environ["ACTION_MODE"],
    "normalization_mode": os.environ["NORMALIZATION_MODE"],
    "use_ddim": parse_bool("USE_DDIM", True),
    "num_ddim_steps": int(os.environ["NUM_DDIM_STEPS"]),
    "action_ensemble": parse_bool("ACTION_ENSEMBLE", False),
    "action_ensemble_horizon": int(os.environ["ACTION_ENSEMBLE_HORIZON"]),
    "adaptive_ensemble_alpha": float(os.environ["ADAPTIVE_ENSEMBLE_ALPHA"]),
}
unnorm_key = os.environ.get("UNNORM_KEY", "").strip()
if unnorm_key:
    payload["unnorm_key"] = unnorm_key
print(json.dumps(payload))
PY
)"

echo "HOST                 = ${HOST}:${PORT}"
echo "STARVLA_CKPT         = ${STARVLA_CKPT}"
echo "TASK_NAME            = ${TASK_NAME}"
echo "TASK_IDX             = ${TASK_IDX}"
echo "DYNAMIC_LEVEL        = ${DYNAMIC_LEVEL}"
echo "TEST_NUM             = ${TEST_NUM}"
echo "EC_TIMEOUT_S         = ${EXPERT_CHECK_TIMEOUT_S}"
echo "OUTPUT_PATH          = ${OUTPUT_PATH}"

cd "${PROJ_ROOT}"
PYTHONWARNINGS=ignore::UserWarning \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python starvla_inference.py \
    --policy_kwargs "${POLICY_KWARGS}" \
    --task_config "${TASK_CONFIG}" \
    --task_idx "${TASK_IDX}" \
    --dynamic_level "${DYNAMIC_LEVEL}" \
    --test_num "${TEST_NUM}" \
    --expert_check_timeout_s "${EXPERT_CHECK_TIMEOUT_S}" \
    --obstacle_scale "${OBSTACLE_SCALE}" \
    --obstacle_target_mode "${OBSTACLE_TARGET_MODE}" \
    --obstacle_selection_mode "${OBSTACLE_SELECTION_MODE}" \
    --l2_start_mode "${L2_START_MODE}" \
    ${L2_FLIP_SIDE_FLAG} \
    --l2_start_min_distance "${L2_START_MIN_DIST}" \
    --fps "${FPS}" \
    --video_dir "${OUTPUT_PATH}"
