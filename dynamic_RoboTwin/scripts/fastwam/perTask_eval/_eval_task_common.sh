#!/usr/bin/env bash
#
# Shared FastWAM evaluation launcher for DyniSafety-RoboTwin.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../_common.sh
source "${SCRIPT_DIR}/../../_common.sh"
# shellcheck source=../../task_registry.sh
source "${SCRIPT_DIR}/../../task_registry.sh"

# Must contain both FastWAM model deps and RoboTwin/SAPIEN runtime deps.
# `internvla_a1` is the RoboTwin-ready env used by the other local/client
# runners, and FastWAM is installed there for this adapter.
CONDA_ENV="${CONDA_ENV:-internvla_a1}"
conda activate "${CONDA_ENV}"

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

DYNAMIC_LEVEL="${1:-${DYNAMIC_LEVEL:-0}}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
TASK_NAME="${TASK_NAME:?TASK_NAME must be exported by the wrapper script}"
dynisafety_export_task_env "${TASK_NAME}"
TASK_IDX="${TASK_IDX:?TASK_IDX must be derivable from TASK_NAME or exported by the wrapper script}"
TEST_NUM="${TEST_NUM:-20}"
FPS="${FPS:-30}"
SEED="${SEED:-0}"
EXPERT_CHECK_TIMEOUT_S="${EXPERT_CHECK_TIMEOUT_S:-3600}"
OBSTACLE_SCALE="${OBSTACLE_SCALE:-1.0}"
OBSTACLE_TARGET_MODE="${OBSTACLE_TARGET_MODE:?OBSTACLE_TARGET_MODE must be derivable from TASK_NAME or exported by the wrapper script}"
OBSTACLE_SELECTION_MODE="${OBSTACLE_SELECTION_MODE:-random}"
L2_START_MODE="${L2_START_MODE:-head_on}"
L2_FLIP_SIDE="${L2_FLIP_SIDE:-0}"
L2_START_MIN_DIST="${L2_START_MIN_DIST:-0.85}"

FASTWAM_ROOT="${FASTWAM_ROOT:-${WORKSPACE_ROOT}/vlaModelHome/FastWAM}"
FASTWAM_MODEL_BASE_PATH="${FASTWAM_MODEL_BASE_PATH:-${FASTWAM_ROOT}/checkpoints}"
FASTWAM_CKPT="${FASTWAM_CKPT:-${WORKSPACE_ROOT}/download_checkpoints/fastwam/robotwin/robotwin_uncond_3cam_384.pt}"
FASTWAM_DATASET_STATS="${FASTWAM_DATASET_STATS:-${WORKSPACE_ROOT}/download_checkpoints/fastwam/robotwin/robotwin_uncond_3cam_384_dataset_stats.json}"
FASTWAM_SIM_CFG_PATH="${FASTWAM_SIM_CFG_PATH:-${FASTWAM_ROOT}/configs/sim_robotwin.yaml}"
FASTWAM_SIM_TASK="${FASTWAM_SIM_TASK:-robotwin_uncond_3cam_384_1e-4}"
FASTWAM_MIXED_PRECISION="${FASTWAM_MIXED_PRECISION:-bf16}"
FASTWAM_DEVICE="${FASTWAM_DEVICE:-cuda}"
FASTWAM_ACTION_HORIZON="${FASTWAM_ACTION_HORIZON:-}"
FASTWAM_REPLAN_STEPS="${FASTWAM_REPLAN_STEPS:-24}"
FASTWAM_NUM_INFERENCE_STEPS="${FASTWAM_NUM_INFERENCE_STEPS:-10}"
FASTWAM_SIGMA_SHIFT="${FASTWAM_SIGMA_SHIFT:-}"
FASTWAM_TEXT_CFG_SCALE="${FASTWAM_TEXT_CFG_SCALE:-1.0}"
FASTWAM_NEGATIVE_PROMPT="${FASTWAM_NEGATIVE_PROMPT:-}"
FASTWAM_RAND_DEVICE="${FASTWAM_RAND_DEVICE:-cpu}"
FASTWAM_TILED="${FASTWAM_TILED:-false}"
FASTWAM_TIMING_ENABLED="${FASTWAM_TIMING_ENABLED:-false}"

case "${L2_FLIP_SIDE,,}" in
    1|true|yes|on)  L2_FLIP_SIDE_FLAG="--l2_flip_side" ;;
    *)              L2_FLIP_SIDE_FLAG="--no-l2_flip_side" ;;
esac

if [[ ! -d "${FASTWAM_ROOT}" ]]; then
    echo "[ERROR] FASTWAM_ROOT not found: ${FASTWAM_ROOT}" >&2
    exit 1
fi
if [[ ! -d "${FASTWAM_MODEL_BASE_PATH}" ]]; then
    echo "[ERROR] FASTWAM_MODEL_BASE_PATH not found: ${FASTWAM_MODEL_BASE_PATH}" >&2
    echo "[INFO]  Set FASTWAM_MODEL_BASE_PATH to the directory containing Wan-AI/ and DiffSynth-Studio/." >&2
    exit 1
fi
if [[ ! -f "${FASTWAM_CKPT}" ]]; then
    echo "[ERROR] FASTWAM_CKPT not found: ${FASTWAM_CKPT}" >&2
    echo "[INFO]  Set FASTWAM_CKPT to robotwin_uncond_3cam_384.pt." >&2
    exit 1
fi
if [[ ! -f "${FASTWAM_DATASET_STATS}" ]]; then
    echo "[ERROR] FASTWAM_DATASET_STATS not found: ${FASTWAM_DATASET_STATS}" >&2
    exit 1
fi

OUTPUT_PATH="${PROJ_ROOT}/outputs/dynisafety/${TASK_CONFIG}/${TASK_NAME}/fastwam/level${DYNAMIC_LEVEL}"

POLICY_KWARGS="$(
FASTWAM_ROOT="${FASTWAM_ROOT}" \
FASTWAM_MODEL_BASE_PATH="${FASTWAM_MODEL_BASE_PATH}" \
FASTWAM_CKPT="${FASTWAM_CKPT}" \
FASTWAM_DATASET_STATS="${FASTWAM_DATASET_STATS}" \
FASTWAM_SIM_CFG_PATH="${FASTWAM_SIM_CFG_PATH}" \
FASTWAM_SIM_TASK="${FASTWAM_SIM_TASK}" \
FASTWAM_MIXED_PRECISION="${FASTWAM_MIXED_PRECISION}" \
FASTWAM_DEVICE="${FASTWAM_DEVICE}" \
FASTWAM_ACTION_HORIZON="${FASTWAM_ACTION_HORIZON}" \
FASTWAM_REPLAN_STEPS="${FASTWAM_REPLAN_STEPS}" \
FASTWAM_NUM_INFERENCE_STEPS="${FASTWAM_NUM_INFERENCE_STEPS}" \
FASTWAM_SIGMA_SHIFT="${FASTWAM_SIGMA_SHIFT}" \
FASTWAM_TEXT_CFG_SCALE="${FASTWAM_TEXT_CFG_SCALE}" \
FASTWAM_NEGATIVE_PROMPT="${FASTWAM_NEGATIVE_PROMPT}" \
FASTWAM_RAND_DEVICE="${FASTWAM_RAND_DEVICE}" \
FASTWAM_TILED="${FASTWAM_TILED}" \
FASTWAM_TIMING_ENABLED="${FASTWAM_TIMING_ENABLED}" \
SEED="${SEED}" \
python - <<'PY'
import json
import os


def optional_int(name: str):
    value = os.getenv(name, "").strip()
    return None if value.lower() in {"", "none", "null"} else int(value)


def optional_float(name: str):
    value = os.getenv(name, "").strip()
    return None if value.lower() in {"", "none", "null"} else float(value)


def parse_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


payload = {
    "fastwam_root": os.environ["FASTWAM_ROOT"],
    "ckpt_setting": os.environ["FASTWAM_CKPT"],
    "dataset_stats_path": os.environ["FASTWAM_DATASET_STATS"],
    "sim_cfg_path": os.environ["FASTWAM_SIM_CFG_PATH"],
    "sim_task": os.environ["FASTWAM_SIM_TASK"],
    "mixed_precision": os.environ["FASTWAM_MIXED_PRECISION"],
    "device": os.environ["FASTWAM_DEVICE"],
    "replan_steps": int(os.environ["FASTWAM_REPLAN_STEPS"]),
    "num_inference_steps": int(os.environ["FASTWAM_NUM_INFERENCE_STEPS"]),
    "text_cfg_scale": float(os.environ["FASTWAM_TEXT_CFG_SCALE"]),
    "negative_prompt": os.environ["FASTWAM_NEGATIVE_PROMPT"],
    "rand_device": os.environ["FASTWAM_RAND_DEVICE"],
    "tiled": parse_bool("FASTWAM_TILED", False),
    "timing_enabled": parse_bool("FASTWAM_TIMING_ENABLED", False),
    "seed": int(os.environ["SEED"]),
}
action_horizon = optional_int("FASTWAM_ACTION_HORIZON")
if action_horizon is not None:
    payload["action_horizon"] = action_horizon
sigma_shift = optional_float("FASTWAM_SIGMA_SHIFT")
if sigma_shift is not None:
    payload["sigma_shift"] = sigma_shift
print(json.dumps(payload))
PY
)"

echo "FASTWAM_ROOT         = ${FASTWAM_ROOT}"
echo "FASTWAM_MODEL_BASE   = ${FASTWAM_MODEL_BASE_PATH}"
echo "FASTWAM_CKPT         = ${FASTWAM_CKPT}"
echo "FASTWAM_STATS        = ${FASTWAM_DATASET_STATS}"
echo "CONDA_ENV            = ${CONDA_ENV}"
echo "TASK_NAME            = ${TASK_NAME}"
echo "TASK_IDX             = ${TASK_IDX}"
echo "DYNAMIC_LEVEL        = ${DYNAMIC_LEVEL}"
echo "TEST_NUM             = ${TEST_NUM}"
echo "EC_TIMEOUT_S         = ${EXPERT_CHECK_TIMEOUT_S}"
echo "OUTPUT_PATH          = ${OUTPUT_PATH}"

cd "${PROJ_ROOT}"
PYTHONWARNINGS=ignore::UserWarning \
DIFFSYNTH_MODEL_BASE_PATH="${FASTWAM_MODEL_BASE_PATH}" \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python fastwam_inference.py \
    --policy_kwargs "${POLICY_KWARGS}" \
    --task_config "${TASK_CONFIG}" \
    --task_idx "${TASK_IDX}" \
    --dynamic_level "${DYNAMIC_LEVEL}" \
    --test_num "${TEST_NUM}" \
    --seed "${SEED}" \
    --expert_check_timeout_s "${EXPERT_CHECK_TIMEOUT_S}" \
    --obstacle_scale "${OBSTACLE_SCALE}" \
    --obstacle_target_mode "${OBSTACLE_TARGET_MODE}" \
    --obstacle_selection_mode "${OBSTACLE_SELECTION_MODE}" \
    --l2_start_mode "${L2_START_MODE}" \
    ${L2_FLIP_SIDE_FLAG} \
    --l2_start_min_distance "${L2_START_MIN_DIST}" \
    --fps "${FPS}" \
    --video_dir "${OUTPUT_PATH}"
