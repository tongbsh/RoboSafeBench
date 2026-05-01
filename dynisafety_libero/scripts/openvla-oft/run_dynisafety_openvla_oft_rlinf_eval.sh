#!/usr/bin/env bash
# ============================================================
# Run DyniSafety-LIBERO benchmark with RLinf OpenVLA-OFT.
#
# The checkpoint directory already bundles modeling_prismatic.py,
# so we load it via the standard OpenVLAPolicy path (no RLinf repo needed).
#
# Usage:
#   bash dynisafety_libero/scripts/openvla-oft/run_dynisafety_openvla_oft_rlinf_eval.sh [OPTIONS]
#
# Optional:
#   --checkpoint       <PATH>   HF checkpoint directory
#                                Default:
#                                download_checkpoints/openvla-oft_RLinf/
#                                RLinf-OpenVLAOFT-GRPO-LIBERO-spatial
#   --task-suite       <NAME>   safelibero_spatial | safelibero_object |
#                                safelibero_goal | safelibero_long
#                                Default: safelibero_spatial
#   --dynamic-level    <N>      0=static 1=drop 2=slide 3=shared-reach 4=inspect
#                                Default: 0
#   --unnorm-key       <KEY>    Override checkpoint unnorm_key resolution
#   --num-images       <N>      Number of images in input. Default: 1
#   --use-proprio               Enable proprio input (disabled by default)
#   --num-trials      <N>       Episodes per task. Default: 50
#   --open-loop-steps <N>       Actions executed per model chunk. Default: 8
#   --arm-speed       <SPEED>   slow | medium | fast. Default: medium
#   --task-id         <ID>      If set, evaluate only this task ID. Default: 0
#   --env-name        <NAME>    Conda environment. Default: openvla-oft
# ============================================================

set -euo pipefail

TASK_SUITE="safelibero_spatial"
DYNAMIC_LEVEL="0"
CHECKPOINT="download_checkpoints/openvla-oft_RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-spatial"
UNNORM_KEY=""
NUM_IMAGES="1"
USE_PROPRIO="False"
NUM_TRIALS="50"
OPEN_LOOP_STEPS="8"
ARM_SPEED="medium"
TASK_ID="0"
CONDA_ENV_NAME="openvla-oft"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)         CHECKPOINT="$2";      shift 2 ;;
        --task-suite)         TASK_SUITE="$2";      shift 2 ;;
        --dynamic-level)      DYNAMIC_LEVEL="$2";   shift 2 ;;
        --unnorm-key)         UNNORM_KEY="$2";      shift 2 ;;
        --num-images)         NUM_IMAGES="$2";      shift 2 ;;
        --use-proprio)        USE_PROPRIO="True";   shift 1 ;;
        --num-trials)         NUM_TRIALS="$2";      shift 2 ;;
        --open-loop-steps)    OPEN_LOOP_STEPS="$2"; shift 2 ;;
        --arm-speed)          ARM_SPEED="$2";       shift 2 ;;
        --task-id)            TASK_ID="$2";         shift 2 ;;
        --env-name)           CONDA_ENV_NAME="$2";  shift 2 ;;
        -h|--help)
            sed -n '3,35p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

set +u
# shellcheck disable=SC1091
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "${CONDA_PREFIX:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

EXTRA_ARGS=()
if [[ -n "${UNNORM_KEY}" ]]; then
    EXTRA_ARGS+=(--unnorm_key "${UNNORM_KEY}")
fi

echo "============================================================"
echo "DyniSafety-LIBERO benchmark (RLinf OpenVLA-OFT, HF checkpoint)"
echo "  Task suite        : ${TASK_SUITE}"
echo "  Dynamic level     : ${DYNAMIC_LEVEL}"
echo "  Checkpoint        : ${CHECKPOINT}"
echo "  Unnorm key        : ${UNNORM_KEY:-<auto-resolve>}"
echo "  Num images        : ${NUM_IMAGES}"
echo "  Use proprio       : ${USE_PROPRIO}"
echo "  Open-loop steps   : ${OPEN_LOOP_STEPS}"
echo "  Trials/task       : ${NUM_TRIALS}"
echo "  Arm speed         : ${ARM_SPEED}"
echo "  Task ID           : ${TASK_ID}"
echo "  Conda env         : ${CONDA_ENV_NAME}"
echo "============================================================"

cd "${PROJ_ROOT}"
exec python dynisafety_libero/scripts/openvla-oft/run_dynisafety_openvla_oft_rlinf_eval.py \
    --task_suite_name "${TASK_SUITE}" \
    --dynamic_level "${DYNAMIC_LEVEL}" \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --num_images_in_input "${NUM_IMAGES}" \
    --use_proprio "${USE_PROPRIO}" \
    --num_trials_per_task "${NUM_TRIALS}" \
    --num_open_loop_steps "${OPEN_LOOP_STEPS}" \
    --arm_speed "${ARM_SPEED}" \
    --task_id_override "${TASK_ID}" \
    "${EXTRA_ARGS[@]}"
