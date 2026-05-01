#!/usr/bin/env bash
# ============================================================
# Run DyniSafety-LIBERO benchmark with OpenVLA-OFT (local checkpoint).
#
# Usage:
#   bash dynisafety_libero/scripts/openvla-oft/run_dynisafety_eval.sh [OPTIONS]
#
# Options:
#   --task-suite       <NAME>   safelibero_spatial | safelibero_object |
#                                safelibero_goal | safelibero_long
#                                Default: safelibero_spatial
#   --dynamic-level    <N>      0=static 1=drop 2=slide 3=shared-reach 4=inspect
#                                Default: 2
#   --checkpoint       <PATH>   HuggingFace id or local path (pretrained_checkpoint)
#                                Default: moojink/openvla-7b-oft-finetuned-libero-spatial
#   --openvla-root     <DIR>    Sets OPENVLA_ROOT for this run (openvla-oft repo)
#   --num-trials       <N>      Episodes per task. Default: 50
#   --open-loop-steps  <N>      Actions executed per model chunk. Default: 8
#   --arm-speed        <SPEED>  slow | medium | fast. Default: medium
#   --task-id          <ID>     If set, evaluate only this task ID. Default: 0
#   --env-name         <NAME>   Conda environment. Default: openvla-oft
#
# Quickstart:
#   bash dynisafety_libero/scripts/openvla-oft/run_dynisafety_eval.sh \\
#       --task-suite safelibero_spatial --dynamic-level 1
#
# Outputs:
#   dynisafety_libero/outputs/openvla_oft/logs/
#   dynisafety_libero/outputs/openvla_oft/videos/
# ============================================================

set -euo pipefail

TASK_SUITE="safelibero_spatial"
DYNAMIC_LEVEL="2"
CHECKPOINT="moojink/openvla-7b-oft-finetuned-libero-spatial"
OPENVLA_ROOT_ARG=""
NUM_TRIALS=50
OPEN_LOOP_STEPS=8
ARM_SPEED="medium"
TASK_ID=0
CONDA_ENV_NAME="openvla-oft"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task-suite)       TASK_SUITE="$2";       shift 2 ;;
        --dynamic-level)    DYNAMIC_LEVEL="$2";    shift 2 ;;
        --checkpoint)       CHECKPOINT="$2";       shift 2 ;;
        --openvla-root)     OPENVLA_ROOT_ARG="$2"; shift 2 ;;
        --num-trials)       NUM_TRIALS="$2";       shift 2 ;;
        --open-loop-steps)  OPEN_LOOP_STEPS="$2";  shift 2 ;;
        --arm-speed)        ARM_SPEED="$2";        shift 2 ;;
        --task-id)          TASK_ID="$2";          shift 2 ;;
        --env-name)         CONDA_ENV_NAME="$2";   shift 2 ;;
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

if [[ -n "${OPENVLA_ROOT_ARG}" ]]; then
    export OPENVLA_ROOT="${OPENVLA_ROOT_ARG}"
fi

echo "============================================================"
echo "DyniSafety-LIBERO benchmark (OpenVLA-OFT)"
echo "  Task suite        : ${TASK_SUITE}"
echo "  Dynamic level     : ${DYNAMIC_LEVEL}"
echo "  Checkpoint        : ${CHECKPOINT}"
echo "  OPENVLA_ROOT      : ${OPENVLA_ROOT:-<default: vlaModelHome/openvla-oft>}"
echo "  Open-loop steps   : ${OPEN_LOOP_STEPS}"
echo "  Trials/task       : ${NUM_TRIALS}"
echo "  Arm speed         : ${ARM_SPEED}"
echo "  Task ID           : ${TASK_ID}"
echo "  Conda env         : ${CONDA_ENV_NAME}"
echo "============================================================"

cd "${PROJ_ROOT}"
exec python dynisafety_libero/scripts/openvla-oft/run_dynisafety_openvla_oft_eval.py \
    --task_suite_name       "${TASK_SUITE}" \
    --dynamic_level         "${DYNAMIC_LEVEL}" \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --num_trials_per_task   "${NUM_TRIALS}" \
    --num_open_loop_steps   "${OPEN_LOOP_STEPS}" \
    --arm_speed             "${ARM_SPEED}" \
    --task_id_override      "${TASK_ID}"
