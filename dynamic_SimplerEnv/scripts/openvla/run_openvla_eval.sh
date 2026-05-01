#!/bin/bash
# DyniSafety-SimplerEnv + OpenVLA (in-process).
#
# Mirrors the structure of upstream:
#   SimplerEnv-OpenVLA/scripts/run_openvla.sh
# (model name, default ckpt list, action_ensemble_temp, multi-task sweep).
# DyniSafety uses a single bench task name per suite (pick_coke_can | move_near |
# drawer) instead of separate variant_agg / visual_matching shell scripts; dynamic
# difficulty is --level 0|1|2|3|all via scripts/run_dynisafety.sh.
#
# Usage (from anywhere):
#   bash dynamic_SimplerEnv/scripts/openvla/run_openvla_eval.sh [options]
#   cd dynamic_SimplerEnv && bash scripts/openvla/run_openvla_eval.sh [options]
#
# Default output layout is bench → outputs/openvla/logs and outputs/openvla/videos/
# under dynamic_SimplerEnv/. For SimplerEnv-OpenVLA tools/calc_metrics_evaluation_videos.py
# (nested results/ tree), use legacy layout:
#   bash scripts/openvla/run_openvla_eval.sh --output-layout legacy --log-root results
#
# Options:
#   (no --task)     Run DEFAULT_TASKS below (default: pick_coke_can only).
#   --task NAME     Run only pick_coke_can | move_near | drawer
#   --level         0 | 1 | 2 | 3 | all  (default: 1; all = 1,2,3)
#   --ckpt PATH     Single checkpoint/HF id (default: openvla/openvla-7b); replaces default ckpt list
#   --gpu N         CUDA device (default: 0)
#   --temp F        Action ensemble temperature (default: -0.8)
#   --output-root   Bench layout root (default: ./outputs)
#   --output-layout bench | legacy  (default: bench)
#   --log-root      Legacy nested log root (default: results)
#   --run-id-note   Optional log filename suffix
#   --env-name      Conda env (default: openvla-oft; same as eval_dynisafety.sh for --vla openvla)
#   Plus any extra args are forwarded to scripts/run_dynisafety.sh
#
# To batch multiple checkpoints, edit CKPTS_DEFAULT in this script (same idea as upstream ckpts array).
set -euo pipefail

MODEL_NAME="openvla"
ACTION_ENSEMBLE_TEMP=-0.8
GPU_ID=0
LEVEL="0"
CONDA_ENV_NAME="simpler_env"
OUTPUT_ROOT="./outputs"
OUTPUT_LAYOUT="bench"
LOG_ROOT="results"
RUN_ID_NOTE=""
SINGLE_TASK=""
EXTRA=()

# Align with upstream run_openvla.sh ckpts=(); add more HF ids here to batch.
CKPTS_DEFAULT=(
  openvla/openvla-7b
)
# Default bench tasks (upstream also has pick_coke_can_* scripts; add move_near drawer to sweep).
DEFAULT_TASKS=(
  pick_coke_can
)

CKPTS=("${CKPTS_DEFAULT[@]}")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)          shift; SINGLE_TASK="$1"; shift ;;
        --level)         shift; LEVEL="$1"; shift ;;
        --ckpt)          shift; CKPTS=("$1"); shift ;;
        --gpu)           shift; GPU_ID="$1"; shift ;;
        --temp)          shift; ACTION_ENSEMBLE_TEMP="$1"; shift ;;
        --output-root)   shift; OUTPUT_ROOT="$1"; shift ;;
        --output-layout) shift; OUTPUT_LAYOUT="$1"; shift ;;
        --log-root)      shift; LOG_ROOT="$1"; shift ;;
        --run-id-note)   shift; RUN_ID_NOTE="$1"; shift ;;
        --env-name)      shift; CONDA_ENV_NAME="$1"; shift ;;
        *)
            EXTRA+=("$1"); shift ;;
    esac
done

if [[ -n "${SINGLE_TASK}" ]]; then
    TASKS_TO_RUN=("${SINGLE_TASK}")
else
    TASKS_TO_RUN=("${DEFAULT_TASKS[@]}")
    if [[ ${#TASKS_TO_RUN[@]} -eq 0 ]]; then
        echo "ERROR: DEFAULT_TASKS is empty (add task names or pass --task)."
        exit 1
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIMPLER_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

set +u
# shellcheck disable=SC1091
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "${CONDA_PREFIX:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

NOTE_ARGS=()
if [[ -n "${RUN_ID_NOTE}" ]]; then
    NOTE_ARGS=(--run-id-note "${RUN_ID_NOTE}")
fi

cd "${SIMPLER_ROOT}"

for ckpt_path in "${CKPTS[@]}"; do
    for task in "${TASKS_TO_RUN[@]}"; do
        echo "============================================================"
        echo "OpenVLA DyniSafety — ckpt=${ckpt_path} task=${task}"
        echo "============================================================"
        bash scripts/run_dynisafety.sh \
            --model "${MODEL_NAME}" \
            --ckpt "${ckpt_path}" \
            --task "${task}" \
            --level "${LEVEL}" \
            --temp "${ACTION_ENSEMBLE_TEMP}" \
            --gpu "${GPU_ID}" \
            --output-root "${OUTPUT_ROOT}" \
            --output-layout "${OUTPUT_LAYOUT}" \
            --log-root "${LOG_ROOT}" \
            "${NOTE_ARGS[@]}" \
            "${EXTRA[@]}"
    done
done

echo "============================================================"
echo "ALL OPENVLA CKPT × TASK RUNS COMPLETE."
echo "============================================================"
