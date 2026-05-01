#!/bin/bash
# DyniSafety-SimplerEnv + OpenVLA — WidowX Bridge suite (in-process).
#
# Mirrors scripts/openvla/run_openvla_eval.sh but drives:
#   scripts/bridge_widowx_dynisafety.sh  (via scripts/run_dynisafety.sh)
# which matches SimplerEnv-OpenVLA/scripts/bridge.sh (widowx / widowx_sink_camera_setup,
# --policy-setup widowx_bridge, control-freq 5, etc.).
#
# Usage (from anywhere):
#   bash dynamic_SimplerEnv/scripts/openvla/run_bridge_openvla_eval.sh [options]
#   cd dynamic_SimplerEnv && bash scripts/openvla/run_bridge_openvla_eval.sh [options]
#
# Options:
#   (no --task)     Run DEFAULT_TASKS below (default: bridge = all four subtasks).
#   --task NAME     bridge | bridge_carrot | bridge_stack | bridge_spoon | bridge_eggplant
#   --model NAME    openvla | openvla_oft  (default: openvla, same as run_openvla_eval.sh)
#   --level         0 | 1 | 2 | 3 | all  (default: 0)
#   --ckpt PATH     Checkpoint/HF id (default: openvla/openvla-7b)
#
#   NOTE: openvla_oft loads simpler_env.policies.openvla_oft — that package is NOT in the
#   stock SimplerEnv-OpenVLA tree. Use --model openvla for base OpenVLA, or install OFT
#   integration separately, then: --model openvla_oft --ckpt moojink/openvla-7b-oft-finetuned-bridge
#   --gpu N         CUDA device (default: 0)
#   --temp F        Action ensemble temperature (default: -0.8)
#   --output-root   Bench layout root (default: ./outputs)
#   --output-layout bench | legacy  (default: bench)
#   --log-root      Legacy nested log root (default: results)
#   --run-id-note   Optional log filename suffix
#   --env-name      Conda env (default: simpler_env; match run_openvla_eval.sh)
#   Plus any extra args are forwarded to scripts/run_dynisafety.sh
#
# Batch checkpoints: edit CKPTS_DEFAULT (same pattern as run_openvla_eval.sh).
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

CKPTS_DEFAULT=(
  openvla/openvla-7b
)
DEFAULT_TASKS=(
  bridge
)

CKPTS=("${CKPTS_DEFAULT[@]}")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)          shift; SINGLE_TASK="$1"; shift ;;
        --model)         shift; MODEL_NAME="$1"; shift ;;
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

for t in "${TASKS_TO_RUN[@]}"; do
    case "${t}" in
        bridge|bridge_carrot|bridge_stack|bridge_spoon|bridge_eggplant)
            ;;
        *)
            echo "ERROR: invalid --task '${t}'. Use: bridge | bridge_carrot | bridge_stack | bridge_spoon | bridge_eggplant"
            exit 1
            ;;
    esac
done

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
        echo "OpenVLA Bridge DyniSafety — model=${MODEL_NAME} ckpt=${ckpt_path} task=${task}"
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
echo "ALL BRIDGE CKPT × TASK RUNS COMPLETE."
echo "============================================================"
