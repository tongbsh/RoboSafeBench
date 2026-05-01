#!/bin/bash
# ============================================================
# DyniSafety-SimplerEnv — VLA registry
#
# Usage:
#   bash scripts/eval_dynisafety.sh [--vla NAME] [--task TASK] [--level N] [options]
#
#   --vla     openvla | openvla_oft | pi0 | pi05 | memoryvla | spatialvla_server |
#             xvla | starvla | univla | gr00t  (default: openvla_oft)
#   --task    pick_coke_can | move_near | drawer | bridge | bridge_carrot | bridge_stack |
#             bridge_spoon | bridge_eggplant  (default: pick_coke_can)
#             For openvla_oft + bridge*: default ckpt = OPENVLA_OFT_BRIDGE_CKPT or
#             moojink/openvla-7b-oft-finetuned-bridge
#   --level   0 | 1 | 2 | 3 | all  (default: 1; all = 1,2,3)
#   --output-layout bench | legacy  (default: bench; legacy for calc_metrics_evaluation_videos.py)
#   --output-root DIR  (default: ./outputs)
#   --run-id-note STR  optional log suffix
#
# Server VLAs: start the matching server under dynamic_SimplerEnv/scripts/<stack>/start_*.sh
#
# Outputs (bench): dynamic_SimplerEnv/outputs/{model}/logs/  and  outputs/{model}/videos/
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIMPLER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VLA="${VLA:-openvla_oft}"
TASK="pick_coke_can"
LEVEL="1"
OUTPUT_LAYOUT="bench"
OUTPUT_ROOT="./outputs"
RUN_ID_NOTE=""
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vla)            shift; VLA="$1"; shift ;;
        --task)           shift; TASK="$1"; shift ;;
        --level)          shift; LEVEL="$1"; shift ;;
        --output-layout)  shift; OUTPUT_LAYOUT="$1"; shift ;;
        --output-root)    shift; OUTPUT_ROOT="$1"; shift ;;
        --run-id-note)    shift; RUN_ID_NOTE="$1"; shift ;;
        *)
            EXTRA+=("$1"); shift ;;
    esac
done

CONDA_ENV="openvla-oft"
CKPT=""
MODEL_NAME=""

case "${VLA}" in
    openvla)
        MODEL_NAME="openvla"
        CKPT="openvla/openvla-7b"
        ;;
    openvla_oft)
        MODEL_NAME="openvla_oft"
        case "${TASK}" in
            drawer)
                CKPT="moojink/openvla-7b-oft-finetuned-libero-goal"
                ;;
            bridge|bridge_carrot|bridge_stack|bridge_spoon|bridge_eggplant)
                CKPT="${OPENVLA_OFT_BRIDGE_CKPT:-moojink/openvla-7b-oft-finetuned-bridge}"
                ;;
            pick_coke_can|move_near|*)
                CKPT="moojink/openvla-7b-oft-finetuned-libero-object"
                ;;
        esac
        ;;
    pi0|pi05)
        MODEL_NAME="${VLA}"
        CKPT="ignored"
        ;;
    memoryvla)
        MODEL_NAME="memoryvla"
        CKPT="ignored"
        ;;
    spatialvla_server)
        MODEL_NAME="spatialvla_server"
        CKPT="ignored"
        ;;
    xvla)
        MODEL_NAME="xvla"
        CKPT="ignored"
        ;;
    starvla)
        MODEL_NAME="starvla"
        CKPT="${STARVLA_CHECKPOINT:-download_checkpoints/starVLA/starvla-libero-4in1/checkpoints/steps_50000_pytorch_model.pt}"
        if [[ "${CKPT}" != /* ]]; then
            CKPT="${WORKSPACE}/${CKPT}"
        fi
        ;;
    univla)
        MODEL_NAME="univla"
        CKPT="ignored"
        ;;
    gr00t)
        MODEL_NAME="gr00t"
        CKPT="${GR00T_CKPT:-}"
        if [[ -z "${CKPT}" ]]; then
            echo "ERROR: gr00t requires GR00T_CKPT env or pre-set CKPT; pass via:"
            echo "  GR00T_CKPT=/path bash scripts/eval_dynisafety.sh --vla gr00t"
            exit 1
        fi
        if [[ "${CKPT}" != /* ]]; then
            CKPT="${WORKSPACE}/${CKPT}"
        fi
        ;;
    *)
        echo "ERROR: unknown --vla '${VLA}'"
        echo "  Supported: openvla | openvla_oft | pi0 | pi05 | memoryvla | spatialvla_server | xvla | starvla | univla | gr00t"
        exit 1
        ;;
esac

export DYNISAFETY_OUTPUT_ROOT="${OUTPUT_ROOT}"
export DYNISAFETY_OUTPUT_LAYOUT="${OUTPUT_LAYOUT}"
export DYNISAFETY_BENCH_TASK="${TASK}"
export DYNISAFETY_RUN_ID_NOTE="${RUN_ID_NOTE}"

set +u
# shellcheck disable=SC1091
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "${CONDA_PREFIX:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u

NOTE_ARGS=()
if [[ -n "${RUN_ID_NOTE}" ]]; then
    NOTE_ARGS=(--run-id-note "${RUN_ID_NOTE}")
fi

cd "${SIMPLER_ROOT}"
exec bash scripts/run_dynisafety.sh \
    --model "${MODEL_NAME}" \
    --ckpt "${CKPT}" \
    --task "${TASK}" \
    --level "${LEVEL}" \
    --output-root "${OUTPUT_ROOT}" \
    --output-layout "${OUTPUT_LAYOUT}" \
    "${NOTE_ARGS[@]}" \
    "${EXTRA[@]}"
