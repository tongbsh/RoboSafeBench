#!/bin/bash
# DyniSafety-SimplerEnv + UniVLA — WidowX Bridge suite (HTTP client to UniVLA server).
#
# Mirrors scripts/x-vla/run_bridge_xvla_eval.sh but uses:
#   --model univla  (see simpler_env/policies/univla/univla_model.py)
#
# Prerequisite: start the UniVLA server (univla conda). This repo's start_univla_server.sh is Bridge-only:
#   bash dynamic_SimplerEnv/scripts/univla/start_univla_server.sh --port 8020
# (LIBERO eval: dynisafety_libero/scripts/univla/start_univla_server.sh)
# Bridge eval: prefer Bridge-tuned weights (see UNIVLA_SIMPLERENV_CHECKPOINT in start_univla_server.sh).
#
# Usage (from anywhere):
#   bash dynamic_SimplerEnv/scripts/univla/run_bridge_univla_eval.sh [options]
#   cd dynamic_SimplerEnv && bash scripts/univla/run_bridge_univla_eval.sh [options]
#
# Options:
#   (no --task)     Run DEFAULT_TASKS below (default: bridge = all four subtasks).
#   --task NAME     bridge | bridge_carrot | bridge_stack | bridge_spoon | bridge_eggplant
#   --level         0 | 1 | 2 | 3 | all  (default: 0)
#   --ckpt PATH     Placeholder for run_dynisafety.sh (default: ignored; inference uses server)
#   --server-host   UniVLA server host (default: 127.0.0.1 or UNIVLA_SERVER_HOST)
#   --server-port   UniVLA server port (default: 8020 or UNIVLA_SERVER_PORT)
#   --gpu N         CUDA device for sim/rendering (default: 0)
#   --temp F        Action ensemble temperature forwarded to main_inference (default: -0.8)
#   --output-root   Bench layout root (default: ./outputs)
#   --output-layout bench | legacy  (default: bench)
#   --log-root      Legacy nested log root (default: results)
#   --run-id-note   Optional log filename suffix
#   --env-name      Conda env for SimplerEnv eval (default: simpler_env)
#   Plus any extra args are forwarded to scripts/run_dynisafety.sh
#
set -euo pipefail

ACTION_ENSEMBLE_TEMP=-0.8
GPU_ID=0
LEVEL="0"
CONDA_ENV_NAME="simpler_env"
OUTPUT_ROOT="./outputs"
OUTPUT_LAYOUT="bench"
LOG_ROOT="results"
RUN_ID_NOTE=""
SINGLE_TASK=""
CKPT="ignored"
SERVER_HOST="${UNIVLA_SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${UNIVLA_SERVER_PORT:-8020}"
EXTRA=()

DEFAULT_TASKS=(
  bridge
)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)          shift; SINGLE_TASK="$1"; shift ;;
        --level)         shift; LEVEL="$1"; shift ;;
        --ckpt)          shift; CKPT="$1"; shift ;;
        --server-host)   shift; SERVER_HOST="$1"; shift ;;
        --server-port)   shift; SERVER_PORT="$1"; shift ;;
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

export UNIVLA_SERVER_HOST="${SERVER_HOST}"
export UNIVLA_SERVER_PORT="${SERVER_PORT}"

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

for task in "${TASKS_TO_RUN[@]}"; do
    echo "============================================================"
    echo "UniVLA Bridge DyniSafety — model=univla ckpt=${CKPT} task=${task}"
    echo "  UniVLA server: ${UNIVLA_SERVER_HOST}:${UNIVLA_SERVER_PORT}"
    echo "============================================================"
    bash scripts/run_dynisafety.sh \
        --model univla \
        --ckpt "${CKPT}" \
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

echo "============================================================"
echo "ALL BRIDGE TASK RUNS COMPLETE."
echo "============================================================"
