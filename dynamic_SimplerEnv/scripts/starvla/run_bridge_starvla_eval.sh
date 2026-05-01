#!/bin/bash
# DyniSafety-SimplerEnv + StarVLA — WidowX Bridge suite (WebSocket client to StarVLA server).
#
# Mirrors scripts/univla/run_bridge_univla_eval.sh but uses:
#   --model starvla  (see simpler_env/policies/starvla/starvla_model.py)
#
# Unlike UniVLA, --ckpt must be a real .../checkpoints/*.pt path: the client loads the run's
# config.yaml and dataset_statistics.json from the same run directory (no torch on client).
# Keep this path aligned with start_starvla_server.sh --ckpt-path.
#
# Prerequisite: start the StarVLA server (starVLA conda), e.g.:
#   bash dynamic_SimplerEnv/scripts/starvla/start_starvla_server.sh --port 10093
# (LIBERO eval lives under dynisafety_libero/; this script is dynamic_SimplerEnv Bridge only.)
#
# Usage (from anywhere):
#   bash dynamic_SimplerEnv/scripts/starvla/run_bridge_starvla_eval.sh [options]
#   cd dynamic_SimplerEnv && bash scripts/starvla/run_bridge_starvla_eval.sh [options]
#
# Options:
#   (no --task)     Run DEFAULT_TASKS below (default: bridge = all four subtasks).
#   --task NAME     bridge | bridge_carrot | bridge_stack | bridge_spoon | bridge_eggplant
#   --level         0 | 1 | 2 | 3 | all  (default: 0)
#   --ckpt PATH     Checkpoint .pt under .../checkpoints/ (default: STARVLA_CHECKPOINT or Qwen-GR00T-Bridge path)
#   --server-host   StarVLA server host (default: 127.0.0.1 or STARVLA_SERVER_HOST)
#   --server-port   StarVLA server port (default: 10093 or STARVLA_SERVER_PORT)
#   --unnorm-key K  STARVLA_UNNORM_KEY for multi-dataset stats (default: oxe_bridge for Bridge-RT1)
#   --gpu N         CUDA device for sim/rendering (default: 0)
#   --temp F        Action ensemble temperature forwarded to main_inference (default: -0.8)
#   --output-root   Bench layout root (default: ./outputs)
#   --output-layout bench | legacy  (default: bench)
#   --log-root      Legacy nested log root (default: results)
#   --run-id-note   Optional log filename suffix
#   --env-name      Conda env for SimplerEnv eval (default: simpler_env)
#   --max-steps N   Override --max-episode-steps for ALL bridge subtasks
#                   (default: 120 — matches official starVLA SimplerEnv eval start_simpler_env.sh).
#                   The repo-wide bridge default for non-StarVLA models stays at carrot/stack/spoon=60,
#                   eggplant=120; we only bump it for StarVLA because the official ckpt typically needs
#                   ~80–110 steps to actually finish placement on carrot/stack/spoon.
#   Plus any extra args are forwarded to scripts/run_dynisafety.sh
#
# Note: run_dynisafety_eval.sh uses --ckpt-path for the same on-disk checkpoint; this entry uses --ckpt
# to match scripts/run_dynisafety.sh.
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
# Default 120 to match official starVLA: examples/SimplerEnv/eval_files/start_simpler_env.sh uses
# --max-episode-steps 120 for ALL four bridge subtasks (PutCarrotOnPlate, StackGreenCubeOnYellowCube,
# PutSpoonOnTableCloth, PutEggplantInBasket). The shared bridge_widowx_dynisafety.sh defaults are
# 60/60/60/120 (matches SimplerEnv-OpenVLA bridge.sh), which truncates StarVLA mid-placement on
# carrot/stack/spoon. Use --max-steps 60 here to fall back to the cross-model default.
MAX_STEPS="${BRIDGE_MAX_EPISODE_STEPS:-120}"
CKPT="${STARVLA_CHECKPOINT:-download_checkpoints/starVLA/Qwen-GR00T-Bridge/checkpoints/steps_45000_pytorch_model.pt}"
SERVER_HOST="${STARVLA_SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${STARVLA_SERVER_PORT:-10093}"
UNNORM_KEY_EXPLICIT=""
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
        --unnorm-key)    shift; UNNORM_KEY_EXPLICIT="$1"; shift ;;
        --gpu)           shift; GPU_ID="$1"; shift ;;
        --temp)          shift; ACTION_ENSEMBLE_TEMP="$1"; shift ;;
        --output-root)   shift; OUTPUT_ROOT="$1"; shift ;;
        --output-layout) shift; OUTPUT_LAYOUT="$1"; shift ;;
        --log-root)      shift; LOG_ROOT="$1"; shift ;;
        --run-id-note)   shift; RUN_ID_NOTE="$1"; shift ;;
        --env-name)      shift; CONDA_ENV_NAME="$1"; shift ;;
        --max-steps)     shift; MAX_STEPS="$1"; shift ;;
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
WORKSPACE="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ "${CKPT}" != /* ]]; then
    CKPT="${WORKSPACE}/${CKPT}"
fi

export STARVLA_SERVER_HOST="${SERVER_HOST}"
export STARVLA_SERVER_PORT="${SERVER_PORT}"

if [[ -n "${UNNORM_KEY_EXPLICIT}" ]]; then
    export STARVLA_UNNORM_KEY="${UNNORM_KEY_EXPLICIT}"
elif [[ -z "${STARVLA_UNNORM_KEY:-}" ]]; then
    export STARVLA_UNNORM_KEY="oxe_bridge"
fi

# Forward to bridge_widowx_dynisafety.sh which honors BRIDGE_MAX_EPISODE_STEPS (overrides per-task default).
export BRIDGE_MAX_EPISODE_STEPS="${MAX_STEPS}"

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
    echo "StarVLA Bridge DyniSafety — model=starvla ckpt=${CKPT} task=${task}"
    echo "  StarVLA server: ${STARVLA_SERVER_HOST}:${STARVLA_SERVER_PORT}"
    echo "  STARVLA_UNNORM_KEY: ${STARVLA_UNNORM_KEY:-}"
    echo "  BRIDGE_MAX_EPISODE_STEPS: ${BRIDGE_MAX_EPISODE_STEPS} (official starVLA SimplerEnv eval = 120)"
    echo "============================================================"
    bash scripts/run_dynisafety.sh \
        --model starvla \
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
