#!/bin/bash
# DyniSafety-SimplerEnv + InternVLA-M1 — WidowX Bridge suite (WebSocket client to InternVLA-M1 server).
#
# Mirrors scripts/starvla/run_bridge_starvla_eval.sh but uses:
#   --model internvla_m1  (see simpler_env/policies/internvla_m1/internvla_m1_model.py)
#
# Like StarVLA, --ckpt must be a real .../checkpoints/*.pt path: the client loads the run's
# config.yaml and dataset_statistics.json from the same run directory (no torch on client).
# Keep this path aligned with start_internvla_m1_server.sh --ckpt-path.
#
# Prerequisite: start the InternVLA-M1 server (internvla_m1 conda), e.g.:
#   bash dynamic_SimplerEnv/scripts/internvla_m1/start_internvla_m1_server.sh --port 10094
#
# Usage (from anywhere):
#   bash dynamic_SimplerEnv/scripts/internvla_m1/run_bridge_internvla_m1_eval.sh [options]
#   cd dynamic_SimplerEnv && bash scripts/internvla_m1/run_bridge_internvla_m1_eval.sh [options]
#
# Options:
#   (no --task)     Run DEFAULT_TASKS below (default: bridge = all four subtasks).
#   --task NAME     bridge | bridge_carrot | bridge_stack | bridge_spoon | bridge_eggplant
#   --level         0 | 1 | 2 | 3 | all  (default: 0)
#   --ckpt PATH     Checkpoint .pt under .../checkpoints/ (default: INTERNVLA_M1_CHECKPOINT or
#                                                          Pretrain-RT-1-Bridge default)
#   --server-host   InternVLA-M1 server host (default: 127.0.0.1 or INTERNVLA_M1_SERVER_HOST)
#   --server-port   InternVLA-M1 server port (default: 10094 or INTERNVLA_M1_SERVER_PORT)
#   --unnorm-key K  INTERNVLA_M1_UNNORM_KEY for multi-dataset stats (default: bridge_dataset)
#   --cfg-scale F   Classifier-free guidance scale (default: 1.5; matches official M1Inference)
#   --ddim-steps N  DDIM denoising steps (default: 5; matches official M1Inference default)
#   --gpu N         CUDA device for sim/rendering (default: 0)
#   --temp F        Action ensemble temperature forwarded to main_inference (default: -0.8)
#   --output-root   Bench layout root (default: ./outputs)
#   --output-layout bench | legacy  (default: bench)
#   --log-root      Legacy nested log root (default: results)
#   --run-id-note   Optional log filename suffix
#   --env-name      Conda env for SimplerEnv eval (default: simpler_env)
#   --max-steps N   Override --max-episode-steps for ALL bridge subtasks
#                   (default: 120 — matches official InternVLA-M1 examples/SimplerEnv/start_simpler_env.sh).
#                   Use --max-steps 60 to fall back to the cross-model default for carrot/stack/spoon.
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
# Default 120 to match official InternVLA-M1: examples/SimplerEnv/start_simpler_env.sh uses
# --max-episode-steps 120 for all four bridge subtasks.
MAX_STEPS="${BRIDGE_MAX_EPISODE_STEPS:-120}"
CKPT="${INTERNVLA_M1_CHECKPOINT:-download_checkpoints/InternVLA/InternVLA-M1-Pretrain-RT-1-Bridge/checkpoints/steps_50000_pytorch_model.pt}"
SERVER_HOST="${INTERNVLA_M1_SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${INTERNVLA_M1_SERVER_PORT:-10094}"
UNNORM_KEY_EXPLICIT=""
CFG_SCALE_EXPLICIT=""
DDIM_STEPS_EXPLICIT=""
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
        --cfg-scale)     shift; CFG_SCALE_EXPLICIT="$1"; shift ;;
        --ddim-steps)    shift; DDIM_STEPS_EXPLICIT="$1"; shift ;;
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

export INTERNVLA_M1_SERVER_HOST="${SERVER_HOST}"
export INTERNVLA_M1_SERVER_PORT="${SERVER_PORT}"

if [[ -n "${UNNORM_KEY_EXPLICIT}" ]]; then
    export INTERNVLA_M1_UNNORM_KEY="${UNNORM_KEY_EXPLICIT}"
elif [[ -z "${INTERNVLA_M1_UNNORM_KEY:-}" ]]; then
    export INTERNVLA_M1_UNNORM_KEY="bridge_dataset"
fi

if [[ -n "${CFG_SCALE_EXPLICIT}" ]]; then
    export INTERNVLA_M1_CFG_SCALE="${CFG_SCALE_EXPLICIT}"
fi

if [[ -n "${DDIM_STEPS_EXPLICIT}" ]]; then
    export INTERNVLA_M1_NUM_DDIM_STEPS="${DDIM_STEPS_EXPLICIT}"
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
    echo "InternVLA-M1 Bridge DyniSafety — model=internvla_m1 ckpt=${CKPT} task=${task}"
    echo "  InternVLA-M1 server: ${INTERNVLA_M1_SERVER_HOST}:${INTERNVLA_M1_SERVER_PORT}"
    echo "  INTERNVLA_M1_UNNORM_KEY     : ${INTERNVLA_M1_UNNORM_KEY:-}"
    echo "  INTERNVLA_M1_CFG_SCALE      : ${INTERNVLA_M1_CFG_SCALE:-1.5 (default)}"
    echo "  INTERNVLA_M1_NUM_DDIM_STEPS : ${INTERNVLA_M1_NUM_DDIM_STEPS:-5 (default)}"
    echo "  BRIDGE_MAX_EPISODE_STEPS    : ${BRIDGE_MAX_EPISODE_STEPS} (official InternVLA-M1 SimplerEnv eval = 120)"
    echo "============================================================"
    bash scripts/run_dynisafety.sh \
        --model internvla_m1 \
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
