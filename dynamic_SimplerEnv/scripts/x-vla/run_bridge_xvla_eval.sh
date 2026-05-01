#!/bin/bash
# DyniSafety-SimplerEnv + X-VLA — WidowX Bridge suite (HTTP client to XVLA server).
#
# Mirrors scripts/openvla/run_bridge_openvla_eval.sh but uses:
#   --model xvla  (see simpler_env/policies/xvla/xvla_model.py)
#   --official    Use --model xvla_official (scripts/x-vla/xvla_official_inference.py),
#                 strict parity with vlaModelHome/X-VLA/evaluation/simpler/WidowX/client_*.py
#
# Prerequisite: start the X-VLA FastAPI server first (XVLA conda env), e.g.:
#   bash dynamic_SimplerEnv/scripts/x-vla/start_xvla_server.sh \
#       --model-path download_checkpoints/xVLA/xvla-simpler-widowx
#
# Usage (from anywhere):
#   bash dynamic_SimplerEnv/scripts/x-vla/run_bridge_xvla_eval.sh [options]
#   cd dynamic_SimplerEnv && bash scripts/x-vla/run_bridge_xvla_eval.sh [options]
#
# Options:
#   (no --task)     Run DEFAULT_TASKS below (default: bridge = all four subtasks).
#   --task NAME     bridge | bridge_carrot | bridge_stack | bridge_spoon | bridge_eggplant
#   --level         0 | 1 | 2 | 3 | all  (default: 0)
#   --ckpt PATH     Placeholder for run_dynisafety.sh (default: ignored; inference uses server)
#   --server-host   XVLA server host (default: 127.0.0.1 or XVLA_SERVER_HOST)
#   --server-port   XVLA server port (default: 8010 or XVLA_SERVER_PORT)
#   --domain-id     XVLA domain id: 0=WidowX/SimplerEnv Bridge (default: 0)
#   --gpu N         CUDA device for sim/rendering (default: 0)
#   --temp F        Action ensemble temperature forwarded to main_inference (default: -0.8)
#   --output-root   Bench layout root (default: ./outputs)
#   --output-layout bench | legacy  (default: bench)
#   --log-root      Legacy nested log root (default: results)
#   --run-id-note   Optional log filename suffix
#   --env-name      Conda env for SimplerEnv eval (default: simpler_env; match run_bridge_openvla_eval.sh)
#   --official      Policy xvla_official (upstream WidowX client decoding + action chunking)
#   --official-simplerenv-root DIR
#                   Optional alternative SimplerEnv root for --official only.
#                   Priority: CLI > SIMPLERENV_ROOT_OFFICIAL > SIMPLERENV_ROOT > bundled root.
#   Plus any extra args are forwarded to scripts/run_dynisafety.sh
#
set -euo pipefail

MODEL_NAME="xvla_official"
USE_OFFICIAL=0
ACTION_ENSEMBLE_TEMP=-0.8
GPU_ID=0
LEVEL="2"
CONDA_ENV_NAME="simpler_env"
OUTPUT_ROOT="./outputs"
OUTPUT_LAYOUT="bench"
LOG_ROOT="results"
RUN_ID_NOTE=""
SINGLE_TASK="bridge_eggplant"
CKPT="ignored"
SERVER_HOST="${XVLA_SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${XVLA_SERVER_PORT:-8010}"
DOMAIN_ID="${XVLA_DOMAIN_ID:-0}"
OFFICIAL_SIMPLERENV_ROOT=""
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
        --domain-id)     shift; DOMAIN_ID="$1"; shift ;;
        --gpu)           shift; GPU_ID="$1"; shift ;;
        --temp)          shift; ACTION_ENSEMBLE_TEMP="$1"; shift ;;
        --output-root)   shift; OUTPUT_ROOT="$1"; shift ;;
        --output-layout) shift; OUTPUT_LAYOUT="$1"; shift ;;
        --log-root)      shift; LOG_ROOT="$1"; shift ;;
        --run-id-note)   shift; RUN_ID_NOTE="$1"; shift ;;
        --env-name)      shift; CONDA_ENV_NAME="$1"; shift ;;
        --official)      USE_OFFICIAL=1; shift ;;
        --official-simplerenv-root) shift; OFFICIAL_SIMPLERENV_ROOT="$1"; shift ;;
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
BUNDLED_SIMPLERENV_ROOT="${SIMPLER_ROOT}/SimplerEnv-OpenVLA"

resolve_dir() {
    local path="$1"
    if [[ ! -d "${path}" ]]; then
        echo "ERROR: SimplerEnv root does not exist: ${path}"
        exit 1
    fi
    (
        cd "${path}"
        pwd
    )
}

export DYNAMIC_SIMPLERENV_ROOT="${DYNAMIC_SIMPLERENV_ROOT:-${SIMPLER_ROOT}}"
OFFICIAL_ROOT_SOURCE=""
if [[ "${USE_OFFICIAL}" -eq 1 ]]; then
    MODEL_NAME="xvla_official"
    if [[ -n "${OFFICIAL_SIMPLERENV_ROOT}" ]]; then
        export SIMPLERENV_ROOT="$(resolve_dir "${OFFICIAL_SIMPLERENV_ROOT}")"
        OFFICIAL_ROOT_SOURCE="cli:--official-simplerenv-root"
    elif [[ -n "${SIMPLERENV_ROOT_OFFICIAL:-}" ]]; then
        export SIMPLERENV_ROOT="$(resolve_dir "${SIMPLERENV_ROOT_OFFICIAL}")"
        OFFICIAL_ROOT_SOURCE="env:SIMPLERENV_ROOT_OFFICIAL"
    elif [[ -n "${SIMPLERENV_ROOT:-}" ]]; then
        export SIMPLERENV_ROOT="$(resolve_dir "${SIMPLERENV_ROOT}")"
        OFFICIAL_ROOT_SOURCE="env:SIMPLERENV_ROOT"
    else
        export SIMPLERENV_ROOT="$(resolve_dir "${BUNDLED_SIMPLERENV_ROOT}")"
        OFFICIAL_ROOT_SOURCE="bundled-default"
    fi

    if [[ -z "${XVLA_EXEC_MODE:-}" ]]; then
        if [[ "${SIMPLERENV_ROOT}" == "$(resolve_dir "${BUNDLED_SIMPLERENV_ROOT}")" ]]; then
            export XVLA_EXEC_MODE="simpler_delta"
        else
            export XVLA_EXEC_MODE="upstream_literal"
        fi
    fi
fi

export XVLA_SERVER_HOST="${SERVER_HOST}"
export XVLA_SERVER_PORT="${SERVER_PORT}"
export XVLA_DOMAIN_ID="${DOMAIN_ID}"

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
    echo "X-VLA Bridge DyniSafety — model=${MODEL_NAME} ckpt=${CKPT} task=${task}"
    echo "  XVLA server: ${XVLA_SERVER_HOST}:${XVLA_SERVER_PORT}  domain_id=${XVLA_DOMAIN_ID}"
    if [[ "${USE_OFFICIAL}" -eq 1 ]]; then
        echo "  SimplerEnv root: ${SIMPLERENV_ROOT}  (${OFFICIAL_ROOT_SOURCE})"
        echo "  Exec mode     : ${XVLA_EXEC_MODE}"
        echo "  Adapter root  : ${DYNAMIC_SIMPLERENV_ROOT}"
    fi
    echo "============================================================"
    bash scripts/run_dynisafety.sh \
        --model "${MODEL_NAME}" \
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
