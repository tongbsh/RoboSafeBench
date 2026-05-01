#!/bin/bash
# DyniSafety-SimplerEnv + UniVLA (HTTP client).
# Server (univla conda): scripts/univla/start_univla_server.sh
#   Default weights: <bench>/download_checkpoints/.../sft-simpler-bridge if present, else under dynamic_SimplerEnv/
#   Override: env UNIVLA_SIMPLERENV_CHECKPOINT
#
# Outputs (bench): outputs/univla/logs/  and  outputs/univla/videos/ (under SimplerEnv root)
#
# Usage:
#   bash dynamic_SimplerEnv/scripts/univla/run_dynisafety_eval.sh [options]
#
# Options:
#   --task          pick_coke_can | move_near | drawer   (default: pick_coke_can)
#   --level         0 | 1 | 2 | 3 | all                 (default: 1; all=1,2,3)
#   --ckpt          Passed as --ckpt (default: ignored)
#   --server-host   UNIVLA server (default: 127.0.0.1 or env UNIVLA_SERVER_HOST)
#   --server-port   (default: 8020)
#   --env-name      Conda env for Simpler eval (default: simpler_env)
#   Plus any extra args are forwarded to scripts/run_dynisafety.sh
set -euo pipefail

TASK="pick_coke_can"
LEVEL="0"
CKPT="ignored"
SERVER_HOST="${UNIVLA_SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${UNIVLA_SERVER_PORT:-8020}"
CONDA_ENV_NAME="simpler_env"
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)          shift; TASK="$1"; shift ;;
        --level)         shift; LEVEL="$1"; shift ;;
        --ckpt)          shift; CKPT="$1"; shift ;;
        --server-host)   shift; SERVER_HOST="$1"; shift ;;
        --server-port)   shift; SERVER_PORT="$1"; shift ;;
        --env-name)      shift; CONDA_ENV_NAME="$1"; shift ;;
        *)
            EXTRA+=("$1"); shift ;;
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

cd "${SIMPLER_ROOT}"
exec bash scripts/run_dynisafety.sh \
    --model univla \
    --ckpt "${CKPT}" \
    --task "${TASK}" \
    --level "${LEVEL}" \
    "${EXTRA[@]}"
