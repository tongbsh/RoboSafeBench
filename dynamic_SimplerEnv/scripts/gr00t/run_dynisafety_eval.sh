#!/bin/bash
# DyniSafety-SimplerEnv + GR00T (in-process; --ckpt-path required).
# Outputs: dynamic_SimplerEnv/outputs/gr00t/{logs,videos}/
set -euo pipefail

TASK="pick_coke_can"
LEVEL="1"
CKPT=""
CONDA_ENV_NAME="openvla-oft"
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)     shift; TASK="$1"; shift ;;
        --level)    shift; LEVEL="$1"; shift ;;
        --ckpt-path) shift; CKPT="$1"; shift ;;
        --env-name) shift; CONDA_ENV_NAME="$1"; shift ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

if [[ -z "${CKPT}" ]]; then
    echo "ERROR: --ckpt-path is required for gr00t"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIMPLER_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

set +u
# shellcheck disable=SC1091
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "${CONDA_PREFIX:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

cd "${SIMPLER_ROOT}"
exec bash scripts/run_dynisafety.sh \
    --model gr00t \
    --ckpt "${CKPT}" \
    --task "${TASK}" \
    --level "${LEVEL}" \
    "${EXTRA[@]}"
