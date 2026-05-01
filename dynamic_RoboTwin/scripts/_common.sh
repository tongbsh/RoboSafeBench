#!/usr/bin/env bash
#
# Common environment + path setup for DyniSafety-RoboTwin shell scripts.
#
# Source this from a per-model script:
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "${SCRIPT_DIR}/../_common.sh"
#
# After sourcing, the following environment variables are available:
#   PROJ_ROOT       - dynamic_RoboTwin/
#   ROBOTWIN_ROOT   - dynamic_RoboTwin/RoboTwin/
#   WORKSPACE_ROOT  - dynamic_safty_bench/
#   CONDA_ROOT      - resolved conda base (may be overridden via _CONDA_ROOT)
#
# It also exports the same NCCL/CUDA flags that the upstream
# InternVLA-A1 RoboTwin scripts rely on, so the two stacks behave
# identically wrt. distributed init.

set -euo pipefail

if [[ -n "${_CONDA_ROOT:-}" ]]; then
    CONDA_ROOT="${_CONDA_ROOT}"
elif [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    CONDA_ROOT="$("${CONDA_EXE}" info --base)"
elif command -v conda >/dev/null 2>&1; then
    CONDA_ROOT="$(conda info --base)"
else
    echo "[ERROR] Cannot determine conda root."
    echo "[ERROR] Please ensure 'conda' is in PATH or export _CONDA_ROOT."
    exit 1
fi

if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    echo "[ERROR] conda.sh not found under ${CONDA_ROOT}"
    exit 1
fi

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

export PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROBOTWIN_ROOT="${PROJ_ROOT}/RoboTwin"
export WORKSPACE_ROOT="$(cd "${PROJ_ROOT}/.." && pwd)"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-6379}"

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
