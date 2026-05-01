#!/bin/bash
# ============================================================
# Start the openpi π₀ / π₀.₅ WebSocket policy server.
#
# This script must be run BEFORE launching the DyniSafety-LIBERO
# benchmark with --model_family pi0 or --model_family pi05.
#
# Usage:
#   bash dynisafety_libero/scripts/start_pi0_server.sh [OPTIONS]
#
# Options:
#   --checkpoint-dir <DIR>   Path to the trained checkpoint directory.
#                            Default: auto-download from GCS
#                            (gs://openpi-assets/checkpoints/pi0_libero).
#   --config <NAME>          openpi training config name.
#                            Default: pi0_libero
#   --port <PORT>            WebSocket server port.
#                            Default: 8000
#   --env-name <NAME>        Conda environment that contains openpi + uv.
#                            Default: openpi
#
# Examples:
#   # Use pre-downloaded checkpoint:
#   bash dynisafety_libero/scripts/start_pi0_server.sh \
#       --checkpoint-dir /root/autodl-tmp/code/dynamic_safty_bench/vlaModelHome/openpi/checkpoints/pi0_libero/my_exp/20000
#
#   # Auto-download from GCS (requires gsutil / internet):
#   bash dynisafety_libero/scripts/start_pi0_server.sh
#
#   # Custom port:
#   bash dynisafety_libero/scripts/start_pi0_server.sh --port 8001
#
# After the server prints "Serving on ws://...", open a second terminal
# and run the benchmark:
#
#   conda activate openvla-oft
#   python dynisafety_libero/run_dynisafety_eval.py \
#       --model_family pi0 \
#       --server_host localhost \
#       --server_port 8000 \
#       --task_suite_name safelibero_spatial \
#       --num_open_loop_steps 8
# ============================================================

set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CHECKPOINT_DIR=""
CONFIG_NAME="pi0_libero"
PORT=8000
CONDA_ENV_NAME="openpi"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
        --config)         CONFIG_NAME="$2";    shift 2 ;;
        --port)           PORT="$2";           shift 2 ;;
        --env-name)       CONDA_ENV_NAME="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,50p' "$0"   # print usage from header
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Locate the openpi repository
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"   # dynamic_safty_bench/
OPENPI_DIR="${PROJ_ROOT}/vlaModelHome/openpi"

if [[ ! -d "${OPENPI_DIR}" ]]; then
    echo "ERROR: openpi directory not found at '${OPENPI_DIR}'."
    echo "  Clone it with: git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git vlaModelHome/openpi"
    exit 1
fi

# ---------------------------------------------------------------------------
# Activate the openpi conda environment
# ---------------------------------------------------------------------------
set +u
# shellcheck disable=SC1091
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "${CONDA_PREFIX:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

# ---------------------------------------------------------------------------
# Build the serve_policy.py command
# ---------------------------------------------------------------------------
cd "${OPENPI_DIR}"

if [[ -n "${CHECKPOINT_DIR}" ]]; then
    # Use a specific local checkpoint directory.
    if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
        echo "ERROR: Checkpoint directory not found: '${CHECKPOINT_DIR}'"
        exit 1
    fi
    echo "============================================================"
    echo "openpi policy server"
    echo "  Config     : ${CONFIG_NAME}"
    echo "  Checkpoint : ${CHECKPOINT_DIR}"
    echo "  Port       : ${PORT}"
    echo "  Conda env  : ${CONDA_ENV_NAME}"
    echo "============================================================"
    python scripts/serve_policy.py \
        --port           "${PORT}" \
        policy:checkpoint \
        --policy.config  "${CONFIG_NAME}" \
        --policy.dir     "${CHECKPOINT_DIR}"
else
    # No explicit checkpoint — auto-download pi0_libero from GCS.
    # NOTE: --env LIBERO in serve_policy.py maps to pi05_libero, so we
    # explicitly specify pi0_libero via policy:checkpoint instead.
    GCS_CKPT="gs://openpi-assets/checkpoints/pi0_libero"
    echo "============================================================"
    echo "openpi policy server (auto-download checkpoint from GCS)"
    echo "  Config     : ${CONFIG_NAME}"
    echo "  Checkpoint : ${GCS_CKPT}"
    echo "  Port       : ${PORT}"
    echo "  Conda env  : ${CONDA_ENV_NAME}"
    echo "  Cache dir  : ${OPENPI_DATA_HOME:-~/.cache/openpi}"
    echo "============================================================"
    python scripts/serve_policy.py \
        --port           "${PORT}" \
        policy:checkpoint \
        --policy.config  "${CONFIG_NAME}" \
        --policy.dir     "${GCS_CKPT}"
fi
