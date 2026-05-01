#!/bin/bash
# ============================================================
# Start the openpi π₀ / π₀.₅ WebSocket policy server.
#
# Run this BEFORE launching DyniSafety-SimplerEnv with --model pi0 or pi05.
#
# Layout:
#   SIMPLER_ROOT   …/dynamic_SimplerEnv  (directory that contains scripts/)
#   BENCH_ROOT     parent of SIMPLER_ROOT (workspace with vlaModelHome/)
#   openpi repo    ${BENCH_ROOT}/vlaModelHome/openpi
#
# Usage:
#   bash dynamic_SimplerEnv/scripts/openpi/start_pi0_server.sh [OPTIONS]
#
# Options:
#   --checkpoint-dir <DIR>   Path to the trained checkpoint directory.
#                            (required unless relying on GCS auto-download)
#   --config <NAME>          openpi training config name.
#                            Common values: pi0_fast, pi0_base
#                            Default: pi0_fast
#   --port <PORT>            WebSocket server port. Default: 8000
#   --env-name <NAME>        Conda environment with openpi + uv. Default: openpi
#   --bench-root <DIR>       Override workspace root (default: parent of dynamic_SimplerEnv)
#
# Examples:
#   # Use a pre-downloaded checkpoint:
#   bash scripts/openpi/start_pi0_server.sh \
#       --checkpoint-dir /path/to/openpi/checkpoints/pi0_fast/my_exp/20000
#
#   # π₀.₅ on a different port:
#   bash scripts/openpi/start_pi0_server.sh \
#       --config pi0_fast --checkpoint-dir /path/to/pi05_ckpt --port 8001
#
# After the server prints "Serving on ws://...", open a second terminal and run:
#   OPENPI_SERVER_HOST=127.0.0.1 OPENPI_SERVER_PORT=8000 \
#   bash scripts/openpi/run_dynisafety_eval.sh --task pick_coke_can --level 1
# ============================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CHECKPOINT_DIR=""
CONFIG_NAME="pi0_fast"
PORT=8000
CONDA_ENV_NAME="openpi"
BENCH_ROOT_OVERRIDE=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
        --config)         CONFIG_NAME="$2";    shift 2 ;;
        --port)           PORT="$2";           shift 2 ;;
        --env-name)       CONDA_ENV_NAME="$2"; shift 2 ;;
        --bench-root)     BENCH_ROOT_OVERRIDE="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,43p' "$0"
            exit 0 ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage."
            exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Locate the openpi repository
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIMPLER_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"      # dynamic_SimplerEnv/
BENCH_ROOT="${BENCH_ROOT_OVERRIDE:-$(cd "${SIMPLER_ROOT}/.." && pwd)}"
OPENPI_DIR="${BENCH_ROOT}/vlaModelHome/openpi"

if [[ ! -d "${OPENPI_DIR}" ]]; then
    echo "ERROR: openpi directory not found at '${OPENPI_DIR}'."
    echo "  Clone it with:"
    echo "    git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git ${OPENPI_DIR}"
    echo "  Or override workspace root with: --bench-root /path/to/workspace"
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
# Build and run the serve_policy.py command
# ---------------------------------------------------------------------------
cd "${OPENPI_DIR}"

if [[ -n "${CHECKPOINT_DIR}" ]]; then
    if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
        echo "ERROR: Checkpoint directory not found: '${CHECKPOINT_DIR}'"
        exit 1
    fi
    echo "============================================================"
    echo "openpi policy server  [DyniSafety-SimplerEnv]"
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
    echo "ERROR: --checkpoint-dir is required."
    echo "  Provide the path to your trained openpi checkpoint directory."
    exit 1
fi
