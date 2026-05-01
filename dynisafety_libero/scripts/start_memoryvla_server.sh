#!/bin/bash
# ============================================================
# Start the MemoryVLA FastAPI inference server.
#
# This script must be run BEFORE launching the DyniSafety-LIBERO
# or DyniSafety-SimplerEnv benchmark with --model_family memoryvla.
#
# Usage:
#   bash dynisafety_libero/scripts/start_memoryvla_server.sh [OPTIONS]
#
# Options:
#   --model-path PATH      Local .pt checkpoint path or HuggingFace repo ID.
#                          (e.g. shihao1895/memvla-libero-spatial)
#   --unnorm-key  KEY      Action un-normalisation key from dataset_statistics.json.
#                          (e.g. libero_spatial_no_noops; default: auto-detect)
#   --port        PORT     Server port (default: 8111)
#   --env-name    NAME     Conda environment with MemoryVLA deps (default: memvla)
#
# Examples:
#   # Using a HuggingFace checkpoint (auto-download):
#   bash dynisafety_libero/scripts/start_memoryvla_server.sh \
#       --model-path shihao1895/memvla-libero-spatial \
#       --unnorm-key libero_spatial_no_noops
#
#   # Using a local checkpoint:
#   bash dynisafety_libero/scripts/start_memoryvla_server.sh \
#       --model-path /path/to/memvla-run/checkpoints/step-150000.pt \
#       --unnorm-key libero_spatial_no_noops
#
# After the server prints "MemoryVLA server ready", open a second terminal
# and run the benchmark:
#
#   conda activate openvla-oft
#   python dynisafety_libero/run_dynisafety_eval.py \
#       --model_family memoryvla \
#       --server_host localhost \
#       --server_port 8111 \
#       --task_suite_name safelibero_spatial
# ============================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL_PATH=""
UNNORM_KEY=""
PORT=8111
CONDA_ENV_NAME="memvla"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)  MODEL_PATH="$2";     shift 2 ;;
        --unnorm-key)  UNNORM_KEY="$2";     shift 2 ;;
        --port)        PORT="$2";            shift 2 ;;
        --env-name)    CONDA_ENV_NAME="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,40p' "$0"
            exit 0 ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage."
            exit 1 ;;
    esac
done

if [[ -z "${MODEL_PATH}" ]]; then
    echo "ERROR: --model-path is required."
    exit 1
fi

# ---------------------------------------------------------------------------
# Locate MemoryVLA server script
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SERVER_SCRIPT="${PROJ_ROOT}/vlaModelHome/MemoryVLA/server.py"

if [[ ! -f "${SERVER_SCRIPT}" ]]; then
    echo "ERROR: Server script not found at '${SERVER_SCRIPT}'."
    exit 1
fi

# ---------------------------------------------------------------------------
# Activate conda environment
# ---------------------------------------------------------------------------
set +u
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "${CONDA_PREFIX:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------
CMD=(python "${SERVER_SCRIPT}" --model-path "${MODEL_PATH}" --port "${PORT}")
if [[ -n "${UNNORM_KEY}" ]]; then
    CMD+=(--unnorm-key "${UNNORM_KEY}")
fi

echo "============================================================"
echo "MemoryVLA Inference Server"
echo "  Model      : ${MODEL_PATH}"
echo "  Unnorm key : ${UNNORM_KEY:-auto-detect}"
echo "  Port       : ${PORT}"
echo "  Conda env  : ${CONDA_ENV_NAME}"
echo "============================================================"

exec "${CMD[@]}"
