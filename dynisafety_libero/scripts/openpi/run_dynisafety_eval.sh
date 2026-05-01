#!/bin/bash
# ============================================================
# Run DyniSafety-LIBERO benchmark with the π₀ policy.
#
# Usage:
#   bash dynisafety_libero/scripts/openpi/run_dynisafety_eval.sh [OPTIONS]
#
# Options:
#   --task-suite  <NAME>   Task suite name. Default: safelibero_spatial
#   --server-host <HOST>   pi0 server host.  Default: localhost
#   --server-port <PORT>   pi0 server port.  Default: 8000
#   --open-loop   <N>      Open-loop steps.  Default: 8
#
# Make sure the pi0 server is running first:
#   bash dynisafety_libero/scripts/openpi/start_pi0_server.sh
# ============================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
TASK_SUITE="safelibero_spatial"
SERVER_HOST="localhost"
SERVER_PORT=8000
OPEN_LOOP_STEPS=5
CONDA_ENV_NAME="openvla-oft"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task-suite)   TASK_SUITE="$2";       shift 2 ;;
        --server-host)  SERVER_HOST="$2";      shift 2 ;;
        --server-port)  SERVER_PORT="$2";      shift 2 ;;
        --open-loop)    OPEN_LOOP_STEPS="$2";  shift 2 ;;
        --env-name)     CONDA_ENV_NAME="$2";   shift 2 ;;
        -h|--help)
            sed -n '3,20p' "$0"
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
# Locate project root (works regardless of where the script is called from)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"   # dynamic_safty_bench/

# ---------------------------------------------------------------------------
# Activate the conda environment
# ---------------------------------------------------------------------------
set +u
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "${CONDA_PREFIX:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
echo "============================================================"
echo "DyniSafety-LIBERO benchmark (pi0)"
echo "  Task suite  : ${TASK_SUITE}"
echo "  Server      : ${SERVER_HOST}:${SERVER_PORT}"
echo "  Open-loop N : ${OPEN_LOOP_STEPS}"
echo "  Conda env   : ${CONDA_ENV_NAME}"
echo "============================================================"

cd "${PROJ_ROOT}"
python dynisafety_libero/run_dynisafety_eval.py \
    --model_family        pi0 \
    --server_host         "${SERVER_HOST}" \
    --server_port         "${SERVER_PORT}" \
    --task_suite_name     "${TASK_SUITE}" \
    --num_open_loop_steps "${OPEN_LOOP_STEPS}"
