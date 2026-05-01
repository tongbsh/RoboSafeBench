#!/bin/bash
# ============================================================
# Run DyniSafety-LIBERO benchmark with the SpatialVLA policy.
#
# Usage:
#   bash dynisafety_libero/scripts/spatialVLA/run_dynisafety_eval.sh [OPTIONS]
#
# Options:
#   --task-suite  <NAME>   Task suite name.
#                          Choices: safelibero_spatial | safelibero_object |
#                                   safelibero_goal | safelibero_long
#                          Default: safelibero_spatial
#   --unnorm-key  <KEY>    Action un-normalisation key.  If omitted, auto-derived
#                          from --task-suite (recommended — see mapping below).
#   --server-host <HOST>   SpatialVLA server host.   Default: localhost
#   --server-port <PORT>   SpatialVLA server port.   Default: 8112
#   --env-name    <NAME>   Conda environment.  Default: openvla-oft
#
# Task suite → correct unnorm-key auto-mapping
# (processor statistics use "_no_noops/1.0.0", NOT the bare dataset name)
#   safelibero_spatial  →  libero_spatial_no_noops/1.0.0
#   safelibero_object   →  libero_object_no_noops/1.0.0
#   safelibero_goal     →  libero_goal_no_noops/1.0.0
#   safelibero_long     →  libero_10_no_noops/1.0.0
#
# Quickstart (--unnorm-key auto-derived from --task-suite):
#   # Terminal 1 — start server with suite auto-config:
#   bash dynisafety_libero/scripts/spatialVLA/start_spatialvla_server.sh \
#       --suite safelibero_spatial
#
#   # Terminal 2 — run eval (unnorm-key auto-derived):
#   bash dynisafety_libero/scripts/spatialVLA/run_dynisafety_eval.sh \
#       --task-suite safelibero_spatial
# ============================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Suite → unnorm-key mapping (mirrors start_spatialvla_server.sh)
# ---------------------------------------------------------------------------
suite_to_unnorm_key() {
    local suite="${1#safe}"   # strip "safe" prefix: safelibero_* → libero_*
    case "${suite}" in
        libero_spatial) echo "libero_spatial_no_noops/1.0.0" ;;
        libero_object)  echo "libero_object_no_noops/1.0.0"  ;;
        libero_goal)    echo "libero_goal_no_noops/1.0.0"    ;;
        libero_10|libero_long) echo "libero_10_no_noops/1.0.0" ;;
        libero_90)      echo "bridge_orig/1.0.0"              ;;
        *)              echo "" ;;   # unknown — fall through to server default
    esac
}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
TASK_SUITE="safelibero_spatial"
SERVER_HOST="localhost"
SERVER_PORT=8112
UNNORM_KEY=""
CONDA_ENV_NAME="openvla-oft"
_UNNORM_EXPLICIT=0

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task-suite)   TASK_SUITE="$2";           shift 2 ;;
        --server-host)  SERVER_HOST="$2";           shift 2 ;;
        --server-port)  SERVER_PORT="$2";           shift 2 ;;
        --unnorm-key)   UNNORM_KEY="$2"; _UNNORM_EXPLICIT=1; shift 2 ;;
        --env-name)     CONDA_ENV_NAME="$2";        shift 2 ;;
        -h|--help)
            sed -n '3,45p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
done

# Auto-derive unnorm-key from task suite if not explicitly set
if [[ ${_UNNORM_EXPLICIT} -eq 0 ]]; then
    UNNORM_KEY="$(suite_to_unnorm_key "${TASK_SUITE}")"
fi

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
# Build eval command
# ---------------------------------------------------------------------------
echo "============================================================"
echo "DyniSafety-LIBERO benchmark (SpatialVLA)"
echo "  Task suite  : ${TASK_SUITE}"
echo "  Server      : ${SERVER_HOST}:${SERVER_PORT}"
echo "  Unnorm key  : ${UNNORM_KEY:-server-default (auto)}"
echo "  Conda env   : ${CONDA_ENV_NAME}"
echo "============================================================"

cd "${PROJ_ROOT}"

EVAL_CMD=(
    python dynisafety_libero/run_dynisafety_eval.py
    --model_family    spatialvla
    --server_host     "${SERVER_HOST}"
    --server_port     "${SERVER_PORT}"
    --task_suite_name "${TASK_SUITE}"
)

# Pass unnorm_key via environment variable so the policy client picks it up
# without needing a new CLI flag in run_dynisafety_eval.py.
if [[ -n "${UNNORM_KEY}" ]]; then
    export SPATIALVLA_UNNORM_KEY="${UNNORM_KEY}"
fi

exec "${EVAL_CMD[@]}"
