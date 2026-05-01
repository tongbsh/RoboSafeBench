#!/bin/bash
# ============================================================
# Start the SpatialVLA FastAPI inference server.
#
# This script must be run BEFORE launching the DyniSafety-LIBERO
# or Source-LIBERO benchmark with --model_family spatialvla.
#
# Usage:
#   bash dynisafety_libero/scripts/spatialVLA/start_spatialvla_server.sh [OPTIONS]
#
# Options:
#   --suite      SUITE     Auto-configure LoRA path and unnorm-key for a task suite.
#                          Supported suite names (bare LIBERO or safelibero prefix both work):
#                            libero_spatial  / safelibero_spatial
#                            libero_object   / safelibero_object
#                            libero_goal     / safelibero_goal
#                            libero_10       / libero_long / safelibero_long
#                            libero_90       (base model, no LoRA)
#   --model-path PATH      Base model path.
#                          Default: download_checkpoints/spatialVLA/spatialvla-4b-224-pt
#   --lora-path  PATH      LoRA adapter directory (overrides --suite LoRA).
#   --unnorm-key KEY       Un-normalisation key (overrides --suite key).
#   --port       PORT      Server port (default: 8112)
#   --env-name   NAME      Conda environment (default: spatialvla)
#
# Suite → LoRA checkpoint & unnorm-key mapping
# (processor statistics use "_no_noops/1.0.0" suffix, NOT the bare dataset name)
#   libero_spatial / safelibero_spatial  →  spatialvla-lora-libero-spatial/checkpoint-16550  |  libero_spatial_no_noops/1.0.0
#   libero_object  / safelibero_object   →  spatialvla-lora-libero-object/checkpoint-20930   |  libero_object_no_noops/1.0.0
#   libero_goal    / safelibero_goal     →  spatialvla-lora-libero-goal/checkpoint-16260     |  libero_goal_no_noops/1.0.0
#   libero_10 / libero_long / safelibero_long → spatialvla-lora-libero-long/checkpoint-31710  |  libero_10_no_noops/1.0.0
#   libero_90                            →  (base model only, no LoRA)                       |  bridge_orig/1.0.0
#
# Quick-start examples:
#   # Auto mode — just specify the suite:
#   bash dynisafety_libero/scripts/spatialVLA/start_spatialvla_server.sh \
#       --suite libero_spatial
#
#   bash dynisafety_libero/scripts/spatialVLA/start_spatialvla_server.sh \
#       --suite safelibero_object
#
#   # Manual override — specify paths explicitly:
#   bash dynisafety_libero/scripts/spatialVLA/start_spatialvla_server.sh \
#       --lora-path download_checkpoints/spatialVLA/spatialvla-lora-libero-spatial/checkpoint-16550 \
#       --unnorm-key libero_spatial_no_noops/1.0.0
#
#   # Base model only (zero-shot):
#   bash dynisafety_libero/scripts/spatialVLA/start_spatialvla_server.sh \
#       --unnorm-key bridge_orig/1.0.0
#
# After the server prints "SpatialVLA server ready", open a second terminal and run:
#   bash dynisafety_libero/scripts/spatialVLA/run_dynisafety_eval.sh   --task-suite safelibero_spatial
#   bash dynisafety_libero/scripts/spatialVLA/source_libero_eval/run_libero_eval.sh --task-suite libero_spatial
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"   # dynamic_safty_bench/

# ---------------------------------------------------------------------------
# Defaults (empty means "not set by user")
# ---------------------------------------------------------------------------
SUITE="safelibero_spatial"
MODEL_PATH=""
LORA_PATH=""
UNNORM_KEY=""
PORT=8112
CONDA_ENV_NAME="spatialvla"

# Flags to track which options were explicitly set by the user
_LORA_EXPLICIT=0
_UNNORM_EXPLICIT=0

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite)       SUITE="$2";          shift 2 ;;
        --model-path)  MODEL_PATH="$2";     shift 2 ;;
        --lora-path)   LORA_PATH="$2";      _LORA_EXPLICIT=1;  shift 2 ;;
        --unnorm-key)  UNNORM_KEY="$2";     _UNNORM_EXPLICIT=1; shift 2 ;;
        --port)        PORT="$2";           shift 2 ;;
        --env-name)    CONDA_ENV_NAME="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,60p' "$0"
            exit 0 ;;
        *)
            echo "ERROR: Unknown option: $1"
            echo "Run with --help for usage."
            exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Suite → LoRA path & unnorm-key auto-mapping
# Explicit --lora-path / --unnorm-key always override the suite mapping.
# ---------------------------------------------------------------------------
CKPT_BASE="${PROJ_ROOT}/download_checkpoints/spatialVLA"

resolve_suite() {
    local suite="$1"
    # Strip "safe" prefix so both "libero_spatial" and "safelibero_spatial" work
    # safelibero_spatial → libero_spatial;  libero_spatial → libero_spatial (unchanged)
    suite="${suite#safe}"

    case "${suite}" in
        libero_spatial)
            echo "${CKPT_BASE}/spatialvla-lora-libero-spatial/checkpoint-16550"
            echo "libero_spatial_no_noops/1.0.0"
            ;;
        libero_object)
            echo "${CKPT_BASE}/spatialvla-lora-libero-object/checkpoint-20930"
            echo "libero_object_no_noops/1.0.0"
            ;;
        libero_goal)
            echo "${CKPT_BASE}/spatialvla-lora-libero-goal/checkpoint-16260"
            echo "libero_goal_no_noops/1.0.0"
            ;;
        libero_10|libero_long)
            echo "${CKPT_BASE}/spatialvla-lora-libero-long/checkpoint-31710"
            echo "libero_10_no_noops/1.0.0"
            ;;
        libero_90)
            echo ""   # no LoRA for libero_90
            echo "bridge_orig/1.0.0"
            ;;
        *)
            echo "ERROR: Unknown suite '${suite}'." >&2
            echo "Supported suites: libero_spatial, libero_object, libero_goal," >&2
            echo "                  libero_10, libero_long, libero_90" >&2
            echo "(You can also use the safelibero_ prefix, e.g. safelibero_spatial)" >&2
            exit 1
            ;;
    esac
}

if [[ -n "${SUITE}" ]]; then
    mapfile -t _SUITE_VALS < <(resolve_suite "${SUITE}")
    SUITE_LORA="${_SUITE_VALS[0]}"
    SUITE_UNNORM="${_SUITE_VALS[1]}"

    [[ ${_LORA_EXPLICIT}  -eq 0 ]] && LORA_PATH="${SUITE_LORA}"
    [[ ${_UNNORM_EXPLICIT} -eq 0 ]] && UNNORM_KEY="${SUITE_UNNORM}"
fi

# ---------------------------------------------------------------------------
# Apply defaults for paths not set by user or suite
# ---------------------------------------------------------------------------
[[ -z "${MODEL_PATH}" ]] && MODEL_PATH="${CKPT_BASE}/spatialvla-4b-224-pt"

# Resolve relative paths relative to PROJ_ROOT
[[ "${MODEL_PATH}" != /* ]] && MODEL_PATH="${PROJ_ROOT}/${MODEL_PATH}"
[[ -n "${LORA_PATH}" && "${LORA_PATH}" != /* ]] && LORA_PATH="${PROJ_ROOT}/${LORA_PATH}"

# ---------------------------------------------------------------------------
# Validate paths
# ---------------------------------------------------------------------------
SERVER_SCRIPT="${PROJ_ROOT}/vlaModelHome/SpatialVLA/server.py"

if [[ ! -f "${SERVER_SCRIPT}" ]]; then
    echo "ERROR: Server script not found at '${SERVER_SCRIPT}'."
    exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: Base model path not found: '${MODEL_PATH}'"
    exit 1
fi

if [[ -n "${LORA_PATH}" && ! -d "${LORA_PATH}" ]]; then
    echo "ERROR: LoRA path not found: '${LORA_PATH}'"
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
# Build & run server command
# ---------------------------------------------------------------------------
CMD=(python "${SERVER_SCRIPT}" --model-path "${MODEL_PATH}" --port "${PORT}")
[[ -n "${LORA_PATH}"  ]] && CMD+=(--lora-path  "${LORA_PATH}")
[[ -n "${UNNORM_KEY}" ]] && CMD+=(--unnorm-key "${UNNORM_KEY}")

echo "============================================================"
echo "SpatialVLA Inference Server"
[[ -n "${SUITE}" ]] && echo "  Suite      : ${SUITE}"
echo "  Base model : ${MODEL_PATH}"
echo "  LoRA path  : ${LORA_PATH:-none}"
echo "  Unnorm key : ${UNNORM_KEY:-auto-detect}"
echo "  Port       : ${PORT}"
echo "  Conda env  : ${CONDA_ENV_NAME}"
echo "============================================================"

exec "${CMD[@]}"
