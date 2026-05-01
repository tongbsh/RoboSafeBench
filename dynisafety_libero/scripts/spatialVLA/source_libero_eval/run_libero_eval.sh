#!/bin/bash
# ============================================================
# Run SpatialVLA evaluation on the original LIBERO benchmark.
#
# Usage:
#   bash dynisafety_libero/scripts/spatialVLA/source_libero_eval/run_libero_eval.sh [OPTIONS]
#
# Options:
#   --task-suite    <NAME>   LIBERO task suite.
#                            Choices: libero_spatial | libero_object |
#                                     libero_goal | libero_10 | libero_90
#                            Default: libero_spatial
#   --unnorm-key    <KEY>    Action un-normalisation key (e.g. libero_spatial).
#                            Leave empty to use the server's default.
#   --num-trials    <N>      Rollouts per task (default: 10)
#   --open-loop     <N>      Open-loop steps before re-querying (default: 1)
#   --server-host   <HOST>   SpatialVLA server host (default: localhost)
#   --server-port   <PORT>   SpatialVLA server port (default: 8112)
#   --no-video               Disable video saving
#   --env-name      <NAME>   Conda env with LIBERO installed (default: openvla)
#   --run-note      <NOTE>   Optional note appended to run ID
#
# The unnorm-key is auto-derived from --task-suite (no manual config needed).
# Task suite → unnorm-key auto-mapping:
#   libero_spatial  →  libero_spatial_no_noops/1.0.0
#   libero_object   →  libero_object_no_noops/1.0.0
#   libero_goal     →  libero_goal_no_noops/1.0.0
#   libero_10       →  libero_10_no_noops/1.0.0
#   libero_90       →  bridge_orig/1.0.0  (base model, no LoRA)
#
# Full two-terminal quickstart:
#
#   # Terminal 1 — start SpatialVLA server (spatialvla env), suite auto-configures LoRA:
#   bash dynisafety_libero/scripts/spatialVLA/start_spatialvla_server.sh \
#       --suite libero_spatial
#
#   # Terminal 2 — run evaluation (openvla env), unnorm-key auto-derived:
#   bash dynisafety_libero/scripts/spatialVLA/source_libero_eval/run_libero_eval.sh \
#       --task-suite libero_spatial
#
# Outputs are saved to:
#   dynisafety_libero/outputs/spatialvla/source_libero/videos/
#   dynisafety_libero/outputs/spatialvla/source_libero/logs/
# ============================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Suite → unnorm-key auto-mapping (mirrors start_spatialvla_server.sh)
# ---------------------------------------------------------------------------
suite_to_unnorm_key() {
    local suite="${1#safe}"   # strip "safe" prefix: safelibero_* → libero_*
    case "${suite}" in
        libero_spatial) echo "libero_spatial_no_noops/1.0.0" ;;
        libero_object)  echo "libero_object_no_noops/1.0.0"  ;;
        libero_goal)    echo "libero_goal_no_noops/1.0.0"    ;;
        libero_10|libero_long) echo "libero_10_no_noops/1.0.0" ;;
        libero_90)      echo "bridge_orig/1.0.0"              ;;
        *)              echo "" ;;
    esac
}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
TASK_SUITE="libero_spatial"
UNNORM_KEY=""
NUM_TRIALS=10
OPEN_LOOP_STEPS=8   # matches action_chunk_size=8 of the fine-tuned LoRA models
SERVER_HOST="localhost"
SERVER_PORT=8112
SAVE_VIDEO="--save-video"
CONDA_ENV_NAME="openvla"
RUN_NOTE=""
_UNNORM_EXPLICIT=0

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task-suite)    TASK_SUITE="$2";                     shift 2 ;;
        --unnorm-key)    UNNORM_KEY="$2"; _UNNORM_EXPLICIT=1; shift 2 ;;
        --num-trials)    NUM_TRIALS="$2";                     shift 2 ;;
        --open-loop)     OPEN_LOOP_STEPS="$2";               shift 2 ;;
        --server-host)   SERVER_HOST="$2";                    shift 2 ;;
        --server-port)   SERVER_PORT="$2";                    shift 2 ;;
        --no-video)      SAVE_VIDEO="--no-save-video";        shift   ;;
        --env-name)      CONDA_ENV_NAME="$2";                 shift 2 ;;
        --run-note)      RUN_NOTE="$2";                       shift 2 ;;
        -h|--help)
            sed -n '3,62p' "$0"
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
# Locate project root
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"   # dynamic_safty_bench/
EVAL_SCRIPT="${SCRIPT_DIR}/run_libero_spatialvla_eval.py"

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
    echo "ERROR: Eval script not found at '${EVAL_SCRIPT}'"
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
echo "============================================================"
echo "SpatialVLA  ×  Source LIBERO Evaluation"
echo "  Task suite  : ${TASK_SUITE}"
echo "  Unnorm key  : ${UNNORM_KEY:-server-default (auto)}"
echo "  Trials/task : ${NUM_TRIALS}"
echo "  Open-loop N : ${OPEN_LOOP_STEPS}"
echo "  Server      : ${SERVER_HOST}:${SERVER_PORT}"
echo "  Conda env   : ${CONDA_ENV_NAME}"
echo "  Outputs     : dynisafety_libero/outputs/spatialvla/source_libero/"
echo "============================================================"

EVAL_CMD=(
    python "${EVAL_SCRIPT}"
    --task-suite           "${TASK_SUITE}"
    --num-trials           "${NUM_TRIALS}"
    --num-open-loop-steps  "${OPEN_LOOP_STEPS}"
    --server-host          "${SERVER_HOST}"
    --server-port          "${SERVER_PORT}"
    "${SAVE_VIDEO}"
)

[[ -n "${UNNORM_KEY}" ]] && EVAL_CMD+=(--unnorm-key "${UNNORM_KEY}")
[[ -n "${RUN_NOTE}"   ]] && EVAL_CMD+=(--run-note   "${RUN_NOTE}")

cd "${PROJ_ROOT}"
exec "${EVAL_CMD[@]}"
