#!/bin/bash
# ============================================================
# Run X-VLA evaluation on the original LIBERO benchmark.
#
# Usage:
#   bash dynisafety_libero/scripts/x-vla/source_libero_eval/run_libero_eval.sh [OPTIONS]
#
# Options:
#   --task-suite    <NAME>   LIBERO task suite.
#                            Choices: libero_spatial | libero_object |
#                                     libero_goal | libero_10 | libero_90
#                            Default: libero_spatial
#   --num-trials    <N>      Rollouts per task (default: 10)
#   --open-loop     <N>      (Deprecated; kept for CLI compatibility. X-VLA
#                            executes the full returned action chunk before
#                            re-querying, matching the original libero_client.py.)
#   --domain-id     <ID>     X-VLA domain ID for LIBERO (default: 3)
#   --diff-steps    <N>      Diffusion denoising steps per inference call
#                            (default: 10)
#   --server-host   <HOST>   X-VLA server host (default: localhost)
#   --server-port   <PORT>   X-VLA server port (default: 8010)
#   --no-video               Disable video saving
#   --env-name      <NAME>   Conda env with LIBERO installed (default: openvla)
#   --run-note      <NOTE>   Optional note appended to run ID
#
# Full two-terminal quickstart:
#
#   # Terminal 1 — start X-VLA server (XVLA conda env):
#   bash dynisafety_libero/scripts/x-vla/start_xvla_server.sh
#
#   # Terminal 2 — run evaluation (openvla conda env):
#   bash dynisafety_libero/scripts/x-vla/source_libero_eval/run_libero_eval.sh \
#       --task-suite libero_spatial
#
# Outputs are saved to:
#   dynisafety_libero/outputs/xvla/source_libero/videos/
#   dynisafety_libero/outputs/xvla/source_libero/logs/
# ============================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
TASK_SUITE="libero_spatial"
NUM_TRIALS=10
OPEN_LOOP_STEPS=10
DOMAIN_ID=3
DIFF_STEPS=10
SERVER_HOST="localhost"
SERVER_PORT=8010
SAVE_VIDEO="--save-video"
CONDA_ENV_NAME="openvla"
RUN_NOTE=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task-suite)   TASK_SUITE="$2";        shift 2 ;;
        --num-trials)   NUM_TRIALS="$2";         shift 2 ;;
        --open-loop)    OPEN_LOOP_STEPS="$2";    shift 2 ;;
        --domain-id)    DOMAIN_ID="$2";          shift 2 ;;
        --diff-steps)   DIFF_STEPS="$2";         shift 2 ;;
        --server-host)  SERVER_HOST="$2";        shift 2 ;;
        --server-port)  SERVER_PORT="$2";        shift 2 ;;
        --no-video)     SAVE_VIDEO="--no-save-video"; shift ;;
        --env-name)     CONDA_ENV_NAME="$2";     shift 2 ;;
        --run-note)     RUN_NOTE="$2";           shift 2 ;;
        -h|--help)
            sed -n '3,55p' "$0"
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
# Locate project root and eval script
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"   # dynamic_safty_bench/
EVAL_SCRIPT="${SCRIPT_DIR}/run_libero_xvla_eval.py"

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
    echo "ERROR: Eval script not found at '${EVAL_SCRIPT}'"
    exit 1
fi

# ---------------------------------------------------------------------------
# Activate conda environment (openvla, where LIBERO is installed)
# ---------------------------------------------------------------------------
set +u
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "${CONDA_PREFIX:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

# ---------------------------------------------------------------------------
# Print run summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "X-VLA  ×  Source LIBERO Evaluation"
echo "  Task suite  : ${TASK_SUITE}"
echo "  Trials/task : ${NUM_TRIALS}"
#echo "  Open-loop N : ${OPEN_LOOP_STEPS}"  # deprecated
echo "  Domain ID   : ${DOMAIN_ID}"
echo "  Diff. steps : ${DIFF_STEPS}"
echo "  Server      : ${SERVER_HOST}:${SERVER_PORT}"
echo "  Conda env   : ${CONDA_ENV_NAME}"
echo "  Outputs     : dynisafety_libero/outputs/xvla/source_libero/"
echo "============================================================"

# ---------------------------------------------------------------------------
# Build and run evaluation command
# ---------------------------------------------------------------------------
EVAL_CMD=(
    python "${EVAL_SCRIPT}"
    --task-suite   "${TASK_SUITE}"
    --num-trials   "${NUM_TRIALS}"
    --domain-id    "${DOMAIN_ID}"
    --steps        "${DIFF_STEPS}"
    --server-host  "${SERVER_HOST}"
    --server-port  "${SERVER_PORT}"
    "${SAVE_VIDEO}"
)

[[ -n "${RUN_NOTE}" ]] && EVAL_CMD+=(--run-note "${RUN_NOTE}")

cd "${PROJ_ROOT}"
exec "${EVAL_CMD[@]}"
