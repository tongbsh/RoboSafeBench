#!/bin/bash
# ============================================================
# Run DyniSafety-LIBERO benchmark with the X-VLA policy.
#
# Usage:
#   bash dynisafety_libero/scripts/x-vla/run_dynisafety_eval.sh [OPTIONS]
#
# Options:
#   --task-suite    <NAME>   Task suite name.
#                            Choices: safelibero_spatial | safelibero_object |
#                                     safelibero_goal | safelibero_long
#                            Default: safelibero_spatial
#   --dynamic-level <N>      Disturbance level: 0=static 1=drop 2=slide 3=shared-reach 4=inspect
#                            Default: 1
#   --server-host   <HOST>   X-VLA server host.  Default: localhost
#   --server-port   <PORT>   X-VLA server port.  Default: 8010
#   --exec-horizon  <N>      Ignored (kept for CLI compatibility). Source-aligned X-VLA
#                            runs the full server chunk before re-querying.
#   --domain-id     <ID>     X-VLA domain ID (XVLA_DOMAIN_ID). Default: 3 (LIBERO)
#   --diff-steps    <N>      Diffusion denoising steps (XVLA_STEPS). Default: 10
#   --num-trials    <N>      Episodes per task. Default: 50
#   --arm-speed     <SPEED>  Human arm speed: slow | medium | fast. Default: medium
#   --task-id       <ID>     If set, evaluate only this task ID. Default: 0
#   --env-name      <NAME>   Conda environment.  Default: openvla-oft
#
# Prerequisites (Terminal 1 — start server in XVLA conda env):
#   bash dynisafety_libero/scripts/x-vla/start_xvla_server.sh
#
# Quickstart (Terminal 2 — run eval in openvla-oft conda env):
#   bash dynisafety_libero/scripts/x-vla/run_dynisafety_eval.sh \
#       --task-suite safelibero_spatial \
#       --dynamic-level 1
#
# Outputs saved to:
#   dynisafety_libero/outputs/xvla/logs/
#   dynisafety_libero/outputs/xvla/videos/
# ============================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
TASK_SUITE="safelibero_spatial"
DYNAMIC_LEVEL="2"
SERVER_HOST="localhost"
SERVER_PORT=8010
EXEC_HORIZON=30
DOMAIN_ID=3
DIFF_STEPS=10
NUM_TRIALS=50
ARM_SPEED="medium"
TASK_ID=0
CONDA_ENV_NAME="openvla-oft"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task-suite)    TASK_SUITE="$2";      shift 2 ;;
        --dynamic-level) DYNAMIC_LEVEL="$2";   shift 2 ;;
        --server-host)   SERVER_HOST="$2";     shift 2 ;;
        --server-port)   SERVER_PORT="$2";     shift 2 ;;
        --exec-horizon)  EXEC_HORIZON="$2";    shift 2 ;;
        --domain-id)     DOMAIN_ID="$2";       shift 2 ;;
        --diff-steps)    DIFF_STEPS="$2";      shift 2 ;;
        --num-trials)    NUM_TRIALS="$2";      shift 2 ;;
        --arm-speed)     ARM_SPEED="$2";       shift 2 ;;
        --task-id)       TASK_ID="$2";         shift 2 ;;
        --env-name)      CONDA_ENV_NAME="$2";  shift 2 ;;
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

# ---------------------------------------------------------------------------
# Locate project root
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"   # dynamic_safty_bench/

# ---------------------------------------------------------------------------
# Activate conda environment
# ---------------------------------------------------------------------------
set +u
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "${CONDA_PREFIX:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

# ---------------------------------------------------------------------------
# Export X-VLA server tuning parameters as environment variables.
# The XVLAPolicy reads these at construction time.
# ---------------------------------------------------------------------------
export XVLA_SERVER_HOST="${SERVER_HOST}"
export XVLA_SERVER_PORT="${SERVER_PORT}"
export XVLA_DOMAIN_ID="${DOMAIN_ID}"
export XVLA_STEPS="${DIFF_STEPS}"

# ---------------------------------------------------------------------------
# Print run summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "DyniSafety-LIBERO benchmark (X-VLA)"
echo "  Task suite    : ${TASK_SUITE}"
echo "  Dynamic level : ${DYNAMIC_LEVEL}"
echo "  Server        : ${SERVER_HOST}:${SERVER_PORT}"
echo "  Exec horizon  : ${EXEC_HORIZON} (ignored; full X-VLA chunk per query)"
echo "  Domain ID     : ${DOMAIN_ID}"
echo "  Diff. steps   : ${DIFF_STEPS}"
echo "  Trials/task   : ${NUM_TRIALS}"
echo "  Arm speed     : ${ARM_SPEED}"
echo "  Task ID       : ${TASK_ID}"
echo "  Conda env     : ${CONDA_ENV_NAME}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
cd "${PROJ_ROOT}"
exec python dynisafety_libero/scripts/x-vla/run_dynisafety_xvla_eval.py \
    --server_host         "${SERVER_HOST}" \
    --server_port         "${SERVER_PORT}" \
    --task_suite_name     "${TASK_SUITE}" \
    --dynamic_level       "${DYNAMIC_LEVEL}" \
    --num_trials_per_task "${NUM_TRIALS}" \
    --arm_speed           "${ARM_SPEED}" \
    --task_id_override    "${TASK_ID}"
