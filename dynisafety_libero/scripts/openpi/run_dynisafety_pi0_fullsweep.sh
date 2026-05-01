#!/usr/bin/env bash
# ============================================================
# Full sweep: DyniSafety-LIBERO × pi0 (openpi server)
#
# 遍历所有 task suite × dynamic level (0-3) × task id (0-3)
# 共 4 suite × 4 level × 4 task = 64 次评估，默认保存视频。
#
# Usage:
#   bash dynisafety_libero/scripts/openpi/run_dynisafety_pi0_fullsweep.sh [OPTIONS]
#
# Optional:
#   --server-host      <HOST>   pi0 server host. Default: localhost
#   --server-port      <PORT>   pi0 server port. Default: 8000
#   --open-loop        <N>      Actions executed per model chunk. Default: 5
#   --open-loop-steps  <N>      Alias of --open-loop
#   --num-trials       <N>      Episodes per task. Default: 50
#   --arm-speed        <SPEED>  slow | medium | fast. Default: medium
#   --env-name         <NAME>   Conda environment. Default: openvla-oft
#   --dry-run                   Print commands without executing them
#
# Make sure the pi0 server is running first:
#   bash dynisafety_libero/scripts/openpi/start_pi0_server.sh
# ============================================================
export CUDA_VISIBLE_DEVICES=0

set -euo pipefail

# ---- defaults -------------------------------------------------------
SERVER_HOST="localhost"
SERVER_PORT="8000"
OPEN_LOOP_STEPS="5"
NUM_TRIALS="50"
ARM_SPEED="medium"
CONDA_ENV_NAME="openvla-oft"
DRY_RUN="false"

# ---- sweep axes -----------------------------------------------------
ALL_SUITES=(
    safelibero_spatial
    safelibero_object
    safelibero_goal
    safelibero_long
)
ALL_LEVELS=( 0 1 2 3)
ALL_TASK_IDS=( 0 1 2 3)

# ---- arg parsing ----------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --server-host)      SERVER_HOST="$2";      shift 2 ;;
        --server-port)      SERVER_PORT="$2";      shift 2 ;;
        --open-loop|--open-loop-steps)
                            OPEN_LOOP_STEPS="$2";  shift 2 ;;
        --num-trials)       NUM_TRIALS="$2";       shift 2 ;;
        --arm-speed)        ARM_SPEED="$2";        shift 2 ;;
        --env-name)         CONDA_ENV_NAME="$2";   shift 2 ;;
        --dry-run)          DRY_RUN="true";        shift 1 ;;
        -h|--help)
            sed -n '3,24p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
done

# ---- paths ----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
EVAL_PY="dynisafety_libero/run_dynisafety_eval.py"

# ---- conda ----------------------------------------------------------
set +u
# shellcheck disable=SC1091
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "${CONDA_PREFIX:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

# ---- summary header -------------------------------------------------
TOTAL=$(( ${#ALL_SUITES[@]} * ${#ALL_LEVELS[@]} * ${#ALL_TASK_IDS[@]} ))

echo "============================================================"
echo "DyniSafety-LIBERO Full Sweep (pi0 / openpi server)"
echo "  Model family    : pi0"
echo "  Suites          : ${ALL_SUITES[*]}"
echo "  Levels          : ${ALL_LEVELS[*]}"
echo "  Task IDs        : ${ALL_TASK_IDS[*]}"
echo "  Total runs      : ${TOTAL}"
echo "  Server          : ${SERVER_HOST}:${SERVER_PORT}"
echo "  Open-loop       : ${OPEN_LOOP_STEPS}"
echo "  Trials/task     : ${NUM_TRIALS}"
echo "  Arm speed       : ${ARM_SPEED}"
echo "  Save video      : True (always)"
echo "  Conda env       : ${CONDA_ENV_NAME}"
if [[ "${DRY_RUN}" == "true" ]]; then
    echo "  *** DRY RUN - commands will be printed, not executed ***"
fi
echo "============================================================"

cd "${PROJ_ROOT}"

# ---- sweep ----------------------------------------------------------
RUN_IDX=0
FAILED=()

for SUITE in "${ALL_SUITES[@]}"; do
    for LEVEL in "${ALL_LEVELS[@]}"; do
        for TASK_ID in "${ALL_TASK_IDS[@]}"; do
            RUN_IDX=$(( RUN_IDX + 1 ))
            TAG="[${RUN_IDX}/${TOTAL}] suite=${SUITE} level=${LEVEL} task=${TASK_ID}"

            CMD=(
                python "${EVAL_PY}"
                    --model_family "pi0"
                    --server_host "${SERVER_HOST}"
                    --server_port "${SERVER_PORT}"
                    --task_suite_name "${SUITE}"
                    --dynamic_level "${LEVEL}"
                    --num_trials_per_task "${NUM_TRIALS}"
                    --num_open_loop_steps "${OPEN_LOOP_STEPS}"
                    --arm_speed "${ARM_SPEED}"
                    --task_id_override "${TASK_ID}"
                    --save_video "True"
            )

            echo ""
            echo "============================================================"
            echo "${TAG}"
            echo "CMD: ${CMD[*]}"
            echo "============================================================"

            if [[ "${DRY_RUN}" == "true" ]]; then
                continue
            fi

            if "${CMD[@]}"; then
                echo "${TAG} - DONE"
            else
                EXIT_CODE=$?
                echo "${TAG} - FAILED (exit ${EXIT_CODE})" >&2
                FAILED+=("${TAG}")
            fi
        done
    done
done

# ---- final report ---------------------------------------------------
echo ""
echo "============================================================"
echo "Sweep complete: ${RUN_IDX} runs attempted."
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "All runs succeeded."
else
    echo "Failed runs (${#FAILED[@]}):"
    for F in "${FAILED[@]}"; do
        echo "  - ${F}"
    done
fi
echo "============================================================"
