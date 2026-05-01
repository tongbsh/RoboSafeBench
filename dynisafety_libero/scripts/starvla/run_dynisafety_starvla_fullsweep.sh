#!/usr/bin/env bash
# ============================================================
# Full sweep: DyniSafety-LIBERO × StarVLA
#
# 遍历所有 task suite × dynamic level (0-3) × task id (0-3)
# 共 4 suite × 4 level × 4 task = 64 次评估，默认保存视频。
#
# Usage:
#   bash dynisafety_libero/scripts/starvla/run_dynisafety_starvla_fullsweep.sh [OPTIONS]
#
# Optional:
#   --server-host      <HOST>   StarVLA server host. Default: localhost
#   --server-port      <PORT>   StarVLA server port. Default: 10093
#   --ckpt-path        <PATH>   Same .pt path as the server uses
#                                Default:
#                                download_checkpoints/starVLA/
#                                starvla-libero-4in1/checkpoints/steps_50000_pytorch_model.pt
#   --unnorm-key       <KEY>    Optional dataset_statistics.json key
#   --num-trials       <N>      Episodes per task. Default: 50
#   --arm-speed        <SPEED>  slow | medium | fast. Default: medium
#   --env-name         <NAME>   Conda env. Default: openvla-oft
#   --dry-run                   Print commands without executing them
#
# Make sure the StarVLA server is running first:
#   bash dynisafety_libero/scripts/starvla/start_starvla_server.sh --ckpt-path ...
# ============================================================

set -euo pipefail

# ---- defaults -------------------------------------------------------
SERVER_HOST="localhost"
SERVER_PORT="10093"
CKPT_PATH="download_checkpoints/starVLA/starvla-libero-4in1/checkpoints/steps_50000_pytorch_model.pt"
UNNORM_KEY=""
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
ALL_LEVELS=(0 1 2 3)
ALL_TASK_IDS=(0 1 2 3)

# ---- arg parsing ----------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --server-host)      SERVER_HOST="$2";      shift 2 ;;
        --server-port)      SERVER_PORT="$2";      shift 2 ;;
        --ckpt-path)        CKPT_PATH="$2";        shift 2 ;;
        --unnorm-key)       UNNORM_KEY="$2";       shift 2 ;;
        --num-trials)       NUM_TRIALS="$2";       shift 2 ;;
        --arm-speed)        ARM_SPEED="$2";        shift 2 ;;
        --env-name)         CONDA_ENV_NAME="$2";   shift 2 ;;
        --dry-run)          DRY_RUN="true";        shift 1 ;;
        -h|--help)
            sed -n '3,28p' "$0"
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
EVAL_PY="dynisafety_libero/scripts/starvla/run_dynisafety_starvla_eval.py"

if [[ "${CKPT_PATH}" != /* ]]; then
    CKPT_PATH="${PROJ_ROOT}/${CKPT_PATH}"
fi

if [[ "${DRY_RUN}" != "true" && ! -f "${CKPT_PATH}" ]]; then
    echo "ERROR: checkpoint file not found: ${CKPT_PATH}"
    echo "Pass --ckpt-path to the same .pt the StarVLA server uses (for client-side config/stats only)."
    exit 1
fi

# ---- conda ----------------------------------------------------------
set +u
# shellcheck disable=SC1091
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "${CONDA_PREFIX:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

export STARVLA_SERVER_HOST="${SERVER_HOST}"
export STARVLA_SERVER_PORT="${SERVER_PORT}"

# ---- extra args -----------------------------------------------------
EXTRA_ARGS=()
if [[ -n "${UNNORM_KEY}" ]]; then
    EXTRA_ARGS+=(--unnorm_key "${UNNORM_KEY}")
fi

# ---- summary header -------------------------------------------------
TOTAL=$(( ${#ALL_SUITES[@]} * ${#ALL_LEVELS[@]} * ${#ALL_TASK_IDS[@]} ))

echo "============================================================"
echo "DyniSafety-LIBERO Full Sweep (StarVLA)"
echo "  Model family     : starvla"
echo "  Suites           : ${ALL_SUITES[*]}"
echo "  Levels           : ${ALL_LEVELS[*]}"
echo "  Task IDs         : ${ALL_TASK_IDS[*]}"
echo "  Total runs       : ${TOTAL}"
echo "  Server           : ${SERVER_HOST}:${SERVER_PORT}"
echo "  Checkpoint       : ${CKPT_PATH}"
echo "  Unnorm key       : ${UNNORM_KEY:-<auto-resolve>}"
echo "  Trials/task      : ${NUM_TRIALS}"
echo "  Arm speed        : ${ARM_SPEED}"
echo "  Save video       : True (always)"
echo "  Conda env        : ${CONDA_ENV_NAME}"
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
                    --server_host "${SERVER_HOST}"
                    --server_port "${SERVER_PORT}"
                    --pretrained_checkpoint "${CKPT_PATH}"
                    --task_suite_name "${SUITE}"
                    --dynamic_level "${LEVEL}"
                    --num_trials_per_task "${NUM_TRIALS}"
                    --arm_speed "${ARM_SPEED}"
                    --task_id_override "${TASK_ID}"
                    --save_video "True"
                    "${EXTRA_ARGS[@]}"
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
