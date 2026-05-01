#!/usr/bin/env bash
# ============================================================
# Full sweep: DyniSafety-LIBERO × X-VLA
#
# 遍历所选 task suite × dynamic level (0-3) × task id (0-3)
# 默认全量为 4 suite × 4 level × 4 task = 64 次评估，默认保存视频。
#
# Usage:
#   bash dynisafety_libero/scripts/x-vla/run_dynisafety_xvla_fullsweep.sh [OPTIONS]
#
# Optional:
#   --suites           <CSV>    Comma-separated suite filter:
#                                spatial,object,goal,long or full names
#                                Default: all 4 suites
#   --server-host      <HOST>   X-VLA server host. Default: localhost
#   --server-port      <PORT>   X-VLA server port. Default: 8010
#   --domain-id        <ID>     XVLA_DOMAIN_ID. Default: 3
#   --diff-steps       <N>      XVLA_STEPS. Default: 10
#   --num-trials       <N>      Episodes per task. Default: 50
#   --arm-speed        <SPEED>  slow | medium | fast. Default: medium
#   --env-name         <NAME>   Conda env. Default: openvla-oft
#   --dry-run                   Print commands without executing them
# ============================================================

set -euo pipefail

if (( BASH_VERSINFO[0] < 4 )); then
    echo "ERROR: bash 4+ required (current: ${BASH_VERSION})" >&2
    exit 1
fi

normalize_suite_name() {
    case "$1" in
        spatial|safelibero_spatial) echo "safelibero_spatial" ;;
        object|safelibero_object)   echo "safelibero_object" ;;
        goal|safelibero_goal)       echo "safelibero_goal" ;;
        long|safelibero_long)       echo "safelibero_long" ;;
        *)
            echo "ERROR: unknown suite '$1'. Use: spatial, object, goal, long," >&2
            echo "       or full names safelibero_spatial/object/goal/long." >&2
            return 1
            ;;
    esac
}

# ---- defaults -------------------------------------------------------
SERVER_HOST="localhost"
SERVER_PORT="8010"
DOMAIN_ID="3"
DIFF_STEPS="10"
NUM_TRIALS="50"
ARM_SPEED="medium"
CONDA_ENV_NAME="openvla-oft"
DRY_RUN="false"
SUITES_CSV=""

# ---- sweep axes -----------------------------------------------------
DEFAULT_SUITES=(
    safelibero_spatial
    safelibero_object
    safelibero_goal
    safelibero_long
)
ALL_SUITES=("${DEFAULT_SUITES[@]}")
ALL_LEVELS=(0 1 2 3)
ALL_TASK_IDS=(0 1 2 3)

# ---- arg parsing ----------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --suites)           SUITES_CSV="$2";       shift 2 ;;
        --server-host)      SERVER_HOST="$2";      shift 2 ;;
        --server-port)      SERVER_PORT="$2";      shift 2 ;;
        --domain-id)        DOMAIN_ID="$2";        shift 2 ;;
        --diff-steps)       DIFF_STEPS="$2";       shift 2 ;;
        --num-trials)       NUM_TRIALS="$2";       shift 2 ;;
        --arm-speed)        ARM_SPEED="$2";        shift 2 ;;
        --env-name)         CONDA_ENV_NAME="$2";   shift 2 ;;
        --dry-run)          DRY_RUN="true";        shift 1 ;;
        -h|--help)
            sed -n '3,23p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
done

if [[ -n "${SUITES_CSV}" ]]; then
    SUITES_CSV="${SUITES_CSV// /}"
    IFS=',' read -r -a SUITE_TOKENS <<< "${SUITES_CSV}"
    ALL_SUITES=()
    declare -A SEEN_SUITES=()
    for TOKEN in "${SUITE_TOKENS[@]}"; do
        [[ -z "${TOKEN}" ]] && continue
        SUITE="$(normalize_suite_name "${TOKEN}")" || exit 1
        if [[ -z "${SEEN_SUITES[${SUITE}]:-}" ]]; then
            ALL_SUITES+=("${SUITE}")
            SEEN_SUITES["${SUITE}"]=1
        fi
    done
    if [[ ${#ALL_SUITES[@]} -eq 0 ]]; then
        echo "ERROR: --suites resolved to an empty suite set." >&2
        exit 1
    fi
fi

# ---- paths ----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
EVAL_PY="dynisafety_libero/scripts/x-vla/run_dynisafety_xvla_eval.py"

# ---- conda ----------------------------------------------------------
set +u
# shellcheck disable=SC1091
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "${CONDA_PREFIX:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

export XVLA_SERVER_HOST="${SERVER_HOST}"
export XVLA_SERVER_PORT="${SERVER_PORT}"
export XVLA_DOMAIN_ID="${DOMAIN_ID}"
export XVLA_STEPS="${DIFF_STEPS}"

# ---- summary header -------------------------------------------------
TOTAL=$(( ${#ALL_SUITES[@]} * ${#ALL_LEVELS[@]} * ${#ALL_TASK_IDS[@]} ))

echo "============================================================"
echo "DyniSafety-LIBERO Full Sweep (X-VLA)"
echo "  Model family      : xvla"
echo "  Suites            : ${ALL_SUITES[*]}"
echo "  Levels            : ${ALL_LEVELS[*]}"
echo "  Task IDs          : ${ALL_TASK_IDS[*]}"
echo "  Total runs        : ${TOTAL}"
echo "  Server            : ${SERVER_HOST}:${SERVER_PORT}"
echo "  XVLA_DOMAIN_ID    : ${DOMAIN_ID}"
echo "  XVLA_STEPS        : ${DIFF_STEPS}"
echo "  Trials/task       : ${NUM_TRIALS}"
echo "  Arm speed         : ${ARM_SPEED}"
echo "  Save video        : True (always)"
echo "  Conda env         : ${CONDA_ENV_NAME}"
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
                    --task_suite_name "${SUITE}"
                    --dynamic_level "${LEVEL}"
                    --num_trials_per_task "${NUM_TRIALS}"
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
