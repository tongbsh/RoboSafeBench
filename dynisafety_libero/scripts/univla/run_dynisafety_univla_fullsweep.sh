#!/usr/bin/env bash
# ============================================================
# Full sweep: DyniSafety-LIBERO × UniVLA
#
# 遍历所有 task suite × dynamic level (0-3) × task id (0-3)
# 共 4 suite × 4 level × 4 task = 64 次评估，默认保存视频。
#
# UniVLA 通常按 suite 使用不同 server 权重，因此本脚本默认给四个 suite
# 分配不同端口，并为日志自动附带对应的 checkpoint/action decoder 提示。
#
# Usage:
#   bash dynisafety_libero/scripts/univla/run_dynisafety_univla_fullsweep.sh [OPTIONS]
#
# Optional:
#   --suites           <CSV>    Comma-separated suite filter:
#                                spatial,object,goal,long or full names
#                                Default: all 4 suites
#   --server-host      <HOST>   UniVLA server host for all suites. Default: localhost
#   --server-port      <PORT>   Use one shared port for all suites
#   --port-spatial     <PORT>   Port for safelibero_spatial. Default: 8020
#   --port-object      <PORT>   Port for safelibero_object.  Default: 8021
#   --port-goal        <PORT>   Port for safelibero_goal.    Default: 8022
#   --port-long        <PORT>   Port for safelibero_long.    Default: 8023
#   --weights-base     <DIR>    Base dir for per-suite UniVLA checkpoints
#                                Default: download_checkpoints/univla/sft-libero
#   --ckpt-spatial     <DIR>    Log-only checkpoint hint for safelibero_spatial
#   --ckpt-object      <DIR>    Log-only checkpoint hint for safelibero_object
#   --ckpt-goal        <DIR>    Log-only checkpoint hint for safelibero_goal
#   --ckpt-long        <DIR>    Log-only checkpoint hint for safelibero_long
#   --decoder-spatial  <PATH>   Log-only action decoder hint for spatial
#   --decoder-object   <PATH>   Log-only action decoder hint for object
#   --decoder-goal     <PATH>   Log-only action decoder hint for goal
#   --decoder-long     <PATH>   Log-only action decoder hint for long
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

# ---- helpers --------------------------------------------------------
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
PORT_SPATIAL="8020"
PORT_OBJECT="8021"
PORT_GOAL="8022"
PORT_LONG="8023"
WEIGHTS_BASE="download_checkpoints/univla/sft-libero"
CKPT_SPATIAL=""
CKPT_OBJECT=""
CKPT_GOAL=""
CKPT_LONG=""
DECODER_SPATIAL=""
DECODER_OBJECT=""
DECODER_GOAL=""
DECODER_LONG=""
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
        --server-port)
            PORT_SPATIAL="$2"
            PORT_OBJECT="$2"
            PORT_GOAL="$2"
            PORT_LONG="$2"
            shift 2
            ;;
        --port-spatial)     PORT_SPATIAL="$2";     shift 2 ;;
        --port-object)      PORT_OBJECT="$2";      shift 2 ;;
        --port-goal)        PORT_GOAL="$2";        shift 2 ;;
        --port-long)        PORT_LONG="$2";        shift 2 ;;
        --weights-base)     WEIGHTS_BASE="$2";     shift 2 ;;
        --ckpt-spatial)     CKPT_SPATIAL="$2";     shift 2 ;;
        --ckpt-object)      CKPT_OBJECT="$2";      shift 2 ;;
        --ckpt-goal)        CKPT_GOAL="$2";        shift 2 ;;
        --ckpt-long)        CKPT_LONG="$2";        shift 2 ;;
        --decoder-spatial)  DECODER_SPATIAL="$2";  shift 2 ;;
        --decoder-object)   DECODER_OBJECT="$2";   shift 2 ;;
        --decoder-goal)     DECODER_GOAL="$2";     shift 2 ;;
        --decoder-long)     DECODER_LONG="$2";     shift 2 ;;
        --num-trials)       NUM_TRIALS="$2";       shift 2 ;;
        --arm-speed)        ARM_SPEED="$2";        shift 2 ;;
        --env-name)         CONDA_ENV_NAME="$2";   shift 2 ;;
        --dry-run)          DRY_RUN="true";        shift 1 ;;
        -h|--help)
            sed -n '3,37p' "$0"
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
EVAL_PY="dynisafety_libero/scripts/univla/run_dynisafety_univla_eval.py"

if [[ "${WEIGHTS_BASE}" != /* ]]; then
    WEIGHTS_BASE="${PROJ_ROOT}/${WEIGHTS_BASE}"
fi

if [[ -z "${CKPT_SPATIAL}" ]]; then
    CKPT_SPATIAL="${WEIGHTS_BASE}/univla-libero-spatial"
fi
if [[ -z "${CKPT_OBJECT}" ]]; then
    CKPT_OBJECT="${WEIGHTS_BASE}/univla-libero-object"
fi
if [[ -z "${CKPT_GOAL}" ]]; then
    CKPT_GOAL="${WEIGHTS_BASE}/univla-libero-goal"
fi
if [[ -z "${CKPT_LONG}" ]]; then
    CKPT_LONG="${WEIGHTS_BASE}/univla-libero-10"
fi

if [[ -z "${DECODER_SPATIAL}" ]]; then
    DECODER_SPATIAL="${CKPT_SPATIAL}/action_decoder.pt"
fi
if [[ -z "${DECODER_OBJECT}" ]]; then
    DECODER_OBJECT="${CKPT_OBJECT}/action_decoder.pt"
fi
if [[ -z "${DECODER_GOAL}" ]]; then
    DECODER_GOAL="${CKPT_GOAL}/action_decoder.pt"
fi
if [[ -z "${DECODER_LONG}" ]]; then
    DECODER_LONG="${CKPT_LONG}/action_decoder.pt"
fi

declare -A SUITE_PORT=(
    [safelibero_spatial]="${PORT_SPATIAL}"
    [safelibero_object]="${PORT_OBJECT}"
    [safelibero_goal]="${PORT_GOAL}"
    [safelibero_long]="${PORT_LONG}"
)

declare -A SUITE_CHECKPOINT=(
    [safelibero_spatial]="${CKPT_SPATIAL}"
    [safelibero_object]="${CKPT_OBJECT}"
    [safelibero_goal]="${CKPT_GOAL}"
    [safelibero_long]="${CKPT_LONG}"
)

declare -A SUITE_ACTION_DECODER=(
    [safelibero_spatial]="${DECODER_SPATIAL}"
    [safelibero_object]="${DECODER_OBJECT}"
    [safelibero_goal]="${DECODER_GOAL}"
    [safelibero_long]="${DECODER_LONG}"
)

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
echo "DyniSafety-LIBERO Full Sweep (UniVLA)"
echo "  Model family      : univla"
echo "  Suites            : ${ALL_SUITES[*]}"
echo "  Levels            : ${ALL_LEVELS[*]}"
echo "  Task IDs          : ${ALL_TASK_IDS[*]}"
echo "  Total runs        : ${TOTAL}"
echo "  Server host       : ${SERVER_HOST}"
echo "  Suite ports       :"
echo "    spatial -> ${SUITE_PORT[safelibero_spatial]}"
echo "    object  -> ${SUITE_PORT[safelibero_object]}"
echo "    goal    -> ${SUITE_PORT[safelibero_goal]}"
echo "    long    -> ${SUITE_PORT[safelibero_long]}"
echo "  Log-only ckpts    :"
echo "    spatial -> ${SUITE_CHECKPOINT[safelibero_spatial]}"
echo "    object  -> ${SUITE_CHECKPOINT[safelibero_object]}"
echo "    goal    -> ${SUITE_CHECKPOINT[safelibero_goal]}"
echo "    long    -> ${SUITE_CHECKPOINT[safelibero_long]}"
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
    SERVER_PORT="${SUITE_PORT[${SUITE}]}"
    PRETRAINED_CHECKPOINT="${SUITE_CHECKPOINT[${SUITE}]}"
    ACTION_DECODER_PATH="${SUITE_ACTION_DECODER[${SUITE}]}"

    EXTRA_ARGS=()
    if [[ -n "${PRETRAINED_CHECKPOINT}" ]]; then
        EXTRA_ARGS+=(--pretrained_checkpoint "${PRETRAINED_CHECKPOINT}")
    fi
    if [[ -n "${ACTION_DECODER_PATH}" ]]; then
        EXTRA_ARGS+=(--action_decoder_path "${ACTION_DECODER_PATH}")
    fi

    for LEVEL in "${ALL_LEVELS[@]}"; do
        for TASK_ID in "${ALL_TASK_IDS[@]}"; do
            RUN_IDX=$(( RUN_IDX + 1 ))
            TAG="[${RUN_IDX}/${TOTAL}] suite=${SUITE} level=${LEVEL} task=${TASK_ID} port=${SERVER_PORT}"

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
