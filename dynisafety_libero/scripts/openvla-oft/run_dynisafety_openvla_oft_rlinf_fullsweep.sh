#!/usr/bin/env bash
# ============================================================
# Full sweep: DyniSafety-LIBERO × RLinf OpenVLA-OFT
#
# 遍历所有 task suite × dynamic level (0-3) × task id (0-3)
# 共 4 suite × 4 level × 4 task = 64 次评估，默认保存视频。
#
# Usage:
#   bash dynisafety_libero/scripts/openvla-oft/run_dynisafety_openvla_oft_rlinf_fullsweep.sh [OPTIONS]
#
# Optional:
#   --suites           <CSV>    Comma-separated suite filter:
#                                spatial,object,goal,long or full names
#                                Default: all 4 suites
#   --ckpt-base        <DIR>    Root dir containing per-suite checkpoint folders
#                                Default: download_checkpoints/openvla-oft_RLinf
#                                Each suite maps to <ckpt-base>/RLinf-OpenVLAOFT-GRPO-LIBERO-{suite_suffix}
#   --unnorm-key       <KEY>    Override checkpoint unnorm_key resolution
#   --num-images       <N>      Number of images in input. Default: 1
#   --use-proprio               Enable proprio input (disabled by default)
#   --num-trials       <N>      Episodes per task. Default: 50
#   --open-loop-steps  <N>      Actions executed per model chunk. Default: 8
#   --arm-speed        <SPEED>  slow | medium | fast. Default: medium
#   --env-name         <NAME>   Conda environment. Default: openvla-oft
#   --dry-run                   Print commands without executing them
# ============================================================

export CUDA_VISIBLE_DEVICES=0

set -euo pipefail
# 关联数组需要 bash 4+
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
CKPT_BASE="download_checkpoints/openvla-oft_RLinf"
UNNORM_KEY=""
NUM_IMAGES="1"
USE_PROPRIO="False"
NUM_TRIALS="50"
OPEN_LOOP_STEPS="8"
ARM_SPEED="medium"
CONDA_ENV_NAME="openvla-oft"
DRY_RUN="false"
SUITES_CSV=""

# ---- sweep axes -----------------------------------------------------
DEFAULT_SUITES=(
    safelibero_spatial
    safelibero_object
    # safelibero_goal
    # safelibero_long
)
ALL_SUITES=("${DEFAULT_SUITES[@]}")
ALL_LEVELS=(0 1 2 3)
ALL_TASK_IDS=(0 1 2 3)

# ---- arg parsing ----------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --suites)          SUITES_CSV="$2";       shift 2 ;;
        --ckpt-base)       CKPT_BASE="$2";      shift 2 ;;
        --unnorm-key)      UNNORM_KEY="$2";     shift 2 ;;
        --num-images)      NUM_IMAGES="$2";     shift 2 ;;
        --use-proprio)     USE_PROPRIO="True";  shift 1 ;;
        --num-trials)      NUM_TRIALS="$2";     shift 2 ;;
        --open-loop-steps) OPEN_LOOP_STEPS="$2"; shift 2 ;;
        --arm-speed)       ARM_SPEED="$2";      shift 2 ;;
        --env-name)        CONDA_ENV_NAME="$2"; shift 2 ;;
        --dry-run)         DRY_RUN="true";      shift 1 ;;
        -h|--help)
            sed -n '3,34p' "$0"
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

# ---- per-suite checkpoint mapping (built after arg parsing so --ckpt-base takes effect) ----
# 每个 suite 有各自独立训练的权重，不可混用。
declare -A SUITE_CHECKPOINT=(
    [safelibero_spatial]="${CKPT_BASE}/RLinf-OpenVLAOFT-GRPO-LIBERO-spatial"
    [safelibero_object]="${CKPT_BASE}/RLinf-OpenVLAOFT-GRPO-LIBERO-object"
    [safelibero_goal]="${CKPT_BASE}/RLinf-OpenVLAOFT-GRPO-LIBERO-goal"
    [safelibero_long]="${CKPT_BASE}/RLinf-OpenVLAOFT-GRPO-LIBERO-long"
)

# ---- paths ----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
EVAL_PY="dynisafety_libero/scripts/openvla-oft/run_dynisafety_openvla_oft_rlinf_eval.py"

# ---- conda ----------------------------------------------------------
set +u
# shellcheck disable=SC1091
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "${CONDA_PREFIX:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

# ---- extra args -----------------------------------------------------
EXTRA_ARGS=()
if [[ -n "${UNNORM_KEY}" ]]; then
    EXTRA_ARGS+=(--unnorm_key "${UNNORM_KEY}")
fi

# ---- summary header -------------------------------------------------
TOTAL=$(( ${#ALL_SUITES[@]} * ${#ALL_LEVELS[@]} * ${#ALL_TASK_IDS[@]} ))

echo "============================================================"
echo "DyniSafety-LIBERO Full Sweep (RLinf OpenVLA-OFT)"
echo "  Suites        : ${ALL_SUITES[*]}"
echo "  Levels        : ${ALL_LEVELS[*]}"
echo "  Task IDs      : ${ALL_TASK_IDS[*]}"
echo "  Total runs    : ${TOTAL}"
echo "  Checkpoint base: ${CKPT_BASE}"
echo "    spatial -> ${SUITE_CHECKPOINT[safelibero_spatial]}"
echo "    object  -> ${SUITE_CHECKPOINT[safelibero_object]}"
echo "    goal    -> ${SUITE_CHECKPOINT[safelibero_goal]}"
echo "    long    -> ${SUITE_CHECKPOINT[safelibero_long]}"
echo "  Unnorm key    : ${UNNORM_KEY:-<auto-resolve>}"
echo "  Num images    : ${NUM_IMAGES}"
echo "  Use proprio   : ${USE_PROPRIO}"
echo "  Open-loop     : ${OPEN_LOOP_STEPS}"
echo "  Trials/task   : ${NUM_TRIALS}"
echo "  Arm speed     : ${ARM_SPEED}"
echo "  Save video    : True (always)"
echo "  Conda env     : ${CONDA_ENV_NAME}"
if [[ "${DRY_RUN}" == "true" ]]; then
    echo "  *** DRY RUN — commands will be printed, not executed ***"
fi
echo "============================================================"

cd "${PROJ_ROOT}"

# ---- sweep ----------------------------------------------------------
RUN_IDX=0
FAILED=()

for SUITE in "${ALL_SUITES[@]}"; do
    CHECKPOINT="${SUITE_CHECKPOINT[${SUITE}]}"
    for LEVEL in "${ALL_LEVELS[@]}"; do
        for TASK_ID in "${ALL_TASK_IDS[@]}"; do
            RUN_IDX=$(( RUN_IDX + 1 ))
            TAG="[${RUN_IDX}/${TOTAL}] suite=${SUITE} level=${LEVEL} task=${TASK_ID}"

            CMD=(
                python "${EVAL_PY}"
                    --task_suite_name  "${SUITE}"
                    --dynamic_level    "${LEVEL}"
                    --pretrained_checkpoint "${CHECKPOINT}"
                    --num_images_in_input   "${NUM_IMAGES}"
                    --use_proprio      "${USE_PROPRIO}"
                    --num_trials_per_task   "${NUM_TRIALS}"
                    --num_open_loop_steps   "${OPEN_LOOP_STEPS}"
                    --arm_speed        "${ARM_SPEED}"
                    --task_id_override "${TASK_ID}"
                    --save_video       "True"
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
                echo "${TAG} — DONE"
            else
                echo "${TAG} — FAILED (exit $?)" >&2
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
