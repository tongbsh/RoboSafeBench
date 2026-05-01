#!/usr/bin/env bash
# ============================================================
# Parallel orchestration: OpenVLA-OFT suite full sweeps
#
# 并行执行每个 suite 的 full sweep 子任务。
# 默认选中 4 个 suite 时，会并行启动 4 个子进程；每个子进程内部继续扫描
# level 0-3 × task 0-3，因此总覆盖仍是 64 组。
#
# Usage:
#   bash dynisafety_libero/scripts/openvla-oft/run_dynisafety_openvla_oft_fullsweep_parallel.sh [OPTIONS]
#
# Optional:
#   --suites           <CSV>    Comma-separated suite filter:
#                                spatial,object,goal,long or full names
#                                Default: all 4 suites
#   --checkpoint-all   <PATH>   Use one shared HF id / local checkpoint for all suites
#   --ckpt-spatial     <PATH>   Checkpoint for safelibero_spatial
#   --ckpt-object      <PATH>   Checkpoint for safelibero_object
#   --ckpt-goal        <PATH>   Checkpoint for safelibero_goal
#   --ckpt-long        <PATH>   Checkpoint for safelibero_long
#   --openvla-root     <DIR>    Override OPENVLA_ROOT for all suite workers
#   --num-trials       <N>      Episodes per task. Default: 50
#   --open-loop-steps  <N>      Actions executed per model chunk. Default: 8
#   --arm-speed        <SPEED>  slow | medium | fast. Default: medium
#   --env-name         <NAME>   Conda env. Default: openvla-oft
#   --cuda-all         <LIST>   Set CUDA_VISIBLE_DEVICES for all suite workers
#   --cuda-spatial     <LIST>   CUDA_VISIBLE_DEVICES for spatial worker
#   --cuda-object      <LIST>   CUDA_VISIBLE_DEVICES for object worker
#   --cuda-goal        <LIST>   CUDA_VISIBLE_DEVICES for goal worker
#   --cuda-long        <LIST>   CUDA_VISIBLE_DEVICES for long worker
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

suite_short_name() {
    case "$1" in
        safelibero_spatial) echo "spatial" ;;
        safelibero_object)  echo "object" ;;
        safelibero_goal)    echo "goal" ;;
        safelibero_long)    echo "long" ;;
        *) return 1 ;;
    esac
}

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    for suite in "${RUNNING_SUITES[@]}"; do
        pid="${EVAL_PID_BY_SUITE[${suite}]:-}"
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
    exit "${exit_code}"
}

# ---- defaults -------------------------------------------------------
CHECKPOINT_ALL=""
CKPT_SPATIAL="moojink/openvla-7b-oft-finetuned-libero-spatial"
CKPT_OBJECT="moojink/openvla-7b-oft-finetuned-libero-object"
CKPT_GOAL="moojink/openvla-7b-oft-finetuned-libero-goal"
CKPT_LONG="moojink/openvla-7b-oft-finetuned-libero-10"
OPENVLA_ROOT_ARG=""
NUM_TRIALS="50"
OPEN_LOOP_STEPS="8"
ARM_SPEED="medium"
CONDA_ENV_NAME="openvla-oft"
CUDA_ALL=""
CUDA_SPATIAL=""
CUDA_OBJECT=""
CUDA_GOAL=""
CUDA_LONG=""
DRY_RUN="false"
SUITES_CSV=""

DEFAULT_SUITES=(
    safelibero_spatial
    safelibero_object
    safelibero_goal
    safelibero_long
)
ALL_SUITES=("${DEFAULT_SUITES[@]}")

# ---- arg parsing ----------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --suites)           SUITES_CSV="$2";        shift 2 ;;
        --checkpoint-all)   CHECKPOINT_ALL="$2";    shift 2 ;;
        --ckpt-spatial)     CKPT_SPATIAL="$2";      shift 2 ;;
        --ckpt-object)      CKPT_OBJECT="$2";       shift 2 ;;
        --ckpt-goal)        CKPT_GOAL="$2";         shift 2 ;;
        --ckpt-long)        CKPT_LONG="$2";         shift 2 ;;
        --openvla-root)     OPENVLA_ROOT_ARG="$2";  shift 2 ;;
        --num-trials)       NUM_TRIALS="$2";        shift 2 ;;
        --open-loop-steps)  OPEN_LOOP_STEPS="$2";   shift 2 ;;
        --arm-speed)        ARM_SPEED="$2";         shift 2 ;;
        --env-name)         CONDA_ENV_NAME="$2";    shift 2 ;;
        --cuda-all)         CUDA_ALL="$2";          shift 2 ;;
        --cuda-spatial)     CUDA_SPATIAL="$2";      shift 2 ;;
        --cuda-object)      CUDA_OBJECT="$2";       shift 2 ;;
        --cuda-goal)        CUDA_GOAL="$2";         shift 2 ;;
        --cuda-long)        CUDA_LONG="$2";         shift 2 ;;
        --dry-run)          DRY_RUN="true";         shift 1 ;;
        -h|--help)
            sed -n '3,30p' "$0"
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
FULLSWEEP_SH="${SCRIPT_DIR}/run_dynisafety_openvla_oft_fullsweep.sh"
RUN_STAMP="$(date +%Y_%m_%d-%H_%M_%S)"
RUN_LOG_DIR="${PROJ_ROOT}/dynisafety_libero/outputs/openvla_oft/logs/fullsweep_parallel/${RUN_STAMP}"
mkdir -p "${RUN_LOG_DIR}"

declare -A SUITE_CUDA=(
    [safelibero_spatial]="${CUDA_SPATIAL:-${CUDA_ALL}}"
    [safelibero_object]="${CUDA_OBJECT:-${CUDA_ALL}}"
    [safelibero_goal]="${CUDA_GOAL:-${CUDA_ALL}}"
    [safelibero_long]="${CUDA_LONG:-${CUDA_ALL}}"
)

RUNNING_SUITES=()
declare -A EVAL_PID_BY_SUITE=()
declare -A EVAL_LOG_BY_SUITE=()

trap cleanup EXIT INT TERM

echo "============================================================"
echo "OpenVLA-OFT Full Sweep Parallel"
echo "  Suites           : ${ALL_SUITES[*]}"
echo "  Checkpoint all   : ${CHECKPOINT_ALL:-<per-suite defaults>}"
echo "  open-loop        : ${OPEN_LOOP_STEPS}"
echo "  Trials/task      : ${NUM_TRIALS}"
echo "  Arm speed        : ${ARM_SPEED}"
echo "  OPENVLA_ROOT     : ${OPENVLA_ROOT_ARG:-<default>}"
echo "  Conda env        : ${CONDA_ENV_NAME}"
echo "  Logs dir         : ${RUN_LOG_DIR}"
if [[ -n "${CUDA_ALL}" ]]; then
    echo "  CUDA all         : ${CUDA_ALL}"
fi
if [[ "${DRY_RUN}" == "true" ]]; then
    echo "  *** DRY RUN - commands will be printed, not executed ***"
fi
echo "============================================================"

for SUITE in "${ALL_SUITES[@]}"; do
    SHORT_SUITE="$(suite_short_name "${SUITE}")"
    SUITE_LOG="${RUN_LOG_DIR}/eval_${SHORT_SUITE}.log"
    EVAL_LOG_BY_SUITE["${SUITE}"]="${SUITE_LOG}"

    EVAL_CMD=(
        bash "${FULLSWEEP_SH}"
            --suites "${SUITE}"
            --num-trials "${NUM_TRIALS}"
            --open-loop-steps "${OPEN_LOOP_STEPS}"
            --arm-speed "${ARM_SPEED}"
            --env-name "${CONDA_ENV_NAME}"
    )
    if [[ -n "${CHECKPOINT_ALL}" ]]; then
        EVAL_CMD+=(--checkpoint-all "${CHECKPOINT_ALL}")
    else
        EVAL_CMD+=(
            --ckpt-spatial "${CKPT_SPATIAL}"
            --ckpt-object "${CKPT_OBJECT}"
            --ckpt-goal "${CKPT_GOAL}"
            --ckpt-long "${CKPT_LONG}"
        )
    fi
    if [[ -n "${OPENVLA_ROOT_ARG}" ]]; then
        EVAL_CMD+=(--openvla-root "${OPENVLA_ROOT_ARG}")
    fi

    echo ""
    echo "============================================================"
    echo "Launch full sweep for ${SUITE}"
    echo "Eval log: ${SUITE_LOG}"
    if [[ -n "${SUITE_CUDA[${SUITE}]}" ]]; then
        echo "CUDA_VISIBLE_DEVICES=${SUITE_CUDA[${SUITE}]}"
    fi
    echo "CMD: ${EVAL_CMD[*]}"
    echo "============================================================"

    if [[ "${DRY_RUN}" == "true" ]]; then
        continue
    fi

    if [[ -n "${SUITE_CUDA[${SUITE}]}" ]]; then
        (
            export CUDA_VISIBLE_DEVICES="${SUITE_CUDA[${SUITE}]}"
            "${EVAL_CMD[@]}"
        ) > "${SUITE_LOG}" 2>&1 &
    else
        "${EVAL_CMD[@]}" > "${SUITE_LOG}" 2>&1 &
    fi
    PID=$!
    RUNNING_SUITES+=("${SUITE}")
    EVAL_PID_BY_SUITE["${SUITE}"]="${PID}"
done

if [[ "${DRY_RUN}" == "true" ]]; then
    exit 0
fi

FAILED=()
for SUITE in "${ALL_SUITES[@]}"; do
    PID="${EVAL_PID_BY_SUITE[${SUITE}]}"
    if wait "${PID}"; then
        echo "Full sweep finished: ${SUITE}"
    else
        EXIT_CODE=$?
        echo "Full sweep FAILED: ${SUITE} (exit ${EXIT_CODE})" >&2
        echo "Check eval log: ${EVAL_LOG_BY_SUITE[${SUITE}]}" >&2
        FAILED+=("${SUITE}")
    fi
done

echo ""
echo "============================================================"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "All selected OpenVLA-OFT suite sweeps succeeded."
else
    echo "Failed suite sweeps (${#FAILED[@]}):"
    for SUITE in "${FAILED[@]}"; do
        echo "  - ${SUITE}"
    done
fi
echo "Logs dir: ${RUN_LOG_DIR}"
echo "============================================================"

if [[ ${#FAILED[@]} -ne 0 ]]; then
    exit 1
fi
