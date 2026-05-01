#!/usr/bin/env bash
# ============================================================
# Serial fullsweep: DyniSafety-SimplerEnv × InternVLA-M1
# WidowX Bridge — first 3 subtasks (carrot, stack, spoon) × 4 dynamic levels.
#
# Total runs: 3 subtasks × 4 levels = 12.
#
# Prerequisite: the InternVLA-M1 WebSocket server is already running, e.g.:
#   bash dynamic_SimplerEnv/scripts/internvla_m1/start_internvla_m1_server.sh \
#       --ckpt-path download_checkpoints/InternVLA/InternVLA-M1-Pretrain-RT-1-Bridge/checkpoints/steps_50000_pytorch_model.pt
#
# Usage:
#   bash dynamic_SimplerEnv/scripts/internvla_m1/run_bridge_internvla_m1_fullsweep.sh [OPTIONS]
#
# Options:
#   --subtasks    <CSV>   carrot,stack,spoon              (default: carrot,stack,spoon)
#   --levels      <CSV>   Dynamic level filter            (default: 0,1,2,3)
#   --gpu         <ID>    CUDA device for sim/rendering   (default: 0)
#   --ckpt        <PATH>  Real .../checkpoints/*.pt path  (default: INTERNVLA_M1_CHECKPOINT or
#                                                          Pretrain-RT-1-Bridge default)
#   --server-host <HOST>  Sets INTERNVLA_M1_SERVER_HOST   (default: 127.0.0.1)
#   --server-port <PORT>  Sets INTERNVLA_M1_SERVER_PORT   (default: 10094)
#   --unnorm-key  <KEY>   Sets INTERNVLA_M1_UNNORM_KEY    (default: bridge_dataset)
#   --cfg-scale   <F>     Sets INTERNVLA_M1_CFG_SCALE     (default: 1.5)
#   --ddim-steps  <N>     Sets INTERNVLA_M1_NUM_DDIM_STEPS (default: 5)
#   --temp        <F>     Action ensemble temperature     (default: -0.8)
#   --output-root   DIR   Bench layout root               (default: ./outputs)
#   --output-layout NAME  bench | legacy                  (default: bench)
#   --log-root      DIR   Legacy nested log root          (default: results)
#   --run-id-note   STR   Optional log filename suffix
#   --env-name      NAME  Conda env                       (default: simpler_env)
#   --max-steps     N     --max-episode-steps for ALL bridge subtasks (default: 120, official InternVLA-M1 value)
#   --dry-run             Print commands without executing them
#   -h | --help
# ============================================================

set -euo pipefail

if (( BASH_VERSINFO[0] < 4 )); then
    echo "ERROR: bash 4+ required (current: ${BASH_VERSION})" >&2
    exit 1
fi

normalize_subtask_name() {
    case "$1" in
        carrot|bridge_carrot)     echo "bridge_carrot" ;;
        stack|bridge_stack)       echo "bridge_stack" ;;
        spoon|bridge_spoon)       echo "bridge_spoon" ;;
        eggplant|bridge_eggplant) echo "bridge_eggplant" ;;
        *)
            echo "ERROR: unknown subtask '$1'. Use carrot|stack|spoon|eggplant or bridge_*." >&2
            return 1
            ;;
    esac
}

# ---- defaults -------------------------------------------------------
SUBTASKS_CSV=""
LEVELS_CSV=""
GPU_ID="0"
CKPT="${INTERNVLA_M1_CHECKPOINT:-download_checkpoints/InternVLA/InternVLA-M1-Pretrain-RT-1-Bridge/checkpoints/steps_50000_pytorch_model.pt}"
SERVER_HOST="${INTERNVLA_M1_SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${INTERNVLA_M1_SERVER_PORT:-10094}"
UNNORM_KEY="${INTERNVLA_M1_UNNORM_KEY:-bridge_dataset}"
CFG_SCALE="${INTERNVLA_M1_CFG_SCALE:-1.5}"
DDIM_STEPS="${INTERNVLA_M1_NUM_DDIM_STEPS:-5}"
ACTION_ENSEMBLE_TEMP="-0.8"
OUTPUT_ROOT="./outputs"
OUTPUT_LAYOUT="bench"
LOG_ROOT="results"
RUN_ID_NOTE=""
CONDA_ENV_NAME="simpler_env"
DRY_RUN="false"
MAX_STEPS="${BRIDGE_MAX_EPISODE_STEPS:-120}"

DEFAULT_SUBTASKS=(bridge_carrot bridge_stack bridge_spoon)
DEFAULT_LEVELS=(0 1 2 3)

# ---- arg parsing ----------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --subtasks)      SUBTASKS_CSV="$2";      shift 2 ;;
        --levels)        LEVELS_CSV="$2";        shift 2 ;;
        --gpu)           GPU_ID="$2";            shift 2 ;;
        --ckpt)          CKPT="$2";              shift 2 ;;
        --server-host)   SERVER_HOST="$2";       shift 2 ;;
        --server-port)   SERVER_PORT="$2";       shift 2 ;;
        --unnorm-key)    UNNORM_KEY="$2";        shift 2 ;;
        --cfg-scale)     CFG_SCALE="$2";         shift 2 ;;
        --ddim-steps)    DDIM_STEPS="$2";        shift 2 ;;
        --temp)          ACTION_ENSEMBLE_TEMP="$2"; shift 2 ;;
        --output-root)   OUTPUT_ROOT="$2";       shift 2 ;;
        --output-layout) OUTPUT_LAYOUT="$2";     shift 2 ;;
        --log-root)      LOG_ROOT="$2";          shift 2 ;;
        --run-id-note)   RUN_ID_NOTE="$2";       shift 2 ;;
        --env-name)      CONDA_ENV_NAME="$2";    shift 2 ;;
        --max-steps)     MAX_STEPS="$2";         shift 2 ;;
        --dry-run)       DRY_RUN="true";         shift 1 ;;
        -h|--help)
            sed -n '3,35p' "$0"
            exit 0 ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage."
            exit 1 ;;
    esac
done

# ---- resolve subtasks / levels -------------------------------------
SUBTASKS=("${DEFAULT_SUBTASKS[@]}")
if [[ -n "${SUBTASKS_CSV}" ]]; then
    SUBTASKS_CSV="${SUBTASKS_CSV// /}"
    IFS=',' read -r -a SUB_TOKENS <<< "${SUBTASKS_CSV}"
    SUBTASKS=()
    declare -A SEEN_SUB=()
    for TOKEN in "${SUB_TOKENS[@]}"; do
        [[ -z "${TOKEN}" ]] && continue
        SUB="$(normalize_subtask_name "${TOKEN}")" || exit 1
        if [[ -z "${SEEN_SUB[${SUB}]:-}" ]]; then
            SUBTASKS+=("${SUB}")
            SEEN_SUB["${SUB}"]=1
        fi
    done
    if [[ ${#SUBTASKS[@]} -eq 0 ]]; then
        echo "ERROR: --subtasks resolved to an empty set." >&2
        exit 1
    fi
fi

LEVELS=("${DEFAULT_LEVELS[@]}")
if [[ -n "${LEVELS_CSV}" ]]; then
    LEVELS_CSV="${LEVELS_CSV// /}"
    IFS=',' read -r -a LEVELS <<< "${LEVELS_CSV}"
    if [[ ${#LEVELS[@]} -eq 0 ]]; then
        echo "ERROR: --levels resolved to an empty set." >&2
        exit 1
    fi
fi

# ---- paths ----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIMPLER_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE="$(cd "${SIMPLER_ROOT}/.." && pwd)"

if [[ "${CKPT}" != /* ]]; then
    CKPT="${WORKSPACE}/${CKPT}"
fi

export INTERNVLA_M1_SERVER_HOST="${SERVER_HOST}"
export INTERNVLA_M1_SERVER_PORT="${SERVER_PORT}"
export INTERNVLA_M1_UNNORM_KEY="${UNNORM_KEY}"
export INTERNVLA_M1_CFG_SCALE="${CFG_SCALE}"
export INTERNVLA_M1_NUM_DDIM_STEPS="${DDIM_STEPS}"

NOTE_ARGS=()
if [[ -n "${RUN_ID_NOTE}" ]]; then
    NOTE_ARGS=(--run-id-note "${RUN_ID_NOTE}")
fi

cd "${SIMPLER_ROOT}"

TOTAL=$(( ${#SUBTASKS[@]} * ${#LEVELS[@]} ))

echo "============================================================"
echo "InternVLA-M1 Bridge fullsweep — serial"
echo "  Subtasks          : ${SUBTASKS[*]}"
echo "  Levels            : ${LEVELS[*]}"
echo "  Total runs        : ${TOTAL}"
echo "  GPU id            : ${GPU_ID}"
echo "  Checkpoint        : ${CKPT}"
echo "  InternVLA-M1 server: ${INTERNVLA_M1_SERVER_HOST}:${INTERNVLA_M1_SERVER_PORT}"
echo "  Unnorm key        : ${INTERNVLA_M1_UNNORM_KEY}"
echo "  Cfg scale         : ${INTERNVLA_M1_CFG_SCALE}"
echo "  DDIM steps        : ${INTERNVLA_M1_NUM_DDIM_STEPS}"
echo "  Action temp       : ${ACTION_ENSEMBLE_TEMP}"
echo "  Max episode steps : ${MAX_STEPS} (official InternVLA-M1 SimplerEnv eval = 120)"
echo "  Output layout     : ${OUTPUT_LAYOUT}"
echo "  Output root       : ${OUTPUT_ROOT}"
echo "  Log root (legacy) : ${LOG_ROOT}"
echo "  Conda env         : ${CONDA_ENV_NAME}"
if [[ "${DRY_RUN}" == "true" ]]; then
    echo "  *** DRY RUN - commands will be printed, not executed ***"
fi
echo "============================================================"

RUN_IDX=0
FAILED=()
PASSED=()

for SUB in "${SUBTASKS[@]}"; do
    for LV in "${LEVELS[@]}"; do
        RUN_IDX=$(( RUN_IDX + 1 ))
        TAG="[${RUN_IDX}/${TOTAL}] subtask=${SUB} level=${LV} gpu=${GPU_ID}"

        CMD=(
            bash scripts/internvla_m1/run_bridge_internvla_m1_eval.sh
                --task "${SUB}"
                --level "${LV}"
                --gpu "${GPU_ID}"
                --ckpt "${CKPT}"
                --server-host "${SERVER_HOST}"
                --server-port "${SERVER_PORT}"
                --unnorm-key "${UNNORM_KEY}"
                --cfg-scale "${CFG_SCALE}"
                --ddim-steps "${DDIM_STEPS}"
                --temp "${ACTION_ENSEMBLE_TEMP}"
                --output-root "${OUTPUT_ROOT}"
                --output-layout "${OUTPUT_LAYOUT}"
                --log-root "${LOG_ROOT}"
                --env-name "${CONDA_ENV_NAME}"
                --max-steps "${MAX_STEPS}"
                "${NOTE_ARGS[@]}"
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
            PASSED+=("${TAG}")
        else
            EXIT_CODE=$?
            echo "${TAG} - FAILED (exit ${EXIT_CODE})" >&2
            FAILED+=("${TAG}")
        fi
    done
done

echo ""
echo "============================================================"
echo "InternVLA-M1 Bridge fullsweep complete: ${RUN_IDX} runs attempted."
echo "  Passed : ${#PASSED[@]}"
echo "  Failed : ${#FAILED[@]}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed runs:"
    for F in "${FAILED[@]}"; do
        echo "  - ${F}"
    done
fi
echo "============================================================"

if [[ ${#FAILED[@]} -ne 0 ]]; then
    exit 1
fi
