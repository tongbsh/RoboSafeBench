#!/usr/bin/env bash
# ============================================================
# Parallel fullsweep: DyniSafety-SimplerEnv x Dream-VLA (best-effort zero-shot Bridge)
# WidowX Bridge - first 3 subtasks (carrot, stack, spoon) x 4 dynamic levels.
#
# 12 (subtask, level) items are dispatched to the GPUs supplied via --gpus
# in round-robin order. Each GPU runs its assigned items sequentially as a
# background worker; logs are written under
#   dynamic_SimplerEnv/outputs/dreamvla/logs/fullsweep_parallel/<timestamp>/
#
# All workers share the SAME Dream-VLA HTTP server (assumed already running).
#
# Usage:
#   bash dynamic_SimplerEnv/scripts/dreamvla/run_bridge_dreamvla_fullsweep_parallel.sh [OPTIONS]
#
# Options:
#   --gpus       <CSV>    Comma-separated GPU ids          (default: 0)
#   --subtasks   <CSV>    Subtask filter                   (default: carrot,stack,spoon)
#   --levels     <CSV>    Dynamic level filter             (default: 0,1,2,3)
#   --ckpt        <PATH>  Forwarded to run_dynisafety.sh   (default: ignored)
#   --server-host <HOST>  Sets DREAMVLA_SERVER_HOST        (default: 127.0.0.1)
#   --server-port <PORT>  Sets DREAMVLA_SERVER_PORT        (default: 8030)
#   --temp        <F>     Action ensemble temperature      (default: -0.8)
#   --output-root   DIR   Bench layout root                (default: ./outputs)
#   --output-layout NAME  bench | legacy                   (default: bench)
#   --log-root      DIR   Legacy nested log root           (default: results)
#   --run-id-note   STR   Optional log filename suffix
#   --env-name      NAME  Conda env                        (default: simpler_env)
#   --max-steps     N     --max-episode-steps for ALL bridge subtasks (default: unset)
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
GPUS_CSV="0"
SUBTASKS_CSV=""
LEVELS_CSV=""
CKPT="ignored"
SERVER_HOST="${DREAMVLA_SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${DREAMVLA_SERVER_PORT:-8030}"
ACTION_ENSEMBLE_TEMP="-0.8"
OUTPUT_ROOT="./outputs"
OUTPUT_LAYOUT="bench"
LOG_ROOT="results"
RUN_ID_NOTE=""
CONDA_ENV_NAME="simpler_env"
DRY_RUN="false"
MAX_STEPS="${BRIDGE_MAX_EPISODE_STEPS:-}"

DEFAULT_SUBTASKS=(bridge_carrot bridge_stack bridge_spoon)
DEFAULT_LEVELS=(0 1 2 3)

WORKER_PIDS=()

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    for pid in "${WORKER_PIDS[@]}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
    exit "${exit_code}"
}

# ---- arg parsing ----------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)          GPUS_CSV="$2";          shift 2 ;;
        --subtasks)      SUBTASKS_CSV="$2";      shift 2 ;;
        --levels)        LEVELS_CSV="$2";        shift 2 ;;
        --ckpt)          CKPT="$2";              shift 2 ;;
        --server-host)   SERVER_HOST="$2";       shift 2 ;;
        --server-port)   SERVER_PORT="$2";       shift 2 ;;
        --temp)          ACTION_ENSEMBLE_TEMP="$2"; shift 2 ;;
        --output-root)   OUTPUT_ROOT="$2";       shift 2 ;;
        --output-layout) OUTPUT_LAYOUT="$2";     shift 2 ;;
        --log-root)      LOG_ROOT="$2";          shift 2 ;;
        --run-id-note)   RUN_ID_NOTE="$2";       shift 2 ;;
        --env-name)      CONDA_ENV_NAME="$2";    shift 2 ;;
        --max-steps)     MAX_STEPS="$2";         shift 2 ;;
        --dry-run)       DRY_RUN="true";         shift 1 ;;
        -h|--help)
            sed -n '3,32p' "$0"
            exit 0 ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage."
            exit 1 ;;
    esac
done

# ---- resolve gpus / subtasks / levels ------------------------------
GPUS_CSV="${GPUS_CSV// /}"
IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
if [[ ${#GPUS[@]} -eq 0 ]]; then
    echo "ERROR: --gpus is empty." >&2
    exit 1
fi

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

ITEMS=()
for SUB in "${SUBTASKS[@]}"; do
    for LV in "${LEVELS[@]}"; do
        ITEMS+=("${SUB}|${LV}")
    done
done
TOTAL=${#ITEMS[@]}
N_GPU=${#GPUS[@]}

# ---- paths ----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIMPLER_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_STAMP="$(date +%Y_%m_%d-%H_%M_%S)"
RUN_LOG_DIR="${SIMPLER_ROOT}/outputs/dreamvla/logs/fullsweep_parallel/${RUN_STAMP}"
mkdir -p "${RUN_LOG_DIR}"

export DREAMVLA_SERVER_HOST="${SERVER_HOST}"
export DREAMVLA_SERVER_PORT="${SERVER_PORT}"

NOTE_ARGS=()
if [[ -n "${RUN_ID_NOTE}" ]]; then
    NOTE_ARGS=(--run-id-note "${RUN_ID_NOTE}")
fi

cd "${SIMPLER_ROOT}"

trap cleanup EXIT INT TERM

echo "============================================================"
echo "Dream-VLA Bridge fullsweep - parallel (best-effort zero-shot)"
echo "  Subtasks          : ${SUBTASKS[*]}"
echo "  Levels            : ${LEVELS[*]}"
echo "  Total items       : ${TOTAL}"
echo "  GPUs              : ${GPUS[*]} (round-robin)"
echo "  Dream-VLA server  : ${DREAMVLA_SERVER_HOST}:${DREAMVLA_SERVER_PORT}"
echo "  Action temp       : ${ACTION_ENSEMBLE_TEMP}"
if [[ -n "${MAX_STEPS}" ]]; then
    echo "  Max episode steps : ${MAX_STEPS} (override)"
else
    echo "  Max episode steps : <unset> (per-task default: carrot/stack/spoon=60, eggplant=120)"
fi
echo "  Output layout     : ${OUTPUT_LAYOUT}"
echo "  Output root       : ${OUTPUT_ROOT}"
echo "  Log root (legacy) : ${LOG_ROOT}"
echo "  Conda env         : ${CONDA_ENV_NAME}"
echo "  Per-GPU log dir   : ${RUN_LOG_DIR}"
if [[ "${DRY_RUN}" == "true" ]]; then
    echo "  *** DRY RUN - commands will be printed, not executed ***"
fi
echo "============================================================"

WORKER_GPU=()
WORKER_LOGS=()

MAX_STEPS_ARGS=()
if [[ -n "${MAX_STEPS}" ]]; then
    MAX_STEPS_ARGS=(--max-steps "${MAX_STEPS}")
fi

for ((i=0; i<N_GPU; i++)); do
    GPU="${GPUS[i]}"
    ITEMS_FOR_GPU=()
    for ((j=i; j<TOTAL; j+=N_GPU)); do
        ITEMS_FOR_GPU+=("${ITEMS[j]}")
    done

    if [[ ${#ITEMS_FOR_GPU[@]} -eq 0 ]]; then
        echo "GPU ${GPU} : no items assigned (skip)"
        continue
    fi

    SUITE_LOG="${RUN_LOG_DIR}/gpu${GPU}.log"
    WORKER_GPU+=("${GPU}")
    WORKER_LOGS+=("${SUITE_LOG}")

    echo ""
    echo "------------------------------------------------------------"
    echo "GPU ${GPU} - ${#ITEMS_FOR_GPU[@]} item(s):"
    for ITEM in "${ITEMS_FOR_GPU[@]}"; do
        echo "    ${ITEM/|/  level=}"
    done
    echo "Log: ${SUITE_LOG}"
    echo "------------------------------------------------------------"

    if [[ "${DRY_RUN}" == "true" ]]; then
        WORKER_PIDS+=("")
        continue
    fi

    (
        set +e
        worker_failed=0
        for ITEM in "${ITEMS_FOR_GPU[@]}"; do
            SUB="${ITEM%%|*}"
            LV="${ITEM##*|}"
            echo "[gpu=${GPU}] >>> subtask=${SUB} level=${LV}  $(date '+%F %T')"
            bash scripts/dreamvla/run_bridge_dreamvla_eval.sh \
                --task "${SUB}" \
                --level "${LV}" \
                --gpu "${GPU}" \
                --ckpt "${CKPT}" \
                --server-host "${SERVER_HOST}" \
                --server-port "${SERVER_PORT}" \
                --temp "${ACTION_ENSEMBLE_TEMP}" \
                --output-root "${OUTPUT_ROOT}" \
                --output-layout "${OUTPUT_LAYOUT}" \
                --log-root "${LOG_ROOT}" \
                --env-name "${CONDA_ENV_NAME}" \
                "${MAX_STEPS_ARGS[@]}" \
                "${NOTE_ARGS[@]}"
            rc=$?
            if [[ ${rc} -ne 0 ]]; then
                echo "[gpu=${GPU}] !!! subtask=${SUB} level=${LV} FAILED exit=${rc}" >&2
                worker_failed=1
            else
                echo "[gpu=${GPU}] <<< subtask=${SUB} level=${LV} DONE"
            fi
        done
        exit "${worker_failed}"
    ) > "${SUITE_LOG}" 2>&1 &
    WORKER_PIDS+=("$!")
done

if [[ "${DRY_RUN}" == "true" ]]; then
    exit 0
fi

FAILED_GPUS=()
for ((i=0; i<${#WORKER_PIDS[@]}; i++)); do
    PID="${WORKER_PIDS[i]}"
    GPU="${WORKER_GPU[i]}"
    LOG="${WORKER_LOGS[i]}"
    if [[ -z "${PID}" ]]; then
        continue
    fi
    if wait "${PID}"; then
        echo "Worker GPU ${GPU} : DONE"
    else
        EXIT_CODE=$?
        echo "Worker GPU ${GPU} : FAILED (exit ${EXIT_CODE}) - see ${LOG}" >&2
        FAILED_GPUS+=("${GPU}")
    fi
done

echo ""
echo "============================================================"
if [[ ${#FAILED_GPUS[@]} -eq 0 ]]; then
    echo "Dream-VLA parallel fullsweep - ALL workers succeeded."
else
    echo "Dream-VLA parallel fullsweep - ${#FAILED_GPUS[@]} worker(s) had failure(s):"
    for G in "${FAILED_GPUS[@]}"; do
        echo "  - GPU ${G}"
    done
fi
echo "Per-GPU logs: ${RUN_LOG_DIR}"
echo "============================================================"

if [[ ${#FAILED_GPUS[@]} -ne 0 ]]; then
    exit 1
fi
