#!/bin/bash
# ============================================================
# DyniSafety Benchmark — full sweep (all suites × tasks × levels 1–3)
#
# Robust overnight runner:
#   - Individual task failures are caught and logged, never abort the whole run
#   - Each task has a configurable timeout (default 30 min)
#   - Summary report printed at the end with pass/fail/timeout per task
#   - Timestamped log file written to logs/
#
# Usage:
#   bash dynisafety_libero/scripts/eval_all.sh --vla openvla
#   bash dynisafety_libero/scripts/eval_all.sh --vla openvla --suites spatial object
#   bash dynisafety_libero/scripts/eval_all.sh --vla openvla --levels 1 2
#   bash dynisafety_libero/scripts/eval_all.sh --vla pi0    --tasks 0 --levels 3
#   bash dynisafety_libero/scripts/eval_all.sh --vla openvla --timeout 3600
#
# LEVEL options: 1 (sudden drop) | 2 (sliding obstacle) | 3 (shared reaching)
# The --vla flag selects the model; all other flags filter suites/tasks/levels.
# Set VLA=<name> in the environment as an alternative to --vla.
# ============================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# --- defaults ---
VLA="${VLA:-openvla}"
SUITES="spatial object goal long"
TASKS="0 1 2 3"
LEVELS="1 2 3"
TIMEOUT_SEC=1800   # 30 minutes per task
SAVE_VIDEO_FLAG=""

# --- parse options ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --vla)
            VLA="$2"; shift 2 ;;
        --suites)
            shift; SUITES=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do SUITES="${SUITES} $1"; shift; done
            SUITES="${SUITES# }" ;;
        --tasks)
            shift; TASKS=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do TASKS="${TASKS} $1"; shift; done
            TASKS="${TASKS# }" ;;
        --levels)
            shift; LEVELS=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do LEVELS="${LEVELS} $1"; shift; done
            LEVELS="${LEVELS# }" ;;
        --timeout)
            TIMEOUT_SEC="$2"; shift 2 ;;
        --save-video)
            SAVE_VIDEO_FLAG="--save-video"; shift ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --vla <VLA> [--suites ...] [--tasks ...] [--levels ...] [--timeout SEC] [--save-video]"
            exit 1 ;;
    esac
done

# --- logging setup ---
LOG_DIR="${PROJ_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/eval_all_${VLA}_${TIMESTAMP}.log"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "${msg}"
    echo "${msg}" >> "${LOG_FILE}"
}

# --- result tracking ---
declare -a RESULTS_PASS=()
declare -a RESULTS_FAIL=()
declare -a RESULTS_TIMEOUT=()
TOTAL=0
for _l in ${LEVELS}; do for _s in ${SUITES}; do for _t in ${TASKS}; do
    TOTAL=$((TOTAL + 1))
done; done; done

log "============================================================"
log "DyniSafety Benchmark — FULL EVALUATION"
log "  VLA    : ${VLA}"
log "  Video  : ${SAVE_VIDEO_FLAG:-disabled}"
log "  Suites : ${SUITES}"
log "  Tasks  : ${TASKS}"
log "  Levels : ${LEVELS}"
log "  Timeout: ${TIMEOUT_SEC}s per task"
log "  Total  : ${TOTAL} evaluations"
log "  Log    : ${LOG_FILE}"
log "============================================================"

CURRENT=0

for level in ${LEVELS}; do
    log ""
    log "======================== Level ${level} ========================"
    for suite in ${SUITES}; do
        log ""
        log "  ---- Suite: ${suite}  Level: ${level} ----"
        for task_id in ${TASKS}; do
            CURRENT=$((CURRENT + 1))
            TAG="${VLA}/${suite}/task${task_id}/level${level}"
            log ""
            log "  [${CURRENT}/${TOTAL}] ${TAG}  — starting..."
            START_TS=$(date +%s)

            timeout "${TIMEOUT_SEC}" \
                bash "${SCRIPT_DIR}/eval_dynisafety.sh" "${suite}" "${task_id}" "${level}" \
                    --vla "${VLA}" ${SAVE_VIDEO_FLAG} \
                >> "${LOG_FILE}" 2>&1
            EXIT_CODE=$?

            ELAPSED=$(( $(date +%s) - START_TS ))
            if [[ ${EXIT_CODE} -eq 0 ]]; then
                RESULTS_PASS+=("${TAG}")
                log "  [${CURRENT}/${TOTAL}] ${TAG}  — PASSED  (${ELAPSED}s)"
            elif [[ ${EXIT_CODE} -eq 124 ]]; then
                RESULTS_TIMEOUT+=("${TAG}")
                log "  [${CURRENT}/${TOTAL}] ${TAG}  — TIMEOUT after ${TIMEOUT_SEC}s"
            else
                RESULTS_FAIL+=("${TAG}")
                log "  [${CURRENT}/${TOTAL}] ${TAG}  — FAILED  exit=${EXIT_CODE} (${ELAPSED}s)"
            fi
        done
        log "  ---- Suite ${suite} Level ${level} — done ----"
    done
    log "======================== Level ${level} — done ========================"
done

# --- summary ---
log ""
log "============================================================"
log "                  EVALUATION SUMMARY  [VLA: ${VLA}]"
log "============================================================"
log "Total  : ${TOTAL}"
log "Passed : ${#RESULTS_PASS[@]}"
log "Failed : ${#RESULTS_FAIL[@]}"
log "Timeout: ${#RESULTS_TIMEOUT[@]}"
log ""

if [[ ${#RESULTS_FAIL[@]} -gt 0 ]]; then
    log "--- FAILED ---"
    for item in "${RESULTS_FAIL[@]}"; do log "  ✗ ${item}"; done
fi
if [[ ${#RESULTS_TIMEOUT[@]} -gt 0 ]]; then
    log "--- TIMEOUT ---"
    for item in "${RESULTS_TIMEOUT[@]}"; do log "  ⏱ ${item}"; done
fi
if [[ ${#RESULTS_PASS[@]} -gt 0 ]]; then
    log "--- PASSED ---"
    for item in "${RESULTS_PASS[@]}"; do log "  ✓ ${item}"; done
fi

log ""
log "Full log: ${LOG_FILE}"
log "============================================================"
log "ALL EVALUATIONS COMPLETE."
log "============================================================"
