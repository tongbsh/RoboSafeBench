#!/bin/bash
# ============================================================
# DyniSafety Benchmark — ALL suites × ALL tasks × Levels 1-3
#
# Robust overnight runner:
#   - Individual task failures are caught and logged, never abort the whole run
#   - Each task has a configurable timeout (default 30 min)
#   - A summary report is printed at the end with pass/fail/timeout status
#   - Timestamped log file written to logs/
#
# Usage:
#   bash scripts/dynamicLibero/eval_all.sh                         # run everything
#   bash scripts/dynamicLibero/eval_all.sh --suites spatial object  # only these suites
#   bash scripts/dynamicLibero/eval_all.sh --levels 1 2            # only level 1 and 2
#   bash scripts/dynamicLibero/eval_all.sh --tasks 0 --levels 3    # only task 0, level 3
#   bash scripts/dynamicLibero/eval_all.sh --timeout 3600          # 60-min timeout per task
# ============================================================

set -uo pipefail
# NOTE: intentionally NOT using set -e — we handle errors per-task

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# --- defaults ---
SUITES="spatial object goal long"
TASKS="0 1 2 3"
LEVELS="1 2 3"
TIMEOUT_SEC=1800   # 30 minutes per task

# --- parse optional filters ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --suites)
            shift; SUITES=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do SUITES="${SUITES} $1"; shift; done
            SUITES="${SUITES# }"
            ;;
        --tasks)
            shift; TASKS=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do TASKS="${TASKS} $1"; shift; done
            TASKS="${TASKS# }"
            ;;
        --levels)
            shift; LEVELS=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do LEVELS="${LEVELS} $1"; shift; done
            LEVELS="${LEVELS# }"
            ;;
        --timeout)
            shift; TIMEOUT_SEC="$1"; shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--suites ...] [--tasks ...] [--levels ...] [--timeout SEC]"
            exit 1
            ;;
    esac
done

# --- logging setup ---
LOG_DIR="${PROJ_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/eval_all_${TIMESTAMP}.log"

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

# --- count total tasks ---
for _l in ${LEVELS}; do for _s in ${SUITES}; do for _t in ${TASKS}; do
    TOTAL=$((TOTAL + 1))
done; done; done

log "============================================================"
log "DyniSafety Benchmark — FULL EVALUATION"
log "Suites : ${SUITES}"
log "Tasks  : ${TASKS}"
log "Levels : ${LEVELS}"
log "Timeout: ${TIMEOUT_SEC}s per task"
log "Total  : ${TOTAL} evaluations"
log "Log    : ${LOG_FILE}"
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
            TAG="${suite}/task${task_id}/level${level}"
            log ""
            log "  [${CURRENT}/${TOTAL}] ${TAG}  — starting..."
            START_TS=$(date +%s)

            # Run with timeout; capture exit code without aborting
            timeout "${TIMEOUT_SEC}" \
                bash "${SCRIPT_DIR}/eval_dynisafety.sh" "${suite}" "${task_id}" "${level}" \
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

# --- summary report ---
log ""
log "============================================================"
log "                  EVALUATION SUMMARY"
log "============================================================"
log "Total : ${TOTAL}"
log "Passed: ${#RESULTS_PASS[@]}"
log "Failed: ${#RESULTS_FAIL[@]}"
log "Timeout: ${#RESULTS_TIMEOUT[@]}"
log ""

if [[ ${#RESULTS_FAIL[@]} -gt 0 ]]; then
    log "--- FAILED ---"
    for item in "${RESULTS_FAIL[@]}"; do
        log "  ✗ ${item}"
    done
fi

if [[ ${#RESULTS_TIMEOUT[@]} -gt 0 ]]; then
    log "--- TIMEOUT ---"
    for item in "${RESULTS_TIMEOUT[@]}"; do
        log "  ⏱ ${item}"
    done
fi

if [[ ${#RESULTS_PASS[@]} -gt 0 ]]; then
    log "--- PASSED ---"
    for item in "${RESULTS_PASS[@]}"; do
        log "  ✓ ${item}"
    done
fi

log ""
log "Full log: ${LOG_FILE}"
log "============================================================"
log "ALL EVALUATIONS COMPLETE."
log "============================================================"
