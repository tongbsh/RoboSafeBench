#!/bin/bash
# ============================================================
# DyniSafety — Level 1 only (sudden drop): all suites × all tasks (sequential)
#
# Order: spatial → object → goal → long; within each suite, task 0-3.
# Note: goal suite has 5 tasks (0-4); use --tasks 0 1 2 3 4 when running goal.
# Invokes scripts/eval_dynisafety.sh (checkpoint & Level-1 flags defined there).
#
# Level 1 obstacle pool (optional):
#   Only named obstacles participate; each episode picks one by rotation
#   (run_dynisafety_eval.py: --l1_drop_obstacle_allowlist).
#
# Usage:
#   bash scripts/dynamicLibero/eval_level1_all_suites.sh
#   bash scripts/dynamicLibero/eval_level1_all_suites.sh --suites spatial spatial --tasks 2
#   bash scripts/dynamicLibero/eval_level1_all_suites.sh --tasks 0 1
#   bash scripts/dynamicLibero/eval_level1_all_suites.sh --allowlist "milk_obstacle_1,wine_bottle_obstacle_1"
#
# Or export before running (applies to all invocations from this script):
#   export L1_DROP_OBSTACLE_ALLOWLIST="milk_obstacle_1,red_coffee_mug_obstacle_1"
#   bash scripts/eval_level1_all_suites.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

SUITES="spatial object  goal long"
TASKS="0 1 2 3"
# If set by --allowlist, passed to eval_dynisafety.sh via L1_DROP_OBSTACLE_ALLOWLIST for Level 1 only.
ALLOWLIST_CLI=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --suites)
            shift
            SUITES=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SUITES="${SUITES} $1"
                shift
            done
            SUITES="${SUITES# }"
            ;;
        --tasks)
            shift
            TASKS=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                TASKS="${TASKS} $1"
                shift
            done
            TASKS="${TASKS# }"
            ;;
        --allowlist|--l1-allowlist)
            shift
            if [[ $# -lt 1 ]]; then
                echo "ERROR: --allowlist requires a comma-separated argument (e.g. milk_obstacle_1,wine_bottle_obstacle_1)"
                exit 1
            fi
            ALLOWLIST_CLI="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--suites spatial object ...] [--tasks 0 1 ...] [--allowlist \"name1,name2,...\"]"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "DyniSafety Level 1 — all suites × tasks (sequential)"
echo "Suites: ${SUITES}"
echo "Tasks : ${TASKS}"
if [[ -n "${ALLOWLIST_CLI}" ]]; then
    echo "L1 allowlist (CLI): ${ALLOWLIST_CLI}"
elif [[ -n "${L1_DROP_OBSTACLE_ALLOWLIST:-}" ]]; then
    echo "L1 allowlist (env): ${L1_DROP_OBSTACLE_ALLOWLIST}"
else
    echo "L1 allowlist: (none — default OBSTACLE_ORDER ∩ visible)"
fi
echo "============================================================"

_run_one_eval() {
    local suite="$1"
    local task_id="$2"
    if [[ -n "${ALLOWLIST_CLI}" ]]; then
        L1_DROP_OBSTACLE_ALLOWLIST="${ALLOWLIST_CLI}" bash "${SCRIPT_DIR}/eval_dynisafety.sh" "${suite}" "${task_id}" 1
    else
        bash "${SCRIPT_DIR}/eval_dynisafety.sh" "${suite}" "${task_id}" 1
    fi
}

for suite in ${SUITES}; do
    echo ""
    echo "============================================================"
    echo "Suite: ${suite} (safelibero_${suite}) — Level 1"
    echo "============================================================"
    for task_id in ${TASKS}; do
        echo ""
        echo "------------------------------------------------------------"
        echo "  ${suite}  task_id=${task_id}  level=1  — starting..."
        echo "------------------------------------------------------------"
        _run_one_eval "${suite}" "${task_id}"
        echo "  ${suite}  task_id=${task_id}  level=1  — done."
    done
    echo ""
    echo "Suite ${suite} — all listed tasks complete."
done

echo ""
echo "============================================================"
echo "Level 1 — all suites complete."
echo "Logs: vlaModelHome/openvla-oft/experiments/logs/"
echo "============================================================"
