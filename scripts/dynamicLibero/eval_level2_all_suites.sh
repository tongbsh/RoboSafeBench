#!/bin/bash
# ============================================================
# DyniSafety — Level 2 only (obstacle sliding): all suites × all tasks (sequential)
#
# Order: spatial → object → goal → long; within each suite, task 0-3.
# Note: goal suite has 5 tasks (0-4); use --tasks 0 1 2 3 4 when running goal.
# Invokes scripts/eval_dynisafety.sh (checkpoint & Level-2 flags defined there).
#
# Usage:
#   bash scripts/dynamicLibero/eval_level2_all_suites.sh
#   bash scripts/dynamicLibero/eval_level2_all_suites.sh --suites spatial object
#   bash scripts/dynamicLibero/eval_level2_all_suites.sh --tasks 0 1
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

SUITES="spatial object goal long"
TASKS="0 1 2 3"

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
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--suites spatial object ...] [--tasks 0 1 ...]"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "DyniSafety Level 2 — all suites × tasks (sequential)"
echo "Suites: ${SUITES}"
echo "Tasks : ${TASKS}"
echo "============================================================"

for suite in ${SUITES}; do
    echo ""
    echo "============================================================"
    echo "Suite: ${suite} (safelibero_${suite}) — Level 2"
    echo "============================================================"
    for task_id in ${TASKS}; do
        echo ""
        echo "------------------------------------------------------------"
        echo "  ${suite}  task_id=${task_id}  level=2  — starting..."
        echo "------------------------------------------------------------"
        bash "${SCRIPT_DIR}/eval_dynisafety.sh" "${suite}" "${task_id}" 2
        echo "  ${suite}  task_id=${task_id}  level=2  — done."
    done
    echo ""
    echo "Suite ${suite} — all listed tasks complete."
done

echo ""
echo "============================================================"
echo "Level 2 — all suites complete."
echo "Logs: vlaModelHome/openvla-oft/experiments/logs/"
echo "============================================================"
