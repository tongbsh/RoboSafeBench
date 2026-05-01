#!/bin/bash
# ============================================================
# DyniSafety Benchmark — safelibero_object: all tasks × all levels
#
# Usage:
#   bash scripts/eval_object_all.sh                    # all 4 tasks × 4 levels
#   bash scripts/eval_object_all.sh --tasks 0 2        # only task 0 and 2
#   bash scripts/eval_object_all.sh --levels 2 3       # only level 1 and 3
#   bash scripts/eval_object_all.sh --tasks 1 --levels 2 4
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUITE="object"

# --- parse optional --tasks and --levels filters ---
TASKS="0 1 2 3"
LEVELS="1 2 3 4"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks)
            shift
            TASKS=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                TASKS="${TASKS} $1"
                shift
            done
            TASKS="${TASKS# }"
            ;;
        --levels)
            shift
            LEVELS=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                LEVELS="${LEVELS} $1"
                shift
            done
            LEVELS="${LEVELS# }"
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "DyniSafety Benchmark — safelibero_object"
echo "Tasks : ${TASKS}"
echo "Levels: ${LEVELS}"
echo "============================================================"

for task_id in ${TASKS}; do
    for level in ${LEVELS}; do
        echo ""
        echo "------------------------------------------------------------"
        echo "  ${SUITE} task=${task_id} level=${level} — starting..."
        echo "------------------------------------------------------------"
        bash "${SCRIPT_DIR}/eval_dynisafety.sh" "${SUITE}" "${task_id}" "${level}"
        echo "  ${SUITE} task=${task_id} level=${level} — done."
    done
done

echo ""
echo "============================================================"
echo "safelibero_object — all evaluations complete."
echo "============================================================"
