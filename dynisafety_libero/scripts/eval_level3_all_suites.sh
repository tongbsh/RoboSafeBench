#!/bin/bash
# ============================================================
# DyniSafety — Level 3 (shared reaching): all suites × all tasks
#
# Usage:
#   bash dynisafety_libero/scripts/eval_level3_all_suites.sh --vla openvla
#   bash dynisafety_libero/scripts/eval_level3_all_suites.sh --vla pi0 --suites spatial
#   bash dynisafety_libero/scripts/eval_level3_all_suites.sh --vla openvla --tasks 0 1
#   bash dynisafety_libero/scripts/eval_level3_all_suites.sh --vla openvla --save-video
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

VLA="${VLA:-openvla}"
SUITES="spatial object goal long"
TASKS="0 1 2 3"
SAVE_VIDEO_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vla)   VLA="$2";   shift 2 ;;
        --suites)
            shift; SUITES=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do SUITES="${SUITES} $1"; shift; done
            SUITES="${SUITES# }" ;;
        --tasks)
            shift; TASKS=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do TASKS="${TASKS} $1"; shift; done
            TASKS="${TASKS# }" ;;
        --save-video)
            SAVE_VIDEO_FLAG="--save-video"; shift ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --vla <VLA> [--suites ...] [--tasks ...] [--save-video]"
            exit 1 ;;
    esac
done

echo "============================================================"
echo "DyniSafety Level 3 — all suites × tasks  [VLA: ${VLA}]"
echo "Suites: ${SUITES}  Tasks: ${TASKS}  Video: ${SAVE_VIDEO_FLAG:-disabled}"
echo "============================================================"

for suite in ${SUITES}; do
    echo ""
    echo "=== Suite: ${suite}  Level: 3  [VLA: ${VLA}] ==="
    for task_id in ${TASKS}; do
        echo "  ${suite} task_id=${task_id} level=3 — starting..."
        bash "${SCRIPT_DIR}/eval_dynisafety.sh" "${suite}" "${task_id}" 3 \
            --vla "${VLA}" ${SAVE_VIDEO_FLAG}
        echo "  ${suite} task_id=${task_id} level=3 — done."
    done
    echo "Suite ${suite} complete."
done

echo ""
echo "Level 3 — all suites complete."
