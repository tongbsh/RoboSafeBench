#!/bin/bash
# ============================================================
# DyniSafety — Level 1 (sudden drop): all suites × all tasks
#
# Usage:
#   bash dynisafety_libero/scripts/eval_level1_all_suites.sh --vla openvla
#   bash dynisafety_libero/scripts/eval_level1_all_suites.sh --vla pi0 --suites spatial object
#   bash dynisafety_libero/scripts/eval_level1_all_suites.sh --vla openvla --tasks 0 1
#   bash dynisafety_libero/scripts/eval_level1_all_suites.sh --vla openvla --save-video
#   bash dynisafety_libero/scripts/eval_level1_all_suites.sh --vla openvla \
#       --allowlist "milk_obstacle_1,wine_bottle_obstacle_1"
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

VLA="${VLA:-openvla}"
SUITES="spatial object goal long"
TASKS="0 1 2 3"
ALLOWLIST_CLI=""
SAVE_VIDEO_FLAG=""   # empty = no video; "--save-video" = save

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vla)       VLA="$2";        shift 2 ;;
        --suites)
            shift; SUITES=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do SUITES="${SUITES} $1"; shift; done
            SUITES="${SUITES# }" ;;
        --tasks)
            shift; TASKS=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do TASKS="${TASKS} $1"; shift; done
            TASKS="${TASKS# }" ;;
        --allowlist|--l1-allowlist)
            ALLOWLIST_CLI="$2"; shift 2 ;;
        --save-video)
            SAVE_VIDEO_FLAG="--save-video"; shift ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --vla <VLA> [--suites ...] [--tasks ...] [--allowlist \"name1,name2,...\"] [--save-video]"
            exit 1 ;;
    esac
done

echo "============================================================"
echo "DyniSafety Level 1 — all suites × tasks  [VLA: ${VLA}]"
echo "Suites: ${SUITES}"
echo "Tasks : ${TASKS}"
echo "Video : ${SAVE_VIDEO_FLAG:-disabled}"
if [[ -n "${ALLOWLIST_CLI}" ]]; then
    echo "L1 allowlist (CLI): ${ALLOWLIST_CLI}"
elif [[ -n "${L1_DROP_OBSTACLE_ALLOWLIST:-}" ]]; then
    echo "L1 allowlist (env): ${L1_DROP_OBSTACLE_ALLOWLIST}"
else
    echo "L1 allowlist: (none — default OBSTACLE_ORDER ∩ visible)"
fi
echo "============================================================"

_run_one() {
    local suite="$1" task_id="$2"
    if [[ -n "${ALLOWLIST_CLI}" ]]; then
        L1_DROP_OBSTACLE_ALLOWLIST="${ALLOWLIST_CLI}" \
            bash "${SCRIPT_DIR}/eval_dynisafety.sh" "${suite}" "${task_id}" 1 \
                --vla "${VLA}" ${SAVE_VIDEO_FLAG}
    else
        bash "${SCRIPT_DIR}/eval_dynisafety.sh" "${suite}" "${task_id}" 1 \
            --vla "${VLA}" ${SAVE_VIDEO_FLAG}
    fi
}

for suite in ${SUITES}; do
    echo ""
    echo "============================================================"
    echo "Suite: ${suite}  Level: 1  [VLA: ${VLA}]"
    echo "============================================================"
    for task_id in ${TASKS}; do
        echo "  ${suite} task_id=${task_id} level=1 — starting..."
        _run_one "${suite}" "${task_id}"
        echo "  ${suite} task_id=${task_id} level=1 — done."
    done
    echo "Suite ${suite} complete."
done

echo ""
echo "Level 1 — all suites complete."
