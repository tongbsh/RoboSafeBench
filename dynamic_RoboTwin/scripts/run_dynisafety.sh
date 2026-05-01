#!/usr/bin/env bash
#
# Top-level orchestration script for DyniSafety-RoboTwin.
#
# Routes a (model, level, task) triple to the appropriate per-model launcher
# under scripts/<model>/. Prefer the generic eval_task.sh entry when present
# so newly added tasks do not require one wrapper script per (task, level).
#
# Usage:
#   bash run_dynisafety.sh --model internvla_a1   --level 0   --task place_a2b_right
#   bash run_dynisafety.sh --model xvla           --level all --task adjust_bottle
#
# Pass --level all to iterate over levels 1, 2, 3 (excludes 0 baseline).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=""
LEVEL=""
TASK="place_a2b_right"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)  MODEL="$2";  shift 2 ;;
        --level)  LEVEL="$2";  shift 2 ;;
        --task)   TASK="$2";   shift 2 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "${MODEL}" || -z "${LEVEL}" ]]; then
    echo "Usage: bash run_dynisafety.sh --model <internvla_a1|xvla|xvla_official|starvla|lingbot_va> --level <0|1|2|3|all> [--task <name>]"
    exit 1
fi

if [[ ! -d "${SCRIPT_DIR}/${MODEL}" ]]; then
    echo "[ERROR] Unknown model '${MODEL}'. Available: $(ls -1 "${SCRIPT_DIR}" | grep -v '^_' | grep -v '\.sh$')"
    exit 1
fi

run_level () {
    local lvl="$1"
    local generic_runner="${SCRIPT_DIR}/${MODEL}/eval_task.sh"
    local script="${SCRIPT_DIR}/${MODEL}/eval_${TASK}_level${lvl}.sh"
    local per_task_script="${SCRIPT_DIR}/${MODEL}/per_task_eval/eval_${TASK}_level${lvl}.sh"
    local legacy_per_task_script="${SCRIPT_DIR}/${MODEL}/perTask_eval/eval_${TASK}_level${lvl}.sh"

    echo "==> ${MODEL} | ${TASK} | level ${lvl}"
    if [[ -f "${generic_runner}" ]]; then
        bash "${generic_runner}" --task "${TASK}" --level "${lvl}" "${EXTRA_ARGS[@]}"
        return
    fi
    if [[ -f "${script}" ]]; then
        bash "${script}" "${EXTRA_ARGS[@]}"
        return
    fi
    if [[ -f "${per_task_script}" ]]; then
        bash "${per_task_script}" "${EXTRA_ARGS[@]}"
        return
    fi
    if [[ -f "${legacy_per_task_script}" ]]; then
        bash "${legacy_per_task_script}" "${EXTRA_ARGS[@]}"
        return
    fi

    echo "[ERROR] Missing launcher for model='${MODEL}', task='${TASK}', level='${lvl}'."
    echo "[INFO]  Tried: ${generic_runner}"
    echo "[INFO]  Tried: ${script}"
    echo "[INFO]  Tried: ${per_task_script}"
    echo "[INFO]  Tried: ${legacy_per_task_script}"
    exit 1
}

case "${LEVEL}" in
    all) for lvl in 1 2 3; do run_level "${lvl}"; done ;;
    *)   run_level "${LEVEL}" ;;
esac
