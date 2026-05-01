#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../task_registry.sh
source "${SCRIPT_DIR}/../task_registry.sh"

TASK_NAME=""
DYNAMIC_LEVEL="${DYNAMIC_LEVEL:-}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)
            TASK_NAME="$2"
            shift 2
            ;;
        --level)
            DYNAMIC_LEVEL="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "${TASK_NAME}" || -z "${DYNAMIC_LEVEL}" ]]; then
    echo "Usage: bash eval_task.sh --task <task_name> --level <0|1|2|3> [-- extra args]"
    exit 1
fi

dynisafety_export_task_env "${TASK_NAME}"
exec bash "${SCRIPT_DIR}/perTask_eval/_eval_task_common.sh" "${DYNAMIC_LEVEL}" "${EXTRA_ARGS[@]}"
