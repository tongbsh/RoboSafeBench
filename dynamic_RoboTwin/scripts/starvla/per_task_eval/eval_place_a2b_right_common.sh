#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TASK_IDX="${TASK_IDX:-21}"
export TASK_NAME="${TASK_NAME:-place_a2b_right}"
exec bash "${SCRIPT_DIR}/_eval_task_common.sh" "$@"
