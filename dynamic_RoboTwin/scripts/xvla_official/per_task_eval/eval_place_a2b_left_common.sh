#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TASK_IDX="${TASK_IDX:-20}"
export TASK_NAME="${TASK_NAME:-place_a2b_left}"
exec bash "${SCRIPT_DIR}/_eval_task_common.sh" "$@"
