#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TASK_IDX="${TASK_IDX:-15}"
export TASK_NAME="${TASK_NAME:-move_stapler_pad}"
exec bash "${SCRIPT_DIR}/_eval_task_common.sh" "$@"
