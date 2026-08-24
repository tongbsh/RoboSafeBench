#!/usr/bin/env bash
# Environment self-check. Any extra args go straight to `dynisafety doctor`.
#
#   bash scripts/run/02_doctor.sh
#   bash scripts/run/02_doctor.sh --backend libero
#   bash scripts/run/02_doctor.sh --backend robocasa
#   bash scripts/run/02_doctor.sh --strict --online
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

dyni_activate "$DYNI_TOOL_ENV"
info "dynisafety doctor $*"
dyni_cli doctor "$@"
