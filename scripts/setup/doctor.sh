#!/usr/bin/env bash
# Environment self-check. Any extra args go straight to `dynisafety doctor`.
#
#   bash scripts/setup/doctor.sh
#   bash scripts/setup/doctor.sh --backend libero
#   bash scripts/setup/doctor.sh --backend robocasa
#   bash scripts/setup/doctor.sh --strict --online
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"

dyni_activate "$DYNI_TOOL_ENV"
info "dynisafety doctor $*"
dyni_cli doctor "$@"
