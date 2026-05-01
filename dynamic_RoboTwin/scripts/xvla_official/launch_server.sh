#!/usr/bin/env bash
#
# Reuse the standard X-VLA server launcher for the official-protocol branch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../past_cache/xvla/launch_server.sh" "$@"
