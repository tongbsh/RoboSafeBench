#!/usr/bin/env bash
# 读 X-VLA walkthrough 的 episodes.jsonl：指标、动态痕迹、视频是否落盘。
#
#   bash scripts/debug/xvla_walkthrough/01_check.sh
#   bash scripts/debug/xvla_walkthrough/01_check.sh --full
#   bash scripts/debug/xvla_walkthrough/01_check.sh --suite results/_sweep/debug_xvla_dynamic/suite.used.yaml
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/lib/env.sh"

dyni_activate "$DYNI_TOOL_ENV"
info "check X-VLA walkthrough  $*"
python "$SCRIPT_DIR/check.py" "$@"
