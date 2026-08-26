#!/usr/bin/env bash
# random × Dynamic RoboTwin main20 L0–L3 场景探测（看扰动/视频，不是成功率）。
#
#   bash scripts/debug/robotwin/launch_random_robotwin_eval.sh
#   EPISODES=3 bash scripts/debug/robotwin/launch_random_robotwin_eval.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
exec bash "$ROOT/scripts/debug/eval_debug.sh" \
  --backend robotwin --model random --suite main20 \
  --level all --task all --episodes "${EPISODES:-1}" "$@"
