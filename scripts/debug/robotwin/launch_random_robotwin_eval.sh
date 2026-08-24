#!/usr/bin/env bash
# random × Dynamic RoboTwin main20 L0–L3 场景探测。
# 对照 artifacts/debug/random_libero_l3_reach_diversity：看扰动/视频，不是成功率。
#
#   bash scripts/debug/robotwin/launch_random_robotwin_eval.sh
#   EPISODES=3 bash scripts/debug/robotwin/launch_random_robotwin_eval.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
exec bash "$ROOT/artifacts/debug/random_robotwin_l0_l3/launch_eval.sh" "$@"
