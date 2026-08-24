#!/usr/bin/env bash
# xvla_official × RoboTwin main20：等 serve ready 后按 task × level 循环 eval。
# --task all = catalog ids（不是 0..19），见 configs/backends/robotwin/suites/main20.yaml
#
#   screen -S xvla_rt_eval
#   bash scripts/debug/robotwin/launch_xvla_official_robotwin_eval.sh --level 0 --task 21 --episodes 1
#   bash scripts/debug/robotwin/launch_xvla_official_robotwin_eval.sh --level 0,1,2,3 --task all
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/../launch_vla_eval.sh" \
  --model xvla_official --backend robotwin "$@"
