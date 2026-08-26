#!/usr/bin/env bash
# xvla_official × RoboTwin：tool env 起 dynisafety serve，权重 X-VLA-RoboTwin2。
# 日志：out/logs/debug/xvla_official_robotwin/serve.screen.log
#
#   screen -S xvla_rt_serve
#   bash scripts/debug/robotwin/launch_xvla_official_robotwin_serve.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/../launch_vla_serve.sh" \
  --model xvla_official --backend robotwin "$@"
