#!/usr/bin/env bash
# xvla_official × SimplerEnv：tool env 起 dynisafety serve，权重 xvla-simpler-widowx。
# 日志：out/logs/debug/xvla_official_simplerenv/serve.screen.log
#
#   screen -S xvla_se_serve
#   bash scripts/debug/simplerenv/launch_xvla_official_simplerenv_serve.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/../launch_vla_serve.sh" \
  --model xvla_official --backend simplerenv "$@"
