#!/usr/bin/env bash
# xvla_official × SimplerEnv bridge：等 serve ready 后按 task × level 循环 eval。
# --task all = 论文 0,1,2（carrot / stack / spoon；eggplant=3 需显式传入）
#
#   screen -S xvla_se_eval
#   bash scripts/debug/simplerenv/launch_xvla_official_simplerenv_eval.sh --level 0 --task 0 --episodes 1
#   bash scripts/debug/simplerenv/launch_xvla_official_simplerenv_eval.sh --level 0,1,2,3 --task all
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/../launch_vla_eval.sh" \
  --model xvla_official --backend simplerenv "$@"
