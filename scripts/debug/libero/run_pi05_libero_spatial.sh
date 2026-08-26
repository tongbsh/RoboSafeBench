#!/usr/bin/env bash
# π₀.₅ × LIBERO safelibero_spatial × L0–L3（单 VLA / 单 suite）
#
# 走 dynisafety sweep（与 paper_main 同一条调度路径）：
#   - serve：conda run -n openpi（configs/models/pi05.yaml serve_env）
#   - eval： conda run -n openvla-oft（configs/models/pi05.yaml eval_env）
#
# 用法:
#   bash scripts/debug/libero/run_pi05_libero_spatial.sh [episodes] [sweep 额外参数...]
#   bash scripts/debug/libero/run_pi05_libero_spatial.sh           # 默认 10 ep + 存视频
#   bash scripts/debug/libero/run_pi05_libero_spatial.sh 1         # 每格 1 ep
#   bash scripts/debug/libero/run_pi05_libero_spatial.sh 1 --dry-run
#   DYNI_GPUS=0 bash scripts/debug/libero/run_pi05_libero_spatial.sh
#   DYNI_SAVE_VIDEO=0 bash scripts/debug/libero/run_pi05_libero_spatial.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/lib/env.sh"

export DYNI_SAVE_VIDEO="${DYNI_SAVE_VIDEO:-1}"

EPISODES=10
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
  EPISODES="$1"
  shift
fi

SUITE_NAME="debug_pi05_libero_spatial"
SUITE_SRC="$ROOT/configs/suites/${SUITE_NAME}.yaml"
[[ -f "$SUITE_SRC" ]] || die "missing sweep suite: $SUITE_SRC"

SUITE_ARG="$SUITE_SRC"
if [[ "$EPISODES" != "10" ]]; then
  SUITE_ARG="$(dyni_paths sweep-root "$SUITE_NAME")/suite_ep${EPISODES}.yaml"
  mkdir -p "$(dirname "$SUITE_ARG")"
  sed "s/^episodes_per_task: .*/episodes_per_task: ${EPISODES}/" \
    "$SUITE_SRC" >"$SUITE_ARG"
fi

LOG_DIR="$(dyni_artifacts_dir debug "$SUITE_NAME")"
dyni_tee_logs "$LOG_DIR/sweep.screen.log"
info "process log: $LOG_DIR/sweep.screen.log"
info "debug sweep: pi05 × safelibero_spatial × task0 × L0–L3 × ${EPISODES} ep"
info "suite=$SUITE_ARG  save_video=${DYNI_SAVE_VIDEO}  extra=[$*]"
export DYNI_SUITE="$SUITE_ARG"
bash "$ROOT/scripts/eval/sweep.sh" "$@"
dyni_print_output_roots debug "$SUITE_NAME"
