#!/usr/bin/env bash
# Start π₀.₅ openpi serve from the tool env (not scripts/eval/serve.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/lib/env.sh"

OUT_DIR="${OUT_DIR:-$(dyni_artifacts_dir feasibility pi05_aegis_vision_full)}"
PORT="${PORT:-8011}"
mkdir -p "$OUT_DIR"
export DYNI_SERVE_LOG_DIR="$OUT_DIR"
dyni_tee_logs "$OUT_DIR/serve.screen.log"

dyni_activate "$DYNI_TOOL_ENV"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"

echo "$(date -Is) starting pi05 serve port=$PORT" | tee -a "$OUT_DIR/serve.boot.log"
exec python -m dynisafety.cli serve --model pi05 --gpu 0 --port "$PORT"
