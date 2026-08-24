#!/usr/bin/env bash
# A2 authorized follow-up: Base / Oracle-SWR / Oracle-SWR+AEGIS small matrix.
#
# DEFAULT: refuse GPU launch (BLOCKED_UNTIL_AUTHORIZED).
# After human approval:
#   AUTHORIZE_GPU=1 bash scripts/feasibility/run_authorized_aegis_feasibility.sh
#
# Dry-run (no GPU) is always safe:
#   python tools/feasibility_runner.py \
#     --matrix configs/feasibility/libero_openvla_oft_aegis_small.yaml \
#     --dry-run --out artifacts/feasibility/aegis_dry_run
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MATRIX="${MATRIX:-configs/feasibility/libero_openvla_oft_aegis_small.yaml}"
OUT_DIR="${OUT_DIR:-$ROOT/artifacts/feasibility/aegis_authorized_run}"
DRY_OUT="${DRY_OUT:-$ROOT/artifacts/feasibility/aegis_dry_run}"

if [[ "${AUTHORIZE_GPU:-0}" != "1" ]]; then
  echo "BLOCKED_UNTIL_AUTHORIZED: refusing GPU eval."
  echo "Emit dry-run plan + commands instead → $DRY_OUT"
  python tools/feasibility_runner.py --matrix "$MATRIX" --dry-run --out "$DRY_OUT"
  echo "To run after approval: AUTHORIZE_GPU=1 $0"
  exit 2
fi

source "$ROOT/scripts/run/00_env.sh"
dyni_activate_for openvla_oft eval libero
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUT_DIR"
MAIN_LOG="$OUT_DIR/run_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$MAIN_LOG") 2>&1

info "authorized A2 aegis matrix log: $MAIN_LOG"
info "matrix=$MATRIX"
info "python=$(command -v python)"

# Refresh command list from the matrix YAML (source of truth).
python tools/feasibility_runner.py --matrix "$MATRIX" --dry-run --out "$OUT_DIR/dry_plan"
CMDS="$OUT_DIR/dry_plan/eval_commands.sh"
if [[ ! -f "$CMDS" ]]; then
  echo "missing $CMDS" >&2
  exit 1
fi

RESULTS_ROOT="$OUT_DIR/results"
mkdir -p "$RESULTS_ROOT"
# Isolate from prior vision/debug runs under $ROOT/results, and bind the
# active interpreter (do not use base's `dynisafety` console script).
python - "$CMDS" "$RESULTS_ROOT" <<'PY'
from pathlib import Path
import sys
cmds, root = Path(sys.argv[1]), sys.argv[2]
lines = []
for line in cmds.read_text(encoding="utf-8").splitlines():
    if line.startswith("dynisafety eval"):
        line = "python -m dynisafety.cli eval" + line[len("dynisafety eval"):]
        line += f" --output-root {root}"
    lines.append(line)
cmds.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

info "=== Executing eval_commands.sh (authorized, results → $RESULTS_ROOT) ==="
bash "$CMDS" || warn "one or more eval jobs failed — continue to analyze"

info "=== Analyze merged episodes ==="
MAP_JSONL="$OUT_DIR/episodes_merged.jsonl"
: > "$MAP_JSONL"
while IFS= read -r ep; do
  cat "$ep" >> "$MAP_JSONL"
done < <(find "$RESULTS_ROOT" -name episodes.jsonl | sort)

n_lines="$(wc -l < "$MAP_JSONL" | tr -d ' ')"
info "merged episode lines: $n_lines"

python tools/feasibility_runner.py \
  --analyze "$MAP_JSONL" \
  --matrix "$MATRIX" \
  --out "$OUT_DIR/analyze" || warn "analyze failed"

ok "authorized A2 run finished — see $OUT_DIR and $MAIN_LOG"
