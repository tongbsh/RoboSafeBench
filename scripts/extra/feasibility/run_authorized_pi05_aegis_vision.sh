#!/usr/bin/env bash
# π₀.₅ + Oracle-SWR vs Oracle-SWR+AEGIS(vision, official-aligned DINO).
# Starts one openpi serve, then 4 eval jobs (L1/L2 × 2 conditions × 3 ep).
#
#   AUTHORIZE_GPU=1 bash scripts/extra/feasibility/run_authorized_pi05_aegis_vision.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

if [[ "${AUTHORIZE_GPU:-0}" != "1" ]]; then
  echo "BLOCKED_UNTIL_AUTHORIZED: refusing GPU eval."
  echo "To run: AUTHORIZE_GPU=1 $0"
  exit 2
fi

# shellcheck disable=SC1091
source "$ROOT/scripts/lib/env.sh"

OUT_DIR="${OUT_DIR:-$ROOT/artifacts/feasibility/pi05_aegis_vision_authorized}"
RESULTS_ROOT="$OUT_DIR/results"
PORT="${PI05_SERVE_PORT:-8010}"
DINO_CFG="${DINO_CFG:-$ROOT/artifacts/feasibility/aegis_vision_gpu_smoke/GroundingDINO_SwinT_OGC.py}"
MATRIX="$ROOT/configs/feasibility/libero_pi05_aegis_vision_small.yaml"

mkdir -p "$OUT_DIR" "$RESULTS_ROOT"
MAIN_LOG="$OUT_DIR/run_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$MAIN_LOG") 2>&1

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export HOST="${HOST:-127.0.0.1}"
export PORT
export DYNI_SAVE_VIDEO="${DYNI_SAVE_VIDEO:-1}"

info "pi05 AEGIS vision run log: $MAIN_LOG"
info "dino_cfg=$DINO_CFG port=$PORT"

if [[ ! -f "$DINO_CFG" ]]; then
  die "missing BERT sidecar DINO config: $DINO_CFG"
fi

# --- serve (openpi via base CLI, same as sweep) ----------------------------
# Do not use scripts/eval/serve.sh here: it activates openpi first, then
# `python -m dynisafety.cli` fails (openpi env has no matplotlib).
SERVE_LOG="$OUT_DIR/serve.log"
info "starting pi05 serve on port $PORT (cli from $DYNI_TOOL_ENV)"
dyni_activate "$DYNI_TOOL_ENV"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
set +e
python -m dynisafety.cli serve --model pi05 --gpu 0 --port "$PORT" >"$SERVE_LOG" 2>&1 &
SERVE_WRAP_PID=$!
set -e

ready=0
for _ in $(seq 1 180); do
  if grep -q '\[serve\] ready' "$SERVE_LOG" 2>/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "$SERVE_WRAP_PID" 2>/dev/null; then
    tail -n 40 "$SERVE_LOG" || true
    die "pi05 serve exited before ready"
  fi
  sleep 2
done
[[ "$ready" == "1" ]] || { tail -n 40 "$SERVE_LOG" || true; die "pi05 serve health timeout"; }
ok "pi05 serve ready (wrap_pid=$SERVE_WRAP_PID)"

cleanup() {
  info "stopping pi05 serve wrap_pid=$SERVE_WRAP_PID"
  kill "$SERVE_WRAP_PID" 2>/dev/null || true
  pkill -P "$SERVE_WRAP_PID" 2>/dev/null || true
}
trap cleanup EXIT

# --- eval (openvla-oft + DINO) --------------------------------------------
dyni_activate_for pi05 eval libero
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

eval_job() {
  local level="$1" cond="$2"
  local extra=(
    --backend libero --model pi05 --suite safelibero_spatial
    --task 0 --level "$level" --seed 7 --episodes 3
    --output-root "$RESULTS_ROOT"
    --set "extra.feasibility_condition=${cond}"
    --set extra.server_host=127.0.0.1
    --set "extra.server_port=${PORT}"
  )
  extra+=(--config configs/backends/libero/oracle/stop_wait_resume.yaml)
  extra+=(--set extra.libero.oracle.enabled=true)
  if [[ "$cond" == "oracle_swr_aegis" ]]; then
    extra+=(--config configs/backends/libero/aegis/enable.yaml)
    extra+=(--config configs/backends/libero/aegis/vision.yaml)
    extra+=(--set extra.libero.aegis.enabled=true)
    extra+=(--set "extra.libero.aegis.vision.grounding_dino_config=${DINO_CFG}")
  else
    extra+=(--set extra.libero.aegis.enabled=false)
  fi
  info "eval L${level} ${cond}"
  python -m dynisafety.cli eval "${extra[@]}"
}

eval_job 1 oracle_swr
eval_job 1 oracle_swr_aegis
eval_job 2 oracle_swr
eval_job 2 oracle_swr_aegis

# --- analyze --------------------------------------------------------------
MAP_JSONL="$OUT_DIR/episodes_merged.jsonl"
: > "$MAP_JSONL"
while IFS= read -r ep; do
  cat "$ep" >> "$MAP_JSONL"
done < <(find "$RESULTS_ROOT" -name episodes.jsonl | sort)
info "merged episode lines: $(wc -l < "$MAP_JSONL" | tr -d ' ')"

python tools/feasibility_runner.py \
  --analyze "$MAP_JSONL" \
  --matrix "$MATRIX" \
  --out "$OUT_DIR/analyze"

ok "pi05 AEGIS vision run finished — see $OUT_DIR"
