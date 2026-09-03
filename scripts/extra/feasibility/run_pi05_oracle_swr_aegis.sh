#!/usr/bin/env bash
# π₀.₅ × DynamicLIBERO：上帝视角 Oracle Stop-Wait-Resume + AEGIS（sim_gt CBF-QP）。
# 自动拉起 openpi serve，再跑 eval。AEGIS 用 enable.yaml，不是 GroundingDINO vision。
#
# 只看计划（不占 GPU）:
#   bash scripts/extra/feasibility/run_pi05_oracle_swr_aegis.sh
# 一键跑:
#   AUTHORIZE_GPU=1 bash scripts/extra/feasibility/run_pi05_oracle_swr_aegis.sh
#
# 常用覆盖:
#   SUITE=safelibero_spatial TASK=0 LEVELS="1 2" EPISODES=1
#   CONDITIONS="oracle_swr oracle_swr_aegis"   # 加上 base 可做三路对照
#   PORT=8011 SKIP_SERVE=1                    # serve 已在跑
#   CAPTURE=1                                 # 同次 rollout 写 robosafe_capture 包
#   DYNI_SAVE_VIDEO=0                         # 关掉评测 mp4（eval 默认就是关）
#
# 输出（与 scripts/eval 同一套 paths.yaml）:
#   评测 JSONL / 视频: out/eval/{backend}/{suite}/l{level}/{run_id}/
#   过程日志:          out/logs/feasibility/pi05_oracle_swr_aegis/
#   capture 包:        out/capture/...（仅 CAPTURE=1；不进 episodes.jsonl）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/lib/env.sh"

SUITE="${SUITE:-safelibero_spatial}"
TASK="${TASK:-0}"
LEVELS="${LEVELS:-1 2}"
EPISODES="${EPISODES:-1}"
SEED="${SEED:-7}"
CONDITIONS="${CONDITIONS:-oracle_swr oracle_swr_aegis}"
HOST="${HOST:-127.0.0.1}"
PORT="${PI05_SERVE_PORT:-${PORT:-8011}}"
SKIP_SERVE="${SKIP_SERVE:-0}"
MATRIX="${MATRIX:-$ROOT/configs/feasibility/libero_pi05_aegis_small.yaml}"
ORACLE_CFG="${ORACLE_CFG:-configs/backends/libero/oracle/stop_wait_resume.yaml}"
AEGIS_CFG="${AEGIS_CFG:-configs/backends/libero/aegis/enable.yaml}"
CAPTURE="${CAPTURE:-${DYNI_CAPTURE:-0}}"
CAPTURE_ROOT="${CAPTURE_ROOT:-}"

_capture_on() {
  case "${CAPTURE,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

print_plan() {
  echo "π₀.₅ DynamicLIBERO 上帝视角 SWR + AEGIS (sim_gt)"
  echo "  suite=$SUITE task=$TASK levels='$LEVELS' episodes=$EPISODES seed=$SEED"
  echo "  conditions='$CONDITIONS' host=$HOST port=$PORT skip_serve=$SKIP_SERVE"
  echo "  oracle=$ORACLE_CFG"
  echo "  aegis=$AEGIS_CFG"
  echo "  capture=$CAPTURE  (packages → ${CAPTURE_ROOT:-<base.yaml dataset_root / out/capture>})"
  echo "  eval products: out/eval/libero/${SUITE}/l{level}/{run_id}/  (same as scripts/eval)"
  echo "  process logs:  out/logs/feasibility/pi05_oracle_swr_aegis/   (kind=feasibility, not eval)"
  echo
  echo "planned jobs:"
  for cond in $CONDITIONS; do
    for level in $LEVELS; do
      echo "  L${level} ${cond}  (pi05 × ${SUITE} t${TASK} × ${EPISODES} ep)"
    done
  done
  echo
  echo "To run: AUTHORIZE_GPU=1 bash $0"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  echo
  print_plan
  exit 0
fi

if [[ "${1:-}" == "--dry-run" || "${AUTHORIZE_GPU:-0}" != "1" ]]; then
  echo "BLOCKED_UNTIL_AUTHORIZED: refusing GPU eval (pass --dry-run or omit AUTHORIZE_GPU)."
  print_plan
  python tools/feasibility_runner.py --matrix "$MATRIX" --dry-run \
    --out "${DRY_OUT:-$ROOT/artifacts/feasibility/pi05_oracle_swr_aegis_dry_run}"
  exit 2
fi

OUT_DIR="${OUT_DIR:-$(dyni_artifacts_dir feasibility pi05_oracle_swr_aegis)}"
RESULTS_ROOT="${RESULTS_ROOT:-$(dyni_results_root)}"
mkdir -p "$OUT_DIR" "$RESULTS_ROOT"
MAIN_LOG="$OUT_DIR/run_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$MAIN_LOG") 2>&1

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export HOST PORT
export DYNI_SAVE_VIDEO="${DYNI_SAVE_VIDEO:-1}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

info "pi05 Oracle-SWR + AEGIS(sim_gt) log: $MAIN_LOG"
info "suite=$SUITE task=$TASK levels='$LEVELS' episodes=$EPISODES conditions='$CONDITIONS' port=$PORT capture=$CAPTURE"
info "eval products → $RESULTS_ROOT/eval/ (same layout as scripts/eval)"
if _capture_on; then
  if [[ ! -f "$ROOT/A_robosafe_capture_datasets/writer/package.py" ]]; then
    warn "CAPTURE=1 but A_robosafe_capture_datasets/ is missing; writer fail-soft, no packages"
  fi
  # Repo-root package is not under src/; keep cwd importable.
  export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi

SERVE_WRAP_PID=""
cleanup() {
  if [[ -n "${SERVE_WRAP_PID:-}" ]]; then
    info "stopping pi05 serve wrap_pid=$SERVE_WRAP_PID"
    kill "$SERVE_WRAP_PID" 2>/dev/null || true
    pkill -P "$SERVE_WRAP_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

wait_serve() {
  local ready=0
  local wait_iters="${DYNI_SERVE_WAIT_ITERS:-180}"
  for _ in $(seq 1 "$wait_iters"); do
    if [[ -n "${SERVE_LOG:-}" ]] && grep -q '\[serve\] ready' "$SERVE_LOG" 2>/dev/null; then
      ready=1
      break
    fi
    if python -c "import socket; socket.create_connection(('${HOST}', int('${PORT}')), 1).close()" \
        >/dev/null 2>&1; then
      ready=1
      break
    fi
    if [[ -n "${SERVE_WRAP_PID:-}" ]] && ! kill -0 "$SERVE_WRAP_PID" 2>/dev/null; then
      tail -n 40 "$SERVE_LOG" || true
      die "pi05 serve exited before ready"
    fi
    sleep 2
  done
  [[ "$ready" == "1" ]] || { tail -n 40 "${SERVE_LOG:-/dev/null}" || true; die "pi05 serve health timeout"; }
}

if [[ "$SKIP_SERVE" == "1" ]]; then
  info "SKIP_SERVE=1 — waiting for existing serve at ${HOST}:${PORT}"
  wait_serve
  ok "existing serve ready at ${HOST}:${PORT}"
else
  SERVE_LOG="$OUT_DIR/serve.log"
  export DYNI_SERVE_LOG_DIR="$OUT_DIR"
  info "starting pi05 serve on port $PORT (cli from $DYNI_TOOL_ENV)"
  dyni_activate "$DYNI_TOOL_ENV"
  export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
  set +e
  python -m dynisafety.cli serve --model pi05 --gpu 0 --port "$PORT" >"$SERVE_LOG" 2>&1 &
  SERVE_WRAP_PID=$!
  set -e
  wait_serve
  ok "pi05 serve ready (wrap_pid=$SERVE_WRAP_PID)"
fi

dyni_activate_for pi05 eval libero
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if _capture_on; then
  export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi

eval_job() {
  local level="$1" cond="$2"
  local extra=(
    --backend libero --model pi05 --suite "$SUITE"
    --task "$TASK" --level "$level" --seed "$SEED" --episodes "$EPISODES"
    --output-root "$RESULTS_ROOT"
    --set "extra.feasibility_condition=${cond}"
    --set extra.server_host="$HOST"
    --set "extra.server_port=${PORT}"
    --set extra.libero.align_obstacle_with_safelibero=true
  )
  if _capture_on; then
    extra+=(--set capture.enabled=true)
    if [[ -n "$CAPTURE_ROOT" ]]; then
      extra+=(--set "capture.dataset_root=${CAPTURE_ROOT}")
    fi
  fi
  case "$cond" in
    base)
      extra+=(--set extra.libero.oracle.enabled=false)
      extra+=(--set extra.libero.aegis.enabled=false)
      ;;
    oracle_swr)
      extra+=(--config "$ORACLE_CFG")
      extra+=(--set extra.libero.oracle.enabled=true)
      extra+=(--set extra.libero.aegis.enabled=false)
      ;;
    oracle_swr_aegis)
      extra+=(--config "$ORACLE_CFG")
      extra+=(--config "$AEGIS_CFG")
      extra+=(--set extra.libero.oracle.enabled=true)
      extra+=(--set extra.libero.aegis.enabled=true)
      ;;
    *)
      die "unknown condition '$cond' (expected base|oracle_swr|oracle_swr_aegis)"
      ;;
  esac
  info "eval L${level} ${cond}"
  python -m dynisafety.cli eval "${extra[@]}"
}

fail=0
for cond in $CONDITIONS; do
  for level in $LEVELS; do
    if ! eval_job "$level" "$cond"; then
      warn "failed: L${level} ${cond}"
      fail=1
    fi
  done
done

MAP_JSONL="$OUT_DIR/episodes_merged.jsonl"
: > "$MAP_JSONL"
while IFS= read -r ep; do
  cat "$ep" >> "$MAP_JSONL"
done < <(find "$RESULTS_ROOT" -name episodes.jsonl -newer "$MAIN_LOG" | sort)
info "merged episode lines: $(wc -l < "$MAP_JSONL" | tr -d ' ')"

if [[ -s "$MAP_JSONL" ]]; then
  python tools/feasibility_runner.py \
    --analyze "$MAP_JSONL" \
    --matrix "$MATRIX" \
    --out "$OUT_DIR/analyze" || warn "analyze failed"
fi

[[ "$fail" -eq 0 ]] || die "one or more eval jobs failed — see $MAIN_LOG"
ok "pi05 Oracle-SWR + AEGIS(sim_gt) finished — see $OUT_DIR"
dyni_print_output_roots feasibility pi05_oracle_swr_aegis
