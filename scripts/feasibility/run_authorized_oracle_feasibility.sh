#!/usr/bin/env bash
# Authorized follow-up: Prompt 11 SIM smoke + Prompt 14 reduced Base/Oracle matrix.
# Waits until GPU has enough free memory, then runs via scripts/run/10_eval.sh.
#
#   bash scripts/feasibility/run_authorized_oracle_feasibility.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/run/00_env.sh"

LOG_DIR="$ROOT/artifacts/feasibility/authorized_run"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/run_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$MAIN_LOG") 2>&1

info "authorized run log: $MAIN_LOG"

wait_for_gpu() {
  local need_mib="${1:-20000}"
  info "waiting for GPU free memory >= ${need_mib} MiB ..."
  while true; do
    local free
    free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    info "GPU free=${free} MiB"
    if [[ "${free:-0}" -ge "$need_mib" ]]; then
      ok "GPU free enough"
      return 0
    fi
    sleep 60
  done
}

run_eval() {
  local level="$1"; shift
  info "EVAL level=$level $*"
  bash "$ROOT/scripts/run/10_eval.sh" libero openvla_oft safelibero_spatial "$level" "$@"
}

wait_for_gpu 20000

info "=== Phase 11 SIM smoke: oracle on, 1 ep x L1/L2/L3, max_steps=80 ==="
for lv in 1 2 3; do
  run_eval "$lv" \
    --task 0 --seed 7 --episodes 1 \
    --config configs/backends/libero/oracle/stop_wait_resume.yaml \
    --set extra.libero.oracle.enabled=true \
    --set runtime.max_steps=80 \
    --set extra.feasibility_condition=oracle_swr_smoke || {
      warn "smoke level $lv failed (exit $?) — continue"
    }
done

info "=== Phase 14 reduced matrix: task0 x L1-L3 x base/oracle x 3 episodes ==="
for lv in 1 2 3; do
  run_eval "$lv" \
    --task 0 --seed 7 --episodes 3 \
    --set extra.libero.oracle.enabled=false \
    --set extra.feasibility_condition=base || warn "base L$lv failed"

  run_eval "$lv" \
    --task 0 --seed 7 --episodes 3 \
    --config configs/backends/libero/oracle/stop_wait_resume.yaml \
    --set extra.libero.oracle.enabled=true \
    --set extra.feasibility_condition=oracle_swr || warn "oracle L$lv failed"
done

info "=== Analyze merged episodes ==="
MAP_JSONL="$LOG_DIR/episodes_merged.jsonl"
: > "$MAP_JSONL"
while IFS= read -r ep; do
  cat "$ep" >> "$MAP_JSONL"
done < <(find "$ROOT/results" -path '*libero-openvla_oft-safelibero_spatial-t0-l[123]*' -name episodes.jsonl | sort)

n_lines="$(wc -l < "$MAP_JSONL" | tr -d ' ')"
info "merged episode lines: $n_lines"

python tools/feasibility_runner.py \
  --analyze "$MAP_JSONL" \
  --matrix configs/feasibility/libero_openvla_oft_small.yaml \
  --out "$LOG_DIR/analyze" || warn "analyze failed"

ok "authorized run finished — see $LOG_DIR and $MAIN_LOG"
