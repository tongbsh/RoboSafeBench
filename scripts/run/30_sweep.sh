#!/usr/bin/env bash
# Large-scale sweep. Replaces the ~276 legacy shell scripts.
#
#   bash scripts/run/30_sweep.sh --dry-run                       # plan only, always do this first
#   bash scripts/run/30_sweep.sh                                 # full paper_main table
#   bash scripts/run/30_sweep.sh --only backend=libero           # one backend
#   bash scripts/run/30_sweep.sh --only model=starvla,level=3    # one cell
#   bash scripts/run/30_sweep.sh --resume                        # skip finished jobs
#   bash scripts/run/30_sweep.sh --retry-failed                  # rerun only failures
#   DYNI_SUITE=smoke bash scripts/run/30_sweep.sh                # different suite
#   DYNI_GPUS=0,1 bash scripts/run/30_sweep.sh                   # pin GPUs
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

SUITE="${DYNI_SUITE:-paper_main}"
dyni_activate "$DYNI_TOOL_ENV"

ARGS=(--suite "$SUITE")
[[ -n "${DYNI_GPUS:-}" ]] && ARGS+=(--gpus "$DYNI_GPUS")
[[ -n "${DYNI_JOBS_PER_GPU:-}" ]] && ARGS+=(--jobs-per-gpu "$DYNI_JOBS_PER_GPU")

warn "sweep launches per-model conda envs itself; server lifecycle is handled by the scheduler."
warn "This is a long job. Consider: nohup bash scripts/run/30_sweep.sh > sweep.log 2>&1 &"
info "dynisafety sweep ${ARGS[*]} $*"
dyni_cli sweep "${ARGS[@]}" "$@"

ok "sweep done. Failures (if any): results/_sweep/$SUITE/failures.jsonl"
ok "Next: bash scripts/run/40_report.sh"
