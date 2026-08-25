#!/usr/bin/env bash
# Large-scale sweep. Replaces the ~276 legacy shell scripts.
#
#   bash scripts/eval/sweep.sh --dry-run                       # plan only, always do this first
#   bash scripts/eval/sweep.sh                                 # full paper_main table
#   bash scripts/eval/sweep.sh --only backend=libero           # one backend
#   bash scripts/eval/sweep.sh --only model=starvla,level=3    # one cell
#   bash scripts/eval/sweep.sh --resume                        # skip finished jobs
#   bash scripts/eval/sweep.sh --retry-failed                  # rerun only failures
#   DYNI_SUITE=smoke bash scripts/eval/sweep.sh                # different suite
#   DYNI_GPUS=0,1 bash scripts/eval/sweep.sh                   # pin GPUs
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"

SUITE="${DYNI_SUITE:-paper_main}"
dyni_activate "$DYNI_TOOL_ENV"

ARGS=(--suite "$SUITE")
[[ -n "${DYNI_GPUS:-}" ]] && ARGS+=(--gpus "$DYNI_GPUS")
[[ -n "${DYNI_JOBS_PER_GPU:-}" ]] && ARGS+=(--jobs-per-gpu "$DYNI_JOBS_PER_GPU")

warn "sweep launches per-model conda envs itself; server lifecycle is handled by the scheduler."
warn "This is a long job. Consider: nohup bash scripts/eval/sweep.sh > sweep.log 2>&1 &"
info "dynisafety sweep ${ARGS[*]} $*"
dyni_cli sweep "${ARGS[@]}" "$@"

ok "sweep done. Failures (if any): results/_sweep/$SUITE/failures.jsonl"
ok "Next: bash scripts/eval/02_report.sh  (or bash scripts/eval/02_report.sh)"
