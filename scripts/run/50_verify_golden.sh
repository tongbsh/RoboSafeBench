#!/usr/bin/env bash
# Refactor-correctness check: replay archived pre-refactor results and compare
# SR / CR / peak force. Proves the refactor did not change physics or metrics.
# NOT about preserving paper numbers — the paper is being rerun from scratch.
#
#   bash scripts/run/50_verify_golden.sh            # run the sim-marked golden tests
#   bash scripts/run/50_verify_golden.sh --extract  # just unpack the archive for inspection
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

[[ -f "$DYNI_GOLDEN_TAR" ]] || die "golden archive not found: $DYNI_GOLDEN_TAR (set DYNI_GOLDEN_TAR)"

if [[ "${1:-}" == "--extract" ]]; then
  DEST="${2:-/tmp/dynisafety_golden}"
  mkdir -p "$DEST"
  tar -xzf "$DYNI_GOLDEN_TAR" -C "$DEST"
  ok "extracted to $DEST"
  info "per-backend log counts:"
  for d in dynisafety_libero dynamic_SimplerEnv dynamic_RoboTwin; do
    printf '    %-22s %s files\n' "$d" "$(find "$DEST/$d" -type f 2>/dev/null | wc -l)"
  done
  exit 0
fi

dyni_activate "$DYNI_TOOL_ENV"

info "golden fixtures committed in the repo:"
ls -1 "$DYNI_ROOT"/tests/golden/*.json 2>/dev/null | sed 's|.*/|    |' || warn "no golden fixtures found"

info "running sim-marked consistency tests"
warn "these need the simulators installed and model weights present; they xfail/skip otherwise"
python -m pytest -q -m sim tests/test_golden_consistency.py "$@" || {
  warn "golden tests did not all pass — see old_cache/docs_internal/HANDOFF.md item P1-1 (local)"
  exit 1
}
ok "golden consistency verified"
