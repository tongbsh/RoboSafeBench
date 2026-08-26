#!/usr/bin/env bash
# 5-minute self-check: config resolves, tests pass, dry-run plans, then a tiny
# live rollout with the dependency-free `random` policy.
#
#   bash scripts/eval/00_smoke.sh              # offline checks + live random rollout
#   bash scripts/eval/00_smoke.sh --offline    # skip the live rollout (no simulator needed)
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"

OFFLINE=0
[[ "${1:-}" == "--offline" ]] && OFFLINE=1

LOG_DIR="$(dyni_artifacts_dir eval smoke)"
dyni_tee_logs "$LOG_DIR/smoke.screen.log"
info "process log: $LOG_DIR/smoke.screen.log"

dyni_activate "$DYNI_TOOL_ENV"

info "1/4 unit tests (simulator-free)"
python -m pytest -q -m "not sim" || die "unit tests failed"

info "2/4 config resolution"
for b in libero simplerenv robotwin; do
  dyni_cli config print --backend "$b" --level 3 >/dev/null || die "config print failed for $b"
  ok "config resolves: $b L3"
done
dyni_cli config print --backend robocasa --level 0 --suite robocasa_nav --model random_mobile >/dev/null \
  || die "config print failed for robocasa"
ok "config resolves: robocasa L0"

info "3/4 eval dry-run per backend"
dyni_cli eval --backend libero     --model openvla_oft  --suite safelibero_spatial --level 3 --dry-run >/dev/null && ok "dry-run libero"
dyni_cli eval --backend simplerenv --model random       --suite bridge             --level 2 --dry-run >/dev/null && ok "dry-run simplerenv"
dyni_cli eval --backend robotwin   --model internvla_a1 --suite main20             --level 0 --dry-run >/dev/null && ok "dry-run robotwin"
dyni_cli eval --backend robocasa   --model random_mobile --suite robocasa_nav      --level 0 --dry-run >/dev/null && ok "dry-run robocasa"

if [[ "$OFFLINE" == 1 ]]; then
  ok "offline smoke passed (live rollout skipped)"
  exit 0
fi

info "4/4 live rollout: simplerenv x random x bridge_carrot x L2 x 1 episode"
bash scripts/eval/cell.sh simplerenv random bridge 2 --task 0 --episodes 1 --no-video \
  || die "live rollout failed — check that the 'simpler_env' conda env has ManiSkill2 installed"

ok "smoke passed."
dyni_print_output_roots eval smoke
