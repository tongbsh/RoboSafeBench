#!/usr/bin/env bash
# One-time repo setup: submodules -> patches -> editable install -> doctor.
#
#   bash scripts/setup/setup.sh              # full setup
#   bash scripts/setup/setup.sh --no-doctor  # skip the final self-check
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"

RUN_DOCTOR=1
[[ "${1:-}" == "--no-doctor" ]] && RUN_DOCTOR=0

info "1/4 git submodules"
if [[ -f .gitmodules ]]; then
  git submodule update --init --recursive || warn "submodule init failed (see scripts/run/README.md: RoboTwin URL is machine-local)"
  ok "submodules processed"
else
  warn ".gitmodules absent; skipping"
fi

info "2/4 apply third-party patches"
if [[ -x scripts/setup/apply_patches.sh ]]; then
  bash scripts/setup/apply_patches.sh || warn "apply_patches reported problems"
  bash scripts/setup/apply_patches.sh --check || warn "patch check not clean"
else
  warn "scripts/setup/apply_patches.sh missing"
fi

info "3/4 editable install into '$DYNI_TOOL_ENV'"
# Only the tool env gets an install. Model envs must NOT: `pip install -e .`
# there resolves our core deps and can upgrade a NumPy-1.x env to NumPy 2,
# breaking tensorflow/sapien with "_ARRAY_API not found". Evaluation in a model
# env works via PYTHONPATH=src (see dyni_cli in 00_env.sh) — no install needed.
if [[ "$DYNI_TOOL_ENV" != "base" && -n "${DYNI_ALLOW_MODEL_ENV_INSTALL:-}" ]]; then
  warn "installing into '$DYNI_TOOL_ENV' because DYNI_ALLOW_MODEL_ENV_INSTALL is set"
fi
dyni_activate "$DYNI_TOOL_ENV"
pip install -e ".[dev]" || die "pip install -e '.[dev]' failed"
ok "dynisafety installed: $(command -v dynisafety || echo '(console script not on PATH)')"

info "4/4 environment audit"
bash scripts/setup/audit_envs.sh || warn "some model conda envs are missing (expected for models you do not run)"

if [[ "$RUN_DOCTOR" == 1 ]]; then
  info "running doctor (non-fatal)"
  dyni_cli doctor || warn "doctor reported failures — copy its FIX column to remediate"
fi

ok "setup finished. Next: bash scripts/eval/00_smoke.sh"
