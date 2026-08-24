#!/usr/bin/env bash
# Audit every configs/models/*.yaml conda env against what exists on this machine.
# Exits non-zero if any referenced env is missing.
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

dyni_conda_init
info "conda envs on this machine:"
conda env list | sed 's/^/    /'
echo
info "per-model env requirements (source of truth: configs/models/*.yaml)"
python "$DYNI_ROOT/tools/model_env.py" --audit
