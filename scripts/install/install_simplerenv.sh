#!/usr/bin/env bash
# Install / sync the SimplerEnv-OpenVLA third_party tree for the simplerenv backend.
#
# Does not hardcode machine paths. Expects to run from a DyniSafety workspace clone.
#
# Steps:
#   1) Ensure submodule third_party/SimplerEnv-OpenVLA @ ccfe380
#   2) Init nested ManiSkill2_real2sim
#   3) Remind: scripts/install/apply_patches.sh simplerenv
#   4) Print pip install hints (user's conda env)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SIMPLERENV_SHA="ccfe3809766839a2fcfb7a3d3c9abff585189188"
MANISKILL2_SHA="91d154bfd864577f8d2e80f3fc2f8b4d9df9ae5c"
TREE="${ROOT}/third_party/SimplerEnv-OpenVLA"

cd "$ROOT"

if [[ ! -e .gitmodules ]]; then
  echo "error: missing .gitmodules — run M8 submodule setup first" >&2
  exit 1
fi

echo "==> syncing submodule third_party/SimplerEnv-OpenVLA"
git submodule update --init --recursive third_party/SimplerEnv-OpenVLA || {
  echo "warn: git submodule update failed; checking existing tree at $TREE" >&2
}

if [[ ! -d "$TREE" ]]; then
  echo "error: $TREE missing after submodule update" >&2
  exit 1
fi

head="$(git -C "$TREE" rev-parse HEAD)"
if [[ "$head" != "$SIMPLERENV_SHA" ]]; then
  echo "==> checking out SimplerEnv pin ${SIMPLERENV_SHA:0:7} (was ${head:0:7})"
  git -C "$TREE" fetch --all --quiet 2>/dev/null || true
  git -C "$TREE" checkout "$SIMPLERENV_SHA"
fi

if [[ -e "${TREE}/ManiSkill2_real2sim" ]]; then
  echo "==> ensuring ManiSkill2_real2sim submodule present"
  git -C "$TREE" submodule update --init ManiSkill2_real2sim || true
  echo "==> ManiSkill2 pin applied by scripts/install/apply_patches.sh (target ${MANISKILL2_SHA:0:7})"
fi

echo
echo "Next steps:"
echo "  1) ${ROOT}/scripts/install/apply_patches.sh simplerenv"
echo "  2) Activate your simplerenv/ManiSkill conda env (needs sapien)."
echo "  3) pip install -r ${TREE}/requirements_full_install.txt   # after patches"
echo "  4) pip install -e ${ROOT}"
echo "  5) Optional: python ${ROOT}/scripts/install/download_assets.py --list"
echo
echo "sim_root (YAML): \${third_party}/SimplerEnv-OpenVLA → ${TREE}"
