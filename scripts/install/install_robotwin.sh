#!/usr/bin/env bash
# Install / sync the RoboTwin third_party tree (private mirror, M8 D1=C).
#
# The .gitmodules URL may be a machine-local bare repo under external/mirrors/RoboTwin.git.
# If the submodule cannot be fetched, this script can rebuild the working tree
# from the still-vendored external/simulators/RoboTwin (dual-track until M9).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBOTWIN_SHA="afd5f1bbc5759ac1512dc42b3cdaf8a218fb7c00"
ROBOTWIN_TAG="dynisafety-pin-20260806"
TREE="${ROOT}/third_party/RoboTwin"
MIRROR="${ROOT}/external/mirrors/RoboTwin.git"
VENDORED="${ROOT}/external/simulators/RoboTwin"

cd "$ROOT"

ensure_from_submodule() {
  git submodule update --init third_party/RoboTwin
}

rebuild_mirror_from_vendored() {
  echo "==> rebuilding local bare mirror from vendored tree (D1=C bootstrap)"
  if [[ ! -d "$VENDORED" ]]; then
    echo "error: vendored tree missing: $VENDORED" >&2
    exit 1
  fi
  local stage
  stage="$(mktemp -d /tmp/dynisafety_robotwin_pin.XXXXXX)"
  # shellcheck disable=SC2064
  trap "rm -rf '$stage'" RETURN
  rsync -a \
    --exclude 'assets/' \
    --exclude 'checkpoints/' \
    --exclude 'outputs/' \
    --exclude 'data/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude 'envs/dynisafety/' \
    --exclude 'envs/curobo/.git/' \
    --exclude '.git/' \
    "${VENDORED}/" "${stage}/"
  # Strip DyniSafety hooks if present so pin stays clean; patches re-apply them.
  if grep -q 'dynisafety_manager' "${stage}/envs/_base_task.py" 2>/dev/null; then
    echo "warn: vendored _base_task still has hooks; mirror pin should be clean." >&2
    echo "      Prefer existing external/mirrors/RoboTwin.git @ ${ROBOTWIN_TAG}." >&2
  fi
  git -C "$stage" init -b main
  git -C "$stage" add -A
  git -C "$stage" \
    -c user.email='dynisafety@local' \
    -c user.name='DyniSafety Mirror' \
    commit -m "dynisafety private pin bootstrap" >/dev/null
  git -C "$stage" tag -f "$ROBOTWIN_TAG" >/dev/null 2>&1 || true
  mkdir -p "${ROOT}/mirrors"
  rm -rf "$MIRROR"
  git clone --bare "$stage" "$MIRROR"
  echo "    wrote $MIRROR"
}

if [[ ! -d "$TREE/.git" && ! -f "$TREE/.git" ]]; then
  if [[ -d "$MIRROR" ]]; then
    echo "==> cloning from local mirror $MIRROR"
    rm -rf "$TREE"
    git clone "$MIRROR" "$TREE"
  else
    echo "==> no checkout and no mirror; trying submodule update"
    if ! ensure_from_submodule; then
      rebuild_mirror_from_vendored
      rm -rf "$TREE"
      git clone "$MIRROR" "$TREE"
    fi
  fi
fi

if [[ -d "$TREE" ]]; then
  echo "==> checking out pin ${ROBOTWIN_SHA:0:7} / tag ${ROBOTWIN_TAG}"
  git -C "$TREE" fetch --all --quiet 2>/dev/null || true
  if git -C "$TREE" rev-parse "$ROBOTWIN_TAG" >/dev/null 2>&1; then
    git -C "$TREE" checkout "$ROBOTWIN_TAG"
  else
    git -C "$TREE" checkout "$ROBOTWIN_SHA" 2>/dev/null || \
      echo "warn: pin SHA not found in this clone; using current HEAD $(git -C "$TREE" rev-parse --short HEAD)" >&2
  fi
fi

echo
echo "Assets note:"
echo "  RoboTwin simulation assets (~16GB) are NOT in the submodule."
echo "  Place them where the backend expects (historically external/simulators/RoboTwin/assets/),"
echo "  or follow upstream RoboTwin asset download docs — do not commit assets to git."
echo
echo "Next steps:"
echo "  1) ${ROOT}/scripts/install/apply_patches.sh robotwin"
echo "  2) Activate a conda env with sapien + RoboTwin deps"
echo "  3) pip install -e ${ROOT}"
echo
echo "sim_root (YAML): \${third_party}/RoboTwin → ${TREE}"
echo "Private mirror URL in .gitmodules is machine-local until replaced with a hosted remote."
