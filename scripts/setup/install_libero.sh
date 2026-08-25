#!/usr/bin/env bash
# Install SafeLIBERO / dyniSafety_libero tree (M8 D2=B — install script only, no submodule).
#
# Preference order for the install source:
#   1) DYNISAFETY_LIBERO_SRC (explicit)
#   2) <workspace>/external/simulators/dyniSafety_libero (local gitignored mirror)
#   3) git clone Lifelong-Robot-Learning/LIBERO @ LIBERO_SHA (upstream; may lack DyniSafety assets)
#
# safelibero/ is deprecated (D15=A); do not install from it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIBERO_SHA="8f1084e3132a39270c3a13ebe37270a43ece2a01"
DEST="${ROOT}/third_party/dyniSafety_libero"
LOCAL_FORK="${ROOT}/external/simulators/dyniSafety_libero"
UPSTREAM_URL="${DYNISAFETY_LIBERO_URL:-https://github.com/Lifelong-Robot-Learning/LIBERO.git}"

cd "$ROOT"
mkdir -p "${ROOT}/third_party"

if [[ -d "${ROOT}/external/simulators/safelibero" ]]; then
  echo "note: external/simulators/safelibero is deprecated (D15=A); install uses dyniSafety_libero only."
fi

SRC="${DYNISAFETY_LIBERO_SRC:-}"
MODE=""
if [[ -n "$SRC" ]]; then
  MODE="env"
elif [[ -d "$LOCAL_FORK" ]]; then
  SRC="$LOCAL_FORK"
  MODE="local_fork"
else
  MODE="upstream_clone"
fi

if [[ "$MODE" == "upstream_clone" ]]; then
  echo "==> cloning upstream LIBERO @ ${LIBERO_SHA:0:7}"
  echo "    warning: upstream lacks DyniSafety safelibero_* BDDL / hand_objects / obstacle assets."
  echo "    Prefer copying external/simulators/dyniSafety_libero when available."
  rm -rf "$DEST"
  git clone "$UPSTREAM_URL" "$DEST"
  git -C "$DEST" checkout "$LIBERO_SHA"
else
  echo "==> installing from ${MODE}: $SRC → $DEST"
  mkdir -p "$DEST"
  rsync -a --delete \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.git/' \
    --exclude 'libero.egg-info/' \
    "${SRC%/}/" "${DEST}/"
  # Preserve .git if source was a full clone with history (optional).
  if [[ -d "${SRC}/.git" && ! -e "${DEST}/.git" ]]; then
    echo "    (source had .git; not copied — destination is a plain tree)"
  fi
fi

echo
echo "Next steps:"
echo "  1) Activate your LIBERO/SafeLIBERO conda env"
echo "  2) pip install -e ${DEST}    # or follow upstream setup.py / requirements.txt"
echo "  3) scripts/setup/apply_patches.sh libero   # no-op today (D2=B, no patches/libero)"
echo "  4) pip install -e ${ROOT}"
echo "  5) python -c 'import libero; print(libero.__file__)'"
echo
echo "sim_root (YAML): \${third_party}/dyniSafety_libero → ${DEST}"
echo "Legacy env DYNISAFETY_SIMENV is superseded by YAML extra.libero.sim_root (D7=A)."
