#!/usr/bin/env bash
# Unified backend install entry (M10) — wraps per-backend install scripts.
#
# Usage:
#   scripts/setup/install_backend.sh <libero|simplerenv|robotwin|robocasa> [--skip-pip] [--skip-patch]
#
# Steps:
#   1) scripts/setup/install_<backend>.sh
#   2) unless --skip-patch: scripts/setup/apply_patches.sh <backend>
#      (robocasa: no DyniSafety patches this window; patch step is skipped)
#   3) unless --skip-pip:
#        libero|simplerenv|robotwin: pip install -e ".[<backend>]"
#        robocasa: pip install -e . --no-deps && pip install -r requirements/robocasa.txt
#        (must not use pip install -e . — pyproject.toml pins numpy<2)
#   4) print next: dynisafety doctor --backend <backend>
#
# Exit codes:
#   0  success
#   1  install/patch/pip failure (propagated)
#   2  usage / unknown backend
#
# Do NOT install multiple backends into one env (sapien 2 vs 3 / MuJoCo mutex).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BACKEND=""
SKIP_PIP=0
SKIP_PATCH=0

usage() {
  sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --skip-pip)
      SKIP_PIP=1
      shift
      ;;
    --skip-patch)
      SKIP_PATCH=1
      shift
      ;;
    libero|simplerenv|robotwin|robocasa)
      if [[ -n "$BACKEND" ]]; then
        echo "error: backend already set to ${BACKEND}" >&2
        exit 2
      fi
      BACKEND="$1"
      shift
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$BACKEND" ]]; then
  echo "error: backend required (libero|simplerenv|robotwin|robocasa)" >&2
  usage >&2
  exit 2
fi

cd "$ROOT"

INSTALL_SCRIPT="${HERE}/install_${BACKEND}.sh"
if [[ ! -x "$INSTALL_SCRIPT" && ! -f "$INSTALL_SCRIPT" ]]; then
  echo "error: missing ${INSTALL_SCRIPT}" >&2
  exit 1
fi

echo "==> install_backend: ${BACKEND}"
echo "==> running ${INSTALL_SCRIPT}"
bash "$INSTALL_SCRIPT"

if [[ "$SKIP_PATCH" -eq 0 ]]; then
  if [[ "$BACKEND" == "robocasa" ]]; then
    echo "==> skip patch (robocasa has no DyniSafety patches this window)"
  else
    echo "==> apply_patches.sh ${BACKEND}"
    bash "${HERE}/apply_patches.sh" "$BACKEND"
  fi
else
  echo "==> skip patch (--skip-patch)"
fi

if [[ "$SKIP_PIP" -eq 0 ]]; then
  if [[ "$BACKEND" == "robocasa" ]]; then
    echo "==> pip install -e . --no-deps  (keep numpy==2.2.5; skip pyproject numpy<2)"
    pip install -e . --no-deps
    echo "==> pip install -r requirements/robocasa.txt"
    pip install -r "${ROOT}/requirements/robocasa.txt"
  else
    echo "==> pip install -e \".[${BACKEND}]\""
    pip install -e ".[${BACKEND}]"
  fi
else
  echo "==> skip pip (--skip-pip)"
fi

echo
echo "Next:"
echo "  # activate the matching conda env (see environment/dynisafety-${BACKEND}.yml)"
echo "  dynisafety doctor --backend ${BACKEND}"
echo "  # optional: dynisafety doctor --backend ${BACKEND} --strict"
echo
echo "Reminder: do not combine libero/simplerenv/robotwin extras in one env."
echo "RoboCasa uses a separate env (dynisafety-robocasa, numpy==2.2.5, MUJOCO_GL=egl)."
echo "See requirements/README.md and docs/user/third_party.md."
