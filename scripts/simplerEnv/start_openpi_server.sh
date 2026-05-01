#!/usr/bin/env bash
# Start OpenPI websocket policy server for π₀ fine-tuned on LIBERO (training config name: pi0_libero).
#
# SimplerEnv / openpi_client expects host:port (default localhost:8000); this script binds 0.0.0.0
# inside serve_policy.py so other containers or hosts can connect if needed.
#
# Checkpoint layout must match OpenPI exports (e.g. same as gs://openpi-assets/checkpoints/pi0_libero).
#
# Usage:
#   export OPENPI_PI0_LIBERO_DIR=/path/to/your/pi0_libero   # required if not using default cache path
#   bash scripts/simplerEnv/start_openpi_server.sh
#
# Default local checkpoint path (same rule as OpenPI downloads):
#   ${OPENPI_DATA_HOME:-~/.cache/openpi}/openpi-assets/checkpoints/pi0_libero
# If you already set OPENPI_DATA_HOME in ~/.bashrc to your custom cache root, this script uses it
# automatically in any shell where that variable is exported (e.g. login shell). From IDE / non-interactive
# bash, run `source ~/.bashrc` first, or export OPENPI_DATA_HOME in that session.
#
# Optional env:
#   OPENPI_ROOT          — openpi repo root (default: <bench>/openpi next to this repo)
#   OPENPI_DATA_HOME     — OpenPI cache root (official env; same as in openpi shared/download.py)
#   OPENPI_SERVER_PORT   — default 8000
#   OPENPI_USE_UV        — if set to 0, use `python` instead of `uv run`
#
# If you actually downloaded π₀.₅-LIBERO instead, use pi05_libero + that checkpoint dir, e.g.:
#   uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=... --port=8000
set -euo pipefail
export OPENPI_USE_UV=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$BENCH_ROOT/vlaModelHome/openpi}"

_CACHE_ROOT="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"
_DEFAULT_PI0_LIBERO_DIR="${_CACHE_ROOT}/openpi-assets/checkpoints/pi0_libero"
POLICY_DIR="${OPENPI_PI0_LIBERO_DIR:-${_DEFAULT_PI0_LIBERO_DIR}}"
PORT="${OPENPI_SERVER_PORT:-8000}"

if [[ ! -d "$OPENPI_ROOT" ]]; then
  echo "ERROR: OPENPI_ROOT is not a directory: $OPENPI_ROOT" >&2
  exit 1
fi

if [[ ! -d "$POLICY_DIR" && "$POLICY_DIR" != gs://* ]]; then
  echo "ERROR: pi0-libero checkpoint not found: $POLICY_DIR" >&2
  if [[ -z "${OPENPI_PI0_LIBERO_DIR:-}" ]]; then
    echo "(Default path uses OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-$HOME/.cache/openpi})" >&2
    echo "If OPENPI_DATA_HOME is only in ~/.bashrc, run: source ~/.bashrc" >&2
  fi
  echo "Set OPENPI_PI0_LIBERO_DIR to the weights directory, or fix OPENPI_DATA_HOME / place files under .../openpi-assets/checkpoints/pi0_libero," >&2
  echo "or use a gs:// URI if you have GCS access." >&2
  exit 1
fi

cd "$OPENPI_ROOT"

# tyro: top-level fields (--port, --env, …) must come *before* the union subcommand `policy:checkpoint`,
# otherwise --port is parsed as part of Checkpoint and rejected.
SERVE_CMD=(
  scripts/serve_policy.py
  --port="$PORT"
  policy:checkpoint
  --policy.config=pi0_libero
  --policy.dir="$POLICY_DIR"
)

if [[ "${OPENPI_USE_UV:-1}" != "0" ]] && command -v uv >/dev/null 2>&1; then
  exec uv run "${SERVE_CMD[@]}"
else
  exec python "${SERVE_CMD[@]}"
fi
