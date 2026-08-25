#!/usr/bin/env bash
# LingBot-VA RoboTwin server wrapper for `dynisafety serve` / sweep.
#
# Mirrors external/model_repos/lingbot-va/evaluation/robotwin/launch_server.sh:
#   python -m torch.distributed.run ... wan_va/wan_va_server.py \
#     --config-name robotwin --port <port> --save_root <dir>
#
# Checkpoint is NOT a CLI flag — va_robotwin_cfg reads LINGBOT_VA_MODEL_ROOT.
#
# Paper eval sets dump_artifacts=false → client sends save_visualization=false.
# patches/save_visualization_gate.patch makes the server honor that (no .pt dumps).
# When dumps are enabled, --save_root stays on the data disk under results/.
#
# Usage: bash scripts/extra/models/lingbot_va/serve.sh <port> <checkpoint_dir>
set -euo pipefail

PORT="${1:?usage: serve.sh <port> <checkpoint_dir>}"
CKPT="${2:?usage: serve.sh <port> <checkpoint_dir>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LINGBOT_ROOT="${LINGBOT_VA_ROOT:-${REPO_ROOT}/external/model_repos/lingbot-va}"

[[ -d "$CKPT" ]] || { echo "lingbot serve: checkpoint dir not found: $CKPT" >&2; exit 1; }
[[ -f "$LINGBOT_ROOT/wan_va/wan_va_server.py" ]] || {
  echo "lingbot serve: wan_va_server.py not under $LINGBOT_ROOT" >&2
  exit 1
}

# Idempotently apply local DyniSafety patches onto the lingbot-va checkout.
PATCH_DIR="${SCRIPT_DIR}/patches"
if [[ -d "$PATCH_DIR" ]]; then
  shopt -s nullglob
  for patch in "$PATCH_DIR"/*.patch; do
    if (cd "$LINGBOT_ROOT" && git apply --reverse --check "$patch") >/dev/null 2>&1; then
      : # already applied
    elif (cd "$LINGBOT_ROOT" && git apply --check "$patch") >/dev/null 2>&1; then
      echo "lingbot serve: applying $(basename "$patch")" >&2
      (cd "$LINGBOT_ROOT" && git apply "$patch")
    else
      echo "lingbot serve: patch drifted or conflict: $patch" >&2
      exit 1
    fi
  done
  shopt -u nullglob
fi

export LINGBOT_VA_MODEL_ROOT="$CKPT"
# Fallback path when save_visualization=true (debug). Not used for paper dumps-off.
# Override with LINGBOT_VA_SAVE_ROOT. Never default to system /tmp.
SAVE_ROOT="${LINGBOT_VA_SAVE_ROOT:-${REPO_ROOT}/results/_lingbot_va_vis}"
mkdir -p "$SAVE_ROOT"

# Websocket listens on PORT; torchrun needs a distinct master port.
MASTER_PORT="${LINGBOT_VA_MASTER_PORT:-$((PORT + 5))}"

cd "$LINGBOT_ROOT"
exec python -m torch.distributed.run \
  --nproc_per_node 1 \
  --master_port "$MASTER_PORT" \
  wan_va/wan_va_server.py \
  --config-name robotwin \
  --port "$PORT" \
  --save_root "$SAVE_ROOT"
