#!/usr/bin/env bash
# Start a model inference server in the conda env that model declares.
# Keep this shell open; run the evaluation from another shell.
#
#   bash scripts/eval/serve.sh starvla
#   bash scripts/eval/serve.sh lingbot_va --gpu 1 --port 39056
#   bash scripts/eval/serve.sh univla --suite safelibero_object   # per-suite weights
#
# Models with per-suite weights load their checkpoint at server startup, so one
# server serves exactly one suite — pass --suite or the CLI refuses to start.
#
# Args: <model> [extra dynisafety serve flags...]
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"

[[ $# -ge 1 ]] || { echo "usage: bash scripts/eval/serve.sh <model> [--gpu N] [--port P]" >&2; exit 2; }
MODEL="$1"; shift

if ! dyni_needs_server "$MODEL"; then
  die "model '$MODEL' is local (transport: local); no server needed — just use scripts/eval/cell.sh"
fi

dyni_activate_for "$MODEL" serve
info "serving $MODEL $*  (Ctrl-C to stop)"
dyni_cli serve --model "$MODEL" "$@"
