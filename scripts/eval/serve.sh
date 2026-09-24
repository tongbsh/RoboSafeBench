#!/usr/bin/env bash
# Start a model inference server in the conda env that model declares.
# Keep this shell open; run the evaluation from another shell.
#
#   bash scripts/eval/serve.sh starvla
#   bash scripts/eval/serve.sh lingbot_va --gpu 1 --port 39056
#   bash scripts/eval/serve.sh univla --suite safelibero_object   # per-suite weights
#   bash scripts/eval/serve.sh turbovla --suite safelibero_spatial --gpu 0
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

LOG_DIR="$(dyni_artifacts_dir eval "serve_${MODEL}")"
mkdir -p "$LOG_DIR"
export DYNI_SERVE_LOG_DIR="$LOG_DIR"
dyni_tee_logs "$LOG_DIR/serve.screen.log"
info "process log: $LOG_DIR/"

# The serve CLI imports pandas via dynisafety.analysis. Serve-only conda envs
# (turbovla) do not ship it. Inference still starts in server.env via
# `conda run` inside start_model_server — only the wrapper CLI moves.
if [[ "$MODEL" == "turbovla" ]]; then
  warn "model '$MODEL' serve env has no pandas; running dynisafety serve in eval env"
  dyni_activate_for "$MODEL" eval
else
  dyni_activate_for "$MODEL" serve
fi
info "serving $MODEL $*  (Ctrl-C to stop)"
dyni_cli serve --model "$MODEL" "$@"
