#!/usr/bin/env bash
# Single evaluation run. Activates the conda env the model declares, then calls
# `dynisafety eval`. Server-based models must already be serving (scripts/eval/serve.sh).
#
#   bash scripts/eval/cell.sh libero      openvla_oft  safelibero_spatial 3
#   bash scripts/eval/cell.sh simplerenv  random       bridge             2 --episodes 3
#   bash scripts/eval/cell.sh robotwin    internvla_a1 main20             0 --task 0 --episodes 2
#   bash scripts/eval/cell.sh robocasa    random_mobile robocasa_nav      0 --task 0 --episodes 1
#
# Args: <backend> <model> <suite> <level> [extra dynisafety eval flags...]
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"

if [[ $# -lt 4 ]]; then
  sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
fi

BACKEND="$1"; MODEL="$2"; SUITE="$3"; LEVEL="$4"; shift 4

dyni_activate_for "$MODEL" eval "$BACKEND"

if dyni_needs_server "$MODEL"; then
  warn "model '$MODEL' is server-based; make sure 'bash scripts/eval/serve.sh $MODEL' is running in another shell"
fi

info "eval: backend=$BACKEND model=$MODEL suite=$SUITE level=$LEVEL $*"
dyni_cli eval --backend "$BACKEND" --model "$MODEL" --suite "$SUITE" --level "$LEVEL" "$@"
ok "results under $DYNI_ROOT/results/"
