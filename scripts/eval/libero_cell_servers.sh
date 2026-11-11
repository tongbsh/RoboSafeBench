#!/usr/bin/env bash
# Start / stop the policy servers of the generated-cell evaluation (hazardhosts.libero.cell_eval) on one GPU.
#
#   libero_cell_servers.sh start MODEL N [BASE_PORT] [DIR]    MODEL = pi05 | pi0 | openvla_oft
#   libero_cell_servers.sh stop DIR
#
# pi05 / pi0: seeded OpenPI websocket servers (hazardhosts.libero.serve_openpi; OpenPI config name and checkpoint
#   are parameters here: pi05 -> pi05_libero, pi0 -> pi0_libero). Default ports 18010.. (pi05), 18020.. (pi0).
# openvla_oft: hazardhosts.libero.serve_oft in the OFT inference environment (default ports 18030..).
# Pids go to DIR/server_<port>.pid, logs to DIR/server_<port>.log. Run from a checkout with hazardlib/src.
# Machine-specific locations come from the environment (no defaults): pi05/pi0 need OPENPI_PYTHON (an interpreter
# with websockets/numpy), OPENPI_REPO and OPENPI_CKPTS (dir holding pi05_libero/ and pi0_libero/; or PI05_CKPT /
# PI0_CKPT); openvla_oft needs OFT_PYTHON, OFT_REPO and OFT_CKPT. Optional: GPU (default 0), OPENPI_SITE,
# OFT_VIEW_ROOT.
set -u
CMD=${1:?start|stop}
GPU=${GPU:-0}
if [ "$CMD" = stop ]; then
  DIR=${2:?DIR}
  for f in "$DIR"/server_*.pid; do
    [ -e "$f" ] || continue
    pid=$(cat "$f")
    if kill -0 "$pid" 2>/dev/null; then kill "$pid"; echo "stopped $pid ($f)"; fi
    rm -f "$f"
  done
  exit 0
fi
MODEL=${2:?model}; N=${3:?number of servers}
case "$MODEL" in pi05) DEFAULT_BASE=18010;; pi0) DEFAULT_BASE=18020;; openvla_oft) DEFAULT_BASE=18030;; *) echo "unknown model $MODEL"; exit 2;; esac
BASE=${4:-$DEFAULT_BASE}
DIR=${5:-$PWD/_logs/servers_$MODEL}
mkdir -p "$DIR"
export PYTHONPATH="$PWD/hazardlib/src"
for ((i = 0; i < N; i++)); do
  port=$((BASE + i))
  case "$MODEL" in
    pi05|pi0)
      OPENPI_REPO=${OPENPI_REPO:?set OPENPI_REPO}; OPENPI_PYTHON=${OPENPI_PYTHON:?set OPENPI_PYTHON}
      OPENPI_SITE=${OPENPI_SITE:-$OPENPI_REPO/.venv/lib/python3.11/site-packages}
      if [ "$MODEL" = pi05 ]; then CONFIG=pi05_libero; CKPT=${PI05_CKPT:-${OPENPI_CKPTS:?set OPENPI_CKPTS or PI05_CKPT}/pi05_libero}
      else CONFIG=pi0_libero; CKPT=${PI0_CKPT:-${OPENPI_CKPTS:?set OPENPI_CKPTS or PI0_CKPT}/pi0_libero}; fi
      setsid nohup "$OPENPI_PYTHON" -m hazardhosts.libero.serve_openpi \
        --repo "$OPENPI_REPO" --site-packages "$OPENPI_SITE" --checkpoint "$CKPT" --config "$CONFIG" \
        --port "$port" --gpu "$GPU" > "$DIR/server_$port.log" 2>&1 &;;
    openvla_oft)
      OFT_PYTHON=${OFT_PYTHON:?set OFT_PYTHON}
      OFT_REPO=${OFT_REPO:?set OFT_REPO}
      # cwd = the OFT checkout: the official loader syncs the repo's modeling code into the (view) checkpoint from ./
      ( cd "$OFT_REPO" && exec setsid nohup "$OFT_PYTHON" -m hazardhosts.libero.serve_oft \
        --repo "$OFT_REPO" \
        --checkpoint "${OFT_CKPT:?set OFT_CKPT}" \
        --view-root "${OFT_VIEW_ROOT:-$DIR/views}" --port "$port" --gpu "$GPU" > "$DIR/server_$port.log" 2>&1 ) &;;
  esac
  echo $! > "$DIR/server_$port.pid"
  echo "started $MODEL server port=$port pid=$(cat "$DIR/server_$port.pid")"
done
