#!/usr/bin/env bash
# Debug VLA serve — screen / log 友好入口。
#
# 改编自 artifacts/feasibility/pi05_base_full_alignv2/launch_serve.sh
# 与 scripts/extra/feasibility/run_pi05_aegis_vision_full_serve.sh：
# 在 tool env 里跑 `python -m dynisafety.cli serve`（不要用 scripts/eval/serve.sh）。
#
# 按 suite 加载权重的模型（xvla_official / univla）必须传 --suite，或传 --backend 推断。
#
#   screen -S vla_serve
#   bash scripts/debug/launch_vla_serve.sh --model pi05 --port 8011
#   bash scripts/debug/launch_vla_serve.sh --model xvla_official --backend robotwin
#   bash scripts/debug/launch_vla_serve.sh --model xvla_official --backend simplerenv
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/lib/env.sh"
cd "$ROOT"

MODEL="pi05"
BACKEND=""
PORT=""
GPU="0"
SUITE=""
OUT_DIR=""

usage() {
  cat <<'EOF'
Debug VLA serve（写 serve.screen.log，适合 screen）

  bash scripts/debug/launch_vla_serve.sh --model pi05 --port 8011
  bash scripts/debug/launch_vla_serve.sh --model xvla_official --backend robotwin
  bash scripts/debug/launch_vla_serve.sh --model xvla_official --backend simplerenv

选项:
  --model STEM     configs/models/<stem>.yaml（默认: pi05）
  --backend NAME   libero | robotwin | simplerenv（给 xvla_official / univla 推断 --suite）
  --suite ID       按 suite 加载权重时必填（univla / xvla_official）
  --port  N        默认: xvla* → 8010，其它 → 8011
  --gpu   N        CUDA_VISIBLE_DEVICES（默认: 0）
  --out   DIR      日志目录（默认: artifacts/debug/<model>[_<backend>]_serve）
  -h | --help

一个 serve 只对应一套权重。xvla_official：
  robotwin   → --suite main20   权重 X-VLA-RoboTwin2
  simplerenv → --suite bridge   权重 xvla-simpler-widowx
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --model) MODEL="${2:?}"; shift 2 ;;
    --backend) BACKEND="${2:?}"; shift 2 ;;
    --port)  PORT="${2:?}"; shift 2 ;;
    --gpu)   GPU="${2:?}"; shift 2 ;;
    --suite) SUITE="${2:?}"; shift 2 ;;
    --out)   OUT_DIR="${2:?}"; shift 2 ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

MODEL="${MODEL:-pi05}"
GPU="${GPU:-0}"

if [[ -n "$BACKEND" ]]; then
  case "$BACKEND" in
    libero|robotwin|simplerenv) ;;
    *) die "invalid --backend '$BACKEND'" ;;
  esac
fi

# per-suite 权重：一个进程只能服务一个 suite
if [[ -z "$SUITE" && -n "$BACKEND" ]]; then
  case "$MODEL:$BACKEND" in
    xvla_official:robotwin)   SUITE="main20" ;;
    xvla_official:simplerenv) SUITE="bridge" ;;
    univla:libero)            SUITE="safelibero_spatial" ;;
  esac
fi

needs_suite=0
case "$MODEL" in
  xvla_official|univla) needs_suite=1 ;;
esac
if [[ "$needs_suite" -eq 1 && -z "$SUITE" ]]; then
  die "model '$MODEL' loads weights at serve start; pass --suite or --backend"
fi

if [[ -z "$PORT" ]]; then
  case "$MODEL" in
    xvla|xvla_official) PORT="8010" ;;
    *) PORT="8011" ;;
  esac
fi

# 与 launch_vla_eval.sh 共用目录，方便 eval 读本目录的 serve.screen.log
if [[ -n "$BACKEND" ]]; then
  OUT_DIR="${OUT_DIR:-$ROOT/artifacts/debug/${MODEL}_${BACKEND}}"
else
  OUT_DIR="${OUT_DIR:-$ROOT/artifacts/debug/${MODEL}_libero}"
fi

mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/serve.screen.log") 2>&1

if ! dyni_needs_server "$MODEL"; then
  die "model '$MODEL' is local (transport: local); no server — just run scripts/debug/launch_vla_eval.sh"
fi

dyni_activate "$DYNI_TOOL_ENV"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"

SERVE_ARGS=(--model "$MODEL" --gpu "$GPU" --port "$PORT")
if [[ -n "$SUITE" ]]; then
  SERVE_ARGS+=(--suite "$SUITE")
fi

echo "$(date -Is) starting serve model=$MODEL backend='${BACKEND:-}' gpu=$GPU port=$PORT suite='${SUITE:-}' out=$OUT_DIR" \
  | tee -a "$OUT_DIR/serve.boot.log"
exec python -m dynisafety.cli serve "${SERVE_ARGS[@]}"
