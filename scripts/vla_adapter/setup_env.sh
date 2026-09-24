#!/usr/bin/env bash
# Clone VLA-Adapter, create conda env vla-adapter, pip install for Pro *eval serve*.
# Does NOT install LIBERO simulation (eval stays in openvla-oft).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="${VLA_ADAPTER_ROOT:-$ROOT/external/model_repos/VLA-Adapter}"
ENV_NAME="${VLA_ADAPTER_ENV:-vla-adapter}"
LOG="${VLA_ADAPTER_SETUP_LOG:-$ROOT/out/setup/vla-adapter-setup.log}"
# pip/tmp caches stay outside the git working tree.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/pip}"
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "$(dirname "$LOG")" "$ROOT/external/model_repos" "$PIP_CACHE_DIR"

exec > >(tee -a "$LOG") 2>&1
echo "===== vla-adapter-setup START $(date -Iseconds) ====="

enable_github_proxy() {
  if [[ -f /etc/network_turbo ]]; then
    # shellcheck disable=SC1091
    source /etc/network_turbo || true
  fi
}

disable_github_proxy() {
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
}

enable_github_proxy

clone_repo() {
  local dest="$1"
  local url="$2"
  if [[ -d "$dest/.git" ]]; then
    echo "[clone] already present: $dest"
    git -C "$dest" rev-parse --short HEAD
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  if git clone --depth 1 "$url" "$dest"; then
    return 0
  fi
  rm -rf "$dest"
  return 1
}

echo "[1/5] clone OpenHelix-Team/VLA-Adapter"
if ! clone_repo "$REPO" "https://github.com/OpenHelix-Team/VLA-Adapter.git"; then
  echo "[1/5] retry via ghfast.top"
  clone_repo "$REPO" "https://ghfast.top/https://github.com/OpenHelix-Team/VLA-Adapter.git"
fi
echo "[1/5] clone done, HEAD=$(git -C "$REPO" rev-parse --short HEAD)"

# GitHub proxy makes pytorch.org / PyPI much slower (AutoDL notice).
disable_github_proxy

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[2/5] conda env $ENV_NAME already exists — skip create"
else
  echo "[2/5] conda create -n $ENV_NAME python=3.10"
  conda create -n "$ENV_NAME" python=3.10 -y
fi

PIP=(conda run -n "$ENV_NAME" --no-capture-output python -m pip)
echo "[3/5] torch==2.2.0 (cu121) without GitHub proxy"
"${PIP[@]}" install --upgrade pip
# --no-deps: torchvision would otherwise pull numpy from the cu121 index
# (very slow). tensorflow 2.15 needs numpy 1.x anyway.
"${PIP[@]}" install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu121 --no-deps
"${PIP[@]}" install filelock typing-extensions sympy networkx jinja2 fsspec \
  pillow requests 'numpy==1.26.4'
# CUDA runtime wheels that --no-deps skipped (libcudnn.so.8 etc.).
"${PIP[@]}" install \
  nvidia-cuda-nvrtc-cu12==12.1.105 \
  nvidia-cuda-runtime-cu12==12.1.105 \
  nvidia-cuda-cupti-cu12==12.1.105 \
  nvidia-cudnn-cu12==8.9.2.26 \
  nvidia-cublas-cu12==12.1.3.1 \
  nvidia-cufft-cu12==11.0.2.54 \
  nvidia-curand-cu12==10.3.2.106 \
  nvidia-cusolver-cu12==11.4.5.107 \
  nvidia-cusparse-cu12==12.1.0.106 \
  nvidia-nccl-cu12==2.19.3 \
  nvidia-nvtx-cu12==12.1.105 \
  triton==2.2.0

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/pip}"
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "$PIP_CACHE_DIR"

echo "[4/5] eval deps from PyPI (skip git dlimp; never clone into the repo)"
# Full `pip install -e .` pulls git+https://github.com/moojink/dlimp_openvla
# and is flaky on AutoDL. Eval only needs the package on PYTHONPATH plus PyPI
# wheels. dlimp may already live in site-packages (import can still fail
# until tensorflow-metadata is pinned). Only clone if the package dir is gone.
if ! conda run -n "$ENV_NAME" python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('dlimp') else 1)" >/dev/null 2>&1; then
  DLIMP="$TMPDIR/dlimp_openvla"
  rm -rf "$DLIMP"
  enable_github_proxy
  git clone --depth 1 "https://ghfast.top/https://github.com/moojink/dlimp_openvla.git" "$DLIMP" \
    || git clone --depth 1 "https://github.com/moojink/dlimp_openvla.git" "$DLIMP"
  disable_github_proxy
  "${PIP[@]}" install --no-deps "$DLIMP"
  rm -rf "$DLIMP"
fi
"${PIP[@]}" install -e "$REPO" --no-deps
# tensorflow 2.15 needs protobuf<5. tfds 4.9.10 pulled metadata 1.21 which
# needs protobuf 5 (`runtime_version`). metadata 1.14.0 wants protobuf<4.21.
# 1.16.1 is generated for protobuf 4. wandb>=0.19 wants protobuf>=5 — pin 0.17.9.
"${PIP[@]}" install \
  "accelerate>=0.25.0" \
  "draccus==0.8.0" \
  "huggingface_hub<1" \
  "json-numpy" \
  "jsonlines" \
  "peft==0.11.1" \
  "rich" \
  "sentencepiece==0.1.99" \
  "timm==0.9.10" \
  "tokenizers==0.19.1" \
  "transformers==4.40.1" \
  "diffusers==0.30.3" \
  "imageio" \
  "tensorflow_graphics==2021.12.3" \
  "wandb==0.17.9" \
  "msgpack" \
  "msgpack-numpy"
"${PIP[@]}" install --no-deps "tensorflow-metadata==1.16.1" "protobuf==4.25.9"

echo "[4/5] packaging ninja + flash-attn==2.5.5 (optional for eval)"
"${PIP[@]}" install packaging ninja
if ! "${PIP[@]}" install "flash-attn==2.5.5" --no-build-isolation; then
  echo "[4/5] flash-attn build failed; eval can use SDPA (get_vla comments it out)"
fi

echo "[5/5] sanity: tensorflow + torch"
conda run -n "$ENV_NAME" --no-capture-output python - <<'PY'
import tensorflow as tf
import torch
print("tensorflow", tf.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY

echo "===== vla-adapter-setup DONE $(date -Iseconds) ====="
