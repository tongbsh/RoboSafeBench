#!/bin/bash
# ============================================================
# Start the StarVLA WebSocket inference server (server_policy.py).
#
# Run this BEFORE launching DyniSafety-SimplerEnv with --model starvla.
#
# Layout:
#   SIMPLER_ROOT   …/dynamic_SimplerEnv  (directory that contains scripts/)
#   BENCH_ROOT     parent of SIMPLER_ROOT (workspace with vlaModelHome/ and download_checkpoints/)
#   Server script  ${BENCH_ROOT}/vlaModelHome/starVLA/deployment/model_server/server_policy.py
#
# Usage:
#   bash dynamic_SimplerEnv/scripts/starvla/start_starvla_server.sh [OPTIONS]
#
# Options:
#   --ckpt-path   PATH   Path to checkpoint file (*.pt or *.safetensors)
#                        Default: download_checkpoints/starVLA/Qwen-GR00T-Bridge/checkpoints/steps_45000_pytorch_model.pt
#   --port        PORT   Server port (default: 10093)
#   --env-name    NAME   Conda environment with StarVLA deps (default: starVLA)
#   --use-bf16           Pass --use_bf16 to server for bfloat16 inference (default: ON, matches the
#                        official starVLA examples/SimplerEnv/eval_files/run_policy_server.sh which
#                        always launches the server with --use_bf16). Use --no-bf16 to disable.
#   --no-bf16            Disable bfloat16 inference (force fp32 — slower, ~2x VRAM, NOT recommended;
#                        the released ckpt was trained in bf16 and is consumed in bf16 by official eval)
#   --bench-root  DIR    Override workspace root (default: parent of dynamic_SimplerEnv)
#
# Example:
#   bash scripts/starvla/start_starvla_server.sh \
#       --ckpt-path /path/to/starvla/checkpoints/steps_45000_pytorch_model.pt
#
# After the server is ready, open a second terminal and run:
#   STARVLA_SERVER_HOST=127.0.0.1 STARVLA_SERVER_PORT=10093 \
#   bash scripts/starvla/run_dynisafety_eval.sh --task pick_coke_can --level 1
# WidowX Bridge (DyniSafety-SimplerEnv): use the same --ckpt-path as the server for eval client metadata:
#   bash scripts/starvla/run_bridge_starvla_eval.sh --level 0
# ============================================================

set -euo pipefail

CKPT_PATH="download_checkpoints/starVLA/Qwen-GR00T-Bridge/checkpoints/steps_45000_pytorch_model.pt"
PORT=10093
CONDA_ENV_NAME="starVLA"
# bf16 ON by default to match official starVLA run_policy_server.sh (always passes --use_bf16).
# The released SimplerEnv-Bridge checkpoint was trained/eval'd in bf16; fp32 is technically valid but
# wastes ~2x VRAM and produces subtly different numerics from the official baseline.
USE_BF16=1
BENCH_ROOT_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt-path)   CKPT_PATH="$2";            shift 2 ;;
        --port)        PORT="$2";                 shift 2 ;;
        --env-name)    CONDA_ENV_NAME="$2";       shift 2 ;;
        --use-bf16)    USE_BF16=1;                shift ;;
        --no-bf16)     USE_BF16=0;                shift ;;
        --bench-root)  BENCH_ROOT_OVERRIDE="$2";  shift 2 ;;
        -h|--help)
            sed -n '3,38p' "$0"
            exit 0 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIMPLER_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"      # dynamic_SimplerEnv/
BENCH_ROOT="${BENCH_ROOT_OVERRIDE:-$(cd "${SIMPLER_ROOT}/.." && pwd)}"

# Apply default checkpoint path
if [[ -z "${CKPT_PATH}" ]]; then
    CKPT_PATH="${BENCH_ROOT}/download_checkpoints/starVLA/Qwen-GR00T-Bridge/checkpoints/steps_45000_pytorch_model.pt"
fi

STARVLA_DIR="${BENCH_ROOT}/vlaModelHome/starVLA"
SERVER_SCRIPT="${STARVLA_DIR}/deployment/model_server/server_policy.py"

if [[ ! -f "${SERVER_SCRIPT}" ]]; then
    echo "ERROR: server_policy.py not found at '${SERVER_SCRIPT}'."
    echo "  Clone StarVLA into: ${BENCH_ROOT}/vlaModelHome/starVLA"
    echo "  Or override workspace root with: --bench-root /path/to/workspace"
    exit 1
fi

# Resolve relative checkpoint path
if [[ "${CKPT_PATH}" != /* ]]; then
    CKPT_PATH="${BENCH_ROOT}/${CKPT_PATH}"
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "ERROR: Checkpoint not found: '${CKPT_PATH}'"
    exit 1
fi

set +u
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "${CONDA_PREFIX:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

CMD=(python "${SERVER_SCRIPT}" --ckpt_path "${CKPT_PATH}" --port "${PORT}")
if [[ "${USE_BF16}" -eq 1 ]]; then
    CMD+=(--use_bf16)
fi

echo "============================================================"
echo "StarVLA Inference Server (WebSocket)  [DyniSafety-SimplerEnv]"
echo "  Checkpoint : ${CKPT_PATH}"
echo "  Port       : ${PORT}"
echo "  Conda env  : ${CONDA_ENV_NAME}"
echo "  Work dir   : ${STARVLA_DIR}"
if [[ ${USE_BF16} -eq 1 ]]; then
    echo "  bfloat16   : on  (matches official run_policy_server.sh)"
else
    echo "  bfloat16   : off (NOT recommended; mismatches official run_policy_server.sh)"
fi
echo "============================================================"

cd "${STARVLA_DIR}"
exec "${CMD[@]}"
