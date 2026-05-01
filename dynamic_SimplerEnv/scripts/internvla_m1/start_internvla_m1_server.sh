#!/bin/bash
# ============================================================
# Start the InternVLA-M1 WebSocket inference server (server_policy_M1.py).
#
# Run this BEFORE launching DyniSafety-SimplerEnv with --model internvla_m1.
#
# Layout:
#   SIMPLER_ROOT   …/dynamic_SimplerEnv  (directory that contains scripts/)
#   BENCH_ROOT     parent of SIMPLER_ROOT (workspace with vlaModelHome/ and download_checkpoints/)
#   Server script  ${BENCH_ROOT}/vlaModelHome/InternVLA-M1/deployment/model_server/server_policy_M1.py
#
# Default port is 10094 to avoid clashing with the StarVLA server (10093);
# the official InternVLA-M1 start_server.sh uses 10093 because it doesn't
# co-exist with any other model.
#
# Usage:
#   bash dynamic_SimplerEnv/scripts/internvla_m1/start_internvla_m1_server.sh [OPTIONS]
#
# Options:
#   --ckpt-path   PATH   Path to checkpoint file (*.pt or *.safetensors)
#                        Default: download_checkpoints/InternVLA/InternVLA-M1-Pretrain-RT-1-Bridge/
#                                 checkpoints/<auto-discover *.pt>
#   --port        PORT   Server port (default: 10094)
#   --env-name    NAME   Conda environment with InternVLA-M1 deps (default: internvla_m1)
#   --use-bf16           Pass --use_bf16 to server for bfloat16 inference (default: ON to match
#                        official examples/SimplerEnv/start_server.sh which always passes --use_bf16).
#                        Use --no-bf16 to disable.
#   --no-bf16            Force fp32 inference (NOT recommended; mismatches official setup and
#                        roughly doubles VRAM usage).
#   --bench-root  DIR    Override workspace root (default: parent of dynamic_SimplerEnv)
#
# Example:
#   bash scripts/internvla_m1/start_internvla_m1_server.sh \
#       --ckpt-path /path/to/RUN/checkpoints/steps_50000_pytorch_model.pt
#
# After the server is ready, open a second terminal and run:
#   INTERNVLA_M1_SERVER_HOST=127.0.0.1 INTERNVLA_M1_SERVER_PORT=10094 \
#   bash scripts/internvla_m1/run_bridge_internvla_m1_eval.sh --task bridge_carrot --level 0
# ============================================================

set -euo pipefail

CKPT_PATH="download_checkpoints/InternVLA/InternVLA-M1-Pretrain-RT-1-Bridge/checkpoints/steps_50000_pytorch_model.pt"
PORT=10094
CONDA_ENV_NAME="internvla-m1"
# bf16 ON by default to match official start_server.sh which always passes --use_bf16.
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
            sed -n '3,42p' "$0"
            exit 0 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIMPLER_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"      # dynamic_SimplerEnv/
BENCH_ROOT="${BENCH_ROOT_OVERRIDE:-$(cd "${SIMPLER_ROOT}/.." && pwd)}"

INTERNVLA_M1_DIR="${BENCH_ROOT}/vlaModelHome/InternVLA-M1"
SERVER_SCRIPT="${INTERNVLA_M1_DIR}/deployment/model_server/server_policy_M1.py"

if [[ ! -f "${SERVER_SCRIPT}" ]]; then
    echo "ERROR: server_policy_M1.py not found at '${SERVER_SCRIPT}'." >&2
    echo "  Clone InternVLA-M1 into: ${BENCH_ROOT}/vlaModelHome/InternVLA-M1" >&2
    echo "  Or override workspace root with: --bench-root /path/to/workspace" >&2
    exit 1
fi

# Auto-discover checkpoint if not provided.
if [[ -z "${CKPT_PATH}" ]]; then
    DEFAULT_RUN_DIR="${BENCH_ROOT}/download_checkpoints/InternVLA/InternVLA-M1-Pretrain-RT-1-Bridge"
    if [[ -d "${DEFAULT_RUN_DIR}/checkpoints" ]]; then
        # Prefer the newest *.pt file
        CKPT_PATH="$(ls -1t "${DEFAULT_RUN_DIR}/checkpoints"/*.pt 2>/dev/null | head -n 1 || true)"
    fi
    if [[ -z "${CKPT_PATH}" ]]; then
        echo "ERROR: No checkpoint discovered under '${DEFAULT_RUN_DIR}/checkpoints/*.pt'." >&2
        echo "  Pass --ckpt-path explicitly, or download the ckpt with download_hf.py first." >&2
        echo "  The InternVLA-M1 download must include: checkpoints/*.pt + config.yaml + dataset_statistics.json" >&2
        exit 1
    fi
fi

# Resolve relative path against BENCH_ROOT.
if [[ "${CKPT_PATH}" != /* ]]; then
    CKPT_PATH="${BENCH_ROOT}/${CKPT_PATH}"
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "ERROR: Checkpoint not found: '${CKPT_PATH}'" >&2
    exit 1
fi

# Validate run directory layout (matches InternVLA.model.framework.share_tools.read_mode_config).
CKPT_DIR="$(dirname "${CKPT_PATH}")"
RUN_DIR="$(dirname "${CKPT_DIR}")"
if [[ "$(basename "${CKPT_DIR}")" != "checkpoints" ]]; then
    echo "ERROR: ckpt must live under '<RUN>/checkpoints/' (got parent: $(basename "${CKPT_DIR}"))" >&2
    exit 1
fi
for f in config.yaml dataset_statistics.json; do
    if [[ ! -f "${RUN_DIR}/${f}" ]]; then
        echo "ERROR: missing required sibling artifact: ${RUN_DIR}/${f}" >&2
        echo "  InternVLA-M1.from_pretrained asserts both files exist next to checkpoints/." >&2
        echo "  Re-download with download_hf.py without restrictive allow_patterns." >&2
        exit 1
    fi
done

set +u
# shellcheck disable=SC1091
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "${CONDA_PREFIX:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

# Match the upstream launch pattern so top-level imports like
# `deployment.*` and `InternVLA.*` resolve from the repo root.
export PYTHONPATH="${INTERNVLA_M1_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
CMD=(python -m deployment.model_server.server_policy_M1 --ckpt_path "${CKPT_PATH}" --port "${PORT}")
if [[ "${USE_BF16}" -eq 1 ]]; then
    CMD+=(--use_bf16)
fi

echo "============================================================"
echo "InternVLA-M1 Inference Server (WebSocket)  [DyniSafety-SimplerEnv]"
echo "  Checkpoint : ${CKPT_PATH}"
echo "  Run dir    : ${RUN_DIR}"
echo "  Port       : ${PORT}"
echo "  Conda env  : ${CONDA_ENV_NAME}"
echo "  Work dir   : ${INTERNVLA_M1_DIR}"
if [[ ${USE_BF16} -eq 1 ]]; then
    echo "  bfloat16   : on  (matches official examples/SimplerEnv/start_server.sh)"
else
    echo "  bfloat16   : off (NOT recommended; mismatches official start_server.sh)"
fi
echo "============================================================"

cd "${INTERNVLA_M1_DIR}"
exec "${CMD[@]}"
