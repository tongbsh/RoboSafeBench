#!/bin/bash
# ============================================================
# DyniSafety-SimplerEnv - Dream-VLA HTTP server (best-effort zero-shot WidowX Bridge)
#
# Wraps experiments/robot/dynamic_SimplerEnv/deploy_dreamvla_server.py inside the
# vendored Dream-VLX repo. The released ``Dream-org/Dream-VLA-7B`` weights have
# no SimplerEnv-Bridge eval upstream, but the README ships a zero-shot
# Widow-X Bridge quick-start (``unnorm_key=bridge_orig``); this server uses
# that recipe.
#
# Layout (defaults):
#   SIMPLER_ROOT     ../dynamic_SimplerEnv
#   BENCH_ROOT       parent of SIMPLER_ROOT (workspace with vlaModelHome/)
#   Default ckpt     <BENCH_ROOT>/download_checkpoints/dream-vla
#                    (or set DREAMVLA_CHECKPOINT)
#   Dream-VLX home   ${DREAMVLA_VLA_HOME:-${BENCH_ROOT}/vlaModelHome/Dream-VLX/vla}
#
# Conda env: dreamvla. Required pip extras (run once in the env):
#   pip install fastapi uvicorn flash-attn
#
# Usage:
#   bash dynamic_SimplerEnv/scripts/dreamvla/start_dreamvla_server.sh [OPTIONS]
#
# Options:
#   --pretrained-checkpoint PATH  HF-format Dream-VLA checkpoint dir
#                                 (default: download_checkpoints/dream-vla; you can also
#                                  set DREAMVLA_CHECKPOINT in the env)
#   --unnorm-key            KEY   Default: bridge_orig (matches README zero-shot Bridge).
#   --policy-setup          NAME  widowx_bridge (default) | google_robot
#   --port                  PORT  Default: 8030
#   --env-name              NAME  Conda env (default: dreamvla)
#   --no-flash-attn               Use attn_implementation=sdpa instead of flash_attention_2.
#                                 Set this if your dreamvla env doesn't have flash-attn built.
#   --no-center-crop              Skip the OpenVLA-style 0.9 center crop. Default is ON
#                                 to match Dream-VLA LIBERO eval (parallel_eval_dreamvla.sh
#                                 uses --center_crop True).
#   --do-sample                   Pass do_sample=True to predict_action (default: False,
#                                 matches README quick-start).
#   --action-chunk          N     predict_action chunk size (default: 1 = closed-loop, matches official example).
#                                 chunk>1 causes open-loop drift on Bridge pick-and-place tasks: the model
#                                 predicts "reach→grasp→retract" in N steps from one observation, but
#                                 the grasp fails and the robot returns empty-handed. Keep at 1 for Bridge.
#   --bench-root            DIR   Override workspace root (default: parent of dynamic_SimplerEnv).
#
# Env:
#   DREAMVLA_ATTN_IMPLEMENTATION   Override attention backend (default: flash_attention_2,
#                                  or sdpa if --no-flash-attn).
#   DREAMVLA_CHECKPOINT            Default checkpoint dir (overrides built-in path).
#   DREAMVLA_VLA_HOME              Override Dream-VLX vla repo root.
#
# After the server is ready, open a second terminal and run:
#   DREAMVLA_SERVER_HOST=127.0.0.1 DREAMVLA_SERVER_PORT=8030 \
#   bash scripts/dreamvla/run_bridge_dreamvla_eval.sh --task bridge_carrot --level 0
# ============================================================

set -euo pipefail

PORT=8030
CONDA_ENV_NAME="dreamvla"
PRETRAINED_CHECKPOINT="download_checkpoints/dream-vla"
UNNORM_KEY="bridge_orig"
POLICY_SETUP="widowx_bridge"
USE_FLASH_ATTN=1
USE_CENTER_CROP=1
DO_SAMPLE=0
ACTION_CHUNK=1   # Bridge: closed-loop step-by-step (official example). chunk>1 causes open-loop drift.
BENCH_ROOT_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pretrained-checkpoint) PRETRAINED_CHECKPOINT="$2"; shift 2 ;;
        --unnorm-key)            UNNORM_KEY="$2";            shift 2 ;;
        --policy-setup)          POLICY_SETUP="$2";          shift 2 ;;
        --port)                  PORT="$2";                  shift 2 ;;
        --env-name)              CONDA_ENV_NAME="$2";        shift 2 ;;
        --no-flash-attn)         USE_FLASH_ATTN=0;           shift ;;
        --no-center-crop)        USE_CENTER_CROP=0;          shift ;;
        --do-sample)             DO_SAMPLE=1;                shift ;;
        --action-chunk)          ACTION_CHUNK="$2";          shift 2 ;;
        --bench-root)            BENCH_ROOT_OVERRIDE="$2";   shift 2 ;;
        -h|--help)
            sed -n '3,55p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIMPLER_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="${BENCH_ROOT_OVERRIDE:-$(cd "${SIMPLER_ROOT}/.." && pwd)}"
DREAMVLA_HOME="${DREAMVLA_VLA_HOME:-${BENCH_ROOT}/vlaModelHome/Dream-VLX/vla}"

if [[ -z "${PRETRAINED_CHECKPOINT}" ]]; then
    if [[ -n "${DREAMVLA_CHECKPOINT:-}" ]]; then
        PRETRAINED_CHECKPOINT="${DREAMVLA_CHECKPOINT}"
    elif [[ -d "${BENCH_ROOT}/download_checkpoints/dream-vla" ]]; then
        PRETRAINED_CHECKPOINT="${BENCH_ROOT}/download_checkpoints/dream-vla"
    else
        PRETRAINED_CHECKPOINT="download_checkpoints/dream-vla"
    fi
fi

# Resolve relative path against BENCH_ROOT.
if [[ "${PRETRAINED_CHECKPOINT}" != /* ]]; then
    PRETRAINED_CHECKPOINT="${BENCH_ROOT}/${PRETRAINED_CHECKPOINT}"
fi

if [[ ! -d "${PRETRAINED_CHECKPOINT}" ]]; then
    echo "ERROR: Dream-VLA checkpoint dir not found: '${PRETRAINED_CHECKPOINT}'" >&2
    echo "  Download via download_hf.py (uncomment the Dream-VLA snapshot_download block)." >&2
    exit 1
fi

DEPLOY_SCRIPT="${DREAMVLA_HOME}/experiments/robot/dynamic_SimplerEnv/deploy_dreamvla_server.py"
if [[ ! -f "${DEPLOY_SCRIPT}" ]]; then
    echo "ERROR: deploy script not found: ${DEPLOY_SCRIPT}" >&2
    echo "  Set DREAMVLA_VLA_HOME to your Dream-VLX/vla repo root." >&2
    exit 1
fi

set +u
# shellcheck disable=SC1091
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "${CONDA_PREFIX:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

# Build CMD
CMD=(python "${DEPLOY_SCRIPT}"
    --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}"
    --unnorm_key "${UNNORM_KEY}"
    --policy_setup "${POLICY_SETUP}"
    --port "${PORT}"
    --action_chunk "${ACTION_CHUNK}"
)
if [[ ${USE_FLASH_ATTN} -eq 0 ]]; then
    CMD+=(--no_flash_attn)
fi
if [[ ${USE_CENTER_CROP} -eq 0 ]]; then
    CMD+=(--no_center_crop)
fi
if [[ ${DO_SAMPLE} -eq 1 ]]; then
    CMD+=(--do_sample)
fi

echo "============================================================"
echo "Dream-VLA HTTP Server  [DyniSafety-SimplerEnv, best-effort Bridge]"
echo "  SIMPLER_ROOT   : ${SIMPLER_ROOT}"
echo "  Dream-VLX home : ${DREAMVLA_HOME}"
echo "  Checkpoint     : ${PRETRAINED_CHECKPOINT}"
echo "  unnorm_key     : ${UNNORM_KEY}"
echo "  policy_setup   : ${POLICY_SETUP}"
echo "  flash_attn     : $([[ ${USE_FLASH_ATTN} -eq 1 ]] && echo on || echo off)"
echo "  center_crop    : $([[ ${USE_CENTER_CROP} -eq 1 ]] && echo on || echo off)"
echo "  do_sample      : $([[ ${DO_SAMPLE} -eq 1 ]] && echo on || echo off)"
echo "  action_chunk   : ${ACTION_CHUNK}"
echo "  Port           : ${PORT}"
echo "  Conda env      : ${CONDA_ENV_NAME}"
echo "  NOTE           : Dream-VLA has no official SimplerEnv-Bridge eval upstream;"
echo "                   this is a zero-shot 'best-effort' setup using bridge_orig stats."
echo "============================================================"

cd "${DREAMVLA_HOME}"
exec "${CMD[@]}"
