#!/usr/bin/env bash
#
# Launch the LingBot-VA inference server (wan_va) inside its own conda env.
#
# Mirrors lingbot-va/evaluation/robotwin/launch_server.sh but resolves the
# repo location from this script's location so it works no matter where
# you copy or symlink it.
#
# Override-able env vars:
#   CONDA_ENV=lingbot_va
#   START_PORT=29056
#   MASTER_PORT=29061
#   SAVE_ROOT=visualization/
#   CUDA_VISIBLE_DEVICES=0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../_common.sh
source "${SCRIPT_DIR}/../_common.sh"

CONDA_ENV="${CONDA_ENV:-lingbot_va}"
conda activate "${CONDA_ENV}"

LINGBOT_ROOT="${WORKSPACE_ROOT}/lingbot-va"

START_PORT="${START_PORT:-29056}"
MASTER_PORT="${MASTER_PORT:-29061}"
SAVE_ROOT="${SAVE_ROOT:-${LINGBOT_ROOT}/visualization}"
mkdir -p "${SAVE_ROOT}"

echo "LINGBOT_ROOT = ${LINGBOT_ROOT}"
echo "START_PORT   = ${START_PORT}"
echo "MASTER_PORT  = ${MASTER_PORT}"
echo "SAVE_ROOT    = ${SAVE_ROOT}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-<inherit all>}"

cd "${LINGBOT_ROOT}"
exec python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port "${MASTER_PORT}" \
    wan_va/wan_va_server.py \
    --config-name robotwin \
    --port "${START_PORT}" \
    --save_root "${SAVE_ROOT}"
