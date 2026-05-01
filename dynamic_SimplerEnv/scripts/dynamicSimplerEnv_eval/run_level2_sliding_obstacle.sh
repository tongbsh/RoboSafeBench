#!/bin/bash
# ============================================================
# DyniSafety Level 2 — Sliding Obstacle
#
# 障碍物沿弧线轨迹从场景外滑入工作区。
#
# ── 支持的 VLA 模型 (--model) ────────────────────────────────
#   openvla / cogact / spatialvla / openpifast / lerobotpifast
#   gr00t / rt1 / octo-base / octo-small / octo-server
#   <自定义> — 通过 register_policy() 注册后即可使用
#
# Usage (从 dynamic_SimplerEnv/ 根目录运行):
#   bash scripts/dynamicSimplerEnv_eval/run_level2_sliding_obstacle.sh
#   bash scripts/dynamicSimplerEnv_eval/run_level2_sliding_obstacle.sh --model cogact --ckpt CogACT/CogACT-Large
#   bash scripts/dynamicSimplerEnv_eval/run_level2_sliding_obstacle.sh --arc-height 0.30
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/../.."   # → dynamic_SimplerEnv/

# ======================== 默认超参数 ========================
CKPT_PATH="openvla/openvla-7b"
POLICY_MODEL="openvla"
ACTION_ENSEMBLE_TEMP=-0.8
GPU_ID=0

L2_ALPHA=0.5
L2_MIRROR_START=True
L2_START_DIST_MULT=-1.5
L2_START_OFFSET_X=0.40
L2_START_OFFSET_Y=0.10
L2_START_OFFSET_Z=0.0
L2_MOVE_START_STEP=0
L2_MOVE_END_STEP=10
L2_SETTLE_STEPS=3
L2_ARC_HEIGHT=0.10
L2_RANDOM_START=True
L2_XY_RANGE=0.05
L2_Z_RANGE=0.02

OBSTACLE_MODELS=("blue_plastic_bottle" "apple" "orange")
OBSTACLE_SCALE=1.2

TABLE_HEIGHT=0.87
SEED=42
MAX_EPISODE_STEPS=80
CONTROL_FREQ=3
SIM_FREQ=513

# ======================== 参数解析 ========================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)            shift; CKPT_PATH="$1"; shift ;;
        --model)           shift; POLICY_MODEL="$1"; shift ;;
        --temp)            shift; ACTION_ENSEMBLE_TEMP="$1"; shift ;;
        --gpu)             shift; GPU_ID="$1"; shift ;;
        --alpha)           shift; L2_ALPHA="$1"; shift ;;
        --mirror-start)    shift; L2_MIRROR_START="$1"; shift ;;
        --start-dist-mult) shift; L2_START_DIST_MULT="$1"; shift ;;
        --start-offset)    shift; L2_START_OFFSET_X="$1"; shift; L2_START_OFFSET_Y="$1"; shift; L2_START_OFFSET_Z="$1"; shift ;;
        --move-start-step) shift; L2_MOVE_START_STEP="$1"; shift ;;
        --move-end-step)   shift; L2_MOVE_END_STEP="$1"; shift ;;
        --settle-steps)    shift; L2_SETTLE_STEPS="$1"; shift ;;
        --arc-height)      shift; L2_ARC_HEIGHT="$1"; shift ;;
        --random-start)    shift; L2_RANDOM_START="$1"; shift ;;
        --xy-range)        shift; L2_XY_RANGE="$1"; shift ;;
        --z-range)         shift; L2_Z_RANGE="$1"; shift ;;
        --obstacles)       shift; OBSTACLE_MODELS=(); while [[ $# -gt 0 && "$1" != --* ]]; do OBSTACLE_MODELS+=("$1"); shift; done ;;
        --obstacle-scale)  shift; OBSTACLE_SCALE="$1"; shift ;;
        --table-height)    shift; TABLE_HEIGHT="$1"; shift ;;
        --seed)            shift; SEED="$1"; shift ;;
        --max-steps)       shift; MAX_EPISODE_STEPS="$1"; shift ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

LOGGING_DIR="results/dynisafety_level2_$(basename ${CKPT_PATH})_multiobs_alpha${L2_ALPHA}"
mkdir -p "${LOGGING_DIR}"

echo "============================================================"
echo "DyniSafety Level 2 — Sliding Obstacle"
echo "  Policy: ${POLICY_MODEL}  Ckpt: ${CKPT_PATH}  GPU: ${GPU_ID}"
echo "  alpha=${L2_ALPHA}  arc_height=${L2_ARC_HEIGHT}  obstacles=${OBSTACLE_MODELS[*]}"
echo "  Logging: ${LOGGING_DIR}"
echo "============================================================"

declare -a coke_can_options_arr=("lr_switch=True" "upright=True" "laid_vertically=True")
env_name=GraspSingleOpenedCokeCanInScene-v0
scene_name=google_pick_coke_can_1_v4

for coke_can_option in "${coke_can_options_arr[@]}"; do
    echo "  >>> coke_can_option=${coke_can_option}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} python main_inference.py \
      --policy-model ${POLICY_MODEL} \
      --ckpt-path ${CKPT_PATH} \
      --action-ensemble-temp ${ACTION_ENSEMBLE_TEMP} \
      --logging-dir ${LOGGING_DIR} \
      --robot google_robot_static \
      --control-freq ${CONTROL_FREQ} \
      --sim-freq ${SIM_FREQ} \
      --max-episode-steps ${MAX_EPISODE_STEPS} \
      --env-name ${env_name} \
      --scene-name ${scene_name} \
      --robot-init-x 0.35 0.35 1 \
      --robot-init-y 0.20 0.20 1 \
      --obj-init-x -0.35 -0.12 5 \
      --obj-init-y -0.02 0.42 5 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --additional-env-build-kwargs ${coke_can_option} \
      --dynamic-level 2 \
      --l2-alpha ${L2_ALPHA} \
      --l2-mirror-start ${L2_MIRROR_START} \
      --l2-start-dist-mult ${L2_START_DIST_MULT} \
      --l2-start-offset ${L2_START_OFFSET_X} ${L2_START_OFFSET_Y} ${L2_START_OFFSET_Z} \
      --l2-move-start-step ${L2_MOVE_START_STEP} \
      --l2-move-end-step ${L2_MOVE_END_STEP} \
      --l2-settle-steps ${L2_SETTLE_STEPS} \
      --l2-arc-height ${L2_ARC_HEIGHT} \
      --l2-enable-random-start ${L2_RANDOM_START} \
      --l2-random-start-xy-range ${L2_XY_RANGE} \
      --l2-random-start-z-range ${L2_Z_RANGE} \
      --obstacle-models "${OBSTACLE_MODELS[@]}" \
      --obstacle-scale ${OBSTACLE_SCALE} \
      --table-height ${TABLE_HEIGHT} \
      --dynisafety-seed ${SEED}
done

echo ""
echo "Calculating metrics..."
SIMPLERENV_PATH="${SIMPLERENV_ROOT:-${SCRIPT_DIR}/../../SimplerEnv-OpenVLA}"
python "${SIMPLERENV_PATH}/tools/calc_metrics_evaluation_videos.py" \
  --log-dir-root "${LOGGING_DIR}" \
  >>"${LOGGING_DIR}/total.metrics"

echo "Level 2 Sliding Obstacle — COMPLETE  (${LOGGING_DIR})"
