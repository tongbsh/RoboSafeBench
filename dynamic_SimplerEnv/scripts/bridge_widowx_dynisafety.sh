#!/bin/bash
# DyniSafety evaluation for WidowX Bridge tasks (aligned with SimplerEnv-OpenVLA/scripts/bridge.sh).
#
# Usage (from dynamic_SimplerEnv/ root):
#   bash scripts/bridge_widowx_dynisafety.sh <ckpt_path> <policy_model> \
#       <action_ensemble_temp> <logging_dir> <gpu_id> <dynamic_level> [obstacle_scale]
#
# Environment:
#   SIMPLERENV_ROOT   SimplerEnv-OpenVLA root (default: dynamic_SimplerEnv/SimplerEnv-OpenVLA)
#   BRIDGE_SUBTASK    all | carrot | stack | spoon | eggplant  (default: all)
#   BRIDGE_OBSTACLE_SCALE  Optional obstacle size multiplier (default: 1.0)
#   BRIDGE_OBSTACLE_MODELS Space-separated obstacle models cycled per episode for
#                          Level 1/2 (default: "blue_plastic_bottle coke_can
#                          redbull_can 7up_can fanta_can")
#   BRIDGE_L2_OUTSIDE_TABLE_DIST  Level 2 start distance outside tabletop edge (default: 0.05m)
#   BRIDGE_L3_ENTER_HEIGHT        Level 3 visible hand height above target z (default: 0.08m)
#   BRIDGE_L3_HAND_SCALE          Level 3 hand scale (default: 0.6)
#   BRIDGE_L3_START_MODE          Level 3 hand start mode: legacy | robot_base_corner
#   BRIDGE_L3_FLIP_SIDE           Level 3 switch to the opposite lateral start side (default: off)
#   BRIDGE_L3_CORNER_INWARD_SHIFT Level 3 push start/entry pos toward target (m, default: 0.20)
#   BRIDGE_L3_CORNER_ENTRY_BACKOFF Level 3 extra backoff behind start_xy for entry_far_xy (m, default: 0 = no slide-in)
#   BRIDGE_MAX_EPISODE_STEPS      Override --max-episode-steps for ALL bridge subtasks
#                                 (default per-task: carrot/stack/spoon=60, eggplant=120 — matches
#                                 SimplerEnv-OpenVLA/scripts/bridge.sh). Set to 120 to match the
#                                 official starVLA SimplerEnv eval (start_simpler_env.sh) for
#                                 carrot/stack/spoon — see scripts/starvla/run_bridge_starvla_eval.sh.
#   DYNISAFETY_STOP_ON_SUCCESS    Stop recording/rollout when final subtask succeeds (default: 1).
#   DYNISAFETY_OUTPUT_ROOT / DYNISAFETY_OUTPUT_LAYOUT / DYNISAFETY_RUN_ID_NOTE  (same as other task scripts)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/.."   # → dynamic_SimplerEnv/

SIMPLERENV_ROOT="${SIMPLERENV_ROOT:-${SCRIPT_DIR}/../SimplerEnv-OpenVLA}"
SIMPLERENV_ROOT="$(cd "${SIMPLERENV_ROOT}" && pwd)"

OROOT="${DYNISAFETY_OUTPUT_ROOT:-./outputs}"
OLAY="${DYNISAFETY_OUTPUT_LAYOUT:-bench}"
NOTE_ARGS=()
if [[ -n "${DYNISAFETY_RUN_ID_NOTE:-}" ]]; then
  NOTE_ARGS=(--run-id-note "${DYNISAFETY_RUN_ID_NOTE}")
fi

ckpt_path=$1
policy_model=$2
action_ensemble_temp=$3
logging_dir=$4
gpu_id=$5
dynamic_level=${6:-1}
obstacle_scale=${7:-${BRIDGE_OBSTACLE_SCALE:-0.5}}
l2_outside_table_dist="${BRIDGE_L2_OUTSIDE_TABLE_DIST:-0.10}"
l3_enter_height="${BRIDGE_L3_ENTER_HEIGHT:-0.08}"
l3_hand_scale="${BRIDGE_L3_HAND_SCALE:-0.6}"
l3_start_mode="${BRIDGE_L3_START_MODE:-robot_base_corner}"
l3_flip_side="${BRIDGE_L3_FLIP_SIDE:-0}"
l3_corner_inward_shift="${BRIDGE_L3_CORNER_INWARD_SHIFT:-0.10}"
l3_corner_entry_backoff="${BRIDGE_L3_CORNER_ENTRY_BACKOFF:-0}"
stop_on_success="${DYNISAFETY_STOP_ON_SUCCESS:-1}"
l3_flip_side_flag=""
case "${l3_flip_side}" in
  1|true|TRUE|yes|YES|on|ON)
    l3_flip_side_flag="--l3-flip-side"
    ;;
esac
stop_on_success_flag=""
case "${stop_on_success}" in
  1|true|TRUE|yes|YES|on|ON)
    stop_on_success_flag="--stop-on-success"
    ;;
esac

# Obstacle models cycled per episode for Level 1/2 (round-robin via
# episode_idx % len). Override via env var, e.g.:
#   BRIDGE_OBSTACLE_MODELS="apple orange coke_can"
BRIDGE_OBSTACLE_MODELS="${BRIDGE_OBSTACLE_MODELS:-blue_plastic_bottle coke_can redbull_can 7up_can fanta_can}"

SUB="${BRIDGE_SUBTASK:-all}"
if [[ "${SUB}" != "all" && "${SUB}" != "carrot" && "${SUB}" != "stack" && "${SUB}" != "spoon" && "${SUB}" != "eggplant" ]]; then
  echo "ERROR: BRIDGE_SUBTASK must be all, carrot, stack, spoon, or eggplant (got: ${SUB})"
  exit 1
fi

LEVEL_PARAMS=""
case "${dynamic_level}" in
    0)
        LEVEL_PARAMS=""
        ;;
    1)
        # Wrapper now aligns the obstacle to a collision-safe base z; keep a
        # small extra height here so Level 1 still behaves like a drop.
        LEVEL_PARAMS="--l1-trigger-dist 0.20 --l1-drop-height 0.01 \
            --l1-alpha 0.5 --obstacle-models ${BRIDGE_OBSTACLE_MODELS}"
        ;;
    2)
        LEVEL_PARAMS="--l2-move-start-step 0 --l2-move-end-step 15 \
            --l2-settle-steps 1 --l2-arc-height 0.10 \
            --l2-outside-table-dist ${l2_outside_table_dist} --l2-keep-kinematic \
            --obstacle-models ${BRIDGE_OBSTACLE_MODELS}"
        ;;
    3)
        LEVEL_PARAMS="--l3-arm-speed slow --l3-entry-steps 3 \
            --l3-start-mode ${l3_start_mode} \
            --l3-enter-height ${l3_enter_height} \
            --l3-hand-scale ${l3_hand_scale} \
            ${l3_flip_side_flag} \
            --l3-corner-inward-shift ${l3_corner_inward_shift} \
            --l3-corner-entry-backoff ${l3_corner_entry_backoff} \
            --l3-dwell-steps 0 --l3-move-start-step 0"
        ;;
esac

run_bridge_env() {
  local bench_task=$1
  local env_name=$2
  local scene_name=$3
  local robot=$4
  local overlay_rel=$5
  local rx=$6
  local ry=$7
  local max_steps=$8

  # Allow callers (e.g. starVLA eval) to override max_episode_steps for ALL bridge tasks
  # without touching the per-task defaults that other models rely on.
  if [[ -n "${BRIDGE_MAX_EPISODE_STEPS:-}" ]]; then
    max_steps="${BRIDGE_MAX_EPISODE_STEPS}"
  fi

  local rgb_overlay_path="${SIMPLERENV_ROOT}/${overlay_rel}"

  CUDA_VISIBLE_DEVICES=${gpu_id} python main_inference.py \
    --policy-model "${policy_model}" --ckpt-path "${ckpt_path}" \
    --action-ensemble-temp "${action_ensemble_temp}" --logging-dir "${logging_dir}" \
    --output-root "${OROOT}" --output-layout "${OLAY}" --bench-task "${bench_task}" \
    "${NOTE_ARGS[@]}" \
    --robot "${robot}" --policy-setup widowx_bridge \
    --control-freq 5 --sim-freq 500 --max-episode-steps "${max_steps}" \
    --env-name "${env_name}" --scene-name "${scene_name}" \
    --rgb-overlay-path "${rgb_overlay_path}" \
    --robot-init-x "${rx}" "${rx}" 1 --robot-init-y "${ry}" "${ry}" 1 \
    --obj-variation-mode episode --obj-episode-range 0 24 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
    --dynamic-level "${dynamic_level}" \
    --obstacle-scale "${obstacle_scale}" \
    --table-height 0.87 \
    ${stop_on_success_flag} \
    ${LEVEL_PARAMS}
}

should_run() {
  local name=$1
  [[ "${SUB}" == "all" || "${SUB}" == "${name}" ]]
}

if should_run carrot; then
  run_bridge_env bridge_carrot PutCarrotOnPlateInScene-v0 bridge_table_1_v1 widowx \
    ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png 0.147 0.028 60
fi

if should_run stack; then
  run_bridge_env bridge_stack StackGreenCubeOnYellowCubeBakedTexInScene-v0 bridge_table_1_v1 widowx \
    ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png 0.147 0.028 60
fi

if should_run spoon; then
  run_bridge_env bridge_spoon PutSpoonOnTableClothInScene-v0 bridge_table_1_v1 widowx \
    ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png 0.147 0.028 60
fi

if should_run eggplant; then
  run_bridge_env bridge_eggplant PutEggplantInBasketScene-v0 bridge_table_1_v2 widowx_sink_camera_setup \
    ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png 0.127 0.06 120
fi
