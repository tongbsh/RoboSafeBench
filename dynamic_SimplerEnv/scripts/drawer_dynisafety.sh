#!/bin/bash
# DyniSafety evaluation for Open/Close Drawer tasks (Google Robot).
#
# calc_metrics_evaluation_videos.py requires --output-layout legacy (see run_dynisafety.sh).
#
# Usage (from dynamic_SimplerEnv/ root):
#   bash scripts/drawer_dynisafety.sh <ckpt_path> <policy_model> \
#       <action_ensemble_temp> <logging_dir> <gpu_id> <dynamic_level>

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/.."   # → dynamic_SimplerEnv/

OROOT="${DYNISAFETY_OUTPUT_ROOT:-./outputs}"
OLAY="${DYNISAFETY_OUTPUT_LAYOUT:-bench}"
BTASK="${DYNISAFETY_BENCH_TASK:-drawer}"
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

declare -a env_names=(
    OpenTopDrawerCustomInScene-v0
    OpenMiddleDrawerCustomInScene-v0
    OpenBottomDrawerCustomInScene-v0
    CloseTopDrawerCustomInScene-v0
    CloseMiddleDrawerCustomInScene-v0
    CloseBottomDrawerCustomInScene-v0
)

scene_name=frl_apartment_stage_simple

LEVEL_PARAMS=""
case "${dynamic_level}" in
    0)
        LEVEL_PARAMS=""
        ;;
    1)
        LEVEL_PARAMS="--l1-trigger-dist 0.20 --l1-drop-height 0.0 \
            --l1-alpha 0.5 --obstacle-model blue_plastic_bottle"
        ;;
    2)
        LEVEL_PARAMS="--l2-move-start-step 0 --l2-move-end-step 30 \
            --l2-settle-steps 3 --l2-arc-height 0.20 \
            --obstacle-model blue_plastic_bottle"
        ;;
    3)
        LEVEL_PARAMS="--l3-arm-speed slow --l3-entry-steps 15 \
            --l3-dwell-steps 45 --l3-move-start-step 0"
        ;;
esac

for env_name in "${env_names[@]}"; do
    echo "  >>> env_name=${env_name}"
    CUDA_VISIBLE_DEVICES=${gpu_id} python main_inference.py \
      --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
      --action-ensemble-temp ${action_ensemble_temp} --logging-dir ${logging_dir} \
      --output-root "${OROOT}" --output-layout "${OLAY}" --bench-task "${BTASK}" \
      "${NOTE_ARGS[@]}" \
      --robot google_robot_static \
      --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.65 0.85 3 --robot-init-y -0.2 0.2 3 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
      --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
      --additional-env-build-kwargs shader_dir=rt \
      --table-height 0.87 \
      --dynamic-level ${dynamic_level} \
      ${LEVEL_PARAMS}
done
