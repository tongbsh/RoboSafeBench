#!/usr/bin/env bash

if [[ -n "${__DYNISAFETY_TASK_REGISTRY_SH:-}" ]]; then
    return 0
fi
__DYNISAFETY_TASK_REGISTRY_SH=1

# Default 20-task DyniSafety evaluation set. Keep the previously supported 10
# tasks at the front so existing run order remains familiar, then append the
# newly enabled tasks.
declare -ga DYNISAFETY_DEFAULT_TASKS=(
    place_a2b_left
    place_a2b_right
    handover_block
    move_stapler_pad
    place_mouse_pad
    adjust_bottle
    beat_block_hammer
    pick_dual_bottles
    move_can_pot
    lift_pot
    open_laptop
    click_bell
    move_pillbottle_pad
    stack_bowls_three
    blocks_ranking_rgb
    stamp_seal
    place_empty_cup
    stack_blocks_two
    move_playingcard_away
    shake_bottle
)

declare -gA DYNISAFETY_TASK_IDX_MAP=(
    [adjust_bottle]=0
    [beat_block_hammer]=1
    [blocks_ranking_rgb]=2
    [click_bell]=5
    [handover_block]=8
    [lift_pot]=11
    [move_can_pot]=12
    [move_pillbottle_pad]=13
    [move_playingcard_away]=14
    [move_stapler_pad]=15
    [open_laptop]=16
    [pick_dual_bottles]=19
    [place_a2b_left]=20
    [place_a2b_right]=21
    [place_empty_cup]=29
    [place_mouse_pad]=31
    [shake_bottle]=42
    [stack_blocks_two]=45
    [stack_bowls_three]=46
    [stamp_seal]=48
)

declare -gA DYNISAFETY_TASK_TARGET_MODE_MAP=(
    [adjust_bottle]=source_object
    [beat_block_hammer]=source_object
    [blocks_ranking_rgb]=source_object
    [click_bell]=source_object
    [handover_block]=source_object
    [lift_pot]=source_object
    [move_can_pot]=source_object
    [move_pillbottle_pad]=source_object
    [move_playingcard_away]=source_object
    [move_stapler_pad]=source_object
    [open_laptop]=source_object
    [pick_dual_bottles]=source_object
    [place_a2b_left]=source_object
    [place_a2b_right]=source_object
    [place_empty_cup]=source_object
    [place_mouse_pad]=source_object
    [shake_bottle]=source_object
    [stack_blocks_two]=source_object
    [stack_bowls_three]=source_object
    [stamp_seal]=source_object
)

dynisafety_has_task() {
    local task_name="$1"
    [[ -n "${DYNISAFETY_TASK_IDX_MAP[${task_name}]+x}" ]]
}

dynisafety_task_idx() {
    local task_name="$1"
    if ! dynisafety_has_task "${task_name}"; then
        return 1
    fi
    printf '%s\n' "${DYNISAFETY_TASK_IDX_MAP[${task_name}]}"
}

dynisafety_task_target_mode() {
    local task_name="$1"
    if ! dynisafety_has_task "${task_name}"; then
        return 1
    fi
    printf '%s\n' "${DYNISAFETY_TASK_TARGET_MODE_MAP[${task_name}]}"
}

dynisafety_print_valid_tasks() {
    printf '%s ' "${DYNISAFETY_DEFAULT_TASKS[@]}"
}

dynisafety_export_task_env() {
    local task_name="$1"
    if ! dynisafety_has_task "${task_name}"; then
        echo "[ERROR] Unknown DyniSafety task: ${task_name}" >&2
        echo -n "[INFO]  Valid tasks: " >&2
        dynisafety_print_valid_tasks >&2
        echo >&2
        return 1
    fi

    export TASK_NAME="${task_name}"
    if [[ -z "${TASK_IDX:-}" ]]; then
        export TASK_IDX
        TASK_IDX="$(dynisafety_task_idx "${task_name}")"
    fi
    if [[ -z "${OBSTACLE_TARGET_MODE:-}" ]]; then
        export OBSTACLE_TARGET_MODE
        OBSTACLE_TARGET_MODE="$(dynisafety_task_target_mode "${task_name}")"
    fi
}
