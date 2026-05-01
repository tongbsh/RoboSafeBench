"""Backward-compatible façade for DyniSafety-LIBERO utilities.

All symbols that were previously defined here have been refactored into
focused sub-modules.  This file re-exports every public name so that
existing ``from dynisafety_utils import …`` statements keep working.
"""

# -- controllers --
from dynisafety_libero.controllers import (  # noqa: F401
    DYNAMIC_OBSTACLE_CONFIG,
    L1_START_OFFSET,
    L3_START_OFFSET,
    REACHING_PEAK_VELOCITY,
    DynamicObstacleController,
    InspectionApproachController,
    SharedReachingController,
    SuddenDropController,
    detect_visible_obstacles,
    read_obstacle_table_pose,
    select_l3_shared_reach_target,
    # backward-compatible aliases
    HumanArmController,
    AggressiveArmController,
)

# -- collision detection --
from dynisafety_libero.collision import (  # noqa: F401
    INJURY_THRESHOLDS,
    classify_injury,
    compute_body_tilt_deg,
    compute_contact_force,
    detect_collisions,
    get_dynamic_obstacle_geom_ids,
    get_obstacle_geom_ids,
    get_robot_geom_ids,
)

# -- environment helpers --
from dynisafety_libero.env_utils import (  # noqa: F401
    get_dynisafety_dummy_action,
    get_dynisafety_env,
    get_dynisafety_image,
    get_dynisafety_wrist_image,
    hide_all_obstacles,
    hide_overlapping_bases,
    pad_init_state_for_hand,
    rescale_obstacle_inplace,
    quat2axisangle,
    save_rollout_video,
)

# -- metrics (NEW) --
from dynisafety_libero.metrics import (  # noqa: F401
    AggregateMetrics,
    EpisodeMetrics,
    ReactivityAnalyzer,
    compute_safety_penalty,
    compute_sasr,
)
