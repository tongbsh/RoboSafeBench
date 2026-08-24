"""Assemble L1–L3 controllers from ``RunConfig`` (no business defaults)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from dynisafety.backends.simplerenv.capabilities import SIMPLERENV_CAPABILITIES
from dynisafety.backends.simplerenv.disturbance._support import DisturbanceController
from dynisafety.backends.simplerenv.disturbance.l1 import SuddenDropController
from dynisafety.backends.simplerenv.disturbance.l2 import SlidingObstacleController
from dynisafety.backends.simplerenv.disturbance.l3 import SharedReachingController
from dynisafety.backends.simplerenv.disturbance.l2_spawn import SLOTS as L2_SLOTS
from dynisafety.backends.simplerenv.disturbance.l3_reach import (
    DEPTHS as L3_DEPTHS,
    SLOTS as L3_SLOTS,
    simplerenv_l3_path_intercept_target,
)
from dynisafety.backends.simplerenv.targets import (
    SceneAccess,
    align_obstacle_target_height,
    compute_l3_start_geometry,
    compute_target_pos,
    get_reference_object_actor,
)
from dynisafety.config.schema import RunConfig
from dynisafety.core.disturbance import require_configured as _require

_SPAWN_DIVERSITY_KEYS: tuple[str, ...] = (
    "enabled",
    "lateral_width_m",
    "linear_z_clearance_m",
    "linear_max_start_z_m",
    "linear_tail_alpha",
    "linear_descent_alpha",
    "clearance_samples",
    "tail_exclude_waypoints",
    "forbidden_slots",
)
_REACH_DIVERSITY_KEYS: tuple[str, ...] = (
    "enabled",
    "lateral_width_m",
    "near_alpha",
    "mid_alpha",
    "hover_rise_m",
    "via_descent_alpha",
    "pin_approach_steps",
    "clearance_samples",
    "min_clearance_m",
    "depth_collapse_m",
    "forbidden_slots",
    "forbidden_depths",
)
_L3_INTERCEPT_KEYS: tuple[str, ...] = (
    "target_mode",
    "intercept_alpha",
    "min_eef_clearance",
    "min_task_clearance",
    "tip_overshoot",
)


def _extra_key(extra: Mapping[str, Any], key: str) -> Any:
    if key not in extra:
        raise ValueError(f"extra.{key} missing from resolved config")
    return extra[key]


def _se_nested(extra: Mapping[str, Any], *keys: str) -> Any:
    cur: Any = extra
    path = ["extra"]
    for key in keys:
        path.append(str(key))
        if not isinstance(cur, Mapping) or key not in cur:
            raise ValueError(f"{'.'.join(path)} missing from resolved config")
        cur = cur[key]
    return cur


def _string_tuple(raw: Any, *, name: str, allowed: tuple[str, ...] | None = None) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        raise TypeError(f"{name} must be a list, got {type(raw)!r}")
    out = tuple(str(item) for item in raw)
    if allowed is not None:
        bad = [item for item in out if item not in allowed]
        if bad:
            raise ValueError(f"{name} entries {bad} not in {allowed}")
    return out


def _l2_spawn_diversity(extra: Mapping[str, Any]) -> dict[str, Any]:
    raw = _se_nested(extra, "simplerenv", "l2", "spawn_diversity")
    if not isinstance(raw, Mapping):
        raise TypeError(f"extra.simplerenv.l2.spawn_diversity must be a mapping, got {type(raw)!r}")
    missing = [key for key in _SPAWN_DIVERSITY_KEYS if key not in raw]
    if missing:
        raise ValueError(
            f"extra.simplerenv.l2.spawn_diversity missing keys {missing} from resolved config"
        )
    enabled = raw["enabled"]
    if not isinstance(enabled, bool):
        raise TypeError(
            f"extra.simplerenv.l2.spawn_diversity.enabled must be bool, got {type(enabled)!r}"
        )
    return {
        "enabled": enabled,
        "lateral_width_m": float(raw["lateral_width_m"]),
        "linear_z_clearance_m": float(raw["linear_z_clearance_m"]),
        "linear_max_start_z_m": float(raw["linear_max_start_z_m"]),
        "linear_tail_alpha": float(raw["linear_tail_alpha"]),
        "linear_descent_alpha": float(raw["linear_descent_alpha"]),
        "clearance_samples": int(raw["clearance_samples"]),
        "tail_exclude_waypoints": int(raw["tail_exclude_waypoints"]),
        "forbidden_slots": _string_tuple(
            raw["forbidden_slots"],
            name="extra.simplerenv.l2.spawn_diversity.forbidden_slots",
            allowed=L2_SLOTS,
        ),
    }


def _l3_reach_block(extra: Mapping[str, Any]) -> dict[str, Any]:
    raw = _se_nested(extra, "simplerenv", "l3")
    if not isinstance(raw, Mapping):
        raise TypeError(f"extra.simplerenv.l3 must be a mapping, got {type(raw)!r}")
    missing = [key for key in _L3_INTERCEPT_KEYS if key not in raw]
    if missing:
        raise ValueError(f"extra.simplerenv.l3 missing keys {missing} from resolved config")
    mode = str(raw["target_mode"])
    if mode not in ("current", "path_intercept"):
        raise ValueError(
            f"extra.simplerenv.l3.target_mode must be 'current' or 'path_intercept', got {mode!r}"
        )
    reach_raw = raw.get("reach_diversity")
    if not isinstance(reach_raw, Mapping):
        raise TypeError(
            f"extra.simplerenv.l3.reach_diversity must be a mapping, got {type(reach_raw)!r}"
        )
    missing_r = [key for key in _REACH_DIVERSITY_KEYS if key not in reach_raw]
    if missing_r:
        raise ValueError(
            f"extra.simplerenv.l3.reach_diversity missing keys {missing_r} from resolved config"
        )
    enabled = reach_raw["enabled"]
    if not isinstance(enabled, bool):
        raise TypeError(
            f"extra.simplerenv.l3.reach_diversity.enabled must be bool, got {type(enabled)!r}"
        )
    pin = reach_raw["pin_approach_steps"]
    if not isinstance(pin, bool):
        raise TypeError(
            "extra.simplerenv.l3.reach_diversity.pin_approach_steps must be bool, "
            f"got {type(pin)!r}"
        )
    return {
        "target_mode": mode,
        "intercept_alpha": float(raw["intercept_alpha"]),
        "min_eef_clearance": float(raw["min_eef_clearance"]),
        "min_task_clearance": float(raw["min_task_clearance"]),
        "tip_overshoot": float(raw["tip_overshoot"]),
        "reach_diversity": {
            "enabled": enabled,
            "lateral_width_m": float(reach_raw["lateral_width_m"]),
            "near_alpha": float(reach_raw["near_alpha"]),
            "mid_alpha": float(reach_raw["mid_alpha"]),
            "hover_rise_m": float(reach_raw["hover_rise_m"]),
            "via_descent_alpha": float(reach_raw["via_descent_alpha"]),
            "pin_approach_steps": pin,
            "clearance_samples": int(reach_raw["clearance_samples"]),
            "min_clearance_m": float(reach_raw["min_clearance_m"]),
            "depth_collapse_m": float(reach_raw["depth_collapse_m"]),
            "forbidden_slots": _string_tuple(
                reach_raw["forbidden_slots"],
                name="extra.simplerenv.l3.reach_diversity.forbidden_slots",
                allowed=L3_SLOTS,
            ),
            "forbidden_depths": _string_tuple(
                reach_raw["forbidden_depths"],
                name="extra.simplerenv.l3.reach_diversity.forbidden_depths",
                allowed=L3_DEPTHS,
            ),
        },
    }


def build_controller(
    level: int,
    actor: Any,
    cfg: RunConfig,
    *,
    access: SceneAccess,
    reset_options: Mapping[str, Any],
    get_eef_pos_fn: Callable[[], np.ndarray],
    seed: int,
    resolved_extra: Mapping[str, Any],
    episode_index: int = 0,
    task_id: Any = 0,
    mode_seed: int | None = None,
) -> DisturbanceController | None:
    """level 0 → None; unsupported level → ValueError."""
    if level not in SIMPLERENV_CAPABILITIES.levels:
        raise ValueError(
            f"level={level} not in SimplerEnv capabilities {SIMPLERENV_CAPABILITIES.levels}"
        )
    if level == 0:
        return None

    disturbance = _require(cfg.disturbance, "cfg.disturbance")
    scene = _require(cfg.scene, "cfg.scene")
    table_height = float(_require(scene.table_height, "scene.table_height"))

    if level == 1:
        return _build_l1(
            actor,
            cfg,
            disturbance=disturbance,
            access=access,
            reset_options=reset_options,
            get_eef_pos_fn=get_eef_pos_fn,
            table_height=table_height,
        )
    if level == 2:
        return _build_l2(
            actor,
            cfg,
            disturbance=disturbance,
            access=access,
            reset_options=reset_options,
            get_eef_pos_fn=get_eef_pos_fn,
            seed=seed,
            table_height=table_height,
            resolved_extra=resolved_extra,
            episode_index=episode_index,
            task_id=task_id,
            mode_seed=mode_seed,
        )
    if level == 3:
        return _build_l3(
            actor,
            cfg,
            disturbance=disturbance,
            access=access,
            reset_options=reset_options,
            get_eef_pos_fn=get_eef_pos_fn,
            seed=seed,
            table_height=table_height,
            resolved_extra=resolved_extra,
            episode_index=episode_index,
            task_id=task_id,
            mode_seed=mode_seed,
        )
    raise ValueError(f"unhandled level={level}")


def _build_l1(
    actor: Any,
    cfg: RunConfig,
    *,
    disturbance: Any,
    access: SceneAccess,
    reset_options: Mapping[str, Any],
    get_eef_pos_fn: Callable[[], np.ndarray],
    table_height: float,
) -> SuddenDropController:
    del cfg
    l1 = _require(disturbance.l1, "disturbance.l1")
    alpha = float(_require(l1.alpha, "disturbance.l1.alpha"))
    target = compute_target_pos(
        access,
        reset_options,
        override=None,
        table_height=table_height,
        level=1,
        alpha=alpha,
    )
    target, _ = align_obstacle_target_height(
        access,
        target,
        level=1,
        mirror_start=True,
        start_dist_mult=1.0,
        start_offset=(0.0, 0.0, 0.0),
        align_start_y_to_eef=False,
        outside_table_dist=0.0,
    )
    return SuddenDropController(
        actor,
        target_pos=target,
        drop_height=float(_require(l1.drop_height, "disturbance.l1.drop_height")),
        trigger_dist=float(_require(l1.trigger_dist, "disturbance.l1.trigger_dist")),
        get_eef_pos_fn=get_eef_pos_fn,
    )


def _build_l2(
    actor: Any,
    cfg: RunConfig,
    *,
    disturbance: Any,
    access: SceneAccess,
    reset_options: Mapping[str, Any],
    get_eef_pos_fn: Callable[[], np.ndarray],
    seed: int,
    table_height: float,
    resolved_extra: Mapping[str, Any],
    episode_index: int,
    task_id: Any,
    mode_seed: int | None,
) -> SlidingObstacleController:
    del cfg
    l2 = _require(disturbance.l2, "disturbance.l2")
    alpha = float(_require(l2.alpha, "disturbance.l2.alpha"))
    outside = float(_require(l2.outside_table_dist, "disturbance.l2.outside_table_dist"))
    mirror = bool(_require(l2.mirror_start, "disturbance.l2.mirror_start"))
    dist_mult = float(_require(l2.start_dist_mult, "disturbance.l2.start_dist_mult"))
    start_offset = list(_require(l2.start_offset, "disturbance.l2.start_offset"))
    align = bool(_require(l2.align_start_y_to_eef, "disturbance.l2.align_start_y_to_eef"))

    target = compute_target_pos(
        access,
        reset_options,
        override=None,
        table_height=table_height,
        level=2,
        alpha=alpha,
    )
    target, l2_off = align_obstacle_target_height(
        access,
        target,
        level=2,
        mirror_start=mirror,
        start_dist_mult=dist_mult,
        start_offset=start_offset,
        align_start_y_to_eef=align and outside <= 0,
        outside_table_dist=outside,
    )
    if l2_off is None:
        from dynisafety.backends.simplerenv.targets import compute_l2_start_offset

        l2_off = compute_l2_start_offset(
            access,
            target,
            mirror_start=mirror,
            start_dist_mult=dist_mult,
            start_offset=start_offset,
            align_start_y_to_eef=align and outside <= 0,
            outside_table_dist=outside,
        )

    enable_random = bool(_require(l2.enable_random_start, "disturbance.l2.enable_random_start"))
    xy_range = float(_require(l2.random_start_xy_range, "disturbance.l2.random_start_xy_range"))
    # Match legacy: disable XY jitter when starting outside the table.
    if outside > 0:
        xy_range = 0.0
        enable_random_xy = False
    else:
        enable_random_xy = enable_random

    spawn_diversity = _l2_spawn_diversity(resolved_extra)
    align_eff = bool(align and outside <= 0)
    if align_eff and bool(spawn_diversity.get("enabled")):
        raise ValueError(
            "disturbance.l2.align_start_y_to_eef is incompatible with "
            "extra.simplerenv.l2.spawn_diversity.enabled: it overwrites start Y "
            "after slot planning and erases lateral offset"
        )

    return SlidingObstacleController(
        actor,
        final_pos=target,
        start_offset=l2_off,
        move_start_step=int(_require(l2.move_start_step, "disturbance.l2.move_start_step")),
        move_end_step=int(_require(l2.move_end_step, "disturbance.l2.move_end_step")),
        settle_steps=int(_require(l2.settle_steps, "disturbance.l2.settle_steps")),
        arc_height=float(_require(l2.arc_height, "disturbance.l2.arc_height")),
        enable_random_start=bool(enable_random_xy),
        random_start_xy_range=float(xy_range),
        random_start_z_range=float(
            _require(l2.random_start_z_range, "disturbance.l2.random_start_z_range")
        ),
        align_start_y_to_eef=align_eff,
        release_after_settle=bool(
            _require(l2.release_after_settle, "disturbance.l2.release_after_settle")
        ),
        get_eef_pos_fn=get_eef_pos_fn,
        seed=int(seed),
        access=access,
        spawn_diversity=spawn_diversity,
        episode_index=int(episode_index),
        task_id=task_id,
        mode_seed=int(seed if mode_seed is None else mode_seed),
    )


def _build_l3(
    actor: Any,
    cfg: RunConfig,
    *,
    disturbance: Any,
    access: SceneAccess,
    reset_options: Mapping[str, Any],
    get_eef_pos_fn: Callable[[], np.ndarray],
    seed: int,
    table_height: float,
    resolved_extra: Mapping[str, Any],
    episode_index: int,
    task_id: Any,
    mode_seed: int | None,
) -> SharedReachingController:
    l3 = _require(disturbance.l3, "disturbance.l3")
    hand = _require(cfg.hand, "cfg.hand")
    runtime = _require(cfg.runtime, "cfg.runtime")
    control_freq = int(_require(runtime.control_freq, "runtime.control_freq"))

    alpha = float(_require(l3.alpha, "disturbance.l3.alpha"))
    l3_block = _l3_reach_block(resolved_extra)
    target = compute_target_pos(
        access,
        reset_options,
        override=None,
        table_height=table_height,
        level=3,
        alpha=alpha,
    )
    task_obj = get_reference_object_actor(access)
    task_obj_pos = None
    if task_obj is not None:
        try:
            task_obj_pos = np.asarray(task_obj.get_pose().p, dtype=np.float64)
        except Exception:
            task_obj_pos = None
    eef_pos = np.asarray(get_eef_pos_fn(), dtype=np.float64)
    if l3_block["target_mode"] == "path_intercept" and task_obj_pos is not None:
        target = simplerenv_l3_path_intercept_target(
            eef_pos,
            task_obj_pos,
            alpha=float(l3_block["intercept_alpha"]),
            min_eef_clearance=float(l3_block["min_eef_clearance"]),
            tip_overshoot=float(l3_block["tip_overshoot"]),
            min_task_clearance=float(l3_block["min_task_clearance"]),
            target_z=float(task_obj_pos[2]),
        )

    hand_scale = float(_require(hand.scale, "hand.scale"))
    half = list(_require(hand.half_size, "hand.half_size"))
    col_off = list(_require(hand.collision_offset, "hand.collision_offset"))
    start_offset, entry_far = compute_l3_start_geometry(
        access,
        target,
        start_mode=str(_require(l3.start_mode, "disturbance.l3.start_mode")),
        mirror_start=bool(_require(l3.mirror_start, "disturbance.l3.mirror_start")),
        start_dist_mult=float(_require(l3.start_dist_mult, "disturbance.l3.start_dist_mult")),
        start_offset=list(_require(l3.start_offset, "disturbance.l3.start_offset")),
        flip_side=bool(_require(l3.flip_side, "disturbance.l3.flip_side")),
        corner_inward_shift=float(
            _require(l3.corner_inward_shift, "disturbance.l3.corner_inward_shift")
        ),
        corner_entry_backoff=float(
            _require(l3.corner_entry_backoff, "disturbance.l3.corner_entry_backoff")
        ),
        enter_height=float(_require(l3.enter_height, "disturbance.l3.enter_height")),
        hand_scale=hand_scale,
        hand_half_size_x=float(half[0]),
        hand_collision_offset_x=float(col_off[0]),
    )

    arm_speed = str(_require(l3.arm_speed, "disturbance.l3.arm_speed"))
    peak_map = _require(hand.peak_velocity, "hand.peak_velocity")
    if not isinstance(peak_map, Mapping) or arm_speed not in peak_map:
        raise KeyError(f"hand.peak_velocity missing key {arm_speed!r}")
    peak_velocity = float(peak_map[arm_speed])

    obb_half = list(_extra_key(resolved_extra, "obb_half"))
    ignore = list(
        _require(
            l3.ignore_collision_link_substrings,
            "disturbance.l3.ignore_collision_link_substrings",
        )
    )

    return SharedReachingController(
        actor,
        target_pos=target,
        start_offset=start_offset,
        entry_far_pos=entry_far,
        alpha=alpha,
        arm_speed=arm_speed,
        peak_velocity=peak_velocity,
        entry_steps=int(_require(l3.entry_steps, "disturbance.l3.entry_steps")),
        dwell_steps=int(_require(l3.dwell_steps, "disturbance.l3.dwell_steps")),
        withdraw_speed_mult=float(
            _require(l3.withdraw_speed_mult, "disturbance.l3.withdraw_speed_mult")
        ),
        move_start_step=int(_require(l3.move_start_step, "disturbance.l3.move_start_step")),
        enter_height=float(_require(l3.enter_height, "disturbance.l3.enter_height")),
        hand_scale=hand_scale,
        enable_pre_physics_collision_guard=bool(
            _require(
                l3.enable_pre_physics_collision_guard,
                "disturbance.l3.enable_pre_physics_collision_guard",
            )
        ),
        flinch_dist=float(_require(hand.flinch_dist, "hand.flinch_dist")),
        pre_physics_obb_half_scale=float(
            _require(hand.pre_physics_obb_half_scale, "hand.pre_physics_obb_half_scale")
        ),
        obb_half=obb_half,
        ignore_collision_link_substrings=ignore,
        control_freq=control_freq,
        seed=int(seed),
        get_eef_pos_fn=get_eef_pos_fn,
        access=access,
        reach_diversity=l3_block["reach_diversity"],
        episode_index=int(episode_index),
        task_id=task_id,
        mode_seed=int(seed if mode_seed is None else mode_seed),
        eef_pos=eef_pos,
        task_obj_pos=task_obj_pos,
        intercept_alpha=float(l3_block["intercept_alpha"]),
        min_eef_clearance=float(l3_block["min_eef_clearance"]),
        min_task_clearance=float(l3_block["min_task_clearance"]),
        tip_overshoot=float(l3_block["tip_overshoot"]),
    )
