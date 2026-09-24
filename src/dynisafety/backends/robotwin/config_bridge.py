"""Bridge ``RunConfig`` + ``extra.robotwin`` → flat legacy attrs for ``DisturbanceManager``."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from dynisafety.backends.robotwin.contact import (
    normalize_l12_force_scheme,
    normalize_l3_force_scheme,
)
from dynisafety.config.schema import MISSING, RunConfig
from dynisafety.core.disturbance import require_configured


def _require(value: Any, field_name: str) -> Any:
    return require_configured(value, field_name)


def _require_nested(obj: Any, attr: str, field_name: str) -> Any:
    _require(obj, field_name)
    if not hasattr(obj, attr):
        raise ValueError(field_name)
    return _require(getattr(obj, attr), field_name)


def _extra(extra_robotwin: Mapping[str, Any], key: str) -> Any:
    field_name = f"extra.robotwin.{key}"
    if key not in extra_robotwin:
        raise ValueError(field_name)
    return _require(extra_robotwin[key], field_name)


def _l3_force_scheme(extra_robotwin: Mapping[str, Any]) -> str:
    return normalize_l3_force_scheme(
        _extra(extra_robotwin, "l3_force_scheme"),
        field_name="extra.robotwin.l3_force_scheme",
    )


def _l12_force_scheme(extra_robotwin: Mapping[str, Any]) -> str:
    return normalize_l12_force_scheme(
        _extra(extra_robotwin, "l12_force_scheme"),
        field_name="extra.robotwin.l12_force_scheme",
    )


def _as_tuple3(value: Any, field_name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(field_name)
    return (float(value[0]), float(value[1]), float(value[2]))


def _as_tuple2(value: Any, field_name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(field_name)
    return (float(value[0]), float(value[1]))


def _as_str_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(field_name)
    out: list[str] = []
    for item in value:
        token = str(item).strip()
        if not token:
            raise ValueError(field_name)
        out.append(token)
    return tuple(out)


def _as_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(field_name)
    return value


def _as_avoid_shape(value: Any, field_name: str) -> str:
    shape = str(value).strip().lower()
    if shape not in ("disk", "aabb"):
        raise ValueError(field_name)
    return shape


def _as_tuple4(value: Any, field_name: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(field_name)
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def _as_str_float_map(value: Any, field_name: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(field_name)
    out: dict[str, float] = {}
    for key, raw in value.items():
        name = str(key).strip()
        if not name:
            raise ValueError(field_name)
        scale = float(raw)
        if scale <= 0.0:
            raise ValueError(field_name)
        out[name] = scale
    return out


def _as_str_str_map(value: Any, field_name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(field_name)
    out: dict[str, str] = {}
    for key, raw in value.items():
        name = str(key).strip()
        family = str(raw).strip()
        if not name or not family:
            raise ValueError(field_name)
        out[name] = family
    return out


def _as_str_quat_map(value: Any, field_name: str) -> dict[str, tuple[float, float, float, float]]:
    if not isinstance(value, Mapping):
        raise ValueError(field_name)
    return {str(key).strip(): _as_tuple4(raw, field_name) for key, raw in value.items()}


def _normalize_hand_geometry(raw: str) -> str:
    geom = str(raw).lower()
    if geom in ("mesh", "mano_mesh"):
        return geom
    if geom == "box":
        return "box"
    raise ValueError("hand.geometry")


def _contact_attrs(contact: Any) -> dict[str, Any]:
    return {
        "topple_threshold_deg": float(
            _require_nested(contact, "topple_threshold_deg", "contact.topple_threshold_deg")
        ),
        "collision_force_equiv_dt": str(
            _require_nested(contact, "force_equiv_dt", "contact.force_equiv_dt")
        ),
        "collision_force_report_window_s": float(
            _require_nested(contact, "force_report_window_s", "contact.force_report_window_s")
        ),
        "collision_impulse_point_reduce": str(
            _require_nested(contact, "impulse_reduce", "contact.impulse_reduce")
        ),
    }


def _scene_attrs(scene: Any) -> dict[str, Any]:
    return {
        "table_surface_z": float(
            _require_nested(scene, "table_surface_z", "scene.table_surface_z")
        ),
        "support_clearance_z": float(
            _require_nested(scene, "support_clearance_z", "scene.support_clearance_z")
        ),
        "obstacle_avoid_margin": float(
            _require_nested(scene, "obstacle_avoid_margin", "scene.obstacle_avoid_margin")
        ),
        "l1_spawn_link_clearance_xy": float(
            _require_nested(scene, "l1_spawn_link_clearance_xy", "scene.l1_spawn_link_clearance_xy")
        ),
        "l1_spawn_z_clearance": float(
            _require_nested(scene, "l1_spawn_z_clearance", "scene.l1_spawn_z_clearance")
        ),
        "l1_spawn_eef_clearance_xy": float(
            _require_nested(scene, "l1_spawn_eef_clearance_xy", "scene.l1_spawn_eef_clearance_xy")
        ),
        "l1_spawn_eef_work_height": float(
            _require_nested(scene, "l1_spawn_eef_work_height", "scene.l1_spawn_eef_work_height")
        ),
        "l1_spawn_eef_radius": float(
            _require_nested(scene, "l1_spawn_eef_radius", "scene.l1_spawn_eef_radius")
        ),
        "l1_home_approach_eps": float(
            _require_nested(scene, "l1_home_approach_eps", "scene.l1_home_approach_eps")
        ),
        "l1_spawn_nudge_step": float(
            _require_nested(scene, "l1_spawn_nudge_step", "scene.l1_spawn_nudge_step")
        ),
        "l1_spawn_nudge_max_tries": int(
            _require_nested(scene, "l1_spawn_nudge_max_tries", "scene.l1_spawn_nudge_max_tries")
        ),
        "held_grasp_dist": float(
            _require_nested(scene, "held_grasp_dist", "scene.held_grasp_dist")
        ),
        "held_gripper_closed": float(
            _require_nested(scene, "held_gripper_closed", "scene.held_gripper_closed")
        ),
        "l12_collision_exclude_substrings": _as_str_tuple(
            _require_nested(
                scene,
                "l12_collision_exclude_substrings",
                "scene.l12_collision_exclude_substrings",
            ),
            "scene.l12_collision_exclude_substrings",
        ),
        "l12_contact_require_impulse": _as_bool(
            _require_nested(
                scene,
                "l12_contact_require_impulse",
                "scene.l12_contact_require_impulse",
            ),
            "scene.l12_contact_require_impulse",
        ),
        "l2_keep_last_free_pose": _as_bool(
            _require_nested(
                scene,
                "l2_keep_last_free_pose",
                "scene.l2_keep_last_free_pose",
            ),
            "scene.l2_keep_last_free_pose",
        ),
        "l2_release_on_penetration": _as_bool(
            _require_nested(
                scene,
                "l2_release_on_penetration",
                "scene.l2_release_on_penetration",
            ),
            "scene.l2_release_on_penetration",
        ),
        "obstacle_avoid_shape": _as_avoid_shape(
            _require_nested(scene, "obstacle_avoid_shape", "scene.obstacle_avoid_shape"),
            "scene.obstacle_avoid_shape",
        ),
        "l2_landing_push_m": float(
            _require_nested(scene, "l2_landing_push_m", "scene.l2_landing_push_m")
        ),
    }


def _obstacle_attrs(obstacle: Any) -> dict[str, Any]:
    mode = str(
        _require_nested(obstacle, "selection_mode", "disturbance.obstacle.selection_mode")
    ).strip().lower()
    if mode not in ("random", "trajectory_order"):
        raise ValueError("disturbance.obstacle.selection_mode")
    pool = _as_str_tuple(
        _require_nested(obstacle, "pool", "disturbance.obstacle.pool"),
        "disturbance.obstacle.pool",
    )
    if not pool:
        raise ValueError("disturbance.obstacle.pool")
    return {
        "obstacle_target_mode": str(
            _require_nested(obstacle, "target_mode", "disturbance.obstacle.target_mode")
        ),
        "obstacle_selection_mode": mode,
        "obstacle_pool": pool,
        "obstacle_rescale": _as_str_float_map(
            _require_nested(obstacle, "rescale", "disturbance.obstacle.rescale"),
            "disturbance.obstacle.rescale",
        ),
        "obstacle_upright_quats": _as_str_quat_map(
            _require_nested(obstacle, "upright_quats", "disturbance.obstacle.upright_quats"),
            "disturbance.obstacle.upright_quats",
        ),
        "obstacle_semantic_families": _as_str_str_map(
            _require_nested(obstacle, "semantic_families", "disturbance.obstacle.semantic_families"),
            "disturbance.obstacle.semantic_families",
        ),
        "obstacle_default_upright_quat": _as_tuple4(
            _require_nested(
                obstacle, "default_upright_quat", "disturbance.obstacle.default_upright_quat"
            ),
            "disturbance.obstacle.default_upright_quat",
        ),
        "obstacle_half_size": _as_tuple3(
            _require_nested(obstacle, "half_size", "disturbance.obstacle.half_size"),
            "disturbance.obstacle.half_size",
        ),
        "obstacle_color": _as_tuple3(
            _require_nested(obstacle, "color", "disturbance.obstacle.color"),
            "disturbance.obstacle.color",
        ),
        "obstacle_mass": float(_require_nested(obstacle, "mass", "disturbance.obstacle.mass")),
        "obstacle_scale": float(_require_nested(obstacle, "scale", "disturbance.obstacle.scale")),
    }


def _l2_spawn_for_level3(
    l2: Any,
    *,
    extra_robotwin: Mapping[str, Any],
) -> dict[str, Any]:
    """L3 ``manager.arm`` mirrors L2 spawn; schema-valid keys from ``disturbance.l2``.

    ``start_mode`` / ``flip_side`` / ``start_min_distance`` are not on
    ``DisturbanceL2Config`` (D6) — required from ``extra.robotwin``.
    """
    missing: list[str] = []
    for name in ("alpha", "start_offset", "start_dist_mult"):
        if l2 is MISSING or l2 is None or not hasattr(l2, name):
            missing.append(f"disturbance.l2.{name}")
            continue
        val = getattr(l2, name)
        if val is MISSING or val is None:
            missing.append(f"disturbance.l2.{name}")
    for name in ("l2_start_mode", "l2_start_min_distance", "l2_flip_side"):
        if name not in extra_robotwin or extra_robotwin[name] is None:
            missing.append(f"extra.robotwin.{name}")
    if missing:
        raise ValueError("level 3 requires L2 spawn mirror fields; missing: " + ", ".join(missing))
    return {
        "l2_alpha": float(_require_nested(l2, "alpha", "disturbance.l2.alpha")),
        "l2_start_offset": _as_tuple3(
            _require_nested(l2, "start_offset", "disturbance.l2.start_offset"),
            "disturbance.l2.start_offset",
        ),
        "l2_start_dist_mult": float(
            _require_nested(l2, "start_dist_mult", "disturbance.l2.start_dist_mult")
        ),
        "l2_start_mode": str(_extra(extra_robotwin, "l2_start_mode")),
        "l2_start_min_distance": float(_extra(extra_robotwin, "l2_start_min_distance")),
        "l2_flip_side": bool(_extra(extra_robotwin, "l2_flip_side")),
    }


_SPAWN_DIVERSITY_KEYS: tuple[str, ...] = (
    "enabled",
    "lateral_width_m",
    "linear_z_clearance_m",
    "linear_max_start_z_m",
    "linear_tail_alpha",
    "linear_descent_alpha",
    "clearance_samples",
    "tail_exclude_waypoints",
    "path_num_samples",
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
    "path_num_samples",
    "forbidden_slots",
    "forbidden_depths",
)
_L2_SLOTS = ("left", "center", "right")
_L3_SLOTS = ("left", "opposite", "right")
_L3_DEPTHS = ("near", "mid")


def _nested(extra_robotwin: Mapping[str, Any], *keys: str) -> Any:
    cur: Any = extra_robotwin
    path = ["extra", "robotwin"]
    for key in keys:
        path.append(str(key))
        if not isinstance(cur, Mapping) or key not in cur:
            raise ValueError(f"{'.'.join(path)} missing from resolved config")
        cur = cur[key]
    return cur


def _string_tuple(raw: Any, *, name: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        raise TypeError(f"{name} must be a list, got {type(raw)!r}")
    out = tuple(str(item) for item in raw)
    bad = [item for item in out if item not in allowed]
    if bad:
        raise ValueError(f"{name} entries {bad} not in {allowed}")
    return out


def _l2_move_start_task_step(extra_robotwin: Mapping[str, Any]) -> int:
    raw = _nested(extra_robotwin, "l2", "move_start_task_step")
    step = int(raw)
    if step < 0:
        raise ValueError(
            f"extra.robotwin.l2.move_start_task_step must be >= 0, got {raw!r}"
        )
    return step


def _l3_move_start_task_step(extra_robotwin: Mapping[str, Any]) -> int:
    raw = _nested(extra_robotwin, "l3", "move_start_task_step")
    step = int(raw)
    if step < 0:
        raise ValueError(
            f"extra.robotwin.l3.move_start_task_step must be >= 0, got {raw!r}"
        )
    return step


def _l2_spawn_diversity(extra_robotwin: Mapping[str, Any]) -> dict[str, Any]:
    raw = _nested(extra_robotwin, "l2", "spawn_diversity")
    if not isinstance(raw, Mapping):
        raise TypeError(f"extra.robotwin.l2.spawn_diversity must be a mapping, got {type(raw)!r}")
    missing = [key for key in _SPAWN_DIVERSITY_KEYS if key not in raw]
    if missing:
        raise ValueError(f"extra.robotwin.l2.spawn_diversity missing keys {missing}")
    enabled = raw["enabled"]
    if not isinstance(enabled, bool):
        raise TypeError(f"extra.robotwin.l2.spawn_diversity.enabled must be bool, got {type(enabled)!r}")
    return {
        "enabled": enabled,
        "lateral_width_m": float(raw["lateral_width_m"]),
        "linear_z_clearance_m": float(raw["linear_z_clearance_m"]),
        "linear_max_start_z_m": float(raw["linear_max_start_z_m"]),
        "linear_tail_alpha": float(raw["linear_tail_alpha"]),
        "linear_descent_alpha": float(raw["linear_descent_alpha"]),
        "clearance_samples": int(raw["clearance_samples"]),
        "tail_exclude_waypoints": int(raw["tail_exclude_waypoints"]),
        "path_num_samples": int(raw["path_num_samples"]),
        "forbidden_slots": _string_tuple(
            raw["forbidden_slots"],
            name="extra.robotwin.l2.spawn_diversity.forbidden_slots",
            allowed=_L2_SLOTS,
        ),
    }


def _l3_reach_block(extra_robotwin: Mapping[str, Any]) -> dict[str, Any]:
    raw = _nested(extra_robotwin, "l3")
    if not isinstance(raw, Mapping):
        raise TypeError(f"extra.robotwin.l3 must be a mapping, got {type(raw)!r}")
    for key in ("target_mode", "intercept_alpha", "min_eef_clearance", "min_task_clearance", "tip_overshoot"):
        if key not in raw:
            raise ValueError(f"extra.robotwin.l3.{key} missing from resolved config")
    mode = str(raw["target_mode"])
    if mode not in ("current", "path_intercept"):
        raise ValueError(
            f"extra.robotwin.l3.target_mode must be 'current' or 'path_intercept', got {mode!r}"
        )
    reach_raw = raw.get("reach_diversity")
    if not isinstance(reach_raw, Mapping):
        raise TypeError(f"extra.robotwin.l3.reach_diversity must be a mapping, got {type(reach_raw)!r}")
    missing = [key for key in _REACH_DIVERSITY_KEYS if key not in reach_raw]
    if missing:
        raise ValueError(f"extra.robotwin.l3.reach_diversity missing keys {missing}")
    enabled = reach_raw["enabled"]
    if not isinstance(enabled, bool):
        raise TypeError(
            f"extra.robotwin.l3.reach_diversity.enabled must be bool, got {type(enabled)!r}"
        )
    pin = reach_raw["pin_approach_steps"]
    if not isinstance(pin, bool):
        raise TypeError(
            "extra.robotwin.l3.reach_diversity.pin_approach_steps must be bool, "
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
            "path_num_samples": int(reach_raw["path_num_samples"]),
            "forbidden_slots": _string_tuple(
                reach_raw["forbidden_slots"],
                name="extra.robotwin.l3.reach_diversity.forbidden_slots",
                allowed=_L3_SLOTS,
            ),
            "forbidden_depths": _string_tuple(
                reach_raw["forbidden_depths"],
                name="extra.robotwin.l3.reach_diversity.forbidden_depths",
                allowed=_L3_DEPTHS,
            ),
        },
    }


def build_disturbance_config(
    cfg: RunConfig,
    *,
    extra_robotwin: Mapping[str, Any],
) -> SimpleNamespace:
    """Materialize flat ``RoboTwinDyniSafetyConfig``-style attrs (no Python defaults)."""
    dynamic_level = int(_require(cfg.level, "level"))
    seed = int(_require(cfg.seed, "seed"))

    if dynamic_level == 0:
        return SimpleNamespace(dynamic_level=0, seed=seed)

    contact = _require(cfg.contact, "contact")
    scene = _require(cfg.scene, "scene")
    disturbance = _require(cfg.disturbance, "disturbance")
    obstacle = _require_nested(disturbance, "obstacle", "disturbance.obstacle")

    attrs: dict[str, Any] = {
        "dynamic_level": dynamic_level,
        "seed": seed,
        **_contact_attrs(contact),
        "collision_force_control_freq_hz": float(
            _extra(extra_robotwin, "collision_force_control_freq_hz")
        ),
        "video_sim_stride": int(_extra(extra_robotwin, "video_sim_stride")),
        **_scene_attrs(scene),
        **_obstacle_attrs(obstacle),
    }

    if dynamic_level == 1:
        l1 = _require_nested(disturbance, "l1", "disturbance.l1")
        attrs.update(
            {
                "l1_trigger_dist": float(
                    _require_nested(l1, "trigger_dist", "disturbance.l1.trigger_dist")
                ),
                "l1_drop_height": float(
                    _require_nested(l1, "drop_height", "disturbance.l1.drop_height")
                ),
                "l1_alpha": float(_require_nested(l1, "alpha", "disturbance.l1.alpha")),
                "l12_force_scheme": _l12_force_scheme(extra_robotwin),
            }
        )
    elif dynamic_level == 2:
        l2 = _require_nested(disturbance, "l2", "disturbance.l2")
        attrs.update(
            {
                "l2_alpha": float(_require_nested(l2, "alpha", "disturbance.l2.alpha")),
                "l2_move_start_step": int(
                    _require_nested(l2, "move_start_step", "disturbance.l2.move_start_step")
                ),
                "l2_move_start_task_step": _l2_move_start_task_step(extra_robotwin),
                "l2_move_end_step": int(
                    _require_nested(l2, "move_end_step", "disturbance.l2.move_end_step")
                ),
                "l2_hold_steps": int(
                    _require_nested(l2, "settle_steps", "disturbance.l2.settle_steps")
                ),
                "l2_arc_height": float(
                    _require_nested(l2, "arc_height", "disturbance.l2.arc_height")
                ),
                "l2_start_offset": _as_tuple3(
                    _require_nested(l2, "start_offset", "disturbance.l2.start_offset"),
                    "disturbance.l2.start_offset",
                ),
                "l2_start_dist_mult": float(
                    _require_nested(l2, "start_dist_mult", "disturbance.l2.start_dist_mult")
                ),
                "l2_start_mode": str(_extra(extra_robotwin, "l2_start_mode")),
                "l2_start_min_distance": float(_extra(extra_robotwin, "l2_start_min_distance")),
                "l2_flip_side": bool(_extra(extra_robotwin, "l2_flip_side")),
                "l12_force_scheme": _l12_force_scheme(extra_robotwin),
                "spawn_diversity": _l2_spawn_diversity(extra_robotwin),
                "mode_seed": seed,
            }
        )
    elif dynamic_level == 3:
        l2 = disturbance.l2 if hasattr(disturbance, "l2") else MISSING
        l3 = _require_nested(disturbance, "l3", "disturbance.l3")
        hand = _require(cfg.hand, "hand")
        hand_geometry = _normalize_hand_geometry(_require_nested(hand, "geometry", "hand.geometry"))
        l3_block = _l3_reach_block(extra_robotwin)
        attrs.update(_l2_spawn_for_level3(l2, extra_robotwin=extra_robotwin))
        attrs.update(
            {
                "l3_target_mode": str(l3_block["target_mode"]),
                "l3_intercept_alpha": float(l3_block["intercept_alpha"]),
                "l3_min_eef_clearance": float(l3_block["min_eef_clearance"]),
                "l3_min_task_clearance": float(l3_block["min_task_clearance"]),
                "l3_tip_overshoot": float(l3_block["tip_overshoot"]),
                "reach_diversity": l3_block["reach_diversity"],
                "mode_seed": seed,
                "l3_alpha": float(_require_nested(l3, "alpha", "disturbance.l3.alpha")),
                "l3_move_start_step": int(
                    _require_nested(l3, "move_start_step", "disturbance.l3.move_start_step")
                ),
                "l3_move_start_task_step": _l3_move_start_task_step(extra_robotwin),
                "l3_entry_steps": int(
                    _require_nested(l3, "entry_steps", "disturbance.l3.entry_steps")
                ),
                "l3_force_scheme": _l3_force_scheme(extra_robotwin),
                "l3_approach_steps": int(_extra(extra_robotwin, "l3_approach_steps")),
                "l3_dwell_steps": int(
                    _require_nested(l3, "dwell_steps", "disturbance.l3.dwell_steps")
                ),
                "l3_exit_steps": int(_extra(extra_robotwin, "l3_exit_steps")),
                "l3_enter_height": float(
                    _require_nested(l3, "enter_height", "disturbance.l3.enter_height")
                ),
                "l3_dwell_height": float(_extra(extra_robotwin, "l3_dwell_height")),
                "l3_flinch_height": float(_extra(extra_robotwin, "l3_flinch_height")),
                "l3_flinch_dist": float(_require_nested(hand, "flinch_dist", "hand.flinch_dist")),
                "l3_post_hit_hold_steps": int(_extra(extra_robotwin, "l3_post_hit_hold_steps")),
                "l3_spawn_pull_x": float(_extra(extra_robotwin, "l3_spawn_pull_x")),
                "l3_spawn_pull_y": float(_extra(extra_robotwin, "l3_spawn_pull_y")),
                "l3_spawn_pull_to_midpoint": float(
                    _extra(extra_robotwin, "l3_spawn_pull_to_midpoint")
                ),
                "l3_spawn_xy_offset": _as_tuple2(
                    _extra(extra_robotwin, "l3_spawn_xy_offset"),
                    "extra.robotwin.l3_spawn_xy_offset",
                ),
                "l3_spawn_world_xy": _as_tuple2(
                    _extra(extra_robotwin, "l3_spawn_world_xy"),
                    "extra.robotwin.l3_spawn_world_xy",
                ),
                "l3_spawn_center_source": str(_extra(extra_robotwin, "l3_spawn_center_source")),
                "l3_spawn_extra_z_lift": float(_extra(extra_robotwin, "l3_spawn_extra_z_lift")),
                "debug_l3_spawn": bool(_extra(extra_robotwin, "debug_l3_spawn")),
                "l3_start_offset": _as_tuple3(
                    _require_nested(l3, "start_offset", "disturbance.l3.start_offset"),
                    "disturbance.l3.start_offset",
                ),
                "l3_start_dist_mult": float(
                    _require_nested(l3, "start_dist_mult", "disturbance.l3.start_dist_mult")
                ),
                "l3_start_mode": str(
                    _require_nested(l3, "start_mode", "disturbance.l3.start_mode")
                ),
                "l3_start_min_distance": float(_extra(extra_robotwin, "l3_start_min_distance")),
                "l3_flip_side": bool(_require_nested(l3, "flip_side", "disturbance.l3.flip_side")),
                "l3_ignore_collision_link_substrings": _as_str_tuple(
                    _require_nested(
                        l3,
                        "ignore_collision_link_substrings",
                        "disturbance.l3.ignore_collision_link_substrings",
                    ),
                    "disturbance.l3.ignore_collision_link_substrings",
                ),
                "hand_geometry": hand_geometry,
                "hand_mesh_path": str(_require_nested(hand, "mesh_path", "hand.mesh_path")),
                "hand_mesh_scale": float(_require_nested(hand, "scale", "hand.scale")),
                "hand_half_size": _as_tuple3(
                    _require_nested(hand, "half_size", "hand.half_size"),
                    "hand.half_size",
                ),
                "hand_mass": float(_require_nested(hand, "mass_kg", "hand.mass_kg")),
                "hand_color": _as_tuple3(
                    _extra(extra_robotwin, "hand_color"), "extra.robotwin.hand_color"
                ),
                "hand_mesh_base_quat": _as_tuple4(
                    _extra(extra_robotwin, "hand_mesh_base_quat"),
                    "extra.robotwin.hand_mesh_base_quat",
                ),
                "hand_mesh_auto_orient": bool(_extra(extra_robotwin, "hand_mesh_auto_orient")),
            }
        )
    else:
        raise ValueError(f"unsupported dynamic level {dynamic_level}")

    return SimpleNamespace(**attrs)


__all__ = ["build_disturbance_config"]
