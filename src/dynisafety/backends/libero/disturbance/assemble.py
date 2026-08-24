"""Assemble level controllers from ``RunConfig`` + ``extra.libero`` (no defaults)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dynisafety.backends.libero.capabilities import LIBERO_CAPABILITIES
from dynisafety.backends.libero.disturbance._support import (
    HandPhysicsParams,
    collect_workspace_entity_poses,
    compute_l3_interpolated_target,
    compute_l3_path_intercept_target,
    detect_visible_obstacles,
    get_robot_base_pos,
    get_table_center_pos,
    get_task_object_positions,
    l3_dwell_at_xy_keep_z,
    libero_goal_source,
    parse_first_on_in,
    pick_workspace_entity_by_substrings,
    relocate_obstacle_clear_of_task_objects,
    select_instruction_first_task_object,
    select_l3_intercept_entity,
    select_l3_shared_reach_target,
)
from dynisafety.backends.libero.disturbance.l1 import SuddenDropController
from dynisafety.backends.libero.disturbance.l2 import DynamicObstacleController
from dynisafety.backends.libero.disturbance.l3 import SharedReachingController
from dynisafety.backends.libero.disturbance.l3_reach import DEPTHS, SLOTS
from dynisafety.backends.libero.obstacle_pack import (
    parse_obstacle_pack,
    plan_for_task,
    planned_instance,
)
from dynisafety.backends.libero.env import (
    LiberoEnvHandle,
    apply_hand_collision_box,
    configure_hand_mesh_visibility,
    hide_all_obstacles,
    hide_overlapping_bases,
)
from dynisafety.config.schema import MISSING, RunConfig
from dynisafety.core.disturbance import require_configured

_require = require_configured


def _libero_extra(extra_libero: Mapping[str, Any], *keys: str, required: bool = True) -> Any:
    cur: Any = extra_libero
    path = "extra.libero"
    for key in keys:
        path = f"{path}.{key}"
        if not isinstance(cur, Mapping) or key not in cur:
            if required:
                raise ValueError(f"{path} missing from resolved config")
            return None
        cur = cur[key]
    return cur


def _l3_target_mode(raw: Any) -> str:
    text = str(raw).strip().lower()
    aliases = {
        "1": "nearest_non_task",
        "nearest_non_task": "nearest_non_task",
        "2": "eef_task_interp",
        "eef_task_interp": "eef_task_interp",
        "3": "path_intercept",
        "path_intercept": "path_intercept",
    }
    if text not in aliases:
        raise ValueError(
            "extra.libero.l3.target_mode must be 1/nearest_non_task, "
            f"2/eef_task_interp, or 3/path_intercept, got {raw!r}"
        )
    return aliases[text]


def _l3_intercept_entity_substrings(l3_extra: Mapping[str, Any]) -> tuple[str, ...]:
    raw = l3_extra.get("intercept_entity_substrings")
    if raw is None:
        return ()
    if isinstance(raw, str):
        return tuple(part.strip() for part in raw.split(",") if part.strip())
    if isinstance(raw, (list, tuple)):
        return tuple(str(item).strip() for item in raw if str(item).strip())
    raise TypeError(
        f"extra.libero.l3.intercept_entity_substrings must be str or list, got {type(raw)!r}"
    )


def _as_xyz_list(pos: np.ndarray) -> list[float]:
    arr = np.asarray(pos, dtype=np.float64).reshape(3)
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _require_xy(raw: Any, name: str) -> list[float]:
    vals = list(_require(raw, name))
    if len(vals) != 2:
        raise ValueError(f"{name} must be length 2, got {vals!r}")
    out = [float(vals[0]), float(vals[1])]
    if out[0] <= 0.0 or out[1] <= 0.0:
        raise ValueError(f"{name} must be > 0, got {out!r}")
    return out


def _optional_xyz(raw: Any, name: str) -> list[float] | None:
    if raw is MISSING or raw is None:
        return None
    vals = list(raw)
    if not vals:
        return None
    if len(vals) != 3:
        raise ValueError(f"{name} must be length 3 or empty, got {vals!r}")
    return [float(x) for x in vals]


def _parse_allowlist(raw: Any) -> tuple[str, ...]:
    if raw is None or raw is MISSING:
        return ()
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return ()
        return tuple(part.strip() for part in text.split(",") if part.strip())
    if isinstance(raw, (list, tuple)):
        return tuple(str(x) for x in raw)
    raise TypeError(f"l1_drop_obstacle_allowlist must be str or list, got {type(raw)!r}")


def _hide_hand_pusher(handle: LiberoEnvHandle) -> None:
    import mujoco

    inner = handle.env.env
    hand_obj = inner.objects_dict.get("hand_pusher_1")
    if hand_obj is None:
        return
    sim = inner.sim
    sim.data.set_joint_qpos(
        hand_obj.joints[-1],
        np.array([10.0, 10.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )
    mujoco.mj_forward(sim.model._model, sim.data._data)


def _swap_obstacle(
    handle: LiberoEnvHandle,
    old_name: str,
    new_name: str,
    pos: np.ndarray,
    quat: np.ndarray,
) -> None:
    import mujoco

    inner = handle.env.env
    sim = inner.sim
    old = inner.objects_dict.get(old_name)
    new = inner.objects_dict.get(new_name)
    if old is None or new is None:
        raise KeyError(f"cannot swap obstacles {old_name!r} → {new_name!r}")
    far = np.array([10.0, 10.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    sim.data.set_joint_qpos(old.joints[-1], far)
    sim.data.set_joint_qpos(new.joints[-1], np.concatenate([pos, quat]))
    mujoco.mj_forward(sim.model._model, sim.data._data)


def _apply_obstacle_pack_identity(
    handle: LiberoEnvHandle,
    extra_libero: Mapping[str, Any],
    *,
    episode_index: int,
    official_name: str,
    pos: np.ndarray,
    quat: np.ndarray,
) -> str:
    """If pack is on, maybe swap an extra similar body onto the official pose."""
    pack = parse_obstacle_pack(extra_libero)
    plan = plan_for_task(pack, handle.suite, handle.task_name)
    desired = planned_instance(plan, episode_index, official_name)
    if desired == official_name:
        return official_name
    objects = handle.env.env.objects_dict
    if desired not in objects:
        return official_name
    _swap_obstacle(handle, official_name, desired, pos, quat)
    return desired


def _task_object_names(handle: LiberoEnvHandle) -> list[str]:
    env = handle.env
    raw = getattr(env, "obj_of_interest", None) or getattr(env.env, "obj_of_interest", None)
    if not raw:
        return []
    return [n for n in raw if n in env.env.objects_dict]


def _resolve_episode_obstacle_name(
    *,
    extra_libero: Mapping[str, Any],
    suite: str,
    task_name: str,
    episode_index: int,
    pool: Sequence[str],
) -> str | None:
    """D3: ``obstacle_episode_names[suite][task_stem][episode_idx % n]`` else None."""
    root = extra_libero.get("obstacle_episode_names")
    if not isinstance(root, Mapping):
        return None
    suite_map = root.get(suite)
    if not isinstance(suite_map, Mapping):
        return None
    names = suite_map.get(task_name)
    if not isinstance(names, (list, tuple)) or not names:
        return None
    return str(names[int(episode_index) % len(names)])


def _task_object_clearance_xy(extra_libero: Mapping[str, Any]) -> float:
    raw = _libero_extra(extra_libero, "task_object_clearance_xy", required=True)
    val = float(raw)
    if val < 0.0:
        raise ValueError(f"extra.libero.task_object_clearance_xy must be >= 0, got {raw!r}")
    return val


def _l1_home_workspace_onset(extra_libero: Mapping[str, Any]) -> bool:
    """YAML-only: defer L1 drop when the home EEF already sits in the circle."""
    raw = _libero_extra(extra_libero, "l1_home_workspace_onset", required=True)
    if not isinstance(raw, bool):
        raise TypeError(f"extra.libero.l1_home_workspace_onset must be bool, got {type(raw)!r}")
    return raw


def _l1_nudge_on_penetration_only(extra_libero: Mapping[str, Any]) -> bool:
    """YAML-only: L1 footprint nudge only after scene-object 穿模."""
    raw = _libero_extra(extra_libero, "l1_nudge_on_penetration_only", required=True)
    if not isinstance(raw, bool):
        raise TypeError(
            f"extra.libero.l1_nudge_on_penetration_only must be bool, got {type(raw)!r}"
        )
    return raw


def _l1_workspace_onset_dist(extra_libero: Mapping[str, Any]) -> float | None:
    """YAML-only L1 3D approach radius. ``null`` follows ``trigger_dist``."""
    raw = _libero_extra(extra_libero, "l1_workspace_onset_dist", required=True)
    if raw is None:
        return None
    val = float(raw)
    if val <= 0.0:
        raise ValueError(f"extra.libero.l1_workspace_onset_dist must be > 0 or null, got {raw!r}")
    return val


def _l2_nudge_landing_xy(extra_libero: Mapping[str, Any]) -> bool:
    """YAML-only: L2 landing XY nudge off overlapping task objects."""
    raw = _libero_extra(extra_libero, "l2", "nudge_landing_xy", required=True)
    if not isinstance(raw, bool):
        raise TypeError(f"extra.libero.l2.nudge_landing_xy must be bool, got {type(raw)!r}")
    return raw


def _l2_arc_height(l2: Any, extra_libero: Mapping[str, Any]) -> float:
    """Task overlay ``extra.libero.l2.arc_height`` wins over ``disturbance.l2``."""
    override = _libero_extra(extra_libero, "l2", "arc_height", required=False)
    if override is not None:
        val = float(override)
        if val < 0.0:
            raise ValueError(f"extra.libero.l2.arc_height must be >= 0, got {override!r}")
        return val
    return float(_require(l2.arc_height, "disturbance.l2.arc_height"))


def _l2_start_offset(l2: Any, extra_libero: Mapping[str, Any]) -> list[float]:
    """``disturbance.l2.start_offset`` plus optional task ``start_z_offset``."""
    raw = list(_require(l2.start_offset, "disturbance.l2.start_offset"))
    if len(raw) != 3:
        raise ValueError(f"disturbance.l2.start_offset must have 3 values, got {raw!r}")
    z_extra = _libero_extra(extra_libero, "l2", "start_z_offset", required=False)
    z_off = 0.0 if z_extra is None else float(z_extra)
    return [float(raw[0]), float(raw[1]), float(raw[2]) + z_off]


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


def _l2_spawn_diversity(extra_libero: Mapping[str, Any]) -> dict[str, Any]:
    """YAML-only L2 spawn-frame diversity block. Every key is required."""
    raw = _libero_extra(extra_libero, "l2", "spawn_diversity", required=True)
    if not isinstance(raw, Mapping):
        raise TypeError(f"extra.libero.l2.spawn_diversity must be a mapping, got {type(raw)!r}")
    missing = [key for key in _SPAWN_DIVERSITY_KEYS if key not in raw]
    if missing:
        raise ValueError(
            f"extra.libero.l2.spawn_diversity missing keys {missing} from resolved config"
        )
    enabled = raw["enabled"]
    if not isinstance(enabled, bool):
        raise TypeError(
            f"extra.libero.l2.spawn_diversity.enabled must be bool, got {type(enabled)!r}"
        )
    forbidden = raw["forbidden_slots"]
    if forbidden is None:
        forbidden_slots: tuple[str, ...] = ()
    elif isinstance(forbidden, str):
        raise TypeError(
            "extra.libero.l2.spawn_diversity.forbidden_slots must be a list, "
            f"got {type(forbidden)!r}"
        )
    else:
        forbidden_slots = tuple(str(item) for item in forbidden)
    return {
        "enabled": enabled,
        "lateral_width_m": float(raw["lateral_width_m"]),
        "linear_z_clearance_m": float(raw["linear_z_clearance_m"]),
        "linear_max_start_z_m": float(raw["linear_max_start_z_m"]),
        "linear_tail_alpha": float(raw["linear_tail_alpha"]),
        "linear_descent_alpha": float(raw["linear_descent_alpha"]),
        "clearance_samples": int(raw["clearance_samples"]),
        "tail_exclude_waypoints": int(raw["tail_exclude_waypoints"]),
        "forbidden_slots": forbidden_slots,
    }


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


def _l3_string_tuple(raw: Any, *, name: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        raise TypeError(f"{name} must be a list, got {type(raw)!r}")
    out = tuple(str(item) for item in raw)
    bad = [item for item in out if item not in allowed]
    if bad:
        raise ValueError(f"{name} entries {bad} not in {allowed}")
    return out


def _l3_reach_diversity(extra_libero: Mapping[str, Any]) -> dict[str, Any]:
    """YAML-only L3 reach-frame diversity block. Every key is required."""
    raw = _libero_extra(extra_libero, "l3", "reach_diversity", required=True)
    if not isinstance(raw, Mapping):
        raise TypeError(f"extra.libero.l3.reach_diversity must be a mapping, got {type(raw)!r}")
    missing = [key for key in _REACH_DIVERSITY_KEYS if key not in raw]
    if missing:
        raise ValueError(
            f"extra.libero.l3.reach_diversity missing keys {missing} from resolved config"
        )
    enabled = raw["enabled"]
    if not isinstance(enabled, bool):
        raise TypeError(
            f"extra.libero.l3.reach_diversity.enabled must be bool, got {type(enabled)!r}"
        )
    pin = raw["pin_approach_steps"]
    if not isinstance(pin, bool):
        raise TypeError(
            "extra.libero.l3.reach_diversity.pin_approach_steps must be bool, "
            f"got {type(pin)!r}"
        )
    near_alpha = float(raw["near_alpha"])
    mid_alpha = float(raw["mid_alpha"])
    if not (0.0 <= near_alpha <= 1.0 and 0.0 <= mid_alpha <= 1.0):
        raise ValueError(
            "extra.libero.l3.reach_diversity near_alpha/mid_alpha must be in [0, 1], "
            f"got {near_alpha!r}, {mid_alpha!r}"
        )
    hover = float(raw["hover_rise_m"])
    via_desc = float(raw["via_descent_alpha"])
    if hover < 0.0:
        raise ValueError(f"extra.libero.l3.reach_diversity.hover_rise_m must be >= 0, got {hover}")
    if via_desc <= 0.0 or via_desc > 1.0:
        raise ValueError(
            "extra.libero.l3.reach_diversity.via_descent_alpha must be in (0, 1], "
            f"got {via_desc}"
        )
    return {
        "enabled": enabled,
        "lateral_width_m": float(raw["lateral_width_m"]),
        "near_alpha": near_alpha,
        "mid_alpha": mid_alpha,
        "hover_rise_m": hover,
        "via_descent_alpha": via_desc,
        "pin_approach_steps": pin,
        "clearance_samples": int(raw["clearance_samples"]),
        "min_clearance_m": float(raw["min_clearance_m"]),
        "depth_collapse_m": float(raw["depth_collapse_m"]),
        "forbidden_slots": _l3_string_tuple(
            raw["forbidden_slots"],
            name="extra.libero.l3.reach_diversity.forbidden_slots",
            allowed=SLOTS,
        ),
        "forbidden_depths": _l3_string_tuple(
            raw["forbidden_depths"],
            name="extra.libero.l3.reach_diversity.forbidden_depths",
            allowed=DEPTHS,
        ),
    }


def _l2_reject_align_with_diversity(
    *,
    align_start_y_to_eef: bool,
    spawn_diversity: Mapping[str, Any],
) -> None:
    """``align_start_y_to_eef`` overwrites slot Y; refuse when diversity is on."""
    if bool(align_start_y_to_eef) and bool(spawn_diversity.get("enabled")):
        raise ValueError(
            "disturbance.l2.align_start_y_to_eef=true is incompatible with "
            "extra.libero.l2.spawn_diversity.enabled: it overwrites the "
            "start Y after slot planning and erases lateral offset"
        )


def _align_obstacle_with_safelibero(extra_libero: Mapping[str, Any]) -> bool:
    """YAML-only switch (iron law 1): follow this episode's SafeLIBERO init identity."""
    raw = _libero_extra(extra_libero, "align_obstacle_with_safelibero", required=True)
    if not isinstance(raw, bool):
        raise TypeError(
            f"extra.libero.align_obstacle_with_safelibero must be bool, got {type(raw)!r}"
        )
    return raw


def select_safelibero_init_obstacle(
    visible: Sequence[tuple[str, np.ndarray, np.ndarray]],
    eef_pos: np.ndarray,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Pick the authored SafeLIBERO obstacle for this init-state episode.

    Level-I ``pruned_init`` hides unused obstacles off-screen; remaining
    ``*obstacle*`` bodies are this trajectory's objects. If several remain,
    take the one closest to the EEF and **keep that instance's identity**
    (never swap another mesh onto its pose).
    """
    if not visible:
        raise ValueError("no visible SafeLIBERO obstacles in the current init state")
    eef = np.asarray(eef_pos, dtype=np.float64)
    chosen = min(
        visible,
        key=lambda item: float(np.linalg.norm(np.asarray(item[1], dtype=np.float64) - eef)),
    )
    return (
        chosen[0],
        np.asarray(chosen[1], dtype=np.float64),
        np.asarray(chosen[2], dtype=np.float64),
    )


def _hand_physics(cfg: RunConfig) -> HandPhysicsParams:
    hand = _require(cfg.hand, "cfg.hand")
    return HandPhysicsParams(
        mass_kg=float(_require(hand.mass_kg, "hand.mass_kg")),
        contact_solref=list(_require(hand.contact_solref, "hand.contact_solref")),
        contact_solimp=list(_require(hand.contact_solimp, "hand.contact_solimp")),
        contact_friction=list(_require(hand.contact_friction, "hand.contact_friction")),
        near_eef_m=float(_require(hand.near_eef_m, "hand.near_eef_m")),
        near_eef_max_step_m=float(_require(hand.near_eef_max_step_m, "hand.near_eef_max_step_m")),
        in_contact_max_step_m=float(
            _require(hand.in_contact_max_step_m, "hand.in_contact_max_step_m")
        ),
        big_jump_m=float(_require(hand.big_jump_m, "hand.big_jump_m")),
    )


def _control_freq(handle: LiberoEnvHandle) -> int:
    inner = handle.env.env
    freq = getattr(inner, "control_freq", None)
    if freq is None:
        raise ValueError("env.env.control_freq missing; cannot invent a default (iron law 1)")
    value = int(freq)
    if value <= 0:
        raise ValueError(f"control_freq must be positive, got {value}")
    return value


def build_controller(
    level: int,
    handle: LiberoEnvHandle,
    cfg: RunConfig,
    *,
    episode_index: int,
    extra_libero: Mapping[str, Any],
) -> SuddenDropController | DynamicObstacleController | SharedReachingController | None:
    """level 0 → None; unsupported level → ValueError."""
    if level not in LIBERO_CAPABILITIES.levels:
        raise ValueError(f"level={level} not in LIBERO capabilities {LIBERO_CAPABILITIES.levels}")

    overlap = float(_libero_extra(extra_libero, "pedestal_overlap_radius", required=True))

    if level == 0:
        hide_all_obstacles(handle, include_bases=False)
        hide_overlapping_bases(handle, overlap_radius=overlap)
        _hide_hand_pusher(handle)
        return None

    disturbance = _require(cfg.disturbance, "cfg.disturbance")
    seed = int(_require(cfg.seed, "cfg.seed"))
    task_id = int(handle.task_id)
    plan_seed = seed + task_id * 10000 + int(episode_index)

    if level == 1:
        return _build_l1(
            handle,
            cfg,
            disturbance=disturbance,
            episode_index=episode_index,
            extra_libero=extra_libero,
            overlap=overlap,
            seed=plan_seed,
        )
    if level == 2:
        return _build_l2(
            handle,
            cfg,
            disturbance=disturbance,
            episode_index=episode_index,
            extra_libero=extra_libero,
            overlap=overlap,
            seed=plan_seed,
        )
    if level == 3:
        return _build_l3(
            handle,
            cfg,
            disturbance=disturbance,
            episode_index=episode_index,
            extra_libero=extra_libero,
            overlap=overlap,
            seed=plan_seed,
        )
    raise ValueError(f"unhandled level={level}")


def _build_l1(
    handle: LiberoEnvHandle,
    cfg: RunConfig,
    *,
    disturbance: Any,
    episode_index: int,
    extra_libero: Mapping[str, Any],
    overlap: float,
    seed: int,
) -> SuddenDropController | None:
    l1 = _require(disturbance.l1, "disturbance.l1")
    obstacle = _require(disturbance.obstacle, "disturbance.obstacle")
    pool = tuple(_require(obstacle.pool, "disturbance.obstacle.pool"))
    if _align_obstacle_with_safelibero(extra_libero):
        visible = detect_visible_obstacles(handle.env)
        if not visible:
            hide_overlapping_bases(handle, overlap_radius=overlap)
            _hide_hand_pusher(handle)
            return None
        eef = handle.env.env.sim.data.site_xpos[handle.env.env.robots[0].eef_site_id].copy()
        chosen_name, chosen_pos, chosen_quat = select_safelibero_init_obstacle(
            visible, eef
        )
        chosen_name = _apply_obstacle_pack_identity(
            handle,
            extra_libero,
            episode_index=episode_index,
            official_name=chosen_name,
            pos=chosen_pos,
            quat=chosen_quat,
        )
        allowlist = (chosen_name,)
    else:
        allowlist = _parse_allowlist(
            _libero_extra(extra_libero, "l1_drop_obstacle_allowlist", required=False) or ""
        )
        named = _resolve_episode_obstacle_name(
            extra_libero=extra_libero,
            suite=handle.suite,
            task_name=handle.task_name,
            episode_index=episode_index,
            pool=pool,
        )
        if named is not None:
            allowlist = (named,)

    clearance = _task_object_clearance_xy(extra_libero)
    ctrl = SuddenDropController(
        handle,
        trigger_dist=float(_require(l1.trigger_dist, "disturbance.l1.trigger_dist")),
        drop_height=float(_require(l1.drop_height, "disturbance.l1.drop_height")),
        drop_z_bias=float(_require(l1.drop_z_bias, "disturbance.l1.drop_z_bias")),
        trigger_dist_bias=float(_require(l1.trigger_dist_bias, "disturbance.l1.trigger_dist_bias")),
        resting_z_mode=str(_libero_extra(extra_libero, "resting_z_mode", required=True)),
        drop_normalize_z=bool(_libero_extra(extra_libero, "drop_normalize_z", required=True)),
        obstacle_pool=pool,
        allowlist=allowlist,
        seed=seed,
        task_object_names=_task_object_names(handle),
        task_clearance_xy=clearance,
        pedestal_overlap_radius=overlap,
        home_workspace_onset=_l1_home_workspace_onset(extra_libero),
        nudge_on_penetration_only=_l1_nudge_on_penetration_only(extra_libero),
        workspace_onset_dist=_l1_workspace_onset_dist(extra_libero),
    )
    ctrl.reset(episode_index)
    if not ctrl._bound:  # noqa: SLF001 — assembly needs bind status
        hide_overlapping_bases(handle, overlap_radius=overlap)
        _hide_hand_pusher(handle)
        return None
    hide_overlapping_bases(handle, overlap_radius=overlap)
    hide_all_obstacles(handle, include_bases=False)
    _hide_hand_pusher(handle)
    # Re-hide chosen (hide_all already moved it; re-bind teleport for consistency).
    ctrl._teleport_hidden()  # noqa: SLF001
    return ctrl


def _build_l2(
    handle: LiberoEnvHandle,
    cfg: RunConfig,
    *,
    disturbance: Any,
    episode_index: int,
    extra_libero: Mapping[str, Any],
    overlap: float,
    seed: int,
) -> DynamicObstacleController | None:
    l2 = _require(disturbance.l2, "disturbance.l2")
    obstacle = _require(disturbance.obstacle, "disturbance.obstacle")
    pool = tuple(_require(obstacle.pool, "disturbance.obstacle.pool"))
    rotation_pool = obstacle.rotation_pool
    if rotation_pool is MISSING or rotation_pool is None:
        rotation_pool = ()
    else:
        rotation_pool = tuple(rotation_pool)

    visible = detect_visible_obstacles(handle.env)
    if not visible:
        hide_overlapping_bases(handle, overlap_radius=overlap)
        _hide_hand_pusher(handle)
        return None

    eef = handle.env.env.sim.data.site_xpos[handle.env.env.robots[0].eef_site_id].copy()
    chosen_name, chosen_pos, chosen_quat = select_safelibero_init_obstacle(visible, eef)
    chosen_name = _apply_obstacle_pack_identity(
        handle,
        extra_libero,
        episode_index=episode_index,
        official_name=chosen_name,
        pos=chosen_pos,
        quat=chosen_quat,
    )

    if not _align_obstacle_with_safelibero(extra_libero):
        named = _resolve_episode_obstacle_name(
            extra_libero=extra_libero,
            suite=handle.suite,
            task_name=handle.task_name,
            episode_index=episode_index,
            pool=pool,
        )
        if named is not None and named in handle.env.env.objects_dict:
            if named != chosen_name:
                _swap_obstacle(handle, chosen_name, named, chosen_pos, chosen_quat)
                chosen_name = named
        elif rotation_pool and "object" in handle.suite:
            desired = rotation_pool[int(episode_index) % len(rotation_pool)]
            if desired != chosen_name and desired in handle.env.env.objects_dict:
                _swap_obstacle(handle, chosen_name, desired, chosen_pos, chosen_quat)
                chosen_name = desired

    l2_extra = _libero_extra(extra_libero, "l2", required=True)
    if not isinstance(l2_extra, Mapping):
        raise TypeError("extra.libero.l2 must be a mapping")

    chosen_pos = np.asarray(chosen_pos, dtype=np.float64).copy()
    if _l2_nudge_landing_xy(extra_libero):
        clearance = _task_object_clearance_xy(extra_libero)
        new_xy, _pedestal, notes = relocate_obstacle_clear_of_task_objects(
            handle,
            chosen_name,
            chosen_pos[:2],
            _task_object_names(handle),
            clearance_xy=clearance,
            pedestal_overlap_radius=overlap,
            require_penetration=True,
        )
        if notes:
            print(
                f"[DynamicObstacleController] landing nudged xy: "
                f"obstacle={chosen_name!r} "
                f"orig=({chosen_pos[0]:+.3f}, {chosen_pos[1]:+.3f}) "
                f"-> ({new_xy[0]:+.3f}, {new_xy[1]:+.3f}) " + "; ".join(notes),
                flush=True,
            )
        chosen_pos[0] = float(new_xy[0])
        chosen_pos[1] = float(new_xy[1])
    hide_overlapping_bases(handle, overlap_radius=overlap)
    _hide_hand_pusher(handle)

    hand = _require(cfg.hand, "cfg.hand")
    peak_map = _require(hand.peak_velocity, "hand.peak_velocity")
    # peak_velocity is unused by the step-indexed L2 path but required by the ctor.
    if not isinstance(peak_map, Mapping) or "medium" not in peak_map:
        raise ValueError("hand.peak_velocity.medium required for L2 assembly")
    peak_velocity = float(peak_map["medium"])

    spawn_diversity = _l2_spawn_diversity(extra_libero)
    align_start_y_to_eef = bool(
        _require(l2.align_start_y_to_eef, "disturbance.l2.align_start_y_to_eef")
    )
    _l2_reject_align_with_diversity(
        align_start_y_to_eef=align_start_y_to_eef,
        spawn_diversity=spawn_diversity,
    )
    mode_seed = int(_require(cfg.seed, "cfg.seed"))

    return DynamicObstacleController(
        handle,
        move_start_step=int(_require(l2.move_start_step, "disturbance.l2.move_start_step")),
        move_end_step=int(_require(l2.move_end_step, "disturbance.l2.move_end_step")),
        settle_steps=int(_require(l2.settle_steps, "disturbance.l2.settle_steps")),
        start_offset=_l2_start_offset(l2, extra_libero),
        arc_height=_l2_arc_height(l2, extra_libero),
        align_start_y_to_eef=align_start_y_to_eef,
        resting_z_mode=str(_libero_extra(extra_libero, "resting_z_mode", required=True)),
        enable_random_start=bool(l2_extra["enable_random_start"]),
        random_start_xy_range=float(l2_extra["random_start_xy_range"]),
        random_start_z_range=float(l2_extra["random_start_z_range"]),
        random_start_max_attempts=int(l2_extra["random_start_max_attempts"]),
        target_z_bias=float(l2_extra["target_z_bias"]),
        obstacle_name=chosen_name,
        rotation_pool=rotation_pool,
        peak_velocity=peak_velocity,
        seed=seed,
        final_pos=np.asarray(chosen_pos, dtype=np.float64),
        final_quat=np.asarray(chosen_quat, dtype=np.float64),
        plan_num_samples=int(l2_extra["plan_num_samples"]),
        target_surface_z=float(chosen_pos[2]),
        task_id=int(handle.task_id),
        episode_index=int(episode_index),
        mode_seed=mode_seed,
        spawn_diversity=spawn_diversity,
    )


def _build_l3(
    handle: LiberoEnvHandle,
    cfg: RunConfig,
    *,
    disturbance: Any,
    episode_index: int,
    extra_libero: Mapping[str, Any],
    overlap: float,
    seed: int,
) -> SharedReachingController | None:
    l3 = _require(disturbance.l3, "disturbance.l3")
    l3_extra = _libero_extra(extra_libero, "l3", required=True)
    if not isinstance(l3_extra, Mapping):
        raise TypeError("extra.libero.l3 must be a mapping")
    reach_diversity = _l3_reach_diversity(extra_libero)

    hand = _require(cfg.hand, "cfg.hand")
    peak_map = _require(hand.peak_velocity, "hand.peak_velocity")
    arm_speed = str(_require(l3.arm_speed, "disturbance.l3.arm_speed"))
    if arm_speed not in peak_map:
        raise KeyError(f"hand.peak_velocity missing key {arm_speed!r}")
    peak_velocity = float(peak_map[arm_speed])

    eef_pos = handle.env.env.sim.data.site_xpos[handle.env.env.robots[0].eef_site_id].copy()

    start_mode = _require(l3.start_mode, "disturbance.l3.start_mode")
    start_z = l3_extra.get("start_z")
    target_mode = _l3_target_mode(l3_extra.get("target_mode", 1))
    target_pos: np.ndarray | None = None
    plan: dict[str, Any] = {
        "start_mode": str(start_mode),
        "target_mode": target_mode,
        "eef_pos": _as_xyz_list(eef_pos),
    }

    task_objs = get_task_object_positions(handle.env, eef_pos)
    task_name: str | None = None
    task_obj_pos: np.ndarray | None = None
    entity_subs = _l3_intercept_entity_substrings(l3_extra)
    dwell_xy_keep_eef_z = bool(l3_extra.get("dwell_at_entity_xy_keep_eef_z", False))
    if target_mode == "path_intercept":
        named_poses = collect_workspace_entity_poses(handle.env)
        if not named_poses:
            named_poses = {name: pos for name, pos in task_objs}
        if entity_subs:
            picked_sub = pick_workspace_entity_by_substrings(named_poses, entity_subs)
            if picked_sub is None:
                raise ValueError(
                    "extra.libero.l3.intercept_entity_substrings="
                    f"{list(entity_subs)!r} matched no workspace entity"
                )
            task_name, task_obj_pos = picked_sub
            plan["task_object"] = task_name
            plan["task_object_pos"] = _as_xyz_list(task_obj_pos)
            plan["task_object_source"] = "task_yaml_substring"
            plan["intercept_entity_substrings"] = list(entity_subs)
        else:
            goal = parse_first_on_in(libero_goal_source(handle.env))
            picked_task = select_l3_intercept_entity(
                instruction=handle.task_description,
                named_poses=named_poses,
                goal_subject=None if goal is None else goal[1],
                goal_dest=None if goal is None else goal[2],
                eef_pos=eef_pos,
            )
            if picked_task is not None:
                task_name, task_obj_pos, task_info = picked_task
                plan["task_object"] = task_name
                plan["task_object_pos"] = _as_xyz_list(task_obj_pos)
                plan["task_object_source"] = task_info.get("source")
                plan["task_object_alias"] = task_info.get("alias")
                plan["task_object_match_index"] = task_info.get("match_index")
                plan["goal_subject"] = task_info.get("goal_subject")
                plan["goal_dest"] = task_info.get("goal_dest")
    elif task_objs:
        picked_task = select_instruction_first_task_object(
            task_objs, handle.task_description, eef_pos
        )
        if picked_task is not None:
            task_name, task_obj_pos, task_info = picked_task
            plan["task_object"] = task_name
            plan["task_object_pos"] = _as_xyz_list(task_obj_pos)
            plan["task_object_source"] = task_info.get("source")
            plan["task_object_alias"] = task_info.get("alias")
            plan["task_object_match_index"] = task_info.get("match_index")

    if dwell_xy_keep_eef_z:
        if task_obj_pos is None:
            raise ValueError(
                "extra.libero.l3.dwell_at_entity_xy_keep_eef_z requires a "
                "resolved task object (set intercept_entity_substrings)"
            )
        target_pos = l3_dwell_at_xy_keep_z(task_obj_pos, float(eef_pos[2]))
        plan["dwell_at_entity_xy_keep_eef_z"] = True
        plan["xy_clearance"] = float(
            np.linalg.norm(target_pos[:2] - np.asarray(eef_pos[:2], dtype=np.float64))
        )
    elif target_mode == "path_intercept":
        if task_obj_pos is not None:
            min_clear = float(_libero_extra(extra_libero, "l3", "min_eef_clearance", required=True))
            tip = float(_libero_extra(extra_libero, "l3", "tip_overshoot", required=True))
            intercept_alpha = float(
                _libero_extra(extra_libero, "l3", "intercept_alpha", required=True)
            )
            min_task_clear = float(
                _libero_extra(extra_libero, "l3", "min_task_clearance", required=True)
            )
            target_pos = compute_l3_path_intercept_target(
                eef_pos,
                task_obj_pos,
                alpha=intercept_alpha,
                min_eef_clearance=min_clear,
                tip_overshoot=tip,
                min_task_clearance=min_task_clear,
                target_z=l3_extra.get("interp_z"),
            )
            plan["min_eef_clearance"] = min_clear
            plan["tip_overshoot"] = tip
            plan["intercept_alpha"] = intercept_alpha
            plan["min_task_clearance"] = min_task_clear
            plan["xy_clearance"] = float(
                np.linalg.norm(target_pos[:2] - np.asarray(eef_pos[:2], dtype=np.float64))
            )
    elif target_mode == "eef_task_interp":
        if task_obj_pos is not None:
            target_pos = compute_l3_interpolated_target(
                eef_pos,
                task_obj_pos,
                alpha=float(_require(l3.alpha, "disturbance.l3.alpha")),
                target_z=l3_extra.get("interp_z"),
            )
    else:
        picked = select_l3_shared_reach_target(handle.env, eef_pos)
        if picked is not None:
            picked_name, chosen_pos, _, info = picked
            target_pos = np.asarray(chosen_pos, dtype=np.float64)
            plan["picked_object"] = picked_name
            plan["picked_source"] = str((info or {}).get("source", ""))

    if target_pos is None:
        # L3 has no drop/arc obstacle; hide pedestals too or the empty
        # stand (box_small_base / box_base) stays on the table.
        hide_all_obstacles(handle, include_bases=True)
        hide_overlapping_bases(handle, overlap_radius=overlap)
        return None

    plan["target_pos"] = _as_xyz_list(target_pos)

    enter_height = float(_require(l3.enter_height, "disturbance.l3.enter_height"))
    dwell_height = float(_libero_extra(extra_libero, "l3", "dwell_height", required=True))
    withdraw_height = float(_libero_extra(extra_libero, "l3", "withdraw_height", required=True))
    plan["enter_height"] = enter_height
    plan["dwell_height"] = dwell_height
    plan["withdraw_height"] = withdraw_height

    start_xy_half = _require_xy(
        _libero_extra(extra_libero, "l3", "start_xy_half", required=True),
        "extra.libero.l3.start_xy_half",
    )
    skip_offscreen_exit = bool(
        _libero_extra(extra_libero, "l3", "skip_offscreen_exit", required=True)
    )
    plan["skip_offscreen_exit"] = skip_offscreen_exit
    plan["start_xy_half"] = list(start_xy_half)

    start_override = None
    if int(start_mode) == 2:
        robot_base = get_robot_base_pos(handle.env)
        table_center = get_table_center_pos(handle.env)
        sym_xy = 2.0 * table_center[:2] - robot_base[:2]
        hx, hy = start_xy_half
        clipped = np.array(
            [
                float(np.clip(sym_xy[0], table_center[0] - hx, table_center[0] + hx)),
                float(np.clip(sym_xy[1], table_center[1] - hy, table_center[1] + hy)),
            ],
            dtype=np.float64,
        )
        if start_z is not None:
            sym_z = float(start_z)
        else:
            # Same height as the parked gripper; approach then descends to alpha.
            sym_z = float(eef_pos[2])
        start_override = np.array([clipped[0], clipped[1], sym_z], dtype=np.float64)
        plan["start_pos"] = _as_xyz_list(start_override)
        plan["start_pos_unclipped"] = [float(sym_xy[0]), float(sym_xy[1]), sym_z]

    hide_all_obstacles(handle, include_bases=True)
    hide_overlapping_bases(handle, overlap_radius=overlap)
    geometry = str(_require(hand.geometry, "hand.geometry"))
    configure_hand_mesh_visibility(handle, geometry=geometry)
    apply_hand_collision_box(
        handle,
        half_size=_optional_xyz(hand.half_size, "hand.half_size"),
        collision_offset=_optional_xyz(hand.collision_offset, "hand.collision_offset"),
    )

    mode_seed = int(_require(cfg.seed, "cfg.seed"))
    hand_half = _optional_xyz(hand.half_size, "hand.half_size")
    hand_off = _optional_xyz(hand.collision_offset, "hand.collision_offset")
    controller = SharedReachingController(
        handle,
        alpha=float(_require(l3.alpha, "disturbance.l3.alpha")),
        dwell_steps=int(_require(l3.dwell_steps, "disturbance.l3.dwell_steps")),
        entry_steps=int(_require(l3.entry_steps, "disturbance.l3.entry_steps")),
        withdraw_speed_mult=float(
            _require(l3.withdraw_speed_mult, "disturbance.l3.withdraw_speed_mult")
        ),
        arm_speed=arm_speed,
        peak_velocity=peak_velocity,
        start_mode=start_mode,
        start_offset=list(_require(l3.start_offset, "disturbance.l3.start_offset")),
        move_start_step=int(_require(l3.move_start_step, "disturbance.l3.move_start_step")),
        target_mode=target_mode,
        interp_z=l3_extra.get("interp_z"),
        start_z=start_z,
        seed=seed,
        target_pos=target_pos,
        hand_physics=_hand_physics(cfg),
        control_freq=_control_freq(handle),
        start_pos_override=start_override,
        enter_height=enter_height,
        dwell_height=dwell_height,
        withdraw_height=withdraw_height,
        skip_offscreen_exit=skip_offscreen_exit,
        reach_diversity=reach_diversity,
        episode_index=int(episode_index),
        task_id=int(handle.task_id),
        mode_seed=mode_seed,
        eef_pos=np.asarray(eef_pos, dtype=np.float64),
        task_obj_pos=None if task_obj_pos is None else np.asarray(task_obj_pos, dtype=np.float64),
        start_xy_half=start_xy_half,
        min_eef_clearance=float(
            _libero_extra(extra_libero, "l3", "min_eef_clearance", required=True)
        ),
        min_task_clearance=float(
            _libero_extra(extra_libero, "l3", "min_task_clearance", required=True)
        ),
        tip_overshoot=float(_libero_extra(extra_libero, "l3", "tip_overshoot", required=True)),
        intercept_alpha=float(_libero_extra(extra_libero, "l3", "intercept_alpha", required=True)),
        hand_half_size=hand_half,
        hand_collision_offset=hand_off,
    )
    if getattr(controller, "reach_plan", None):
        plan.update(controller.reach_plan)
        plan["start_pos"] = list(plan.get("start") or plan.get("start_pos") or [])
        if controller.reach_plan.get("dwell") is not None:
            plan["target_pos"] = list(controller.reach_plan["dwell"])
    controller.l3_plan = plan
    return controller
