"""Target pose / start-geometry helpers for SimplerEnv L1–L3.

All geometric hyperparameters are caller-injected (no business defaults).
Migrated from ``dynamic_SimplerEnv/dynisafety/wrapper.py`` geometry methods.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dynisafety.backends.simplerenv.actor import compute_obstacle_quat
from dynisafety.core.disturbance import interpolate_toward, perpendicular_2d

logger = logging.getLogger(__name__)

# Scene-geometry solver constants (not eval hyperparameters).
_OBSTACLE_CLEARANCE_Z = 0.005
_SUPPORT_SCAN_MAX_DELTA_Z = 0.25
_BLOCKING_HEIGHT_MARGIN_Z = 0.02
_OBSTACLE_XY_CLEARANCE = 0.01
_MAX_TARGET_XY_ADJUST_ITERS = 8
_MAX_TARGET_XY_SHIFT = 0.35
_TABLETOP_Z_TOL = 0.01
# Pixel slack so the palm sits just off-screen, not clipped on the exact edge.
_VIEW_ENTRY_MARGIN_PX = 12
# After the palm is just off-screen, step this far back toward the corner.
_VIEW_ENTRY_EXTRA_OUTWARD_M = 0.05
_VIEW_ENTRY_SEARCH_ITERS = 24


@dataclass
class SceneAccess:
    """Minimal env surface needed by target geometry (avoids Gym wrapper coupling)."""

    get_eef_pos: Callable[[], np.ndarray]
    get_scene: Callable[[], Any]
    get_robot: Callable[[], Any]
    get_inner_env: Callable[[], Any]
    get_dynamic_actor: Callable[[], Any | None]
    table_height: float
    tabletop_xy_bounds_cache: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)


def get_actor_world_aabb(actor: Any) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        from mani_skill2_real2sim.utils.geometry import get_axis_aligned_bbox_for_actor

        mins, maxs = get_axis_aligned_bbox_for_actor(actor)
    except Exception as exc:
        logger.debug("[DyniSafety] Failed to compute actor AABB: %s", exc)
        return None
    mins = np.asarray(mins, dtype=np.float64)
    maxs = np.asarray(maxs, dtype=np.float64)
    if not np.all(np.isfinite(mins)) or not np.all(np.isfinite(maxs)):
        return None
    return mins, maxs


def xy_aabb_overlaps(
    mins_a: np.ndarray,
    maxs_a: np.ndarray,
    mins_b: np.ndarray,
    maxs_b: np.ndarray,
) -> bool:
    return not (
        maxs_a[0] < mins_b[0]
        or mins_a[0] > maxs_b[0]
        or maxs_a[1] < mins_b[1]
        or mins_a[1] > maxs_b[1]
    )


def get_reference_object_actor(access: SceneAccess) -> Any | None:
    inner = access.get_inner_env()
    for attr in (
        "episode_source_obj",
        "obj",
        "target_object",
        "object",
        "cube",
        "apple",
        "target_obj",
        "episode_target_obj",
        "main_object",
    ):
        obj = getattr(inner, attr, None)
        if obj is None:
            continue
        try:
            pos = np.array(obj.get_pose().p, dtype=np.float64)
        except Exception:
            continue
        if not np.any(np.isnan(pos)):
            return obj

    skip_keywords = {
        "robot",
        "ground",
        "table",
        "wall",
        "camera",
        "light",
        "link",
        "base",
        "dynisafety",
    }
    try:
        for actor in access.get_scene().get_all_actors():
            name = actor.name.lower()
            if any(k in name for k in skip_keywords):
                continue
            pos = np.array(actor.get_pose().p, dtype=np.float64)
            th = float(access.table_height)
            if th - 0.05 < pos[2] < th + 0.30:
                return actor
    except Exception:
        pass
    return None


def get_tabletop_xy_bounds(
    access: SceneAccess,
) -> tuple[np.ndarray, np.ndarray] | None:
    if access.tabletop_xy_bounds_cache is not None:
        return access.tabletop_xy_bounds_cache

    inner = access.get_inner_env()
    arena = getattr(inner, "arena", None)
    if arena is None:
        return None

    table_height = float(getattr(inner, "scene_table_height", access.table_height))
    tabletop_points = []
    try:
        for shape in arena.get_collision_shapes():
            scaled_vertices = shape.geometry.vertices * shape.geometry.scale
            local_pose = shape.get_local_pose()
            mat = (arena.get_pose() * local_pose).to_transformation_matrix()
            world_vertices = scaled_vertices @ mat[:3, :3].T + mat[:3, 3]
            mask = np.abs(world_vertices[:, 2] - table_height) <= _TABLETOP_Z_TOL
            if np.any(mask):
                tabletop_points.append(world_vertices[mask][:, :2])
    except Exception as exc:
        logger.debug("[DyniSafety] Failed to estimate tabletop bounds: %s", exc)
        return None

    if not tabletop_points:
        return None

    tabletop_points = np.vstack(tabletop_points)
    mins = tabletop_points.min(axis=0)
    maxs = tabletop_points.max(axis=0)
    access.tabletop_xy_bounds_cache = (
        np.asarray(mins, dtype=np.float64),
        np.asarray(maxs, dtype=np.float64),
    )
    return access.tabletop_xy_bounds_cache


def compute_l2_outside_table_start_offset(
    access: SceneAccess,
    target_pos: np.ndarray,
    *,
    outside_table_dist: float,
) -> tuple[float, float, float] | None:
    outside_dist = float(outside_table_dist)
    if outside_dist <= 0.0:
        return None

    bounds = get_tabletop_xy_bounds(access)
    if bounds is None:
        logger.warning(
            "[DyniSafety] Could not infer tabletop bounds; falling back to legacy L2 start logic."
        )
        return None

    table_mins, table_maxs = bounds
    target_xy = np.array(target_pos[:2], dtype=np.float64)
    table_center = 0.5 * (table_mins + table_maxs)
    direction = target_xy - table_center
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        eef_xy = access.get_eef_pos()[:2]
        direction = target_xy - eef_xy
        norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        direction = np.array([1.0, 0.0], dtype=np.float64)
        norm = 1.0
    direction /= norm

    edge_ts = []
    if direction[0] > 1e-8:
        edge_ts.append((table_maxs[0] - target_xy[0]) / direction[0])
    elif direction[0] < -1e-8:
        edge_ts.append((table_mins[0] - target_xy[0]) / direction[0])
    if direction[1] > 1e-8:
        edge_ts.append((table_maxs[1] - target_xy[1]) / direction[1])
    elif direction[1] < -1e-8:
        edge_ts.append((table_mins[1] - target_xy[1]) / direction[1])

    edge_ts = [float(t) for t in edge_ts if np.isfinite(t) and t >= 0.0]
    t_edge = min(edge_ts) if edge_ts else 0.0
    start_xy = target_xy + direction * (t_edge + outside_dist)
    return (
        float(start_xy[0] - target_xy[0]),
        float(start_xy[1] - target_xy[1]),
        0.0,
    )


def compute_l2_start_offset(
    access: SceneAccess,
    target_pos: np.ndarray,
    *,
    mirror_start: bool,
    start_dist_mult: float,
    start_offset: Sequence[float],
    align_start_y_to_eef: bool,
    outside_table_dist: float,
) -> tuple[float, float, float]:
    outside = compute_l2_outside_table_start_offset(
        access, target_pos, outside_table_dist=outside_table_dist
    )
    if outside is not None:
        return outside

    if mirror_start:
        eef_pos = access.get_eef_pos()
        dir_to_eef = eef_pos - target_pos
        dir_to_eef[2] = 0.0
        eef_dist = float(np.linalg.norm(dir_to_eef))
        if eef_dist > 1e-8:
            unit_dir = dir_to_eef / eef_dist
        else:
            unit_dir = np.array([1.0, 0.0, 0.0])
        perp_dir = perpendicular_2d(unit_dir)
        offset = np.array(perp_dir * eef_dist * float(start_dist_mult), dtype=np.float64)
        if align_start_y_to_eef:
            offset[1] = float(eef_pos[1] - target_pos[1])
        return float(offset[0]), float(offset[1]), float(offset[2])

    offset = np.asarray(start_offset, dtype=np.float64).copy()
    if offset.shape != (3,):
        raise ValueError(f"start_offset must have length 3, got {start_offset!r}")
    if align_start_y_to_eef:
        eef_pos = access.get_eef_pos()
        offset[1] = float(eef_pos[1] - target_pos[1])
    return float(offset[0]), float(offset[1]), float(offset[2])


def _get_robot_base_xy(access: SceneAccess) -> np.ndarray | None:
    robot = access.get_robot()
    try:
        return np.asarray(robot.pose.p[:2], dtype=np.float64)
    except Exception:
        try:
            return np.asarray(robot.get_pose().p[:2], dtype=np.float64)
        except Exception:
            return None


def project_world_to_cv(
    xyz: np.ndarray,
    K: np.ndarray,
    T_cw: np.ndarray,
) -> tuple[float, float, float] | None:
    """OpenCV pixel ``(u, v)`` and camera-Z, or ``None`` if behind the camera."""
    point = np.ones(4, dtype=np.float64)
    point[:3] = np.asarray(xyz, dtype=np.float64).reshape(3)
    cam = np.asarray(T_cw, dtype=np.float64).reshape(4, 4) @ point
    if float(cam[2]) <= 1e-6:
        return None
    kin = np.asarray(K, dtype=np.float64).reshape(3, 3)
    u = float(kin[0, 0] * cam[0] / cam[2] + kin[0, 2])
    v = float(kin[1, 1] * cam[1] / cam[2] + kin[1, 2])
    return u, v, float(cam[2])


def _palm_view_samples(
    xy: np.ndarray,
    z: float,
    inward_dir: np.ndarray,
    palm_radius_m: float,
) -> list[np.ndarray]:
    radius = max(float(palm_radius_m), 1e-4)
    inward = np.asarray(inward_dir, dtype=np.float64).reshape(2)
    lateral = np.array([-inward[1], inward[0]], dtype=np.float64)
    origin = np.array([float(xy[0]), float(xy[1]), float(z)], dtype=np.float64)
    return [
        origin,
        origin + np.array([inward[0], inward[1], 0.0], dtype=np.float64) * radius,
        origin + np.array([lateral[0], lateral[1], 0.0], dtype=np.float64) * radius,
        origin - np.array([lateral[0], lateral[1], 0.0], dtype=np.float64) * radius,
        origin + np.array([0.0, 0.0, radius], dtype=np.float64),
        origin - np.array([0.0, 0.0, radius], dtype=np.float64),
    ]


def _palm_outside_view(
    samples: Sequence[np.ndarray],
    *,
    K: np.ndarray,
    T_cw: np.ndarray,
    width: int,
    height: int,
    margin_px: float,
) -> bool:
    left = float(margin_px)
    top = float(margin_px)
    right = float(width) - float(margin_px)
    bottom = float(height) - float(margin_px)
    for sample in samples:
        hit = project_world_to_cv(sample, K, T_cw)
        if hit is None:
            continue
        u, v, _ = hit
        if left <= u < right and top <= v < bottom:
            return False
    return True


def start_inward_just_outside_view(
    *,
    anchor_xy: np.ndarray,
    inward_dir: np.ndarray,
    start_z: float,
    max_inward_m: float,
    palm_radius_m: float,
    K: np.ndarray,
    T_cw: np.ndarray,
    width: int,
    height: int,
    margin_px: float = _VIEW_ENTRY_MARGIN_PX,
) -> float:
    """Largest inward offset in ``[0, max]`` whose palm samples stay off-screen."""
    cap = max(float(max_inward_m), 0.0)
    anchor = np.asarray(anchor_xy, dtype=np.float64).reshape(2)
    inward = np.asarray(inward_dir, dtype=np.float64).reshape(2)

    def outside(shift: float) -> bool:
        xy = anchor + inward * float(shift)
        return _palm_outside_view(
            _palm_view_samples(xy, start_z, inward, palm_radius_m),
            K=K,
            T_cw=T_cw,
            width=width,
            height=height,
            margin_px=margin_px,
        )

    if cap <= 1e-9 or not outside(0.0):
        return 0.0
    if outside(cap):
        return cap
    lo, hi = 0.0, cap
    for _ in range(_VIEW_ENTRY_SEARCH_ITERS):
        mid = 0.5 * (lo + hi)
        if outside(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _read_eval_camera_view(
    access: SceneAccess,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    try:
        inner = access.get_inner_env()
    except Exception:
        return None
    if inner is None:
        return None
    try:
        from dynisafety.backends.simplerenv.env import resolve_primary_camera_name

        name = resolve_primary_camera_name(inner, None)
    except Exception:
        name = None
    try:
        params = inner.get_camera_params()
    except Exception:
        return None
    if not isinstance(params, Mapping) or not params:
        return None
    if name is None or name not in params:
        name = next(iter(params))
    cam = params.get(name)
    if not isinstance(cam, Mapping):
        return None
    try:
        K = np.asarray(cam["intrinsic_cv"], dtype=np.float64)
        T_cw = np.asarray(cam["extrinsic_cv"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    if K.shape != (3, 3) or T_cw.shape != (4, 4):
        return None
    width = 0
    height = 0
    cameras = getattr(inner, "_cameras", None)
    wrapper = cameras.get(name) if isinstance(cameras, Mapping) else None
    sapien_cam = getattr(wrapper, "camera", wrapper)
    try:
        width = int(getattr(sapien_cam, "width", 0) or 0)
        height = int(getattr(sapien_cam, "height", 0) or 0)
    except (TypeError, ValueError):
        width, height = 0, 0
    if width <= 0 or height <= 0:
        return None
    return K, T_cw, width, height


def compute_l3_legacy_start_offset(
    access: SceneAccess,
    target_pos: np.ndarray,
    *,
    mirror_start: bool,
    start_dist_mult: float,
    start_offset: Sequence[float],
    flip_side: bool,
) -> tuple[float, float, float]:
    side_sign = -1.0 if flip_side else 1.0
    if mirror_start:
        eef_pos = access.get_eef_pos()
        dir_to_eef = eef_pos - target_pos
        dir_to_eef[2] = 0.0
        eef_dist = float(np.linalg.norm(dir_to_eef))
        if eef_dist > 1e-8:
            unit_dir = dir_to_eef / eef_dist
        else:
            unit_dir = np.array([1.0, 0.0, 0.0])
        perp_dir = perpendicular_2d(unit_dir, flip=flip_side)
        offset = perp_dir * eef_dist * float(start_dist_mult)
        return float(offset[0]), float(offset[1]), float(offset[2])

    arr = np.asarray(start_offset, dtype=np.float64).copy()
    if arr.shape != (3,):
        raise ValueError(f"start_offset must have length 3, got {start_offset!r}")
    arr[:2] *= side_sign
    return float(arr[0]), float(arr[1]), float(arr[2])


def compute_l3_robot_base_corner_start_geometry(
    access: SceneAccess,
    target_pos: np.ndarray,
    *,
    flip_side: bool,
    corner_inward_shift: float,
    corner_entry_backoff: float,
    hand_scale: float,
    hand_half_size_x: float,
    hand_collision_offset_x: float,
    enter_height: float,
) -> tuple[tuple[float, float, float], np.ndarray | None] | None:
    """Return corner geometry, or ``None`` if table/base pose cannot be resolved.

    Callers must fall back to :func:`compute_l3_legacy_start_offset` with YAML params.
    """
    bounds = get_tabletop_xy_bounds(access)
    if bounds is None:
        return None

    robot_base_xy = _get_robot_base_xy(access)
    if robot_base_xy is None:
        return None

    table_mins, table_maxs = bounds
    corner_x = (
        table_mins[0]
        if abs(robot_base_xy[0] - table_mins[0]) <= abs(robot_base_xy[0] - table_maxs[0])
        else table_maxs[0]
    )
    near_y = (
        table_mins[1]
        if abs(robot_base_xy[1] - table_mins[1]) <= abs(robot_base_xy[1] - table_maxs[1])
        else table_maxs[1]
    )
    far_y = table_maxs[1] if np.isclose(near_y, table_mins[1]) else table_mins[1]
    corner_y = far_y if flip_side else near_y

    corner_xy = np.array([corner_x, corner_y], dtype=np.float64)
    inward_dir = np.array(target_pos[:2], dtype=np.float64) - corner_xy
    inward_norm = float(np.linalg.norm(inward_dir))
    if inward_norm <= 1e-8:
        center_xy = 0.5 * (table_mins + table_maxs)
        inward_dir = center_xy - corner_xy
        inward_norm = float(np.linalg.norm(inward_dir))
    if inward_norm <= 1e-8:
        inward_dir = np.array([-1.0, 0.0], dtype=np.float64)
        inward_norm = 1.0
    inward_dir /= inward_norm

    hand_front_anchor_depth = max(float(hand_half_size_x + hand_collision_offset_x), 0.0) * float(
        hand_scale
    )
    anchor_xy = corner_xy - inward_dir * hand_front_anchor_depth
    inward_m = float(corner_inward_shift)
    camera = _read_eval_camera_view(access)
    if camera is not None:
        K, T_cw, width, height = camera
        palm_r = max(0.25 * float(hand_half_size_x), abs(float(hand_collision_offset_x))) * float(
            hand_scale
        )
        inward_m = start_inward_just_outside_view(
            anchor_xy=anchor_xy,
            inward_dir=inward_dir,
            start_z=float(target_pos[2]) + float(enter_height),
            max_inward_m=inward_m,
            palm_radius_m=palm_r,
            K=K,
            T_cw=T_cw,
            width=width,
            height=height,
        )
        inward_m = max(0.0, inward_m - _VIEW_ENTRY_EXTRA_OUTWARD_M)
    start_xy = anchor_xy + inward_dir * inward_m
    entry_far_xy = start_xy - inward_dir * float(corner_entry_backoff)
    start_offset = (
        float(start_xy[0] - target_pos[0]),
        float(start_xy[1] - target_pos[1]),
        0.0,
    )
    entry_far_pos = np.array([entry_far_xy[0], entry_far_xy[1], target_pos[2]], dtype=np.float64)
    return start_offset, entry_far_pos


def compute_l3_start_geometry(
    access: SceneAccess,
    target_pos: np.ndarray,
    *,
    start_mode: str,
    mirror_start: bool,
    start_dist_mult: float,
    start_offset: Sequence[float],
    flip_side: bool,
    corner_inward_shift: float,
    corner_entry_backoff: float,
    enter_height: float,
    hand_scale: float,
    hand_half_size_x: float,
    hand_collision_offset_x: float,
) -> tuple[tuple[float, float, float], np.ndarray | None]:
    if start_mode == "robot_base_corner":
        corner = compute_l3_robot_base_corner_start_geometry(
            access,
            target_pos,
            flip_side=flip_side,
            corner_inward_shift=corner_inward_shift,
            corner_entry_backoff=corner_entry_backoff,
            hand_scale=hand_scale,
            hand_half_size_x=hand_half_size_x,
            hand_collision_offset_x=hand_collision_offset_x,
            enter_height=enter_height,
        )
        if corner is not None:
            return corner
        logger.warning("[L3 robot-base-corner] Falling back to legacy L3 start logic.")
        return (
            compute_l3_legacy_start_offset(
                access,
                target_pos,
                mirror_start=mirror_start,
                start_dist_mult=start_dist_mult,
                start_offset=start_offset,
                flip_side=flip_side,
            ),
            None,
        )
    if start_mode != "legacy":
        raise ValueError(
            f"l3 start_mode must be 'legacy' or 'robot_base_corner', got {start_mode!r}"
        )
    return (
        compute_l3_legacy_start_offset(
            access,
            target_pos,
            mirror_start=mirror_start,
            start_dist_mult=start_dist_mult,
            start_offset=start_offset,
            flip_side=flip_side,
        ),
        None,
    )


def compute_target_pos(
    access: SceneAccess,
    reset_options: Mapping[str, Any],
    *,
    override: Sequence[float] | None,
    table_height: float,
    level: int,
    alpha: float,
) -> np.ndarray:
    """Compute obstacle/hand reference target for this episode."""
    if override is not None:
        arr = np.asarray(override, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"target_pos override must have length 3, got {override!r}")
        return arr

    obj_pos = None
    obj_opts = reset_options.get("obj_init_options", {}) or {}
    if "init_xy" in obj_opts:
        obj_xy = np.array(obj_opts["init_xy"], dtype=np.float64)
        obj_pos = np.array([obj_xy[0], obj_xy[1], float(table_height) + 0.10], dtype=np.float64)

    if obj_pos is None:
        actor = get_reference_object_actor(access)
        if actor is not None:
            try:
                obj_pos = np.array(actor.get_pose().p, dtype=np.float64)
            except Exception:
                obj_pos = None

    if obj_pos is None:
        raise ValueError(
            "Cannot determine object position; set task/scene target_pos override "
            "or ensure the ManiSkill2 env exposes a graspable actor"
        )

    obj_pos = np.asarray(obj_pos, dtype=np.float64)
    eef_pos = access.get_eef_pos()
    target = interpolate_toward(eef_pos, obj_pos, alpha)
    target[2] = obj_pos[2]
    if level not in (1, 2, 3):
        raise ValueError(f"level must be 1–3 for target computation, got {level}")
    return target


def compute_obstacle_spawn_quat(
    access: SceneAccess,
    *,
    level: int,
    target_pos: np.ndarray,
    start_offset: Sequence[float] | None,
    mirror_start: bool,
    start_dist_mult: float,
    default_start_offset: Sequence[float],
    align_start_y_to_eef: bool,
    outside_table_dist: float,
) -> np.ndarray:
    if level == 1:
        return compute_obstacle_quat(0.0)
    if level != 2:
        raise ValueError(f"Obstacle spawn quaternion only applies to levels 1/2, got {level}")
    if start_offset is None:
        start_offset = compute_l2_start_offset(
            access,
            target_pos,
            mirror_start=mirror_start,
            start_dist_mult=start_dist_mult,
            start_offset=default_start_offset,
            align_start_y_to_eef=align_start_y_to_eef,
            outside_table_dist=outside_table_dist,
        )
    start_pos = np.array(target_pos, dtype=np.float64) + np.array(start_offset, dtype=np.float64)
    push_dir = np.array(target_pos, dtype=np.float64) - start_pos
    push_dir[2] = 0.0
    norm = float(np.linalg.norm(push_dir))
    yaw = float(np.arctan2(push_dir[1], push_dir[0])) if norm > 1e-8 else 0.0
    return compute_obstacle_quat(yaw)


def align_obstacle_target_height(
    access: SceneAccess,
    target_pos: np.ndarray,
    *,
    level: int,
    mirror_start: bool,
    start_dist_mult: float,
    start_offset: Sequence[float],
    align_start_y_to_eef: bool,
    outside_table_dist: float,
) -> tuple[np.ndarray, tuple[float, float, float] | None]:
    """Refine L1/L2 target z using obstacle/support collision geometry."""
    adjusted = np.array(target_pos, dtype=np.float64)
    if level not in (1, 2):
        return adjusted, None

    l2_off = (
        compute_l2_start_offset(
            access,
            adjusted,
            mirror_start=mirror_start,
            start_dist_mult=start_dist_mult,
            start_offset=start_offset,
            align_start_y_to_eef=align_start_y_to_eef,
            outside_table_dist=outside_table_dist,
        )
        if level == 2
        else None
    )
    spawn_quat = compute_obstacle_spawn_quat(
        access,
        level=level,
        target_pos=adjusted,
        start_offset=l2_off,
        mirror_start=mirror_start,
        start_dist_mult=start_dist_mult,
        default_start_offset=start_offset,
        align_start_y_to_eef=align_start_y_to_eef,
        outside_table_dist=outside_table_dist,
    )
    adjusted, _ = _resolve_obstacle_target_xy_overlap(access, adjusted, spawn_quat)
    l2_off = (
        compute_l2_start_offset(
            access,
            adjusted,
            mirror_start=mirror_start,
            start_dist_mult=start_dist_mult,
            start_offset=start_offset,
            align_start_y_to_eef=align_start_y_to_eef,
            outside_table_dist=outside_table_dist,
        )
        if level == 2
        else None
    )
    spawn_quat = compute_obstacle_spawn_quat(
        access,
        level=level,
        target_pos=adjusted,
        start_offset=l2_off,
        mirror_start=mirror_start,
        start_dist_mult=start_dist_mult,
        default_start_offset=start_offset,
        align_start_y_to_eef=align_start_y_to_eef,
        outside_table_dist=outside_table_dist,
    )
    obstacle_aabb = _probe_obstacle_aabb(access, adjusted[:2], spawn_quat)
    support_z = _get_support_surface_z(access, adjusted, obstacle_aabb=obstacle_aabb)

    safe_z = support_z + _OBSTACLE_CLEARANCE_Z
    if obstacle_aabb is not None:
        obstacle_min_z_at_zero = float(obstacle_aabb[0][2])
        safe_z = support_z + _OBSTACLE_CLEARANCE_Z - obstacle_min_z_at_zero
    adjusted[2] = safe_z
    return adjusted, l2_off


def _probe_obstacle_aabb(
    access: SceneAccess,
    target_xy: np.ndarray,
    spawn_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    import sapien.core as sapien

    actor = access.get_dynamic_actor()
    if actor is None:
        return None
    probe_pos = np.array([target_xy[0], target_xy[1], 0.0], dtype=np.float64)
    prev_pose = actor.get_pose()
    try:
        actor.set_pose(sapien.Pose(probe_pos, spawn_quat))
        return get_actor_world_aabb(actor)
    finally:
        actor.set_pose(prev_pose)


def _get_reference_actor_bottom_z(access: SceneAccess) -> float | None:
    actor = get_reference_object_actor(access)
    if actor is None:
        return None
    aabb = get_actor_world_aabb(actor)
    if aabb is None:
        return None
    return float(aabb[0][2])


def _get_support_surface_z(
    access: SceneAccess,
    reference_pos: np.ndarray,
    *,
    obstacle_aabb: tuple[np.ndarray, np.ndarray] | None,
) -> float:
    actor = get_reference_object_actor(access)
    reference_actor_bottom_z = _get_reference_actor_bottom_z(access)

    if obstacle_aabb is not None:
        xy_mins = obstacle_aabb[0][:2]
        xy_maxs = obstacle_aabb[1][:2]
        support_cap = (
            max(float(reference_pos[2]), float(access.table_height)) + _SUPPORT_SCAN_MAX_DELTA_Z
        )
        overlap_support_z = None
        dynamic = access.get_dynamic_actor()
        for scene_actor in access.get_scene().get_all_actors():
            if scene_actor is dynamic:
                continue
            name = scene_actor.name.lower()
            if any(k in name for k in ("robot", "camera", "light", "dynisafety", "goal_site")):
                continue
            scene_aabb = get_actor_world_aabb(scene_actor)
            if scene_aabb is None:
                continue
            if not xy_aabb_overlaps(scene_aabb[0][:2], scene_aabb[1][:2], xy_mins, xy_maxs):
                continue
            candidate_top_z = float(scene_aabb[1][2])
            if candidate_top_z > support_cap:
                continue
            if overlap_support_z is None or candidate_top_z > overlap_support_z:
                overlap_support_z = candidate_top_z
        if overlap_support_z is not None:
            return float(overlap_support_z)

    if reference_actor_bottom_z is not None:
        return float(reference_actor_bottom_z)

    if actor is not None:
        try:
            return float(actor.get_pose().p[2])
        except Exception:
            pass

    fallback_z = float(access.table_height)
    if reference_pos is not None and reference_pos.shape[0] >= 3 and np.isfinite(reference_pos[2]):
        fallback_z = max(fallback_z, float(reference_pos[2]))
    logger.warning(
        "[DyniSafety] Cannot determine support surface height; using fallback z=%.3f",
        fallback_z,
    )
    return fallback_z


def _get_obstacle_xy_blockers(
    access: SceneAccess,
    obstacle_aabb: tuple[np.ndarray, np.ndarray],
    reference_actor_bottom_z: float | None,
) -> list[tuple[Any, tuple[np.ndarray, np.ndarray]]]:
    blockers = []
    blocker_floor = (
        reference_actor_bottom_z + _BLOCKING_HEIGHT_MARGIN_Z
        if reference_actor_bottom_z is not None
        else float(access.table_height) + _BLOCKING_HEIGHT_MARGIN_Z
    )
    xy_mins = obstacle_aabb[0][:2]
    xy_maxs = obstacle_aabb[1][:2]
    dynamic = access.get_dynamic_actor()
    for scene_actor in access.get_scene().get_all_actors():
        if scene_actor is dynamic:
            continue
        name = scene_actor.name.lower()
        if any(
            k in name
            for k in ("robot", "camera", "light", "dynisafety", "goal_site", "arena", "ground")
        ):
            continue
        scene_aabb = get_actor_world_aabb(scene_actor)
        if scene_aabb is None:
            continue
        if not xy_aabb_overlaps(scene_aabb[0][:2], scene_aabb[1][:2], xy_mins, xy_maxs):
            continue
        if float(scene_aabb[1][2]) <= blocker_floor:
            continue
        blockers.append((scene_actor, scene_aabb))
    return blockers


def _compute_xy_separation_shift(
    obstacle_aabb: tuple[np.ndarray, np.ndarray],
    blocker_aabb: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    obs_mins, obs_maxs = obstacle_aabb
    blk_mins, blk_maxs = blocker_aabb
    obs_center = 0.5 * (obs_mins[:2] + obs_maxs[:2])
    blk_center = 0.5 * (blk_mins[:2] + blk_maxs[:2])

    if obs_center[0] <= blk_center[0]:
        dx = blk_mins[0] - obs_maxs[0] - _OBSTACLE_XY_CLEARANCE
    else:
        dx = blk_maxs[0] - obs_mins[0] + _OBSTACLE_XY_CLEARANCE

    if obs_center[1] <= blk_center[1]:
        dy = blk_mins[1] - obs_maxs[1] - _OBSTACLE_XY_CLEARANCE
    else:
        dy = blk_maxs[1] - obs_mins[1] + _OBSTACLE_XY_CLEARANCE

    if abs(dx) < abs(dy):
        return np.array([dx, 0.0], dtype=np.float64)
    return np.array([0.0, dy], dtype=np.float64)


def _resolve_obstacle_target_xy_overlap(
    access: SceneAccess,
    target_pos: np.ndarray,
    spawn_quat: np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray] | None]:
    adjusted = np.array(target_pos, dtype=np.float64)
    total_shift = np.zeros(2, dtype=np.float64)
    reference_actor_bottom_z = _get_reference_actor_bottom_z(access)
    obstacle_aabb = _probe_obstacle_aabb(access, adjusted[:2], spawn_quat)
    for _ in range(_MAX_TARGET_XY_ADJUST_ITERS):
        if obstacle_aabb is None:
            break
        blockers = _get_obstacle_xy_blockers(access, obstacle_aabb, reference_actor_bottom_z)
        if not blockers:
            break
        _blocker_actor, blocker_aabb = max(blockers, key=lambda item: float(item[1][1][2]))
        shift_xy = _compute_xy_separation_shift(obstacle_aabb, blocker_aabb)
        if np.linalg.norm(shift_xy) < 1e-8:
            break
        adjusted[:2] += shift_xy
        total_shift += shift_xy
        obstacle_aabb = _probe_obstacle_aabb(access, adjusted[:2], spawn_quat)
        if np.linalg.norm(total_shift) > _MAX_TARGET_XY_SHIFT:
            break
    return adjusted, obstacle_aabb
