"""RoboCasa L1/L2/L3 geometry helpers (locked formulas; numbers come from YAML).

参考系 mobilebase0_base；z=floor。禁止追击当前底座，禁止台面高度。
L1：落点在 start→dest 上、dest 前。出现点=终点。
    p_spawn = p_dest - (goal_clearance + k*retry_step) * u
    触发 ||base.xy - p_spawn.xy|| <= trigger_dist。
L2：同一路径落点；沿 n 的有界区间横穿来回。开局停在一端。
    p_land = p_dest - (goal_clearance + k*retry_step) * u
    p_plus = p_land + span_plus * n ; p_minus = p_land - span_minus * n
    n = rot90(u)。区间不含 dest 成功盘。
    触发 ||base.xy - p_land.xy|| <= trigger_dist 后再横穿。无视角触发。
L3：同一围裙。p_dwell = p_land；p_start = p_land + lateral * n。
    人走近、站住、离开（清场）。触发还需底座仍在 p_land 起点一侧：
    ||base.xy - p_land.xy|| <= trigger_dist AND (p_land - base) · u > 0。
z 不进这些 XY 公式；调用方用 floor_z + half_z 贴地。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dynisafety.backends.robocasa.env import (
    HIDDEN_OBSTACLE_BODY,
    read_base_xy_yaw,
)
from dynisafety.core.disturbance import Phase as Phase
from dynisafety.core.disturbance import lerp, perpendicular_2d
from dynisafety.core.disturbance import minimum_jerk as minimum_jerk

# MuJoCo free-joint quat is wxyz. Upright box; not a YAML tunable.
UPRIGHT_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
# Same underground stash Window C used for the L0 actor (not a spawn formula).
HIDDEN_POS = np.array([0.0, 0.0, -10.0], dtype=np.float64)
# Contact solver epsilon for "is penetrating", not a disturbance hyperparameter.
PENETRATION_EPS = 1e-4


def heading_from_yaw(yaw0: float) -> np.ndarray:
    """h = (cos yaw0, sin yaw0)."""
    yaw = float(yaw0)
    return np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)


def lateral_from_yaw(yaw0: float) -> np.ndarray:
    """n = (-sin yaw0, cos yaw0). Unused by L1; exported for L2/L3."""
    return perpendicular_2d(heading_from_yaw(yaw0))


def path_unit_and_length(
    p_start_xy: Sequence[float],
    p_dest_xy: Sequence[float],
) -> tuple[np.ndarray | None, float]:
    """Unit vector start→dest and path length. None unit when the segment degenerates."""
    p_start = np.asarray(p_start_xy, dtype=np.float64).reshape(2)
    p_dest = np.asarray(p_dest_xy, dtype=np.float64).reshape(2)
    delta = p_dest - p_start
    path_len = float(np.linalg.norm(delta))
    if path_len <= 1e-8:
        return None, path_len
    return delta / path_len, path_len


def l1_path_spawn_xy(
    p_start_xy: Sequence[float],
    p_dest_xy: Sequence[float],
    goal_clearance: float,
    retry_step: float,
    attempt: int,
    min_start_clearance: float,
) -> np.ndarray | None:
    """Candidate on the open start→dest segment, *goal_clearance* before dest.

    Attempt *attempt* (0-based) walks toward the start by *retry_step*.
    The point stays strictly before dest and at least *min_start_clearance*
    from the start (do not spawn on the chassis). Numbers come from the caller.
    """
    if int(attempt) < 0:
        raise ValueError(f"attempt must be >= 0, got {attempt!r}")
    unit, path_len = path_unit_and_length(p_start_xy, p_dest_xy)
    if unit is None:
        return None
    min_start = float(min_start_clearance)
    if min_start < 0.0:
        raise ValueError(f"min_start_clearance must be >= 0, got {min_start}")
    max_back = path_len - min_start
    if max_back <= 1e-8:
        return None
    back = float(goal_clearance) + int(attempt) * float(retry_step)
    if back <= 0.0:
        return None
    if back > max_back:
        if int(attempt) != 0:
            return None
        back = max_back
    p_dest = np.asarray(p_dest_xy, dtype=np.float64).reshape(2)
    return p_dest - back * unit


def path_left_normal(unit_xy: Sequence[float]) -> np.ndarray:
    """n = rot90 CCW of path unit u: (-uy, ux). Left when looking along start→dest."""
    u = np.asarray(unit_xy, dtype=np.float64).reshape(2)
    return np.array([-float(u[1]), float(u[0])], dtype=np.float64)


def l2_lateral_start_xy(
    p_land_xy: Sequence[float],
    unit_xy: Sequence[float],
    start_lateral: float,
    *,
    flip_side: bool,
) -> np.ndarray:
    """p_start_xy = p_land + start_lateral * n. n flipped if flip_side."""
    if float(start_lateral) < 0.0:
        raise ValueError(f"start_lateral must be >= 0, got {start_lateral!r}")
    n = path_left_normal(unit_xy)
    if flip_side:
        n = -n
    land = np.asarray(p_land_xy, dtype=np.float64).reshape(2)
    return land + float(start_lateral) * n


def l2_lerp_sample_xys(
    p_start_xy: Sequence[float],
    p_land_xy: Sequence[float],
    n_mid: int,
) -> list[np.ndarray]:
    """Interior XY samples on start→land (excludes endpoints)."""
    if int(n_mid) < 0:
        raise ValueError(f"n_mid must be >= 0, got {n_mid!r}")
    if int(n_mid) == 0:
        return []
    out: list[np.ndarray] = []
    for i in range(1, int(n_mid) + 1):
        alpha = float(i) / float(int(n_mid) + 1)
        out.append(lerp_xy(p_start_xy, p_land_xy, alpha))
    return out


def l1_should_appear(
    base_xy: Sequence[float],
    spawn_xy: Sequence[float],
    trigger_dist: float,
) -> bool:
    """True when chassis XY is within *trigger_dist* of the planned spawn."""
    radius = float(trigger_dist)
    if radius < 0.0:
        raise ValueError(f"trigger_dist must be >= 0, got {radius}")
    base = np.asarray(base_xy, dtype=np.float64).reshape(2)
    spawn = np.asarray(spawn_xy, dtype=np.float64).reshape(2)
    return float(np.linalg.norm(base - spawn)) <= radius


def base_before_land(
    base_xy: Sequence[float],
    land_xy: Sequence[float],
    path_unit_xy: Sequence[float],
) -> bool:
    """True when the chassis is still on the start side of *land* along *u*.

    ``(p_land - base) · u > 0`` means the apron is still ahead. After the
    base passes *p_land* toward dest the dot is ≤ 0 — L3 must not appear.
    """
    base = np.asarray(base_xy, dtype=np.float64).reshape(2)
    land = np.asarray(land_xy, dtype=np.float64).reshape(2)
    unit = np.asarray(path_unit_xy, dtype=np.float64).reshape(2)
    return float(np.dot(land - base, unit)) > 1e-8


def l3_should_appear(
    base_xy: Sequence[float],
    land_xy: Sequence[float],
    trigger_dist: float,
    path_unit_xy: Sequence[float],
) -> bool:
    """L3 onset: close enough to the apron and still before it along *u*."""
    return l1_should_appear(base_xy, land_xy, trigger_dist) and base_before_land(
        base_xy, land_xy, path_unit_xy
    )


def point_in_opengl_frustum(
    point_xyz: Sequence[float],
    cam_pos: Sequence[float],
    cam_mat: Sequence[float],
    *,
    fovy_deg: float,
    aspect: float,
) -> bool:
    """True if *point_xyz* is in front of an OpenGL camera (looks along −Z).

    *cam_mat* is the 3×3 world-from-camera rotation (MuJoCo ``cam_xmat``).
    """
    fov = float(fovy_deg)
    asp = float(aspect)
    if fov <= 0.0:
        raise ValueError(f"fovy_deg must be > 0, got {fov}")
    if asp <= 0.0:
        raise ValueError(f"aspect must be > 0, got {asp}")
    point = np.asarray(point_xyz, dtype=np.float64).reshape(3)
    origin = np.asarray(cam_pos, dtype=np.float64).reshape(3)
    mat = np.asarray(cam_mat, dtype=np.float64).reshape(3, 3)
    local = mat.T @ (point - origin)
    depth = -float(local[2])
    if depth <= 1e-8:
        return False
    half_y = math.tan(math.radians(fov) * 0.5)
    half_x = half_y * asp
    return abs(float(local[0])) <= half_x * depth + 1e-9 and abs(
        float(local[1])
    ) <= half_y * depth + 1e-9


def _camera_id(model: Any, name: str) -> int | None:
    if hasattr(model, "camera_name2id"):
        try:
            return int(model.camera_name2id(name))
        except Exception:
            pass
    names = getattr(model, "camera_names", None)
    if names is not None:
        try:
            return int(list(names).index(name))
        except ValueError:
            return None
    return None


def cameras_seeing_point(
    env: Any,
    point_xyz: Sequence[float],
    camera_names: Sequence[str],
    *,
    aspect: float,
) -> list[str]:
    """Names of MuJoCo cameras whose OpenGL frustum contains *point_xyz*."""
    model = env.sim.model
    data = env.sim.data
    seen: list[str] = []
    for raw in camera_names:
        name = str(raw).strip()
        if not name:
            continue
        cam_id = _camera_id(model, name)
        if cam_id is None:
            continue
        if point_in_opengl_frustum(
            point_xyz,
            data.cam_xpos[cam_id],
            data.cam_xmat[cam_id],
            fovy_deg=float(model.cam_fovy[cam_id]),
            aspect=float(aspect),
        ):
            seen.append(name)
    return seen


def l2_pingpong_alpha(elapsed: int, one_way_steps: int) -> float:
    """Triangle wave in [0, 1]: 0 at the start end, 1 at the far end."""
    steps = int(one_way_steps)
    if steps < 1:
        raise ValueError(f"one_way_steps must be >= 1, got {steps}")
    t = max(int(elapsed), 0) % (2 * steps)
    if t <= steps:
        return float(t) / float(steps)
    return 2.0 - float(t) / float(steps)


def l2_pingpong_xy(
    p_a_xy: Sequence[float],
    p_b_xy: Sequence[float],
    elapsed: int,
    one_way_steps: int,
) -> np.ndarray:
    """Lerp *p_a*→*p_b*→*p_a* using l2_pingpong_alpha."""
    return lerp_xy(p_a_xy, p_b_xy, l2_pingpong_alpha(elapsed, one_way_steps))


def l1_probe_failure_action(require_clear_ahead: bool) -> str:
    """After a failed frontal probe: wait for a later heading, or block now.

    ``wait`` = keep hidden and do not snapshot p0/yaw0 (onset has not happened).
    ``block`` = first motion snapshot failed; spawn_failed, never retry.
    """
    return "wait" if bool(require_clear_ahead) else "block"


def l1_spawn_failed_flag(
    *,
    dropped: bool,
    motion_armed: bool,
    blocked: bool,
) -> bool:
    """Episode-end spawn_failed: moved/blocked but never placed on free floor."""
    if dropped:
        return False
    return bool(blocked or motion_armed)


def body_z_on_floor(floor_z: float, half_size_z: float) -> float:
    """Box origin sits on the floor: floor_z + half_z (bottom flush, not buried)."""
    return float(floor_z) + float(half_size_z)


def l12_meta(env: Any) -> Mapping[str, Any]:
    meta = getattr(env, "_dynisafety_l12", None)
    if not isinstance(meta, Mapping):
        raise RuntimeError(f"env missing _dynisafety_l12 merge metadata for {HIDDEN_OBSTACLE_BODY}")
    return meta


def read_floor_z(env: Any) -> float:
    """Floor plane Z from Floor fixtures or floor-named geoms. Never invent 0."""
    zs: list[float] = []
    fixtures = getattr(env, "fixtures", None) or {}
    for _name, fxtr in fixtures.items():
        if type(fxtr).__name__ != "Floor":
            continue
        pos = getattr(fxtr, "pos", None)
        if pos is None:
            continue
        zs.append(float(np.asarray(pos, dtype=np.float64).reshape(-1)[2]))
    if zs:
        return float(max(zs))

    model = env.sim.model
    for gid in range(int(model.ngeom)):
        gname = model.geom_id2name(gid)
        if gname is None:
            continue
        if "floor" not in str(gname).lower():
            continue
        zs.append(float(env.sim.data.geom_xpos[gid][2]))
    if zs:
        return float(max(zs))
    raise RuntimeError("no Floor fixture or floor geom; cannot invent floor_z (FIX：窗口 D)")


def layout_has_island(env: Any) -> bool:
    fixtures = getattr(env, "fixtures", None) or {}
    for name in fixtures:
        if "island" in str(name).lower():
            return True
    return False


def _raw_model(sim: Any) -> Any:
    model = sim.model
    return getattr(model, "_model", model)


def _raw_data(sim: Any) -> Any:
    data = sim.data
    return getattr(data, "_data", data)


def _joint_dof_addr(env: Any, joint_name: str) -> int:
    model = env.sim.model
    jid = int(model.joint_name2id(joint_name))
    return int(model.jnt_dofadr[jid])


def set_free_joint_pose(
    env: Any,
    joint_name: str,
    pos: Sequence[float],
    quat_wxyz: Sequence[float] | None = None,
) -> None:
    import mujoco

    quat = (
        UPRIGHT_QUAT_WXYZ
        if quat_wxyz is None
        else np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    )
    qpos = np.concatenate([np.asarray(pos, dtype=np.float64).reshape(3), quat])
    sim = env.sim
    sim.data.set_joint_qpos(joint_name, qpos)
    addr = _joint_dof_addr(env, joint_name)
    sim.data.qvel[addr : addr + 6] = 0.0
    mujoco.mj_forward(_raw_model(sim), _raw_data(sim))


def teleport_obstacle_hidden(env: Any) -> None:
    meta = l12_meta(env)
    set_free_joint_pose(env, str(meta["joint_name"]), HIDDEN_POS, UPRIGHT_QUAT_WXYZ)


def read_obstacle_xy(env: Any) -> np.ndarray:
    meta = l12_meta(env)
    body_id = int(env.sim.model.body_name2id(str(meta["body_name"])))
    return np.asarray(env.sim.data.body_xpos[body_id][:2], dtype=np.float64).copy()


def read_obstacle_pos(env: Any) -> np.ndarray:
    meta = l12_meta(env)
    body_id = int(env.sim.model.body_name2id(str(meta["body_name"])))
    return np.asarray(env.sim.data.body_xpos[body_id], dtype=np.float64).copy()


def _is_floor_geom(env: Any, geom_id: int) -> bool:
    model = env.sim.model
    gname = model.geom_id2name(int(geom_id))
    if gname is not None and "floor" in str(gname).lower():
        return True
    body_id = int(model.geom_bodyid[int(geom_id)])
    bname = model.body_id2name(body_id)
    if bname is None:
        return False
    text = str(bname).lower()
    return "floor" in text or text.endswith("_ground")


_MJGEOM_PLANE = 0
_MJGEOM_BOX = 6


def _xy_on_floor_geom(env: Any, geom_id: int, xy: Sequence[float], *, margin: float) -> bool:
    """Whether (x, y) lies over a floor geom (thin-axis ignored)."""
    sim = env.sim
    model = sim.model
    gtype = int(model.geom_type[int(geom_id)])
    pos = np.asarray(sim.data.geom_xpos[int(geom_id)], dtype=np.float64)
    x, y = float(xy[0]), float(xy[1])
    if gtype == _MJGEOM_PLANE:
        return True
    if gtype != _MJGEOM_BOX:
        r = float(model.geom_rbound[int(geom_id)])
        return (x - pos[0]) ** 2 + (y - pos[1]) ** 2 <= (r + margin) ** 2
    xmat = np.asarray(sim.data.geom_xmat[int(geom_id)], dtype=np.float64).reshape(3, 3)
    size = np.asarray(model.geom_size[int(geom_id)], dtype=np.float64)
    query = np.array([x, y, pos[2]], dtype=np.float64)
    local = xmat.T @ (query - pos)
    thin = int(np.argmin(size))
    for i in range(3):
        if i == thin:
            continue
        if abs(float(local[i])) > float(size[i]) + float(margin):
            return False
    return True


def _floor_geom_top_z(env: Any, geom_id: int) -> float:
    sim = env.sim
    model = sim.model
    pos = np.asarray(sim.data.geom_xpos[int(geom_id)], dtype=np.float64)
    gtype = int(model.geom_type[int(geom_id)])
    if gtype != _MJGEOM_BOX:
        return float(pos[2])
    xmat = np.asarray(sim.data.geom_xmat[int(geom_id)], dtype=np.float64).reshape(3, 3)
    size = np.asarray(model.geom_size[int(geom_id)], dtype=np.float64)
    thin = int(np.argmin(size))
    axis = xmat[:, thin]
    return float(pos[2] + abs(float(axis[2])) * float(size[thin]))


def floor_support_at_xy(env: Any, xy: Sequence[float]) -> float | None:
    """Top Z of a Floor geom covering *xy*, or None if the point is off the floor."""
    model = env.sim.model
    tops: list[float] = []
    for gid in range(int(model.ngeom)):
        if not _is_floor_geom(env, gid):
            continue
        if not _xy_on_floor_geom(env, gid, xy, margin=0.02):
            continue
        tops.append(_floor_geom_top_z(env, gid))
    if not tops:
        return None
    return float(max(tops))


def obstacle_has_scene_penetration(
    env: Any,
    *,
    obstacle_geom_ids: set[int],
) -> bool:
    """True if the obstacle is overlapping non-floor scene geoms (cabinets/islands)."""
    sim = env.sim
    ncon = int(sim.data.ncon)
    for i in range(ncon):
        contact = sim.data.contact[i]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        g1_obs = g1 in obstacle_geom_ids
        g2_obs = g2 in obstacle_geom_ids
        if g1_obs == g2_obs:
            continue
        other = g2 if g1_obs else g1
        if _is_floor_geom(env, other):
            continue
        body_id = int(sim.model.geom_bodyid[other])
        bname = sim.model.body_id2name(body_id)
        if bname is not None and str(bname).startswith(
            ("robot0_", "gripper0_", "mobilebase0_", "panda0_")
        ):
            continue
        if float(contact.dist) < -PENETRATION_EPS:
            return True
    return False


def set_obstacle_contact_enabled(env: Any, *, enabled: bool) -> None:
    meta = l12_meta(env)
    model = env.sim.model
    raw = _raw_model(env.sim)
    body_id = int(model.body_name2id(str(meta["body_name"])))
    saved = meta.get("geom_contact")
    for gid in range(int(model.ngeom)):
        if int(model.geom_bodyid[gid]) != body_id:
            continue
        if not hasattr(raw, "geom_contype"):
            continue
        if enabled:
            if isinstance(saved, Mapping) and gid in saved:
                contype, conaff, rgba3 = saved[gid]
                raw.geom_contype[gid] = int(contype)
                raw.geom_conaffinity[gid] = int(conaff)
                if hasattr(raw, "geom_rgba"):
                    raw.geom_rgba[gid, 3] = float(rgba3)
            else:
                raw.geom_contype[gid] = 1
                raw.geom_conaffinity[gid] = 1
                if hasattr(raw, "geom_rgba"):
                    raw.geom_rgba[gid, 3] = 1.0
        else:
            raw.geom_contype[gid] = 0
            raw.geom_conaffinity[gid] = 0
            if hasattr(raw, "geom_rgba"):
                raw.geom_rgba[gid, 3] = 0.0
    if enabled:
        from dynisafety.backends.robocasa.visuals import reapply_l12_visual

        reapply_l12_visual(env)
    try:
        env.sim.forward()
    except Exception:
        pass


def snapshot_geom_contact(env: Any) -> dict[int, tuple[int, int, float]]:
    """Record collision+alpha so spawn can restore after a hidden episode start."""
    meta = l12_meta(env)
    model = env.sim.model
    raw = _raw_model(env.sim)
    body_id = int(model.body_name2id(str(meta["body_name"])))
    out: dict[int, tuple[int, int, float]] = {}
    for gid in range(int(model.ngeom)):
        if int(model.geom_bodyid[gid]) != body_id:
            continue
        contype = int(raw.geom_contype[gid]) if hasattr(raw, "geom_contype") else 1
        conaff = int(raw.geom_conaffinity[gid]) if hasattr(raw, "geom_conaffinity") else 1
        rgba3 = float(raw.geom_rgba[gid, 3]) if hasattr(raw, "geom_rgba") else 1.0
        out[int(gid)] = (contype, conaff, rgba3)
    return out


def base_speed_xy(prev_xy: np.ndarray, curr_xy: np.ndarray, *, control_freq: float) -> float:
    if control_freq <= 0:
        raise ValueError(f"control_freq must be > 0, got {control_freq!r}")
    delta = np.asarray(curr_xy, dtype=np.float64).reshape(2) - np.asarray(
        prev_xy, dtype=np.float64
    ).reshape(2)
    return float(np.linalg.norm(delta) * float(control_freq))


def read_base_pose(env: Any) -> tuple[np.ndarray, float]:
    return read_base_xy_yaw(env)


# ---------------------------------------------------------------------------
# L2 / L3 XY formulas (z is applied by the caller via body_z_on_floor)
# ---------------------------------------------------------------------------


def l2_start_xy(
    p0_xy: Sequence[float],
    yaw0: float,
    start_ahead: float,
    start_lateral: float,
    *,
    flip_side: bool,
) -> np.ndarray:
    """p_start_xy = p0 + start_ahead * h + start_lateral * n (n flipped if flip_side)."""
    p0 = np.asarray(p0_xy, dtype=np.float64).reshape(2)
    h = heading_from_yaw(yaw0)
    n = lateral_from_yaw(yaw0)
    if flip_side:
        n = -n
    return p0 + float(start_ahead) * h + float(start_lateral) * n


def l2_goal_xy(
    p0_xy: Sequence[float],
    yaw0: float,
    stop_clearance: float,
) -> np.ndarray:
    """p_goal_xy = p0 + stop_clearance * h. Stop on the floor in front of the base."""
    p0 = np.asarray(p0_xy, dtype=np.float64).reshape(2)
    return p0 + float(stop_clearance) * heading_from_yaw(yaw0)


def l2_attempt_start_xy(
    p0_xy: Sequence[float],
    yaw0: float,
    start_ahead: float,
    start_lateral: float,
    retry_step: float,
    attempt: int,
    *,
    flip_side: bool,
) -> np.ndarray:
    """Attempt *attempt* (0-based): extra *retry_step* along h for p_start only."""
    if int(attempt) < 0:
        raise ValueError(f"attempt must be >= 0, got {attempt!r}")
    dist = float(start_ahead) + int(attempt) * float(retry_step)
    return l2_start_xy(p0_xy, yaw0, dist, start_lateral, flip_side=bool(flip_side))


def l3_start_xy(
    p0_xy: Sequence[float],
    yaw0: float,
    human_start_ahead: float,
    human_start_lateral: float,
    *,
    flip_side: bool,
) -> np.ndarray:
    """p_start_xy = p0 + human_start_ahead * h + human_start_lateral * n."""
    p0 = np.asarray(p0_xy, dtype=np.float64).reshape(2)
    h = heading_from_yaw(yaw0)
    n = lateral_from_yaw(yaw0)
    if flip_side:
        n = -n
    return p0 + float(human_start_ahead) * h + float(human_start_lateral) * n


def l3_dwell_xy(
    p0_xy: Sequence[float],
    yaw0: float,
    human_standoff: float,
) -> np.ndarray:
    """p_dwell_xy = p0 + human_standoff * h. Pedestrian stands on the floor ahead."""
    p0 = np.asarray(p0_xy, dtype=np.float64).reshape(2)
    return p0 + float(human_standoff) * heading_from_yaw(yaw0)


def l3_attempt_start_xy(
    p0_xy: Sequence[float],
    yaw0: float,
    human_start_ahead: float,
    human_start_lateral: float,
    retry_step: float,
    attempt: int,
    *,
    flip_side: bool,
) -> np.ndarray:
    """Attempt *attempt* (0-based): extra *retry_step* along h for L3 p_start."""
    if int(attempt) < 0:
        raise ValueError(f"attempt must be >= 0, got {attempt!r}")
    dist = float(human_start_ahead) + int(attempt) * float(retry_step)
    return l3_start_xy(p0_xy, yaw0, dist, human_start_lateral, flip_side=bool(flip_side))


def lerp_xy(p_start_xy: Sequence[float], p_goal_xy: Sequence[float], alpha: float) -> np.ndarray:
    """Ground-plane linear interpolation. No Z, no parabola."""
    p0 = np.asarray(p_start_xy, dtype=np.float64).reshape(2)
    p1 = np.asarray(p_goal_xy, dtype=np.float64).reshape(2)
    return lerp(p0, p1, alpha)


def l3_meta(env: Any) -> Mapping[str, Any]:
    meta = getattr(env, "_dynisafety_l3", None)
    if not isinstance(meta, Mapping):
        raise RuntimeError("env missing _dynisafety_l3 merge metadata for the pedestrian proxy")
    return meta


def pos_from_xy_floor(
    xy: Sequence[float],
    floor_z: float,
    half_size_z: float,
) -> np.ndarray:
    """Body origin at floor_z + half_z. Caller supplies floor_z (never invent 0)."""
    xy_a = np.asarray(xy, dtype=np.float64).reshape(2)
    return np.array(
        [float(xy_a[0]), float(xy_a[1]), body_z_on_floor(floor_z, half_size_z)],
        dtype=np.float64,
    )


def read_body_pos(env: Any, body_name: str) -> np.ndarray:
    body_id = int(env.sim.model.body_name2id(str(body_name)))
    return np.asarray(env.sim.data.body_xpos[body_id], dtype=np.float64).copy()


def snapshot_actor_geom_contact(
    env: Any, meta: Mapping[str, Any]
) -> dict[int, tuple[int, int, float]]:
    model = env.sim.model
    raw = _raw_model(env.sim)
    body_id = int(model.body_name2id(str(meta["body_name"])))
    out: dict[int, tuple[int, int, float]] = {}
    for gid in range(int(model.ngeom)):
        if int(model.geom_bodyid[gid]) != body_id:
            continue
        contype = int(raw.geom_contype[gid]) if hasattr(raw, "geom_contype") else 1
        conaff = int(raw.geom_conaffinity[gid]) if hasattr(raw, "geom_conaffinity") else 1
        rgba3 = float(raw.geom_rgba[gid, 3]) if hasattr(raw, "geom_rgba") else 1.0
        out[int(gid)] = (contype, conaff, rgba3)
    return out


def set_actor_contact_enabled(env: Any, meta: Mapping[str, Any], *, enabled: bool) -> None:
    model = env.sim.model
    raw = _raw_model(env.sim)
    body_id = int(model.body_name2id(str(meta["body_name"])))
    saved = meta.get("geom_contact")
    for gid in range(int(model.ngeom)):
        if int(model.geom_bodyid[gid]) != body_id:
            continue
        if not hasattr(raw, "geom_contype"):
            continue
        if enabled:
            if isinstance(saved, Mapping) and gid in saved:
                contype, conaff, rgba3 = saved[gid]
                raw.geom_contype[gid] = int(contype)
                raw.geom_conaffinity[gid] = int(conaff)
                if hasattr(raw, "geom_rgba"):
                    raw.geom_rgba[gid, 3] = float(rgba3)
            else:
                raw.geom_contype[gid] = 1
                raw.geom_conaffinity[gid] = 1
                if hasattr(raw, "geom_rgba"):
                    raw.geom_rgba[gid, 3] = 1.0
        else:
            raw.geom_contype[gid] = 0
            raw.geom_conaffinity[gid] = 0
            if hasattr(raw, "geom_rgba"):
                raw.geom_rgba[gid, 3] = 0.0
    if enabled:
        from dynisafety.backends.robocasa.visuals import reapply_l12_visual

        reapply_l12_visual(env)
    try:
        env.sim.forward()
    except Exception:
        pass


def teleport_actor_hidden(env: Any, meta: Mapping[str, Any]) -> None:
    set_free_joint_pose(env, str(meta["joint_name"]), HIDDEN_POS, UPRIGHT_QUAT_WXYZ)


def collect_body_geom_ids(env: Any, body_name: str) -> set[int]:
    model = env.sim.model
    body_id = int(model.body_name2id(str(body_name)))
    body_ids = {body_id}
    for child in range(int(model.nbody)):
        if int(model.body_parentid[child]) in body_ids:
            body_ids.add(int(child))
    ids: set[int] = set()
    for gid in range(int(model.ngeom)):
        if int(model.geom_bodyid[gid]) in body_ids:
            ids.add(int(gid))
    return ids


_ROBOT_BODY_PREFIXES = ("robot0_", "gripper0_", "mobilebase0_", "panda0_")


def actor_has_robot_contact(
    env: Any,
    *,
    actor_geom_ids: set[int],
    robot_geom_ids: set[int] | None = None,
) -> bool:
    sim = env.sim
    ncon = int(sim.data.ncon)
    for i in range(ncon):
        contact = sim.data.contact[i]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        a1, a2 = g1 in actor_geom_ids, g2 in actor_geom_ids
        if a1 == a2:
            continue
        other = g2 if a1 else g1
        if robot_geom_ids is not None:
            if other in robot_geom_ids:
                return True
            continue
        body_id = int(sim.model.geom_bodyid[other])
        bname = sim.model.body_id2name(body_id)
        if bname is not None and str(bname).startswith(_ROBOT_BODY_PREFIXES):
            return True
    return False


def footprint_on_floor(
    env: Any,
    xy: Sequence[float],
    half_x: float,
    half_y: float,
) -> float | None:
    """Require the full box footprint (center + 4 corners) to lie on Floor geoms.

    Prevents a tall pedestrian AABB from overlapping a counter in XY.
    Returns the center's floor top-z, or None.
    """
    x0, y0 = float(xy[0]), float(xy[1])
    hx, hy = float(half_x), float(half_y)
    samples = (
        (x0, y0),
        (x0 - hx, y0 - hy),
        (x0 - hx, y0 + hy),
        (x0 + hx, y0 - hy),
        (x0 + hx, y0 + hy),
    )
    center_z: float | None = None
    for i, (sx, sy) in enumerate(samples):
        z = floor_support_at_xy(env, (sx, sy))
        if z is None:
            return None
        if i == 0:
            center_z = float(z)
    return center_z


def probe_floor_placement(
    env: Any,
    *,
    joint_name: str,
    xy: Sequence[float],
    half_size_z: float,
    obstacle_geom_ids: set[int],
    half_xy: Sequence[float] | None = None,
) -> np.ndarray | None:
    """Place at floor(xy)+half_z; return pos if on Floor and not in furniture.

    Restores the previous free-joint pose. Never uses counter/table support_z.
    If *half_xy* is given, the full footprint must sit on Floor (L3 pedestrian).
    """
    if half_xy is not None:
        support_z = footprint_on_floor(env, xy, half_xy[0], half_xy[1])
    else:
        support_z = floor_support_at_xy(env, xy)
    if support_z is None:
        return None
    pos = pos_from_xy_floor(xy, support_z, half_size_z)
    return _probe_pose(
        env,
        joint_name=joint_name,
        pos=pos,
        obstacle_geom_ids=obstacle_geom_ids,
    )


def probe_xy_at_z(
    env: Any,
    *,
    joint_name: str,
    xy: Sequence[float],
    origin_z: float,
    obstacle_geom_ids: set[int],
    half_xy: Sequence[float] | None = None,
) -> np.ndarray | None:
    """Place at (xy, origin_z) if the footprint is on Floor and not in furniture.

    XY still has to sit on a Floor geom (aisle, not a counter). Z is the
    caller-supplied body origin (e.g. EEF height), not floor+half_z.
    """
    if half_xy is not None:
        support_z = footprint_on_floor(env, xy, half_xy[0], half_xy[1])
    else:
        support_z = floor_support_at_xy(env, xy)
    if support_z is None:
        return None
    xy_a = np.asarray(xy, dtype=np.float64).reshape(2)
    pos = np.array(
        [float(xy_a[0]), float(xy_a[1]), float(origin_z)],
        dtype=np.float64,
    )
    return _probe_pose(
        env,
        joint_name=joint_name,
        pos=pos,
        obstacle_geom_ids=obstacle_geom_ids,
    )


def _probe_pose(
    env: Any,
    *,
    joint_name: str,
    pos: np.ndarray,
    obstacle_geom_ids: set[int],
) -> np.ndarray | None:
    prev = np.asarray(env.sim.data.get_joint_qpos(joint_name), dtype=np.float64).copy()
    try:
        set_free_joint_pose(env, joint_name, pos, UPRIGHT_QUAT_WXYZ)
        if obstacle_has_scene_penetration(env, obstacle_geom_ids=obstacle_geom_ids):
            return None
        return np.asarray(pos, dtype=np.float64).reshape(3).copy()
    finally:
        prev_pos = prev[:3]
        prev_quat = prev[3:7] if prev.size >= 7 else UPRIGHT_QUAT_WXYZ
        set_free_joint_pose(env, joint_name, prev_pos, prev_quat)
