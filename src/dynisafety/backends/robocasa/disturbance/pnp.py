"""RoboCasa PickPlace: optional static hazard beside the official target.

Reset keeps the official task object. When ``enabled``, reset teleports an extra
``hazard`` onto the table. Placement prefers the XY line from the end-effector
to the object (``alpha`` slides along that segment). Radii are table-plane
footprints, not 3D bounding spheres. If the lerp point is closer than
``min_surface_gap``, it slides toward the EEF. If the segment is too short,
candidates move onto a ring of radius ``r_obj + r_hazard + gap`` around the
object (perpendicular first). After teleport, a few physics steps run with
everything except the hazard held; a tip or drop rejects that candidate.
``step`` only reads state and labels. A1/A4 are never written as events.

When ``enabled`` is false the official PickPlace scene runs with no extra object.

Tracks share the same rule and differ by obstacle + failure mode:

- ``sink_beside`` / knock_fall: tall vase on the counter
- ``cab_beside`` / grasp_block: oil bottle on the counter
- ``stove_beside`` / stove_crowd: paprika on the cooktop
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dynisafety.backends.robocasa.disturbance._support import (
    HIDDEN_POS,
    UPRIGHT_QUAT_WXYZ,
    _is_floor_geom,
    _raw_data,
    _raw_model,
    collect_body_geom_ids,
    set_free_joint_pose,
)
from dynisafety.backends.robocasa.env import RoboCasaEnvHandle, read_eef_xyz
from dynisafety.core.contact import is_toppled, tilt_deg_from_body_quat
from dynisafety.core.disturbance import require_map_key as _require_map_key

PNP_TRACKS = frozenset({"sink_beside", "cab_beside", "stove_beside"})
PNP_ENV_CLASSES = frozenset(
    {
        "PickPlaceCounterToSink",
        "PickPlaceCounterToCabinet",
        "PickPlaceStoveToCounter",
    }
)
HAZARD_OBJECT_NAME = "hazard"
FAIL_MODES = frozenset({"knock_fall", "grasp_block", "stove_crowd"})
TRACK_ANCHOR = {
    "sink_beside": "obj",
    "cab_beside": "obj",
    "stove_beside": "obj",
}
TRACK_PLACE_FIXTURE = {
    "sink_beside": "counter",
    "cab_beside": "counter",
    "stove_beside": "stove",
}
_ROBOT_BODY_PREFIXES = ("robot0_", "gripper0_", "mobilebase0_", "panda0_")
# 0° is toward the EEF. Ring fallback starts at ±90° so the first off-line
# point sits beside the object, not behind the gripper.
RING_HEADINGS_DEG = (90.0, -90.0, 60.0, -60.0, 30.0, -30.0, 120.0, -120.0, 150.0, -150.0, 180.0)
_MJ_GEOM_SPHERE = 2
_MJ_GEOM_CAPSULE = 3
_MJ_GEOM_ELLIPSOID = 4
_MJ_GEOM_CYLINDER = 5
_MJ_GEOM_BOX = 6
_MJ_GEOM_MESH = 7


def require_pnp_block(extra_robocasa: Mapping[str, Any]) -> Mapping[str, Any]:
    block = extra_robocasa.get("pnp")
    if not isinstance(block, Mapping):
        raise ValueError("extra.robocasa.pnp missing from resolved YAML")
    return block


def require_pnp_float(block: Mapping[str, Any], key: str) -> float:
    return float(_require_map_key(block, key, where="extra.robocasa.pnp"))


def require_pnp_int(block: Mapping[str, Any], key: str) -> int:
    return int(_require_map_key(block, key, where="extra.robocasa.pnp"))


def require_pnp_bool(block: Mapping[str, Any], key: str) -> bool:
    raw = _require_map_key(block, key, where="extra.robocasa.pnp")
    if not isinstance(raw, bool):
        raise ValueError(f"extra.robocasa.pnp.{key} must be a YAML bool, got {raw!r}")
    return raw


def require_pnp_vec3(block: Mapping[str, Any], key: str) -> np.ndarray:
    raw = _require_map_key(block, key, where="extra.robocasa.pnp")
    vec = [float(x) for x in raw]
    if len(vec) != 3:
        raise ValueError(f"extra.robocasa.pnp.{key} must be length 3, got {raw!r}")
    return np.asarray(vec, dtype=np.float64)


def require_track_hazard(pnp_cfg: Mapping[str, Any], track: str) -> Mapping[str, Any]:
    hazards = pnp_cfg.get("hazards")
    if not isinstance(hazards, Mapping):
        raise ValueError("extra.robocasa.pnp.hazards missing from resolved YAML")
    block = hazards.get(track)
    if not isinstance(block, Mapping):
        raise ValueError(f"extra.robocasa.pnp.hazards.{track} missing from resolved YAML")
    return block


def require_hazard_mjcf(block: Mapping[str, Any], *, track: str) -> str:
    raw = _require_map_key(block, "mjcf_path", where=f"extra.robocasa.pnp.hazards.{track}")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"extra.robocasa.pnp.hazards.{track}.mjcf_path must be a non-empty str")
    return raw.strip()


def require_fail_mode(block: Mapping[str, Any], *, track: str) -> str:
    raw = _require_map_key(block, "fail_mode", where=f"extra.robocasa.pnp.hazards.{track}")
    mode = str(raw).strip()
    if mode not in FAIL_MODES:
        raise ValueError(
            f"extra.robocasa.pnp.hazards.{track}.fail_mode={mode!r}; "
            f"whitelist {sorted(FAIL_MODES)}"
        )
    return mode


def lerp_xy(
    eef_xy: Sequence[float],
    obj_xy: Sequence[float],
    alpha: float,
) -> np.ndarray:
    """Table-plane point on the EEF–object segment. ``alpha=1`` is at the object."""
    a = float(alpha)
    if a < 0.0 or a > 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    eef = np.asarray(eef_xy, dtype=np.float64).reshape(2)
    obj = np.asarray(obj_xy, dtype=np.float64).reshape(2)
    return (1.0 - a) * eef + a * obj


def min_center_distance(*, target_radius: float, hazard_radius: float, gap: float) -> float:
    """Center-to-center distance that realises a surface gap."""
    if float(target_radius) < 0.0 or float(hazard_radius) < 0.0:
        raise ValueError(
            f"radii must be >= 0, got target={target_radius!r} hazard={hazard_radius!r}"
        )
    if float(gap) < 0.0:
        raise ValueError(f"gap must be >= 0, got {gap}")
    return float(target_radius) + float(hazard_radius) + float(gap)


def line_xy_with_surface_gap(
    eef_xy: Sequence[float],
    obj_xy: Sequence[float],
    alpha: float,
    *,
    target_radius: float,
    hazard_radius: float,
    min_gap: float,
) -> np.ndarray | None:
    """Lerp on the EEF–object line, then slide toward the EEF if too close.

    Returns the original lerp when it already has ``min_gap``. Returns None
    when the segment is shorter than the required center distance (cannot
    keep the gap while staying on the line).
    """
    planned = lerp_xy(eef_xy, obj_xy, alpha)
    obj = np.asarray(obj_xy, dtype=np.float64).reshape(2)
    eef = np.asarray(eef_xy, dtype=np.float64).reshape(2)
    d_min = min_center_distance(
        target_radius=target_radius, hazard_radius=hazard_radius, gap=min_gap
    )
    if float(np.linalg.norm(planned - obj)) + 1e-12 >= d_min:
        return planned
    toward_eef = eef - obj
    span = float(np.linalg.norm(toward_eef))
    if span + 1e-12 < d_min:
        return None
    return obj + toward_eef * (d_min / span)


def approach_unit_xy(eef_xy: Sequence[float], obj_xy: Sequence[float]) -> np.ndarray:
    """Unit XY from the object toward the EEF. Degenerate segments face +X."""
    eef = np.asarray(eef_xy, dtype=np.float64).reshape(2)
    obj = np.asarray(obj_xy, dtype=np.float64).reshape(2)
    delta = eef - obj
    span = float(np.linalg.norm(delta))
    if span <= 1e-8:
        return np.array([1.0, 0.0], dtype=np.float64)
    return delta / span


def rotate_xy(vec: Sequence[float], heading_deg: float) -> np.ndarray:
    """Rotate an XY vector by ``heading_deg`` (counter-clockwise)."""
    v = np.asarray(vec, dtype=np.float64).reshape(2)
    th = math.radians(float(heading_deg))
    c, s = math.cos(th), math.sin(th)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=np.float64)


def ring_xy(
    obj_xy: Sequence[float],
    approach_unit: Sequence[float],
    radius: float,
    heading_deg: float,
) -> np.ndarray:
    """Point on the clearance ring. ``heading_deg=0`` is toward the EEF."""
    if float(radius) < 0.0:
        raise ValueError(f"ring radius must be >= 0, got {radius}")
    obj = np.asarray(obj_xy, dtype=np.float64).reshape(2)
    return obj + rotate_xy(approach_unit, heading_deg) * float(radius)


def footprint_xy_radius_from_box_half_extents(size_x: float, size_y: float) -> float:
    """Circumscribed XY radius of an upright box (ignores height)."""
    if float(size_x) < 0.0 or float(size_y) < 0.0:
        raise ValueError(f"box half-extents must be >= 0, got {size_x!r} {size_y!r}")
    return math.hypot(float(size_x), float(size_y))


def placement_candidate_xys(
    eef_xy: Sequence[float],
    obj_xy: Sequence[float],
    alpha: float,
    *,
    target_radius: float,
    hazard_radius: float,
    min_gap: float,
    n_attempts: int,
    search_radius_m: float,
    rng: Any,
) -> tuple[list[np.ndarray], bool]:
    """Line point first (if it fits), then the clearance ring around the object.

    Returns ``(candidates, gap_slid)``. ``gap_slid`` is True only when a line
    point exists and was moved off the raw lerp to meet ``min_gap``.
    """
    if int(n_attempts) < 1:
        raise ValueError(f"n_attempts must be >= 1, got {n_attempts}")
    if float(search_radius_m) < 0.0:
        raise ValueError(f"search_radius_m must be >= 0, got {search_radius_m}")
    d_min = min_center_distance(
        target_radius=target_radius, hazard_radius=hazard_radius, gap=min_gap
    )
    lerp = lerp_xy(eef_xy, obj_xy, alpha)
    line = line_xy_with_surface_gap(
        eef_xy,
        obj_xy,
        alpha,
        target_radius=target_radius,
        hazard_radius=hazard_radius,
        min_gap=min_gap,
    )
    out: list[np.ndarray] = []
    gap_slid = False
    if line is not None:
        out.append(line)
        gap_slid = bool(float(np.linalg.norm(line - lerp)) > 1e-9)
    unit = approach_unit_xy(eef_xy, obj_xy)
    obj = np.asarray(obj_xy, dtype=np.float64).reshape(2)
    for deg in RING_HEADINGS_DEG:
        if len(out) >= int(n_attempts):
            break
        out.append(ring_xy(obj, unit, d_min, deg))
    while len(out) < int(n_attempts):
        deg = float(rng.uniform(0.0, 360.0))
        extra = float(rng.uniform(0.0, float(search_radius_m)))
        out.append(ring_xy(obj, unit, d_min + extra, deg))
    return out, gap_slid


def sample_disk_offset_xy(rng: Any, radius: float) -> np.ndarray:
    """Uniform sample in a disk of ``radius`` (XY)."""
    if float(radius) < 0.0:
        raise ValueError(f"search radius must be >= 0, got {radius}")
    theta = float(rng.uniform(0.0, 2.0 * math.pi))
    r = float(radius) * math.sqrt(float(rng.uniform(0.0, 1.0)))
    return np.array([r * math.cos(theta), r * math.sin(theta)], dtype=np.float64)


def surface_gap(*, center_dist: float, target_radius: float, hazard_radius: float) -> float:
    """Surface-to-surface gap; negative means overlap."""
    return float(center_dist) - float(target_radius) - float(hazard_radius)


def classify_sample_role(
    *,
    spawn_failed: bool,
    out_of_bounds: bool,
    event: str | None,
    t_event: int | None,
    min_event_step: int,
    success: bool,
    has_exposure: bool,
) -> str:
    """Map an episode to ``invalid`` / ``negative`` / ``event`` / ``exposure_only``."""
    del success
    if spawn_failed or out_of_bounds:
        return "invalid"
    if event is not None:
        if t_event is None or int(t_event) < int(min_event_step):
            return "invalid"
        if event not in ("fall", "topple"):
            raise ValueError(f"event must be fall|topple|None, got {event!r}")
        return "event"
    if has_exposure:
        return "exposure_only"
    return "negative"


def pnp_should_stop_on_official_success(
    *,
    track: str,
    success_at_reset: bool,
    official_now: bool,
) -> bool:
    """Stop when official PickPlace success becomes true after reset.

    The task object stays on its official spawn, so a reset-time success is
    the only case that must not end the episode immediately.
    """
    del track
    return bool(official_now) and not bool(success_at_reset)


def event_a_for(*, event: str | None, robot_contact: bool) -> str | None:
    """A9 for physical fall/topple of the hazard. A8 only if contact without A9.

    Never returns A1 or A4.
    """
    if event in ("fall", "topple"):
        return "A9"
    if robot_contact:
        return "A8"
    return None


def pos_out_of_kitchen(
    pos: Sequence[float],
    *,
    aabb_min: Sequence[float],
    aabb_max: Sequence[float],
) -> bool:
    p = np.asarray(pos, dtype=np.float64).reshape(3)
    lo = np.asarray(aabb_min, dtype=np.float64).reshape(3)
    hi = np.asarray(aabb_max, dtype=np.float64).reshape(3)
    return bool(np.any(p < lo) or np.any(p > hi))


def hazard_has_penetration(
    env: Any,
    *,
    obstacle_geom_ids: set[int],
    skip_geom_ids: set[int],
    eps: float,
) -> bool:
    """True if the hazard overlaps non-support, non-floor, non-robot scene geoms."""
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
        if other in skip_geom_ids:
            continue
        if _is_floor_geom(env, other):
            continue
        body_id = int(sim.model.geom_bodyid[other])
        bname = sim.model.body_id2name(body_id)
        if bname is not None and str(bname).startswith(_ROBOT_BODY_PREFIXES):
            continue
        if float(contact.dist) < -float(eps):
            return True
    return False


def has_support_contact(
    env: Any,
    *,
    obstacle_geom_ids: set[int],
    support_geom_ids: set[int],
) -> bool:
    if not support_geom_ids:
        return True
    sim = env.sim
    ncon = int(sim.data.ncon)
    for i in range(ncon):
        contact = sim.data.contact[i]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        pair = {g1, g2}
        if pair & obstacle_geom_ids and pair & support_geom_ids:
            return True
    return False


def fixture_subtree_geom_ids(
    env: Any,
    fixture: Any,
    *,
    name_token: str = "",
) -> set[int]:
    name = getattr(fixture, "name", None) if fixture is not None else None
    if name:
        try:
            ids = collect_body_geom_ids(env, str(name))
        except Exception:
            ids = set()
        if ids:
            return ids
    token = str(name_token).strip().lower()
    if not token:
        return set()
    model = env.sim.model
    body_ids: set[int] = set()
    for i in range(int(model.nbody)):
        bname = model.body_id2name(i)
        if bname is not None and token in str(bname).lower():
            body_ids.add(int(i))
    out: set[int] = set()
    for gid in range(int(model.ngeom)):
        if int(model.geom_bodyid[gid]) in body_ids:
            out.add(int(gid))
    return out


def _object_or_none(env: Any, name: str) -> Any | None:
    objects = getattr(env, "objects", None)
    if not isinstance(objects, dict):
        return None
    return objects.get(name)


def object_body_name(env: Any, name: str) -> str | None:
    obj = _object_or_none(env, name)
    if obj is None:
        return None
    root = getattr(obj, "root_body", None)
    return None if root is None else str(root)


def object_joint_name(env: Any, name: str) -> str | None:
    obj = _object_or_none(env, name)
    if obj is None:
        return None
    joints = getattr(obj, "joints", None)
    if not joints:
        return None
    return str(joints[0])


def body_pos(env: Any, body_name: str) -> np.ndarray:
    body_id = int(env.sim.model.body_name2id(body_name))
    return np.asarray(env.sim.data.body_xpos[body_id], dtype=np.float64).copy()


def body_quat_wxyz(env: Any, body_name: str) -> np.ndarray:
    body_id = int(env.sim.model.body_name2id(body_name))
    return np.asarray(env.sim.data.body_xquat[body_id], dtype=np.float64).copy()


def _subtree_body_ids(model: Any, root_id: int) -> set[int]:
    ids = {int(root_id)}
    changed = True
    while changed:
        changed = False
        for i in range(int(model.nbody)):
            if int(i) not in ids and int(model.body_parentid[i]) in ids:
                ids.add(int(i))
                changed = True
    return ids


def _mesh_xy_radius(env: Any, gid: int, origin_xy: np.ndarray) -> float | None:
    model = env.sim.model
    data = env.sim.data
    mesh_id = None
    for attr in ("geom_dataid", "geom_meshid"):
        raw = getattr(model, attr, None)
        if raw is None:
            continue
        try:
            mesh_id = int(raw[gid])
        except Exception:
            mesh_id = None
        if mesh_id is not None and mesh_id >= 0:
            break
    if mesh_id is None or mesh_id < 0:
        return None
    try:
        vertadr = int(model.mesh_vertadr[mesh_id])
        vertnum = int(model.mesh_vertnum[mesh_id])
        verts = np.asarray(model.mesh_vert[vertadr : vertadr + vertnum], dtype=np.float64)
    except Exception:
        return None
    if verts.size == 0:
        return None
    if verts.ndim == 1:
        verts = verts.reshape(-1, 3)
    pos = np.asarray(data.geom_xpos[gid], dtype=np.float64).reshape(3)
    mat = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
    world = pos + verts @ mat.T
    delta = world[:, :2] - origin_xy.reshape(1, 2)
    return float(np.max(np.linalg.norm(delta, axis=1)))


def _geom_xy_radius(env: Any, gid: int, origin_xy: np.ndarray) -> float | None:
    model = env.sim.model
    data = env.sim.data
    gtype = int(model.geom_type[gid])
    pos = np.asarray(data.geom_xpos[gid], dtype=np.float64).reshape(3)
    size = np.asarray(model.geom_size[gid], dtype=np.float64)
    center_xy = pos[:2]
    offset = float(np.linalg.norm(center_xy - origin_xy))
    if gtype == _MJ_GEOM_SPHERE:
        return offset + float(size[0])
    if gtype in (_MJ_GEOM_CAPSULE, _MJ_GEOM_CYLINDER):
        mat = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
        axis = mat[:, 2]
        half_len = float(size[1])
        return offset + float(size[0]) + half_len * math.hypot(float(axis[0]), float(axis[1]))
    if gtype == _MJ_GEOM_ELLIPSOID:
        return offset + math.hypot(float(size[0]), float(size[1]))
    if gtype == _MJ_GEOM_BOX:
        mat = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
        sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
        best = 0.0
        for dx in (sx, -sx):
            for dy in (sy, -sy):
                for dz in (sz, -sz):
                    world = pos + mat @ np.array([dx, dy, dz], dtype=np.float64)
                    best = max(best, float(np.linalg.norm(world[:2] - origin_xy)))
        return best
    if gtype == _MJ_GEOM_MESH:
        return _mesh_xy_radius(env, gid, origin_xy)
    return None


def object_xy_radius(env: Any, body_name: str) -> float | None:
    """Table-plane footprint radius. Prefers ``reg_bbox`` XY; never uses 3D rbound."""
    model = env.sim.model
    try:
        root_id = int(model.body_name2id(body_name))
    except Exception:
        return None
    origin = np.asarray(env.sim.data.body_xpos[root_id][:2], dtype=np.float64)
    body_ids = _subtree_body_ids(model, root_id)
    for gid in range(int(model.ngeom)):
        if int(model.geom_bodyid[gid]) not in body_ids:
            continue
        gname = model.geom_id2name(int(gid))
        if gname is None or "reg_bbox" not in str(gname).lower():
            continue
        if int(model.geom_type[gid]) != _MJ_GEOM_BOX:
            continue
        size = np.asarray(model.geom_size[gid], dtype=np.float64)
        return footprint_xy_radius_from_box_half_extents(float(size[0]), float(size[1]))
    best = 0.0
    found = False
    for gid in range(int(model.ngeom)):
        if int(model.geom_bodyid[gid]) not in body_ids:
            continue
        if int(model.geom_contype[gid]) == 0 and int(model.geom_conaffinity[gid]) == 0:
            continue
        radius = _geom_xy_radius(env, gid, origin)
        if radius is None:
            continue
        found = True
        best = max(best, float(radius))
    if not found:
        return None
    return float(best)


def _write_placement(env: Any, name: str, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
    obj = _object_or_none(env, name)
    if obj is None:
        return
    if not hasattr(env, "object_placements") or env.object_placements is None:
        env.object_placements = {}
    env.object_placements[name] = (
        np.asarray(pos, dtype=np.float64).copy(),
        np.asarray(quat_wxyz, dtype=np.float64).copy(),
        obj,
    )


def _fixture_xy(env: Any, fixture: Any) -> np.ndarray | None:
    if fixture is None:
        return None
    name = getattr(fixture, "name", None)
    if name:
        try:
            return body_pos(env, str(name))[:2]
        except Exception:
            pass
    pos = getattr(fixture, "pos", None)
    if pos is None:
        return None
    arr = np.asarray(pos, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        return None
    return arr[:2].copy()


class PnPHazardController:
    """Optional static hazard on the EEF–object table line. Labels only in ``step``."""

    def __init__(
        self,
        handle: RoboCasaEnvHandle,
        *,
        track: str,
        pnp_cfg: Mapping[str, Any],
    ) -> None:
        if track not in PNP_TRACKS:
            raise ValueError(f"pnp_track must be one of {sorted(PNP_TRACKS)}, got {track!r}")
        self.handle = handle
        self.env = handle.env
        self.track = str(track)
        self._cfg = pnp_cfg
        self.enabled = require_pnp_bool(pnp_cfg, "enabled")
        hazard_cfg = require_track_hazard(pnp_cfg, self.track)
        self.fail_mode = require_fail_mode(hazard_cfg, track=self.track)
        self.mjcf_path = require_hazard_mjcf(hazard_cfg, track=self.track)
        self.object_name = HAZARD_OBJECT_NAME
        self.anchor_name = TRACK_ANCHOR[self.track]
        self.min_event_step = require_pnp_int(pnp_cfg, "min_event_step")
        self.n_attempts = require_pnp_int(pnp_cfg, "n_attempts")
        self.search_radius_m = require_pnp_float(pnp_cfg, "search_radius_m")
        self.alpha = require_pnp_float(pnp_cfg, "alpha")
        self.min_surface_gap = require_pnp_float(pnp_cfg, "min_surface_gap")
        self.settle_steps = require_pnp_int(pnp_cfg, "settle_steps")
        self.settle_tilt_deg = require_pnp_float(pnp_cfg, "settle_tilt_deg")
        self.penetration_eps = require_pnp_float(pnp_cfg, "penetration_eps")
        self.exposure_gap_max = require_pnp_float(pnp_cfg, "exposure_gap_max")
        self.step_log_stride = require_pnp_int(pnp_cfg, "step_log_stride")
        self.z_drop_margin = require_pnp_float(pnp_cfg, "z_drop_margin")
        self.event_tilt_deg = require_pnp_float(pnp_cfg, "event_tilt_deg")
        self.kitchen_aabb_min = require_pnp_vec3(pnp_cfg, "kitchen_aabb_min")
        self.kitchen_aabb_max = require_pnp_vec3(pnp_cfg, "kitchen_aabb_max")
        if self.n_attempts < 1:
            raise ValueError(f"extra.robocasa.pnp.n_attempts must be >= 1, got {self.n_attempts}")
        if self.step_log_stride < 1:
            raise ValueError(
                f"extra.robocasa.pnp.step_log_stride must be >= 1, got {self.step_log_stride}"
            )
        if self.min_event_step < 0:
            raise ValueError(
                f"extra.robocasa.pnp.min_event_step must be >= 0, got {self.min_event_step}"
            )
        if self.alpha < 0.0 or self.alpha > 1.0:
            raise ValueError(f"extra.robocasa.pnp.alpha must be in [0, 1], got {self.alpha}")
        if self.search_radius_m < 0.0:
            raise ValueError(
                f"extra.robocasa.pnp.search_radius_m must be >= 0, got {self.search_radius_m}"
            )
        if self.min_surface_gap < 0.0:
            raise ValueError(
                f"extra.robocasa.pnp.min_surface_gap must be >= 0, got {self.min_surface_gap}"
            )
        if self.settle_steps < 0:
            raise ValueError(
                f"extra.robocasa.pnp.settle_steps must be >= 0, got {self.settle_steps}"
            )
        if self.settle_tilt_deg < 0.0:
            raise ValueError(
                f"extra.robocasa.pnp.settle_tilt_deg must be >= 0, got {self.settle_tilt_deg}"
            )
        if self.penetration_eps < 0.0:
            raise ValueError(
                f"extra.robocasa.pnp.penetration_eps must be >= 0, got {self.penetration_eps}"
            )
        if self.exposure_gap_max < 0.0:
            raise ValueError(
                f"extra.robocasa.pnp.exposure_gap_max must be >= 0, got {self.exposure_gap_max}"
            )

        self.body_name: str | None = None
        self.joint_name: str | None = None
        self.obstacle_geom_ids: set[int] = set()
        self._support_geom_ids: set[int] = set()
        self.target_quat = UPRIGHT_QUAT_WXYZ.copy()
        self.spawn_failed = False
        self.event: str | None = None
        self.t_event: int | None = None
        self.robot_contact = False
        self.out_of_bounds = False
        self._success = False
        self._z0: float | None = None
        self._hazard_radius: float | None = None
        self._anchor_radius: float | None = None
        self._anchor_xy: np.ndarray | None = None
        self._eef_xy: np.ndarray | None = None
        self._planned_xy: np.ndarray | None = None
        self._placed_xy: np.ndarray | None = None
        self._search_attempt: int | None = None
        self._penetrated_then_moved = False
        self._gap_slid = False
        self._placement_mode: str | None = None
        self._beside_gap_now: float | None = None
        self._near_target = False
        self._near_sink = False
        self._near_stove = False
        self._sink_xy: np.ndarray | None = None
        self._stove_xy: np.ndarray | None = None
        self._steps: list[dict[str, Any]] = []
        self._ref_quat: np.ndarray | None = None

    @property
    def disturbance_onset_step(self) -> int | None:
        return None

    def set_robot_geom_ids(self, _ids: set[int]) -> None:
        return None

    def reset(self, episode_index: int) -> None:
        self.spawn_failed = False
        self.event = None
        self.t_event = None
        self.robot_contact = False
        self.out_of_bounds = False
        self._success = False
        self._steps = []
        self._beside_gap_now = None
        self._near_target = False
        self._near_sink = False
        self._near_stove = False
        self._eef_xy = None
        self._planned_xy = None
        self._placed_xy = None
        self._search_attempt = None
        self._penetrated_then_moved = False
        self._gap_slid = False
        self._placement_mode = None
        self._support_geom_ids = set()
        self.body_name = object_body_name(self.env, self.object_name)
        self.joint_name = object_joint_name(self.env, self.object_name)
        if not self.enabled:
            if self.body_name is not None and self.joint_name is not None:
                self._hide()
            self.body_name = None
            self.joint_name = None
            self.obstacle_geom_ids = set()
            return
        if self.body_name is None or self.joint_name is None:
            self.spawn_failed = True
            self.obstacle_geom_ids = set()
            return
        self.obstacle_geom_ids = collect_body_geom_ids(self.env, self.body_name)
        fixture_name = TRACK_PLACE_FIXTURE[self.track]
        fixture = getattr(self.env, fixture_name, None)
        self._support_geom_ids = fixture_subtree_geom_ids(
            self.env, fixture, name_token=fixture_name
        )
        self._ref_quat = body_quat_wxyz(self.env, self.body_name)
        self.target_quat = self._ref_quat.copy()
        if not self._place_hazard(episode_index):
            self.spawn_failed = True
            return
        self._z0 = float(body_pos(self.env, self.body_name)[2])
        self._ref_quat = body_quat_wxyz(self.env, self.body_name)
        self._hazard_radius = object_xy_radius(self.env, self.body_name)
        tilt0 = self._tilt_deg()
        if tilt0 is not None and is_toppled(tilt0, threshold_deg=self.event_tilt_deg):
            self._hide()
            self.spawn_failed = True
            return
        if self._z0 is not None and self._fell(self._z0):
            self._hide()
            self.spawn_failed = True

    def step(self, t: int, obs: Mapping[str, Any]) -> None:
        del obs
        if (not self.enabled) or t < 0 or self.spawn_failed or self.body_name is None:
            return
        pos = body_pos(self.env, self.body_name)
        if pos_out_of_kitchen(
            pos, aabb_min=self.kitchen_aabb_min, aabb_max=self.kitchen_aabb_max
        ):
            self.out_of_bounds = True
        tilt = self._tilt_deg()
        self._update_exposure(pos[:2])
        if self.event is None:
            if self._fell(float(pos[2])):
                self.event = "fall"
                self.t_event = int(t)
            elif tilt is not None and is_toppled(tilt, threshold_deg=self.event_tilt_deg):
                self.event = "topple"
                self.t_event = int(t)
        if int(t) % int(self.step_log_stride) == 0:
            self._steps.append(
                {
                    "t": int(t),
                    "obj_z": float(pos[2]),
                    "tilt_deg": None if tilt is None else float(tilt),
                    "beside_gap": self._beside_gap_now,
                    "near_target": bool(self._near_target),
                    "in_contact": bool(self.robot_contact),
                    "search_attempt": self._search_attempt,
                }
            )

    def note_robot_contact(self, in_contact: bool) -> None:
        if in_contact:
            self.robot_contact = True

    def finalize_labels(self, *, success: bool) -> None:
        self._success = bool(success)

    def extra_fields(self) -> dict[str, Any]:
        exposure = {
            "fail_mode": self.fail_mode,
            "beside_gap": self._beside_gap_now,
            "near_target": bool(self._near_target),
            "near_sink": bool(self._near_sink),
            "near_stove": bool(self._near_stove),
            "anchor_name": self.anchor_name,
        }
        has_exposure = bool(self.enabled and (self._near_target or self._near_sink or self._near_stove))
        role = classify_sample_role(
            spawn_failed=self.spawn_failed,
            out_of_bounds=self.out_of_bounds,
            event=self.event,
            t_event=self.t_event,
            min_event_step=self.min_event_step,
            success=self._success,
            has_exposure=has_exposure,
        )
        hazard = {
            "enabled": bool(self.enabled),
            "alpha": float(self.alpha),
            "event": self.event,
            "event_a": event_a_for(event=self.event, robot_contact=self.robot_contact),
            "t_event": self.t_event,
            "exposure": exposure,
            "spawn_failed": bool(self.spawn_failed),
            "sample_role": role,
            "pnp_track": self.track,
            "fail_mode": self.fail_mode,
            "object_name": self.object_name,
            "anchor_name": self.anchor_name,
            "robot_contact": bool(self.robot_contact),
            "out_of_bounds": bool(self.out_of_bounds),
            "eef_xy": None if self._eef_xy is None else [float(x) for x in self._eef_xy],
            "obj_xy": None if self._anchor_xy is None else [float(x) for x in self._anchor_xy],
            "planned_xy": None if self._planned_xy is None else [float(x) for x in self._planned_xy],
            "placed_xy": None if self._placed_xy is None else [float(x) for x in self._placed_xy],
            "search_attempt": self._search_attempt,
            "penetrated_then_moved": bool(self._penetrated_then_moved),
            "gap_slid": bool(self._gap_slid),
            "placement_mode": self._placement_mode,
            "min_surface_gap": float(self.min_surface_gap),
            "settle_steps": int(self.settle_steps),
            "settle_tilt_deg": float(self.settle_tilt_deg),
            "steps": list(self._steps),
        }
        return {"spawn_failed": bool(self.spawn_failed), "hazard": hazard}

    def _tilt_deg(self) -> float | None:
        if self.body_name is None:
            return None
        quat = body_quat_wxyz(self.env, self.body_name)
        return float(tilt_deg_from_body_quat(quat, ref_quat=self._ref_quat))

    def _fell(self, z: float) -> bool:
        if self._z0 is None:
            return False
        return float(z) < float(self._z0) - float(self.z_drop_margin)

    def _update_exposure(self, xy: np.ndarray) -> None:
        if self._anchor_xy is None or self._hazard_radius is None or self._anchor_radius is None:
            return
        dist = float(np.linalg.norm(xy - self._anchor_xy))
        gap = surface_gap(
            center_dist=dist,
            target_radius=self._anchor_radius,
            hazard_radius=self._hazard_radius,
        )
        self._beside_gap_now = gap
        self._near_target = bool(gap <= self.exposure_gap_max)
        if self.track == "sink_beside":
            self._near_sink = bool(self._near_target)
        if self.track == "stove_beside":
            self._near_stove = bool(self._near_target)

    def _teleport(self, pos: np.ndarray, quat: np.ndarray) -> None:
        assert self.joint_name is not None
        set_free_joint_pose(self.env, self.joint_name, pos, quat)
        _write_placement(self.env, self.object_name, pos, quat)

    def _hide(self) -> None:
        if self.joint_name is None:
            return
        self._teleport(HIDDEN_POS, UPRIGHT_QUAT_WXYZ)

    def _read_eef_xy(self) -> np.ndarray | None:
        try:
            return read_eef_xyz(self.env)[:2]
        except Exception:
            return None

    def _episode_rng(self, episode_index: int) -> Any:
        rng = getattr(self.env, "rng", None)
        if rng is not None and hasattr(rng, "uniform"):
            return rng
        return np.random.default_rng(int(episode_index))

    def _surface_gap_ok(self, cand: np.ndarray) -> bool:
        if self._anchor_xy is None or self._anchor_radius is None or self._hazard_radius is None:
            return False
        gap = surface_gap(
            center_dist=float(np.linalg.norm(cand - self._anchor_xy)),
            target_radius=self._anchor_radius,
            hazard_radius=self._hazard_radius,
        )
        return gap + 1e-12 >= float(self.min_surface_gap)

    def _candidate_clear(self) -> bool:
        if hazard_has_penetration(
            self.env,
            obstacle_geom_ids=self.obstacle_geom_ids,
            skip_geom_ids=self._support_geom_ids,
            eps=self.penetration_eps,
        ):
            return False
        if not has_support_contact(
            self.env,
            obstacle_geom_ids=self.obstacle_geom_ids,
            support_geom_ids=self._support_geom_ids,
        ):
            return False
        return True

    def _snapshot_state(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self.env.sim.data.qpos, dtype=np.float64).copy(),
            np.asarray(self.env.sim.data.qvel, dtype=np.float64).copy(),
        )

    def _restore_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        import mujoco

        self.env.sim.data.qpos[:] = qpos
        self.env.sim.data.qvel[:] = qvel
        mujoco.mj_forward(_raw_model(self.env.sim), _raw_data(self.env.sim))

    def _hold_non_hazard(self, snap_qpos: np.ndarray, snap_qvel: np.ndarray) -> None:
        assert self.joint_name is not None
        model = self.env.sim.model
        jid = int(model.joint_name2id(self.joint_name))
        qadr = int(model.jnt_qposadr[jid])
        dadr = int(model.jnt_dofadr[jid])
        haz_q = np.asarray(self.env.sim.data.qpos[qadr : qadr + 7], dtype=np.float64).copy()
        haz_v = np.asarray(self.env.sim.data.qvel[dadr : dadr + 6], dtype=np.float64).copy()
        self.env.sim.data.qpos[:] = snap_qpos
        self.env.sim.data.qvel[:] = snap_qvel
        self.env.sim.data.qpos[qadr : qadr + 7] = haz_q
        self.env.sim.data.qvel[dadr : dadr + 6] = haz_v

    def _settle_upright(self, snap_qpos: np.ndarray, snap_qvel: np.ndarray) -> bool:
        import mujoco

        assert self.body_name is not None
        if self.settle_steps < 1:
            return True
        z0 = float(body_pos(self.env, self.body_name)[2])
        raw_model = _raw_model(self.env.sim)
        raw_data = _raw_data(self.env.sim)
        for _ in range(int(self.settle_steps)):
            mujoco.mj_step(raw_model, raw_data)
            self._hold_non_hazard(snap_qpos, snap_qvel)
            mujoco.mj_forward(raw_model, raw_data)
        z = float(body_pos(self.env, self.body_name)[2])
        if z < z0 - float(self.z_drop_margin):
            return False
        tilt = self._tilt_deg()
        if tilt is not None and is_toppled(tilt, threshold_deg=self.settle_tilt_deg):
            return False
        if not has_support_contact(
            self.env,
            obstacle_geom_ids=self.obstacle_geom_ids,
            support_geom_ids=self._support_geom_ids,
        ):
            return False
        pos = body_pos(self.env, self.body_name)
        quat = body_quat_wxyz(self.env, self.body_name)
        _write_placement(self.env, self.object_name, pos, quat)
        return True

    def _place_hazard(self, episode_index: int) -> bool:
        assert self.body_name is not None
        anchor_body = object_body_name(self.env, self.anchor_name)
        if anchor_body is None:
            self._hide()
            return False
        eef_xy = self._read_eef_xy()
        if eef_xy is None:
            self._hide()
            return False
        obj_xy = body_pos(self.env, anchor_body)[:2]
        anchor_r = object_xy_radius(self.env, anchor_body)
        hazard_r = object_xy_radius(self.env, self.body_name)
        if anchor_r is None or hazard_r is None:
            self._hide()
            return False
        rng = self._episode_rng(episode_index)
        candidates, gap_slid = placement_candidate_xys(
            eef_xy,
            obj_xy,
            self.alpha,
            target_radius=anchor_r,
            hazard_radius=hazard_r,
            min_gap=self.min_surface_gap,
            n_attempts=self.n_attempts,
            search_radius_m=self.search_radius_m,
            rng=rng,
        )
        line = line_xy_with_surface_gap(
            eef_xy,
            obj_xy,
            self.alpha,
            target_radius=anchor_r,
            hazard_radius=hazard_r,
            min_gap=self.min_surface_gap,
        )
        pos0 = body_pos(self.env, self.body_name)
        quat = body_quat_wxyz(self.env, self.body_name)
        snap_qpos, snap_qvel = self._snapshot_state()
        self._eef_xy = eef_xy.copy()
        self._anchor_xy = obj_xy.copy()
        self._planned_xy = candidates[0].copy()
        self._gap_slid = bool(gap_slid)
        self._anchor_radius = float(anchor_r)
        self._hazard_radius = float(hazard_r)
        sink = getattr(self.env, "sink", None)
        stove = getattr(self.env, "stove", None)
        self._sink_xy = _fixture_xy(self.env, sink)
        self._stove_xy = _fixture_xy(self.env, stove)
        for attempt, cand in enumerate(candidates):
            self._restore_state(snap_qpos, snap_qvel)
            if not self._surface_gap_ok(cand):
                continue
            pos = np.array([float(cand[0]), float(cand[1]), float(pos0[2])], dtype=np.float64)
            self._teleport(pos, quat)
            if not self._candidate_clear():
                continue
            if not self._settle_upright(snap_qpos, snap_qvel):
                continue
            placed = body_pos(self.env, self.body_name)[:2]
            self._placed_xy = placed.copy()
            self._search_attempt = int(attempt)
            self._penetrated_then_moved = bool(attempt > 0)
            if line is not None and attempt == 0:
                self._placement_mode = "line"
            else:
                self._placement_mode = "ring"
            self._update_exposure(placed)
            return True
        self._restore_state(snap_qpos, snap_qvel)
        self._hide()
        return False


def build_pnp_controller(
    level: int,
    handle: RoboCasaEnvHandle,
    extra_robocasa: Mapping[str, Any],
    *,
    episode_index: int = 0,
) -> PnPHazardController:
    if int(level) != 0:
        raise ValueError(f"robocasa_pnp only allows level 0, got {level}")
    pnp_cfg = require_pnp_block(extra_robocasa)
    track = handle.extra.get("pnp_track")
    if not isinstance(track, str) or track not in PNP_TRACKS:
        raise ValueError(
            f"handle.extra['pnp_track'] must be one of {sorted(PNP_TRACKS)}, got {track!r}"
        )
    ctrl = PnPHazardController(handle, track=track, pnp_cfg=pnp_cfg)
    ctrl.reset(int(episode_index))
    return ctrl
