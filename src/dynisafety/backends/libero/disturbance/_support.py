"""Shared MuJoCo helpers for LIBERO disturbance controllers."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from dynisafety.backends._engine.mujoco_contact import (
    collect_geom_ids_for_bodies,
    expand_body_descendants,
)
from dynisafety.backends.libero.env import (
    LiberoEnvHandle,
    _is_base_name,
    _is_visible_xy,
    _root_body_xy,
)
from dynisafety.core.disturbance import (
    Phase,
    arm_quat_from_yaw,
    minimum_jerk,
)
from dynisafety.core.disturbance import compute_approach_steps as compute_approach_steps

RESTING_Z_MODE_SETTLE = "settle"
RESTING_Z_MODE_ANCHOR = "anchor"
VALID_RESTING_Z_MODES = (RESTING_Z_MODE_SETTLE, RESTING_Z_MODE_ANCHOR)


@dataclass(frozen=True)
class HandPhysicsParams:
    mass_kg: float
    contact_solref: Sequence[float]
    contact_solimp: Sequence[float]
    contact_friction: Sequence[float]
    near_eef_m: float
    near_eef_max_step_m: float
    in_contact_max_step_m: float
    big_jump_m: float


class DisturbanceController(Protocol):
    def reset(self, episode_index: int) -> None: ...
    def step(self, t: int, obs: Mapping[str, Any]) -> None: ...

    @property
    def disturbance_onset_step(self) -> int | None: ...

    @property
    def active_obstacle_name(self) -> str | None: ...


def render_env(handle: LiberoEnvHandle):
    return handle.env


def _is_pedestal_body_name(name) -> bool:
    """Whether a body is a decorative pedestal (``box_base`` /
    ``box_small_base``) that — although technically free-jointed —
    should still be eligible as a support surface for
    ``_scene_persistent_support_z``.

    These pedestals are authored at episode init specifically to host
    other obstacles on top of them (e.g. ``wine_bottle_obstacle_1``
    sitting on ``box_small_base_1`` at the same XY).  Treating them as
    "persistent" support during the geometric anchor lookup matches
    that authoring intent — without this exception the obstacle would
    be anchored to the floor and visually intersect the pedestal's
    side wall, then topple as physics resolves the contact.
    """
    if not name:
        return False
    nl = name.lower()
    return ("box_base" in nl or "box_small_base" in nl) and "hand" not in nl


def _body_is_world_fixed(model, bid: int) -> bool:
    """True iff the body is rigidly welded to the worldbody.

    A body is "world-fixed" only when **it and every ancestor up to the
    worldbody** have *no joints at all*.  Any joint along the chain (free
    *or* hinge / slide / ball — i.e. articulated robot links, cabinet
    doors, drawers, the obstacle's own free joint, …) means the body can
    move during simulation and must not be treated as a stable support
    surface for ``_scene_persistent_support_z``.
    """
    if bid <= 0:
        return True  # worldbody itself
    cur = int(bid)
    while cur > 0:
        if int(model.body_jntnum[cur]) > 0:
            return False
        cur = int(model.body_parentid[cur])
    return True


def _body_is_movable(model, bid: int) -> bool:
    """Convenience inverse of ``_body_is_world_fixed``."""
    return not _body_is_world_fixed(model, bid)


def _geom_world_z_extent(model, data, gid: int):
    """Return (top_z, bottom_z) of geom ``gid`` in world coordinates.

    Exact for box / sphere / plane.  Conservative bounding-sphere fallback for
    cylinder / capsule / mesh / ellipsoid (slightly over-estimates the extent,
    which is the safe direction for a "find the supporting surface" query).
    """
    import mujoco

    gtype = int(model.geom_type[gid])
    gpos = data.geom_xpos[gid]
    gxmat = data.geom_xmat[gid].reshape(3, 3)
    gsize = model.geom_size[gid]
    z = float(gpos[2])

    if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        ext = float(
            abs(gxmat[2, 0]) * gsize[0] + abs(gxmat[2, 1]) * gsize[1] + abs(gxmat[2, 2]) * gsize[2]
        )
        return z + ext, z - ext

    if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        r = float(gsize[0])
        return z + r, z - r

    if gtype == int(mujoco.mjtGeom.mjGEOM_PLANE):
        # MuJoCo planes are infinite in XY; treat as a flat sheet at gpos.z.
        return z, z

    # Cylinder / capsule / mesh / ellipsoid: use bounding sphere fallback.
    try:
        rbound = float(model.geom_rbound[gid])
    except Exception:
        rbound = 0.0
    if rbound <= 0:
        rbound = float(max(gsize)) if len(gsize) else 0.0
    return z + rbound, z - rbound


def _geom_xy_radius(model, data, gid: int) -> float:
    """Conservative upper bound on the geom's XY half-extent at gpos.

    Used to decide whether a candidate support-surface geom actually
    extends under the obstacle's drop XY.  We use the bounding-sphere
    radius (``geom_rbound``) when available — that's an over-estimate of
    the true XY footprint, which is the safe direction (false positives
    here just allow more candidates; false negatives would discard real
    surfaces).
    """
    try:
        rb = float(model.geom_rbound[gid])
        if rb > 0:
            return rb
    except Exception:
        pass
    # Fallback: largest dimension of the geom_size half-extents.
    gsize = model.geom_size[gid]
    if len(gsize) == 0:
        return 0.0
    return float(max(abs(s) for s in gsize))


def _scene_persistent_support_z(sim, hint_z: float, target_xy=None, slack: float = 0.10):
    """Find a stable support-surface Z near ``hint_z`` (and ``target_xy``).

    Scans the scene for the *highest* world-fixed (no joints anywhere
    along its parent chain) collision geom whose top is at or below
    ``hint_z + slack`` and whose XY footprint plausibly covers
    ``target_xy``.  This deliberately ignores:

    * Robot links / drawer doors / cabinet doors (they have hinge/slide
      joints, so they're not world-fixed) — otherwise the gripper or arm
      can be picked as the "support" and the obstacle ends up dangling
      mid-air.
    * Free-jointed task / obstacle objects (they move) — same reason.

    The optional ``table_top`` site (present in main-table scenes) is
    treated as one *additional* candidate alongside the geom scan so
    that cabinet-shelf scenes (where ``hint_z`` is well above the table
    top) can still pick the cabinet shelf rather than blindly anchoring
    to the table.  The "highest below ``hint_z + slack``" rule then
    naturally selects the right surface in:

    * floor scenes (only ``floor`` plane is < ``hint_z``);
    * tabletop scenes (table-top geom wins over floor);
    * cabinet-on-table scenes (cabinet shelf wins over table top);
    * data-quality outliers where ``hint_z`` is implausibly high
      (no fixture authored at that height) — we trust the *geometry*
      and snap to the highest real surface, rather than leaving the
      obstacle suspended in the air.

    Falls back to ``hint_z`` only when the scan finds **no** suitable
    candidate at all (extremely rare — would require a scene with no
    world-fixed collision geom anywhere below ``hint_z + slack`` whose
    footprint covers ``target_xy``).

    Args:
        sim: robosuite ``MjSim`` wrapper (``sim.model`` / ``sim.data``).
        hint_z: caller's expected support Z (typically the obstacle's
            ``pruned_init_z``).  Surfaces above this + ``slack`` are
            skipped.
        target_xy: optional ``(x, y)`` of the obstacle's drop XY.  When
            provided, candidate geoms must lie within their own XY-radius
            of this point (i.e. their footprint must plausibly extend
            under the obstacle).
        slack: how far above ``hint_z`` a surface is still allowed to be
            picked, to absorb authoring noise (default 0.10 m).

    Returns:
        ``(support_z, source_label)``.  ``source_label`` is a short
        string for logging, e.g. ``"table_top_site@z=0.900"`` or
        ``"fixture_geom('cabinet_shelf_col')@z=0.987"``.  If no suitable
        fixture is found, falls back to ``hint_z`` with label
        ``"fallback(no_fixture_below_hint)"``.
    """
    import mujoco

    txy = None
    if target_xy is not None:
        txy = (float(target_xy[0]), float(target_xy[1]))

    upper = float(hint_z) + float(slack)
    candidates = []  # list of (top_z, source_label)

    # Candidate 0: the dedicated ``table_top`` site (when present).  No
    # XY-coverage check — it's a workspace designation marker, not a
    # collision geom, so its 1 mm size doesn't tell us its footprint.
    try:
        sid = sim.model.site_name2id("table_top")
        z_site = float(sim.data.site_xpos[sid][2])
        if z_site <= upper:
            candidates.append((z_site, "table_top_site"))
    except (KeyError, ValueError, Exception):
        pass

    # Candidates 1..N: scan of all world-fixed collision geoms.
    # Decorative pedestals (`box_base` / `box_small_base`) are special-
    # cased through the world-fixed gate: although they have a free
    # joint, SafeLIBERO authors them to serve as static supports for
    # other obstacles on top, so we want them to win over the floor /
    # table when an obstacle's target XY happens to coincide with a
    # pedestal's XY.
    for gid in range(int(sim.model.ngeom)):
        bid = int(sim.model.geom_bodyid[gid])
        is_pedestal = False
        if not _body_is_world_fixed(sim.model, bid):
            try:
                bname = mujoco.mj_id2name(sim.model._model, int(mujoco.mjtObj.mjOBJ_BODY), bid)
            except Exception:
                bname = None
            if not _is_pedestal_body_name(bname):
                continue
            is_pedestal = True
        # Skip pure-visual geoms (no collision footprint).
        if int(sim.model.geom_contype[gid]) == 0 and int(sim.model.geom_conaffinity[gid]) == 0:
            continue
        top_z, _ = _geom_world_z_extent(sim.model, sim.data, gid)
        if top_z > upper:
            continue
        # XY coverage: target_xy must lie within the geom's bounding
        # radius around its centre (over-approximate; safe direction).
        if txy is not None:
            gpos = sim.data.geom_xpos[gid]
            dx = float(gpos[0]) - txy[0]
            dy = float(gpos[1]) - txy[1]
            if (dx * dx + dy * dy) ** 0.5 > _geom_xy_radius(sim.model, sim.data, gid):
                continue
        try:
            gname = mujoco.mj_id2name(sim.model._model, int(mujoco.mjtObj.mjOBJ_GEOM), int(gid))
        except Exception:
            gname = None
        prefix = "pedestal_geom" if is_pedestal else "fixture_geom"
        label = f"{prefix}('{gname}')" if gname else f"{prefix}_{gid}"
        candidates.append((top_z, label))

    if not candidates:
        return float(hint_z), "fallback(no_fixture_below_hint)"

    # Pick the highest candidate (closest to hint_z from below).
    # No "max_drop_below_hint" safety net any more: in practice
    # ``hint_z`` (= ``pruned_init_z``) can be authored anywhere — even
    # in mid-air for some SafeLIBERO data outliers — and trusting it
    # over the actual scene geometry leaves the obstacle suspended.
    # Trust the geometry: snap to the highest real surface beneath.
    best_z, best_label = max(candidates, key=lambda c: c[0])
    return float(best_z), f"{best_label}@z={best_z:.3f}"


def _compute_body_to_bottom_offset(
    sim, joint_name: str, dof_addr: int, body_id: int, geom_ids, quat
) -> float:
    """How far the obstacle's lowest collision-geom point is below body origin
    when the obstacle is oriented at ``quat``.

    Temporarily teleports the obstacle to a known, off-screen pose to read
    geom world positions; full ``qpos`` / ``qvel`` is saved and restored so
    the caller's simulation state is untouched.
    """
    import mujoco

    saved_qpos = sim.data.qpos.copy()
    saved_qvel = sim.data.qvel.copy()
    try:
        ref_pos = np.array([5.0, 5.0, 1.0], dtype=np.float64)
        sim.data.set_joint_qpos(
            joint_name, np.concatenate([ref_pos, np.asarray(quat, dtype=np.float64)])
        )
        sim.data.qvel[dof_addr : dof_addr + 6] = 0.0
        mujoco.mj_forward(sim.model._model, sim.data._data)

        body_z = float(sim.data.body_xpos[body_id][2])
        min_z = None
        for gid in geom_ids:
            # Skip pure-visual geoms (group=1, no collisions).
            if int(sim.model.geom_contype[gid]) == 0 and int(sim.model.geom_conaffinity[gid]) == 0:
                continue
            _, bottom_z = _geom_world_z_extent(sim.model, sim.data, gid)
            if min_z is None or bottom_z < min_z:
                min_z = bottom_z
        if min_z is None:
            return 0.0
        return body_z - min_z
    finally:
        sim.data.qpos[:] = saved_qpos
        sim.data.qvel[:] = saved_qvel
        mujoco.mj_forward(sim.model._model, sim.data._data)


def detect_visible_obstacles(env):
    """Return obstacle objects that are currently visible in the scene.

    SafeLIBERO hides inactive obstacles by teleporting them far off-screen
    (e.g. x=±10 or y=±10).  An obstacle is considered visible only when
    both |x| < 5 and |y| < 5.

    Returns:
        list of (name, pos, quat) tuples for each visible obstacle.
        ``pos``/``quat`` are read from the obstacle free-joint qpos when
        available, so the pose can be written back via ``set_joint_qpos``
        without frame mismatch.
    """
    inner = env.env
    sim = inner.sim
    visible = []
    for name, obj in inner.objects_dict.items():
        if "obstacle" not in name.lower():
            continue
        if "hand" in name.lower():
            continue
        body_id = sim.model.body_name2id(obj.root_body)
        body_pos = sim.data.body_xpos[body_id].copy()
        if abs(body_pos[0]) < 5.0 and abs(body_pos[1]) < 5.0:
            # Use free-joint world pose (qpos layout [x, y, z, qw, qx, qy, qz])
            # so later set_joint_qpos(...) writes back to the same frame.
            try:
                joint_name = obj.joints[-1]
                qpos = sim.data.get_joint_qpos(joint_name).copy()
                pos = qpos[:3]
                quat = qpos[3:7]
            except Exception:
                # Fallback for non-free-joint objects.
                pos = body_pos
                quat = sim.data.body_xquat[body_id].copy()
            visible.append((name, pos, quat))
    return visible


def clear_xy_from_task_objects(
    xy: np.ndarray,
    obstacle_radius: float,
    tasks: Sequence[tuple[str, np.ndarray, float]],
    margin: float,
    *,
    max_iters: int = 8,
) -> tuple[np.ndarray, list[str]]:
    """Push ``xy`` away from task objects until geom radii + ``margin`` clear.

    ``tasks`` is ``(name, task_xy, task_radius)``. Direction is the existing
    obstacle→task vector (SafeLIBERO's authored offset); exact overlap uses +x.
    Returns the (possibly unchanged) XY and human-readable nudge notes.
    """
    out = np.asarray(xy, dtype=np.float64).reshape(-1)[:2].copy()
    notes: list[str] = []
    r_obs = max(0.0, float(obstacle_radius))
    extra = max(0.0, float(margin))
    for _ in range(int(max_iters)):
        moved = False
        for name, task_xy, task_radius in tasks:
            txy = np.asarray(task_xy, dtype=np.float64).reshape(-1)[:2]
            need = r_obs + max(0.0, float(task_radius)) + extra
            delta = out - txy
            dist = float(np.hypot(delta[0], delta[1]))
            if dist >= need:
                continue
            if dist < 1e-6:
                ux, uy = 1.0, 0.0
            else:
                ux, uy = float(delta[0] / dist), float(delta[1] / dist)
            new = np.array([txy[0] + ux * need, txy[1] + uy * need], dtype=np.float64)
            notes.append(
                f"task={name!r} {dist:.3f}m -> {need:.3f}m "
                f"(r_obs={r_obs:.3f} r_task={float(task_radius):.3f} "
                f"margin={extra:.3f})"
            )
            out = new
            moved = True
        if not moved:
            break
    return out, notes


def _geom_is_collision(model: Any, gid: int) -> bool:
    return not (int(model.geom_contype[gid]) == 0 and int(model.geom_conaffinity[gid]) == 0)


def _geom_collision_xy_half_extent(model: Any, data: Any, gid: int) -> float:
    """Tight XY half-extent of a collision geom (not the 3D bounding sphere)."""
    import mujoco

    gtype = int(model.geom_type[gid])
    gsize = model.geom_size[gid]
    gxmat = data.geom_xmat[gid].reshape(3, 3)
    if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        hx = abs(gxmat[0, 0]) * gsize[0] + abs(gxmat[0, 1]) * gsize[1] + abs(gxmat[0, 2]) * gsize[2]
        hy = abs(gxmat[1, 0]) * gsize[0] + abs(gxmat[1, 1]) * gsize[1] + abs(gxmat[1, 2]) * gsize[2]
        return float(max(hx, hy))
    if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return float(gsize[0])
    if gtype in (
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
    ):
        return float(gsize[0])
    if len(gsize) >= 2:
        return float(max(abs(float(gsize[0])), abs(float(gsize[1]))))
    if len(gsize) == 1:
        return float(abs(gsize[0]))
    return 0.0


def object_subtree_geom_ids(sim: Any, root_body_id: int) -> set[int]:
    """All geoms on a free-joint object (root + every descendant body)."""
    return collect_geom_ids_for_bodies(
        sim.model, expand_body_descendants(sim.model, [int(root_body_id)])
    )


def collision_xy_radius(
    sim: Any,
    body_id: int,
    geom_ids: set[int] | None = None,
) -> float:
    """XY radius of a body's collision geoms about the body origin."""
    body_xy = np.asarray(sim.data.body_xpos[int(body_id)][:2], dtype=np.float64)
    model, data = sim.model, sim.data
    best = 0.0
    allowed = None if geom_ids is not None else expand_body_descendants(model, [int(body_id)])
    if geom_ids is None:
        gids = range(int(model.ngeom))
    else:
        gids = geom_ids
    for gid in gids:
        gid_i = int(gid)
        bid = int(model.geom_bodyid[gid_i])
        if allowed is not None and bid not in allowed:
            continue
        if not _geom_is_collision(model, gid_i):
            continue
        gxy = np.asarray(data.geom_xpos[gid_i][:2], dtype=np.float64)
        cand = float(np.linalg.norm(gxy - body_xy)) + _geom_collision_xy_half_extent(
            model, data, gid_i
        )
        if cand > best:
            best = cand
    return best


def task_object_xy_radii(
    env: Any, task_object_names: Sequence[str]
) -> list[tuple[str, np.ndarray, float]]:
    """Visible BDDL task objects as ``(name, xy, collision_xy_radius)``."""
    inner = env.env
    sim = inner.sim
    out: list[tuple[str, np.ndarray, float]] = []
    for name in task_object_names:
        obj = inner.objects_dict.get(name)
        if obj is None:
            continue
        try:
            body_id = sim.model.body_name2id(obj.root_body)
        except Exception:
            continue
        pos = sim.data.body_xpos[body_id]
        if abs(float(pos[0])) >= 5.0 or abs(float(pos[1])) >= 5.0:
            continue
        out.append(
            (
                str(name),
                np.asarray(pos[:2], dtype=np.float64).copy(),
                collision_xy_radius(sim, int(body_id)),
            )
        )
    return out


def companion_pedestal_name(
    handle: LiberoEnvHandle,
    obstacle_xy: np.ndarray,
    overlap_radius: float,
) -> str | None:
    """Visible pedestal whose XY coincides with the authored obstacle pose."""
    inner = handle.env.env
    sim = inner.sim
    src = np.asarray(obstacle_xy, dtype=np.float64).reshape(-1)[:2]
    best_name: str | None = None
    best_d = float(overlap_radius)
    for name, obj in inner.objects_dict.items():
        if not _is_base_name(name):
            continue
        try:
            xy = _root_body_xy(sim, obj)
        except Exception:
            continue
        if not _is_visible_xy(xy):
            continue
        d = float(np.linalg.norm(xy - src))
        if d < best_d:
            best_d = d
            best_name = str(name)
    return best_name


def teleport_free_joint_xy(handle: LiberoEnvHandle, object_name: str, xy: np.ndarray) -> None:
    """Rewrite a free-joint object's XY, keeping Z and quaternion."""
    import mujoco

    inner = handle.env.env
    sim = inner.sim
    obj = inner.objects_dict[object_name]
    joint_name = obj.joints[-1]
    qpos = sim.data.get_joint_qpos(joint_name).copy()
    qpos[0] = float(xy[0])
    qpos[1] = float(xy[1])
    sim.data.set_joint_qpos(joint_name, qpos)
    jid = sim.model.joint_name2id(joint_name)
    dof = sim.model.jnt_dofadr[jid]
    sim.data.qvel[dof : dof + 6] = 0.0
    mujoco.mj_forward(sim.model._model, sim.data._data)


def hide_free_joint_object(handle: LiberoEnvHandle, object_name: str) -> None:
    """Teleport a free-joint object off-screen (same pose as ``hide_all_obstacles``)."""
    import mujoco

    inner = handle.env.env
    sim = inner.sim
    obj = inner.objects_dict.get(object_name)
    if obj is None:
        raise KeyError(f"unknown object {object_name!r}")
    sim.data.set_joint_qpos(
        obj.joints[-1],
        np.array([10.0, 10.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )
    jid = sim.model.joint_name2id(obj.joints[-1])
    dof = sim.model.jnt_dofadr[jid]
    sim.data.qvel[dof : dof + 6] = 0.0
    mujoco.mj_forward(sim.model._model, sim.data._data)


NUDGE_PENETRATION_EPS = 1e-4
_TABLE_FLOOR_TOKENS = ("table", "floor", "ground")
# Whole-word only: "stable" / "vegetable" must not match "table".
_TABLE_FLOOR_TOKEN_RE = re.compile(
    r"(?<![a-z])(?:" + "|".join(re.escape(t) for t in _TABLE_FLOOR_TOKENS) + r")(?![a-z])"
)


def geom_looks_like_table_or_floor(name: str) -> bool:
    """True for arena table / floor / ground names (not props or pedestals).

    Tokens must be complete words (bounded by non-letters) so asset names
    that merely contain those letters — e.g. ``stable``, ``vegetable`` —
    are not treated as table/floor geometry.
    """
    return _TABLE_FLOOR_TOKEN_RE.search((name or "").lower()) is not None


def _geom_name(sim: Any, gid: int) -> str:
    try:
        return str(sim.model.geom_id2name(int(gid)) or "")
    except Exception:
        return ""


def _body_name_for_geom(sim: Any, gid: int) -> str:
    bid = int(sim.model.geom_bodyid[int(gid)])
    try:
        return str(sim.model.body_id2name(bid) or "")
    except Exception:
        return ""


def _geom_is_table_or_floor(sim: Any, gid: int) -> bool:
    """Skip sitting contacts against the table, floor, or infinite planes."""
    import mujoco

    if geom_looks_like_table_or_floor(_geom_name(sim, gid)):
        return True
    if geom_looks_like_table_or_floor(_body_name_for_geom(sim, gid)):
        return True
    try:
        if int(sim.model.geom_type[int(gid)]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            return True
    except Exception:
        pass
    return False


def contacts_show_penetration(
    contacts: Sequence[tuple[int, int, float]],
    obstacle_gids: set[int],
    task_gids: set[int],
    *,
    eps: float = NUDGE_PENETRATION_EPS,
) -> bool:
    """True when an obstacle↔task contact has ``dist < -eps`` (mesh interpenetration)."""
    if not obstacle_gids or not task_gids:
        return False
    thresh = -float(eps)
    for g1, g2, dist in contacts:
        paired = (g1 in obstacle_gids and g2 in task_gids) or (
            g2 in obstacle_gids and g1 in task_gids
        )
        if paired and float(dist) < thresh:
            return True
    return False


def named_object_geom_ids(handle: LiberoEnvHandle, object_names: Sequence[str]) -> set[int]:
    """Collision+visual geom IDs for named free-joint objects (full subtree)."""
    inner = handle.env.env
    sim = inner.sim
    gids: set[int] = set()
    for name in object_names:
        obj = inner.objects_dict.get(name)
        if obj is None:
            continue
        try:
            body_id = int(sim.model.body_name2id(obj.root_body))
        except Exception:
            continue
        gids |= object_subtree_geom_ids(sim, body_id)
    return gids


def _iter_sim_contacts(sim: Any) -> list[tuple[int, int, float]]:
    out: list[tuple[int, int, float]] = []
    for i in range(int(sim.data.ncon)):
        contact = sim.data.contact[i]
        out.append((int(contact.geom1), int(contact.geom2), float(contact.dist)))
    return out


def visible_other_object_names(handle: LiberoEnvHandle, stack_names: Sequence[str]) -> list[str]:
    """Visible ``objects_dict`` props excluding the obstacle stack / table / floor."""
    inner = handle.env.env
    sim = inner.sim
    skip = {str(n) for n in stack_names}
    out: list[str] = []
    for name, obj in inner.objects_dict.items():
        if name in skip:
            continue
        nl = str(name).lower()
        if geom_looks_like_table_or_floor(str(name)):
            continue
        if "hand_pusher" in nl:
            continue
        try:
            xy = _root_body_xy(sim, obj)
        except Exception:
            continue
        if not _is_visible_xy(xy):
            continue
        out.append(str(name))
    return out


def obstacle_xy_penetrates_task_objects(
    handle: LiberoEnvHandle,
    obstacle_name: str,
    obstacle_xy: np.ndarray,
    task_object_names: Sequence[str],
    *,
    eps: float = NUDGE_PENETRATION_EPS,
) -> bool:
    """Probe landing XY for MuJoCo penetration; restore the free-joint pose after."""
    import mujoco

    inner = handle.env.env
    sim = inner.sim
    obj = inner.objects_dict.get(obstacle_name)
    if obj is None:
        raise KeyError(f"unknown obstacle {obstacle_name!r}")
    joint_name = obj.joints[-1]
    saved_qpos = np.asarray(sim.data.get_joint_qpos(joint_name), dtype=np.float64).copy()
    jid = sim.model.joint_name2id(joint_name)
    dof = int(sim.model.jnt_dofadr[jid])
    saved_qvel = np.asarray(sim.data.qvel[dof : dof + 6], dtype=np.float64).copy()
    try:
        teleport_free_joint_xy(handle, obstacle_name, obstacle_xy)
        return contacts_show_penetration(
            _iter_sim_contacts(sim),
            named_object_geom_ids(handle, (obstacle_name,)),
            named_object_geom_ids(handle, task_object_names),
            eps=eps,
        )
    finally:
        sim.data.set_joint_qpos(joint_name, saved_qpos)
        sim.data.qvel[dof : dof + 6] = saved_qvel
        mujoco.mj_forward(sim.model._model, sim.data._data)


def obstacle_xy_penetrates_scene_objects(
    handle: LiberoEnvHandle,
    obstacle_name: str,
    obstacle_xy: np.ndarray,
    *,
    pedestal_overlap_radius: float,
    eps: float = NUDGE_PENETRATION_EPS,
) -> bool:
    """True iff the obstacle stack interpenetrates another scene object.

    The stack is the obstacle plus its companion pedestal (hollow stands
    clip pudding / milk even when the bottle body does not). Table, floor,
    and infinite planes are ignored so a sitting contact never counts as
    穿模. Nearby-but-not-clipped footprints (moka vs bowl) stay put.
    """
    import mujoco

    pedestal = companion_pedestal_name(handle, obstacle_xy, pedestal_overlap_radius)
    stack = [obstacle_name]
    if pedestal is not None:
        stack.append(pedestal)
    others = visible_other_object_names(handle, stack)
    if not others:
        return False
    inner = handle.env.env
    sim = inner.sim
    obj = inner.objects_dict.get(obstacle_name)
    if obj is None:
        raise KeyError(f"unknown obstacle {obstacle_name!r}")
    joint_name = obj.joints[-1]
    saved_qpos = np.asarray(sim.data.get_joint_qpos(joint_name), dtype=np.float64).copy()
    jid = sim.model.joint_name2id(joint_name)
    dof = int(sim.model.jnt_dofadr[jid])
    saved_qvel = np.asarray(sim.data.qvel[dof : dof + 6], dtype=np.float64).copy()
    try:
        teleport_free_joint_xy(handle, obstacle_name, obstacle_xy)
        stack_gids = named_object_geom_ids(handle, stack)
        other_gids = {
            gid
            for gid in named_object_geom_ids(handle, others)
            if not _geom_is_table_or_floor(sim, gid)
        }
        return contacts_show_penetration(
            _iter_sim_contacts(sim),
            stack_gids,
            other_gids,
            eps=eps,
        )
    finally:
        sim.data.set_joint_qpos(joint_name, saved_qpos)
        sim.data.qvel[dof : dof + 6] = saved_qvel
        mujoco.mj_forward(sim.model._model, sim.data._data)


def relocate_obstacle_clear_of_task_objects(
    handle: LiberoEnvHandle,
    obstacle_name: str,
    obstacle_xy: np.ndarray,
    task_object_names: Sequence[str],
    *,
    clearance_xy: float,
    pedestal_overlap_radius: float,
    require_penetration: bool = False,
) -> tuple[np.ndarray, str | None, list[str]]:
    """Nudge landing XY only when footprints overlap; do not move the pedestal.

    If the authored pose already clears ``r_obs + r_task + clearance_xy``,
    XY and the companion stand are left untouched (other suites / far
    trajectories stay as SafeLIBERO authored them). When a nudge does fire,
    the empty stand is hidden at its *original* XY — moving it with the
    landing clips clutter, and a table-anchored drop onto the stand topples.
    ``hide_overlapping_bases`` still runs afterwards for leftover clips.

    ``require_penetration``: skip the footprint circle unless MuJoCo reports
    the obstacle stack (obstacle + companion pedestal) interpenetrating
    another scene object at the authored XY. Table / floor sitting contacts
    do not count. Nearby-but-not-clipped landings stay put.
    """
    if require_penetration and not obstacle_xy_penetrates_scene_objects(
        handle,
        obstacle_name,
        obstacle_xy,
        pedestal_overlap_radius=pedestal_overlap_radius,
    ):
        return (
            np.asarray(obstacle_xy, dtype=np.float64).reshape(-1)[:2].copy(),
            None,
            [],
        )
    inner = handle.env.env
    sim = inner.sim
    obj = inner.objects_dict.get(obstacle_name)
    if obj is None:
        raise KeyError(f"unknown obstacle {obstacle_name!r}")
    body_id = sim.model.body_name2id(obj.root_body)
    r_obs = collision_xy_radius(sim, int(body_id))
    tasks = task_object_xy_radii(handle.env, task_object_names)
    new_xy, notes = clear_xy_from_task_objects(obstacle_xy, r_obs, tasks, clearance_xy)
    pedestal = companion_pedestal_name(handle, obstacle_xy, pedestal_overlap_radius)
    if notes and pedestal is not None:
        hide_free_joint_object(handle, pedestal)
    return new_xy, pedestal, notes


def select_l3_shared_reach_target(env, eef_pos: np.ndarray):
    """Pick reach target per ``human_hand_improvement_plan.md`` §2.2.

    1. Collect movable objects in the main workspace (|x| < 5, same as visible
       obstacles — excludes SafeLIBERO objects teleported off-screen).
    2. Exclude the human hand and every name listed in BDDL ``obj_of_interest``
       (task-relevant objects — not only the grammar heuristic).
    3. From the remainder, choose the one **closest to the current EEF**
       (proxy for “near the robot’s operating region” at episode start).
    4. If no object remains, **fallback** to the nearest **visible obstacle**
       (``detect_visible_obstacles``), preserving previous behaviour.

    Returns:
        ``(name, pos, quat, info)`` with ``info`` containing at least
        ``source`` ∈ ``{"non_task_object", "obstacle_fallback"}``, or
        ``None`` if there is no usable target at all.
    """
    inner = env.env
    sim = inner.sim

    raw_ooi = getattr(env, "obj_of_interest", None)
    if raw_ooi is None:
        raw_ooi = getattr(inner, "obj_of_interest", None)
    ooi = set(raw_ooi) if raw_ooi else set()

    candidates = []
    for name, obj in inner.objects_dict.items():
        nl = name.lower()
        if "hand_pusher" in nl:
            continue
        try:
            body_id = sim.model.body_name2id(obj.root_body)
        except Exception:
            continue
        pos = sim.data.body_xpos[body_id].copy()
        quat = sim.data.body_xquat[body_id].copy()
        if abs(pos[0]) >= 5.0 or abs(pos[1]) >= 5.0:
            continue
        if name in ooi:
            continue
        candidates.append((name, pos, quat))

    eef = np.asarray(eef_pos, dtype=np.float64)
    if candidates:
        candidates.sort(key=lambda item: float(np.linalg.norm(item[1] - eef)))
        name, pos, quat = candidates[0]
        dists = [(n, f"{float(np.linalg.norm(p - eef)):.3f}m") for n, p, _ in candidates]
        return (
            name,
            pos,
            quat,
            {
                "source": "non_task_object",
                "n_candidates": len(candidates),
                "dists": dists,
            },
        )

    visible_obs = detect_visible_obstacles(env)
    if not visible_obs:
        return None
    visible_obs.sort(key=lambda item: float(np.linalg.norm(item[1] - eef)))
    name, pos, quat = visible_obs[0]
    dists = [(n, f"{float(np.linalg.norm(p - eef)):.3f}m") for n, p, _ in visible_obs]
    return (
        name,
        pos,
        quat,
        {
            "source": "obstacle_fallback",
            "n_candidates": len(visible_obs),
            "dists": dists,
        },
    )


def get_task_object_positions(env, eef_pos: np.ndarray):
    """Return positions of BDDL ``obj_of_interest`` objects in the workspace.

    Returns:
        list of (name, pos) for each task-relevant object currently in the
        workspace (|x| < 5, |y| < 5).  Empty if none found.
    """
    inner = env.env
    sim = inner.sim

    raw_ooi = getattr(env, "obj_of_interest", None)
    if raw_ooi is None:
        raw_ooi = getattr(inner, "obj_of_interest", None)
    ooi = set(raw_ooi) if raw_ooi else set()
    if not ooi:
        return []

    results = []
    for name in ooi:
        obj = inner.objects_dict.get(name)
        if obj is None:
            continue
        try:
            body_id = sim.model.body_name2id(obj.root_body)
        except Exception:
            continue
        pos = sim.data.body_xpos[body_id].copy()
        if abs(pos[0]) >= 5.0 or abs(pos[1]) >= 5.0:
            continue
        results.append((name, pos))
    return results


_INSTANCE_SUFFIX = re.compile(r"_\d+$")
_PLACE_PUT = re.compile(r"\b(?:place|put)\b")
_DEST_PREP = re.compile(r"\b(?:on top of|to the (?:right|left) of|inside|into|onto|in|on)\b")


def object_name_aliases(name: str) -> list[str]:
    """Most-specific first: ``akita_black_bowl_1`` → black bowl / bowl."""
    stem = str(name).strip().lower().replace("__", "_")
    stem = _INSTANCE_SUFFIX.sub("", stem).replace("_", " ")
    stem = re.sub(r"\s+", " ", stem).strip()
    if not stem:
        return []
    tokens = stem.split()
    aliases: list[str] = []
    for i in range(len(tokens)):
        if tokens[i] in {"and", "or", "the", "a", "an"}:
            continue
        phrase = " ".join(tokens[i:])
        if phrase and phrase not in aliases:
            aliases.append(phrase)
    return aliases


def instruction_destination_spans(instruction: str) -> list[tuple[int, int]]:
    """Character spans after ``place/put … in/on/inside`` (receptacle side)."""
    text = str(instruction or "").lower()
    verbs = list(_PLACE_PUT.finditer(text))
    spans: list[tuple[int, int]] = []
    for i, verb in enumerate(verbs):
        clause_end = verbs[i + 1].start() if i + 1 < len(verbs) else len(text)
        clause = text[verb.end() : clause_end]
        prep = _DEST_PREP.search(clause)
        if prep is None:
            continue
        spans.append((verb.end() + prep.end(), clause_end))
    return spans


def _alias_match_index(instruction: str, alias: str) -> int | None:
    if not alias:
        return None
    found = re.search(rf"\b{re.escape(alias)}\b", instruction)
    return int(found.start()) if found else None


def select_instruction_first_task_object(
    task_objs: Sequence[tuple[str, np.ndarray]],
    instruction: str,
    eef_pos: np.ndarray,
) -> tuple[str, np.ndarray, dict[str, Any]] | None:
    """Pick the first manipulated OOI mentioned in the language instruction.

    Uses the longest alias that appears in ``instruction`` (so
    ``yellow_and_white_mug`` does not steal the earlier ``white mug``).
    Mentions that fall in a place/put destination span are skipped.
    Falls back to the OOI nearest the EEF when nothing matches.
    """
    if not task_objs:
        return None
    text = str(instruction or "").lower()
    dest = instruction_destination_spans(text)
    eef = np.asarray(eef_pos, dtype=np.float64).reshape(3)

    def _in_dest(idx: int) -> bool:
        return any(start <= idx < end for start, end in dest)

    scored: list[tuple[int, int, float, str, np.ndarray, str]] = []
    for name, pos in task_objs:
        chosen_alias = None
        chosen_idx = None
        for alias in object_name_aliases(name):
            idx = _alias_match_index(text, alias)
            if idx is not None:
                chosen_alias = alias
                chosen_idx = idx
                break
        if chosen_alias is None or chosen_idx is None:
            continue
        if _in_dest(chosen_idx):
            continue
        dist = float(np.linalg.norm(np.asarray(pos, dtype=np.float64).reshape(-1)[:3] - eef))
        scored.append((chosen_idx, -len(chosen_alias), dist, name, pos, chosen_alias))

    if scored:
        scored.sort()
        idx, _nlen, dist, name, pos, alias = scored[0]
        return (
            name,
            pos,
            {
                "source": "instruction_first",
                "alias": alias,
                "match_index": int(idx),
                "eef_dist": dist,
            },
        )

    nearest = min(
        task_objs,
        key=lambda item: float(
            np.linalg.norm(np.asarray(item[1], dtype=np.float64).reshape(-1)[:3] - eef)
        ),
    )
    name, pos = nearest
    return (
        name,
        pos,
        {
            "source": "nearest_fallback",
            "alias": None,
            "match_index": None,
            "eef_dist": float(
                np.linalg.norm(np.asarray(pos, dtype=np.float64).reshape(-1)[:3] - eef)
            ),
        },
    )


_ON_IN_ATOM = re.compile(
    r"\(\s*(On|In)\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\s*\)",
    re.I,
)
_REGION_TAILS = (
    "_contain_region",
    "_cook_region",
    "_top_region",
    "_init_region",
    "_top_side",
    "_region",
    "_side",
)
_OPEN_SKIP_SUBSTR = (
    "table",
    "floor",
    "stove",
    "wine_rack",
    "hand_pusher",
    "plate",
    "basket",
)


def parse_first_on_in(source: Any) -> tuple[str, str, str] | None:
    """First ``(On|In subject dest)`` from BDDL text or a parsed goal tree."""
    if isinstance(source, (list, tuple)):
        return _first_on_in_tree(source)
    if isinstance(source, Mapping):
        for key in ("goal", ":goal", "Goal"):
            if key in source and source[key]:
                found = parse_first_on_in(source[key])
                if found is not None:
                    return found
        return None
    text = str(source or "")
    match = _ON_IN_ATOM.search(text)
    if match is None:
        return None
    return match.group(1).lower(), match.group(2), match.group(3)


def _first_on_in_tree(node: Any) -> tuple[str, str, str] | None:
    if not isinstance(node, (list, tuple)) or not node:
        return None
    head = str(node[0]).lower()
    if head in {"on", "in"} and len(node) >= 3:
        return head, str(node[1]), str(node[2])
    for child in node:
        found = _first_on_in_tree(child)
        if found is not None:
            return found
    return None


def instruction_opens_first(instruction: str) -> bool:
    """True when the first verb is ``open`` (drawer / cabinet first action)."""
    tokens = re.sub(r"[^a-z0-9\s]", " ", str(instruction or "").lower()).split()
    if tokens and tokens[0] == "please":
        tokens = tokens[1:]
    return bool(tokens) and tokens[0] == "open"


def _looks_like_region(name: str) -> bool:
    lower = name.lower()
    return any(lower.endswith(tail) for tail in _REGION_TAILS)


def resolve_entity_name(name: str, available: Sequence[str]) -> str | None:
    """Map a BDDL atom (object or region) onto a posed body/fixture name."""
    avail = [str(item) for item in available]
    if name in avail:
        return name
    stem = name
    for tail in _REGION_TAILS:
        if stem.endswith(tail):
            stem = stem[: -len(tail)]
            break
    if stem in avail:
        return stem
    matches = [
        item
        for item in avail
        if item == stem or item.startswith(stem + "_") or stem.startswith(item + "_")
    ]
    if not matches:
        return None
    bodies = [item for item in matches if not _looks_like_region(item)]
    pool = bodies or matches
    return min(pool, key=len)


def pick_open_fixture(instruction: str, names: Sequence[str]) -> str | None:
    """Choose the fixture being opened (cabinet/drawer), never the later put object."""
    text = str(instruction or "").lower()
    keywords: list[str] = []
    if re.search(r"\bdrawer\b", text):
        keywords.extend(["drawer", "cabinet"])
    if re.search(r"\bcabinet\b", text):
        keywords.extend(["cabinet", "drawer"])
    if re.search(r"\bfridge\b|\brefrigerator\b", text):
        keywords.extend(["fridge", "refrigerator"])
    if not keywords:
        keywords = ["drawer", "cabinet"]
    candidates: list[str] = []
    for name in names:
        lower = name.lower()
        if any(skip in lower for skip in _OPEN_SKIP_SUBSTR):
            continue
        if any(key in lower for key in keywords):
            candidates.append(name)
    if not candidates:
        return None
    bodies = [name for name in candidates if not _looks_like_region(name)]
    return (bodies or candidates)[0]


def pick_workspace_entity_by_substrings(
    named_poses: Mapping[str, np.ndarray],
    substrings: Sequence[str],
) -> tuple[str, np.ndarray] | None:
    """First non-region workspace body whose name contains any substring."""
    needles = [str(item).strip().lower() for item in substrings if str(item).strip()]
    if not needles or not named_poses:
        return None
    for name, pos in named_poses.items():
        if _looks_like_region(name):
            continue
        lower = str(name).lower()
        if any(needle in lower for needle in needles):
            return str(name), np.asarray(pos, dtype=np.float64).reshape(3).copy()
    return None


def l3_dwell_at_xy_keep_z(entity_pos: np.ndarray, z: float) -> np.ndarray:
    """Dwell at the entity XY while keeping an explicit world-Z (e.g. EEF height).

    Does not read ``dwell_height`` — that field stays unused for L3 pose.
    """
    pos = np.asarray(entity_pos, dtype=np.float64).reshape(3)
    return np.array([float(pos[0]), float(pos[1]), float(z)], dtype=np.float64)


def select_l3_intercept_entity(
    *,
    instruction: str,
    named_poses: Mapping[str, np.ndarray],
    goal_subject: str | None,
    goal_dest: str | None,
    eef_pos: np.ndarray,
) -> tuple[str, np.ndarray, dict[str, Any]] | None:
    """L3 path-intercept entity: open-fixture, else first BDDL ``On``/``In`` subject."""
    if not named_poses:
        return None
    names = list(named_poses)
    if instruction_opens_first(instruction):
        opened = pick_open_fixture(instruction, names)
        if opened is not None:
            pos = np.asarray(named_poses[opened], dtype=np.float64)
            return (
                opened,
                pos,
                {
                    "source": "open_override",
                    "goal_subject": goal_subject,
                    "goal_dest": goal_dest,
                },
            )
    if goal_subject:
        resolved = resolve_entity_name(goal_subject, names)
        if resolved is not None:
            pos = np.asarray(named_poses[resolved], dtype=np.float64)
            return (
                resolved,
                pos,
                {
                    "source": "bddl_goal",
                    "goal_subject": goal_subject,
                    "goal_dest": goal_dest,
                },
            )
    dests: set[str] = set()
    if goal_dest:
        dests.add(goal_dest)
        lifted = resolve_entity_name(goal_dest, names)
        if lifted is not None:
            dests.add(lifted)
    remaining = [
        (name, named_poses[name])
        for name in names
        if name not in dests and not _looks_like_region(name)
    ]
    return select_instruction_first_task_object(remaining, instruction, eef_pos)


def libero_goal_source(env: Any) -> Any:
    """Best available BDDL goal payload on a LIBERO env / wrapper."""
    for obj in (env, getattr(env, "env", None)):
        if obj is None:
            continue
        parsed = getattr(obj, "parsed_problem", None)
        if isinstance(parsed, Mapping):
            for key in ("goal", ":goal", "Goal"):
                if parsed.get(key):
                    return parsed[key]
        path = getattr(obj, "bddl_file_name", None)
        if path:
            file_path = Path(path)
            if file_path.is_file():
                return file_path.read_text()
    return ""


def collect_workspace_entity_poses(env: Any) -> dict[str, np.ndarray]:
    """Body poses for objects and fixtures still in the workspace (|x|,|y| < 5)."""
    inner = getattr(env, "env", env)
    sim = inner.sim
    poses: dict[str, np.ndarray] = {}
    for mapping in (
        getattr(inner, "objects_dict", {}) or {},
        getattr(inner, "fixtures_dict", {}) or {},
    ):
        if not isinstance(mapping, Mapping):
            continue
        for name, obj in mapping.items():
            if "hand_pusher" in str(name).lower():
                continue
            root = getattr(obj, "root_body", None)
            if not root:
                continue
            try:
                body_id = sim.model.body_name2id(root)
            except Exception:
                continue
            pos = sim.data.body_xpos[body_id].copy()
            if abs(float(pos[0])) >= 5.0 or abs(float(pos[1])) >= 5.0:
                continue
            poses[str(name)] = np.asarray(pos, dtype=np.float64)
    return poses


def compute_l3_path_intercept_target(
    eef_pos: np.ndarray,
    task_obj_pos: np.ndarray,
    *,
    alpha: float,
    min_eef_clearance: float,
    tip_overshoot: float,
    min_task_clearance: float,
    target_z: float | None,
) -> np.ndarray:
    """Place the hand on the home-EEF → task-object segment.

    ``alpha`` is a lerp on that segment: 0 at the EEF, 0.5 midpoint, 1 at the
    task object. ``min_eef_clearance + tip_overshoot`` is a minimum XY distance
    from the EEF (0 disables it); if the object is closer than that, the point
    extends past the object along the same ray. ``min_task_clearance`` caps
    how close the body may sit to the task object when the segment is long
    enough (0 disables the cap).
    Z lerps with the same ``alpha`` (EEF → task), unless ``target_z`` is set.
    """
    eef = np.asarray(eef_pos, dtype=np.float64).reshape(3)
    task = np.asarray(task_obj_pos, dtype=np.float64).reshape(3)
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f"intercept alpha must be in [0, 1], got {alpha!r}")
    if float(min_eef_clearance) < 0.0:
        raise ValueError(f"min_eef_clearance must be >= 0, got {min_eef_clearance!r}")
    if float(tip_overshoot) < 0.0:
        raise ValueError(f"tip_overshoot must be >= 0, got {tip_overshoot!r}")
    if float(min_task_clearance) < 0.0:
        raise ValueError(f"min_task_clearance must be >= 0, got {min_task_clearance!r}")

    delta = task[:2] - eef[:2]
    dist = float(np.linalg.norm(delta))
    floor = float(min_eef_clearance) + float(tip_overshoot)
    task_cap = float(min_task_clearance)
    if dist < 1e-8:
        xy = eef[:2] + np.array([floor, 0.0], dtype=np.float64)
    else:
        unit = delta / dist
        along = float(alpha) * dist
        hi = dist - task_cap if task_cap > 0.0 else dist
        if hi < floor:
            along = floor if floor > 0.0 else max(hi, 0.0)
        else:
            along = min(max(along, floor), hi)
        xy = eef[:2] + along * unit
    if target_z is None:
        z = (1.0 - float(alpha)) * float(eef[2]) + float(alpha) * float(task[2])
    else:
        z = float(target_z)
    return np.array([xy[0], xy[1], z], dtype=np.float64)


def compute_l3_interpolated_target(
    eef_pos: np.ndarray,
    task_obj_pos: np.ndarray,
    alpha: float = 0.5,
    target_z: float = None,
):
    """Compute a hand target on the EEF→task-object line in the XY plane.

    Args:
        eef_pos:      Current end-effector position (3,).
        task_obj_pos:  Task target object position (3,).
        alpha:        Interpolation factor ∈ [0, 1].
                      0 → at EEF,  1 → at task object.
        target_z:     Desired Z height for the hand target.
                      *None* → midpoint of eef_z and task_obj_z.

    Returns:
        np.ndarray of shape (3,) — the hand target position.
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    target = np.empty(3, dtype=np.float64)
    target[0] = eef_pos[0] + alpha * (task_obj_pos[0] - eef_pos[0])
    target[1] = eef_pos[1] + alpha * (task_obj_pos[1] - eef_pos[1])
    if target_z is not None:
        target[2] = float(target_z)
    else:
        target[2] = (eef_pos[2] + task_obj_pos[2]) / 2.0
    return target


def get_robot_base_pos(env):
    """Return the robot base body position in world frame.

    Tries common robosuite body name patterns (``{prefix}base``,
    ``{prefix}link0``) and falls back to the EEF position if none are
    found.
    """
    sim = env.env.sim
    robot = env.env.robots[0]
    prefix = getattr(robot.robot_model, "naming_prefix", "robot0_")
    for suffix in ("base", "link0"):
        try:
            bid = sim.model.body_name2id(prefix + suffix)
            return sim.data.body_xpos[bid].copy()
        except Exception:
            continue
    return sim.data.site_xpos[robot.eef_site_id].copy()


def get_table_center_pos(env):
    """Return the workspace center position in world frame.

    Tries ``table_top`` first (table / kitchen arenas).  If that site does
    not exist (e.g. floor arenas in safelibero_object), falls back to the
    centroid of all ``floor_*_region`` sites as a workspace-center estimate.
    """
    sim = env.env.sim
    try:
        table_sid = sim.model.site_name2id("table_top")
        return sim.data.site_xpos[table_sid].copy()
    except Exception:
        pass

    region_positions = []
    for sid in range(sim.model.nsite):
        name = sim.model.site_id2name(sid) or ""
        if name.startswith("floor_") and name.endswith("_region"):
            if "hand_pusher" not in name:
                region_positions.append(sim.data.site_xpos[sid].copy())
    if region_positions:
        return np.mean(region_positions, axis=0)

    return sim.data.site_xpos[env.env.robots[0].eef_site_id].copy()


def read_obstacle_table_pose(env, obstacle_name):
    """Read the obstacle's current world position and orientation.

    In Level-I init_states the visible obstacle is already on the table at its
    correct position.  Call this AFTER env.set_init_state().

    Returns:
        (pos, quat): np arrays of shape (3,) and (4,).
    """
    inner_env = env.env
    obj = inner_env.objects_dict[obstacle_name]
    body_id = inner_env.sim.model.body_name2id(obj.root_body)
    pos = inner_env.sim.data.body_xpos[body_id].copy()
    quat = inner_env.sim.data.body_xquat[body_id].copy()
    return pos, quat


class HandBase:
    """Kinematic hand bookkeeping; physics params injected from YAML."""

    hides_obstacles: bool = True

    def __init__(
        self,
        handle: LiberoEnvHandle,
        *,
        hand_name: str,
        seed: int,
        hand_physics: HandPhysicsParams,
        control_freq: int,
    ):
        self.handle = handle
        self.env = handle.env
        self.inner_env = self.env.env
        self.hand_name = hand_name
        if control_freq <= 0:
            raise ValueError(f"control_freq must be positive, got {control_freq!r}")
        self._control_freq = int(control_freq)
        self._hand = hand_physics

        obj = self.inner_env.objects_dict[hand_name]
        self._joint = obj.joints[-1]
        self._body_id = self.inner_env.sim.model.body_name2id(obj.root_body)
        jid = self.inner_env.sim.model.joint_name2id(self._joint)
        self._dof_addr = self.inner_env.sim.model.jnt_dofadr[jid]

        self.obstacle_geom_ids = self._collect_geom_ids()
        self._soften_hand_collision_geoms()
        self.inner_env.sim.model.body_mass[self._body_id] = float(hand_physics.mass_kg)

        self._hidden_pos = np.array([10.0, 10.0, 0.0])
        self._arm_quat = arm_quat_from_yaw(0.0)
        self.blocked = False
        self._last_task_step = 0
        self._rng = np.random.default_rng(seed)
        self._robot_geom_ids = None

    def set_robot_geom_ids(self, robot_geom_ids):
        self._robot_geom_ids = frozenset(robot_geom_ids) if robot_geom_ids else None

    def _collect_geom_ids(self):
        gids = set()
        model = self.inner_env.sim.model
        for gid in range(model.ngeom):
            bid = model.geom_bodyid[gid]
            if bid == self._body_id or model.body_parentid[bid] == self._body_id:
                gids.add(gid)
        return gids

    def _soften_hand_collision_geoms(self):
        model = self.inner_env.sim.model
        solref = np.asarray(self._hand.contact_solref, dtype=np.float64)
        solimp = np.asarray(self._hand.contact_solimp, dtype=np.float64)
        friction = np.asarray(self._hand.contact_friction, dtype=np.float64)
        for gid in self.obstacle_geom_ids:
            if model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0:
                continue
            model.geom_solref[gid] = solref
            nimp = min(len(solimp), model.geom_solimp.shape[1])
            model.geom_solimp[gid, :nimp] = solimp[:nimp]
            model.geom_friction[gid, : len(friction)] = friction

    def _teleport(self, pos, relax: bool = True):
        import mujoco

        sim = self.inner_env.sim
        desired = np.asarray(pos, dtype=np.float64).copy()
        cur = self._get_pos()
        delta = desired - cur
        dist = float(np.linalg.norm(delta))
        to_hidden = np.allclose(desired, self._hidden_pos, atol=0.02)
        big_jump = dist > float(self._hand.big_jump_m)
        rid = self._robot_geom_ids
        apply_relax = bool(relax and rid is not None and not to_hidden and not big_jump)
        if apply_relax:
            max_step = None
            if self._detect_contact(rid):
                max_step = float(self._hand.in_contact_max_step_m)
            elif np.linalg.norm(cur - self._get_eef_pos()) < float(self._hand.near_eef_m):
                max_step = float(self._hand.near_eef_max_step_m)
            if max_step is not None and dist > max_step > 0.0:
                desired = cur + delta * (max_step / dist)
        qpos = np.concatenate([desired, self._arm_quat])
        sim.data.set_joint_qpos(self._joint, qpos)
        sim.data.qvel[self._dof_addr : self._dof_addr + 6] = 0.0
        mujoco.mj_forward(sim.model._model, sim.data._data)

    def _begin_exit_offscreen(self, task_step: int):
        self._phase = Phase.EXIT
        self._phase_start = task_step
        self._exit_from = self._get_pos().copy()
        self._exit_to = self._hidden_pos.copy()
        d = float(np.linalg.norm(self._exit_to - self._exit_from))
        v_exit = 1.0
        self._exit_steps = max(22, min(72, int(d / max(v_exit, 0.15) * self._control_freq)))

    def _step_exit_offscreen(self, task_step: int) -> bool:
        dur = max(self._exit_steps, 2)
        alpha = min((task_step - self._phase_start) / dur, 1.0)
        pos = self._exit_from + minimum_jerk(alpha) * (self._exit_to - self._exit_from)
        self._teleport(pos, relax=False)
        return alpha >= 1.0

    def _get_pos(self):
        return self.inner_env.sim.data.body_xpos[self._body_id].copy()

    def _get_eef_pos(self):
        return self.inner_env.sim.data.site_xpos[self.inner_env.robots[0].eef_site_id].copy()

    def _update_yaw_toward(self, target_xy):
        current = self._get_pos()[:2]
        d = target_xy - current
        norm = np.linalg.norm(d)
        if norm > 1e-8:
            yaw = math.atan2(d[1], d[0])
            self._arm_quat = arm_quat_from_yaw(yaw)

    def _tremor(self, amplitude: float):
        return self._rng.normal(0, amplitude, 3)

    def _detect_contact(self, robot_geom_ids) -> bool:
        sim = self.inner_env.sim
        for i in range(sim.data.ncon):
            c = sim.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 in robot_geom_ids and g2 in self.obstacle_geom_ids) or (
                g2 in robot_geom_ids and g1 in self.obstacle_geom_ids
            ):
                return True
        return False

    def check_robot_collision(self, robot_geom_ids) -> bool:
        if self._detect_contact(robot_geom_ids):
            self.blocked = True
            return True
        return False

    def update_before_step(self, task_step: int):
        raise NotImplementedError

    @property
    def disturbance_onset_step(self):
        raise NotImplementedError
