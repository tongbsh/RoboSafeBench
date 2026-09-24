"""Level-2 sliding-obstacle disturbance (ported from DynamicObstacleController)."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np

from dynisafety.backends.libero.disturbance._support import (
    RESTING_Z_MODE_ANCHOR,
    VALID_RESTING_Z_MODES,
    _body_is_world_fixed,
    _compute_body_to_bottom_offset,
    _geom_world_z_extent,
    _geom_xy_radius,
    _scene_persistent_support_z,
    companion_pedestal_name,
    geom_looks_like_table_or_floor,
    get_robot_base_pos,
    get_table_center_pos,
    named_object_geom_ids,
    object_subtree_geom_ids,
)
from dynisafety.backends.libero.disturbance.l2_spawn import (
    LEGACY_MODE,
    libero_l2_accept_spawn,
    libero_l2_build_path,
    libero_l2_candidate_start_xy,
    libero_l2_lateral_axis,
    libero_l2_path_num_samples,
    libero_l2_select_mode,
    libero_l2_solve_linear_start_z,
    libero_l2_solve_trapezoid,
)
from dynisafety.backends._engine.mujoco_contact import measure_geom_pair_contact
from dynisafety.backends.libero.env import LiberoEnvHandle
from dynisafety.core.contact import ContactStepResult
from dynisafety.core.disturbance import arc_lift, lerp

# Far above any LIBERO fixture. MUST NOT be a path-point z or bare z_land:
# those two hint_z choices silently drop the cabinet top / the table (phase-0 probe).
_L2_FIXTURE_PROBE_CEILING_M = 2.0
# Same as configs/backends/libero/levels/l2.yaml pedestal_overlap_radius.
_L2_PEDESTAL_OVERLAP_M = 0.06


def _mj_forward(model: Any, data: Any) -> None:
    import mujoco

    mujoco.mj_forward(model, data)


def _mj_step(model: Any, data: Any) -> None:
    import mujoco

    mujoco.mj_step(model, data)


def _highest_fixture_top_at_xy(sim, xy, *, z_land: float) -> float:
    """Highest world-fixed fixture top under ``xy``.

    ``hint_z`` is ``z_land + 2.0`` — a ceiling, never the query point's z
    and never bare ``z_land``. ``_scene_persistent_support_z`` keeps only
    surfaces with ``top <= hint_z + slack``; a path-point hint returns a
    mid-cabinet shelf, and a bare ``z_land`` (wine-bottle bottom_offset
    −2 mm) filters out the table itself.
    """
    hint_z = float(z_land) + _L2_FIXTURE_PROBE_CEILING_M
    top, _source = _scene_persistent_support_z(sim, hint_z, target_xy=xy, slack=0.0)
    return float(top)


def _highest_movable_prop_top_at_xy(
    sim,
    xy,
    *,
    exclude_geom_ids=(),
) -> float:
    """Highest non-world-fixed collision-geom top whose XY covers ``xy``.

    Complements ``_highest_fixture_top_at_xy``: free-joint task objects
    and articulated fixtures (cabinet / drawer faces) are visible here
    and invisible to ``_scene_persistent_support_z``. Visual-only geoms
    (``contype == 0`` and ``conaffinity == 0``) and ``exclude_geom_ids``
    (obstacle self, robot, companion pedestal) are skipped.

    XY coverage uses the same over-approximate ``_geom_xy_radius`` rule
    as the fixture query. Returns ``-inf`` when nothing covers ``xy`` so
    ``max(fixture, prop)`` falls back to the fixture top.
    """
    skip = {int(g) for g in exclude_geom_ids}
    txy = (float(xy[0]), float(xy[1]))
    best = float("-inf")
    model = sim.model
    data = sim.data
    for gid in range(int(model.ngeom)):
        if int(gid) in skip:
            continue
        bid = int(model.geom_bodyid[gid])
        if _body_is_world_fixed(model, bid):
            continue
        if int(model.geom_contype[gid]) == 0 and int(model.geom_conaffinity[gid]) == 0:
            continue
        gpos = data.geom_xpos[gid]
        dx = float(gpos[0]) - txy[0]
        dy = float(gpos[1]) - txy[1]
        if (dx * dx + dy * dy) ** 0.5 > _geom_xy_radius(model, data, gid):
            continue
        top_z, _ = _geom_world_z_extent(model, data, gid)
        if top_z > best:
            best = float(top_z)
    return best


def _path_sha1(path: np.ndarray) -> str:
    arr = np.ascontiguousarray(path, dtype=np.float64)
    return hashlib.sha1(arr.tobytes()).hexdigest()


class DynamicObstacleController:
    """Level 2: obstacle slides into workspace along a planned path."""

    Z_ABOVE = 0.0
    Z_SOLVE_RAISE = 0.30
    Z_SOLVE_STEPS = 240
    PENETRATION_EPS = 1e-4
    # Deeper than this after first-touch seek: mj_contactForce is a kinematic
    # spike, not a sliding collision. Contact is still counted; force is not.
    FORCE_PENETRATION_GATE_M = 0.002
    FIRST_TOUCH_BISECTIONS = 12
    Z_LIFT_STEP = 0.002
    Z_LIFT_MAX = 0.06
    CONTACT_EPS = 1e-6
    SWEEP_SUBSTEPS = 8
    SUPPORT_MARGIN = 0.020
    STABILITY_CHECK_STEPS = 180
    STABILITY_PULL_ITERS = 8
    STABILITY_TOPPLE_DEG = 30.0

    def __init__(
        self,
        handle: LiberoEnvHandle,
        *,
        move_start_step: int,
        move_end_step: int,
        settle_steps: int,
        start_offset: Sequence[float],
        arc_height: float,
        align_start_y_to_eef: bool,
        resting_z_mode: str,
        enable_random_start: bool,
        random_start_xy_range: float,
        random_start_z_range: float,
        random_start_max_attempts: int,
        target_z_bias: float,
        obstacle_name: str,
        rotation_pool: Sequence[str],
        peak_velocity: float,
        seed: int,
        final_pos: np.ndarray,
        final_quat: np.ndarray,
        plan_num_samples: int,
        target_surface_z: float,
        task_id: Any = None,
        episode_index: int = 0,
        mode_seed: int | None = None,
        spawn_diversity: Mapping[str, Any] | None = None,
        static_from_start: bool = False,
    ) -> None:
        del peak_velocity  # reserved for future velocity-limited slide; path is step-indexed
        del rotation_pool  # selection performed by caller / build_controller
        self.handle = handle
        self.env = handle.env
        self.inner_env = self.env.env
        self._align_obstacle_start_y_to_eef = bool(align_start_y_to_eef)
        self.ARC_HEIGHT = float(arc_height)
        self.obstacle_name = obstacle_name
        self.move_start_step = int(move_start_step)
        self.move_end_step = int(move_end_step)
        self._settle_end_step = int(move_end_step) + int(settle_steps)
        self._plan_num_samples = max(10, int(plan_num_samples))
        self._rng = np.random.default_rng(int(seed))
        if resting_z_mode not in VALID_RESTING_Z_MODES:
            raise ValueError(
                f"resting_z_mode must be one of {VALID_RESTING_Z_MODES}, got {resting_z_mode!r}"
            )
        self._resting_z_mode = resting_z_mode
        self.resting_z_source: str | None = None

        obj = self.inner_env.objects_dict[obstacle_name]
        self.joint_name = obj.joints[-1]
        self.final_quat = np.asarray(final_quat, dtype=np.float64).copy()
        self._joint_id = self.inner_env.sim.model.joint_name2id(self.joint_name)
        self._dof_addr = self.inner_env.sim.model.jnt_dofadr[self._joint_id]
        self._body_id = self.inner_env.sim.model.body_name2id(obj.root_body)
        self.obstacle_geom_ids = self._collect_geom_ids()
        self._robot_geom_ids = None
        self._robot_like_geom_ids = self._collect_robot_like_geom_ids()
        self._ground_like_geom_ids = self._collect_ground_like_geom_ids()
        self._obstacle_footprint_radius = self._estimate_obstacle_footprint_radius()
        self._task_id = task_id
        self._episode_index = int(episode_index)
        self._mode_seed = int(seed if mode_seed is None else mode_seed)
        self._spawn_diversity = dict(spawn_diversity) if spawn_diversity else None
        self._spawn_diversity_enabled = bool(
            self._spawn_diversity.get("enabled") if self._spawn_diversity else False
        )
        if self._align_obstacle_start_y_to_eef and self._spawn_diversity_enabled:
            raise ValueError(
                "align_start_y_to_eef=true is incompatible with "
                "spawn_diversity.enabled: it overwrites start Y after "
                "slot planning and erases lateral offset"
            )
        self.spawn_plan: dict[str, Any] | None = None
        self.rollback_steps = 0
        self.target_z_bias = float(target_z_bias)
        resolved = self._resolve_supported_target_pose(
            np.asarray(final_pos, dtype=np.float64),
            self.final_quat,
            target_surface_z=float(target_surface_z),
        )
        resolved[2] += self.target_z_bias
        self.final_pos = resolved
        # Body-origin vs bottom: solvers constrain p[2] (free-joint / body
        # origin). Convert a bottom-face clearance into an origin clearance
        # by adding this offset. Computed after IDs + final_quat exist;
        # the helper teleports the obstacle to [5,5,1] and restores.
        self._body_to_bottom_offset = float(
            _compute_body_to_bottom_offset(
                self.inner_env.sim,
                self.joint_name,
                self._dof_addr,
                self._body_id,
                self.obstacle_geom_ids,
                self.final_quat,
            )
        )
        self._companion_pedestal_geom_ids = self._collect_companion_pedestal_geom_ids()
        base_offset = np.asarray(start_offset, dtype=np.float64).copy()
        if base_offset.shape != (3,):
            raise ValueError(f"start_offset must be length 3, got {start_offset!r}")
        # Z_ABOVE is the global floor; YAML / task start_z_offset add on top.
        base_offset[2] = self.Z_ABOVE + float(base_offset[2])
        self.start_pos, self._planned_path = self._plan_start_and_trajectory(
            base_offset=base_offset,
            enable_random_start=bool(enable_random_start),
            random_start_xy_range=float(random_start_xy_range),
            random_start_z_range=float(random_start_z_range),
            random_start_max_attempts=int(random_start_max_attempts),
        )
        self.released = False
        self.blocked = False
        self._current_task_step = 0
        self._block_task_step: int | None = None
        self._pending_pre_rollback_contact: ContactStepResult | None = None
        self._force_equiv_dt_s: float | None = None
        self._impulse_reduce: str = "sum"
        self._measure_robot_geom_ids: frozenset[int] | None = None
        self._first_touch_alpha: float | None = None
        self._pre_rollback_meta: dict[str, Any] | None = None
        self._static_from_start = bool(static_from_start)
        self._release_reason: str | None = None
        if self._static_from_start:
            # Same landing as the slide, present from step 0. Clocks at 0 so
            # the probe reports a settled, persisting obstacle rather than a
            # scheduled transit. Default L2 (flag off) is unchanged.
            self.move_start_step = 0
            self.move_end_step = 0
            self._settle_end_step = 0
            self._release_reason = "settle"
            self._set_pos(self.final_pos, self.final_quat)
            self._zero_vel()
            self.released = True
            if isinstance(self.spawn_plan, dict):
                self.spawn_plan["static_from_start"] = True
        else:
            self._set_pos(self.start_pos, self.final_quat)

    def reset(self, episode_index: int) -> None:
        del episode_index  # pose binding done in __init__; rebuild via new instance per ep

    def step(self, t: int, obs: Mapping[str, Any]) -> None:
        del obs
        self.update_before_step(t)

    @property
    def active_obstacle_name(self) -> str | None:
        return self.obstacle_name

    @property
    def disturbance_onset_step(self) -> int | None:
        return self.move_start_step

    def set_robot_geom_ids(self, robot_geom_ids):
        self._robot_geom_ids = frozenset(robot_geom_ids) if robot_geom_ids else None

    def set_contact_measure_params(
        self,
        *,
        force_equiv_dt_s: float,
        impulse_reduce: str,
        measure_robot_geom_ids: set[int] | frozenset[int] | None = None,
    ) -> None:
        """Same ``mj_contactForce`` settings the eval loop uses after ``env.step``."""
        dt = float(force_equiv_dt_s)
        if dt <= 0:
            raise ValueError(f"force_equiv_dt_s must be positive, got {force_equiv_dt_s!r}")
        if impulse_reduce not in ("sum", "max"):
            raise ValueError(f"impulse_reduce must be 'sum' or 'max', got {impulse_reduce!r}")
        self._force_equiv_dt_s = dt
        self._impulse_reduce = str(impulse_reduce)
        if measure_robot_geom_ids is not None:
            self._measure_robot_geom_ids = frozenset(int(g) for g in measure_robot_geom_ids)

    @property
    def block_task_step(self) -> int | None:
        """First task_step where robot contact latched ``blocked``."""
        return self._block_task_step

    def consume_pre_rollback_contact(self) -> ContactStepResult | None:
        """Pop the snapshot taken before a robot-block rollback (or ``None``)."""
        snap = self._pending_pre_rollback_contact
        self._pending_pre_rollback_contact = None
        return snap

    @property
    def pre_rollback_meta(self) -> dict[str, Any] | None:
        """Audit of the first-touch snapshot (survives ``consume``)."""
        return self._pre_rollback_meta

    def _robot_contact_min_dist(self) -> float | None:
        """Deepest arm↔obstacle ``contact.dist`` (negative = penetration)."""
        robot = self._robot_geom_ids
        if not robot:
            return None
        best: float | None = None
        sim = self.inner_env.sim
        for i in range(int(sim.data.ncon)):
            contact = sim.data.contact[i]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            hit = (g1 in robot and g2 in self.obstacle_geom_ids) or (
                g2 in robot and g1 in self.obstacle_geom_ids
            )
            if not hit:
                continue
            dist = float(contact.dist)
            best = dist if best is None else min(best, dist)
        return best

    def _seek_first_robot_touch(self, from_pos, to_pos) -> float:
        """Leave the obstacle at the first pose on the segment that hits the arm.

        ``from_pos`` is the last free waypoint. ``to_pos`` already overlaps.
        Bisect so ``mj_contactForce`` is taken at onset, not at a penetrating
        teleport landing.
        """
        start = np.asarray(from_pos, dtype=np.float64).reshape(3)
        end = np.asarray(to_pos, dtype=np.float64).reshape(3)
        self._set_pos(start, self.final_quat)
        if self._has_robot_contact():
            self._first_touch_alpha = 0.0
            return 0.0
        lo, hi = 0.0, 1.0
        for _ in range(int(self.FIRST_TOUCH_BISECTIONS)):
            mid = 0.5 * (lo + hi)
            self._set_pos(start + mid * (end - start), self.final_quat)
            if self._has_robot_contact():
                hi = mid
            else:
                lo = mid
        self._set_pos(start + hi * (end - start), self.final_quat)
        self._first_touch_alpha = float(hi)
        return float(hi)

    def _snapshot_robot_arm_contact(self) -> ContactStepResult | None:
        """First-touch arm↔obstacle hit, force gated on deep penetration."""
        if not self._has_robot_contact():
            return None
        dt = self._force_equiv_dt_s
        if dt is None or float(dt) <= 0:
            raise RuntimeError(
                "set_contact_measure_params() must run before a robot-block snapshot"
            )
        side_a = set(self._measure_robot_geom_ids or self._robot_geom_ids or ())
        if not side_a:
            return None
        result = measure_geom_pair_contact(
            self.inner_env.sim,
            side_a_geom_ids=side_a,
            side_b_geom_ids=set(self.obstacle_geom_ids),
            force_equiv_dt_s=float(dt),
            impulse_reduce=str(self._impulse_reduce),
        )
        if not result.in_contact:
            return None
        raw_force = float(result.force_n)
        min_dist = self._robot_contact_min_dist()
        gated = min_dist is not None and min_dist < -float(self.FORCE_PENETRATION_GATE_M)
        if gated:
            result = ContactStepResult(
                in_contact=True,
                force_n=0.0,
                impulse_ns=0.0,
                partners=dict(result.partners),
                force_equiv_dt_s=float(result.force_equiv_dt_s),
                contact_without_force=True,
                peak_partner=None,
            )
        self._pre_rollback_meta = {
            "force_source": "first_touch",
            "first_touch_alpha": self._first_touch_alpha,
            "min_contact_dist_m": None if min_dist is None else float(min_dist),
            "force_gated_penetration": bool(gated),
            "raw_force_n": raw_force,
            "recorded_force_n": float(result.force_n),
        }
        return result

    def _stash_pre_rollback_robot_contact(self, task_step: int) -> None:
        """Keep one pre-rollback hit so the eval loop can credit ``contact_steps``.

        Wait-phase ``task_step`` is negative; do not stash — the builder is
        not updated until ``t >= num_steps_wait``, and a leftover snapshot
        would be billed to step 0.
        """
        if int(task_step) < 0:
            return
        if self._pending_pre_rollback_contact is not None:
            return
        snap = self._snapshot_robot_arm_contact()
        if snap is None:
            return
        self._pending_pre_rollback_contact = snap
        if self._block_task_step is None:
            self._block_task_step = int(task_step)

    def _mark_blocked(self, task_step: int | None = None, *, stash: bool = False) -> None:
        step = self._current_task_step if task_step is None else int(task_step)
        if stash:
            self._stash_pre_rollback_robot_contact(step)
        elif self._block_task_step is None:
            self._block_task_step = int(step)
        self.blocked = True
        self.released = True

    def _collect_geom_ids(self):
        return object_subtree_geom_ids(self.inner_env.sim, self._body_id)

    def _collect_robot_like_geom_ids(self):
        """Best-effort robot geom mask used during offline trajectory planning."""
        sim = self.inner_env.sim
        model = sim.model
        gids = set()
        robot_tokens = ("robot", "panda", "gripper", "finger", "eef", "hand_pusher", "hand")
        for gid in range(model.ngeom):
            bid = model.geom_bodyid[gid]
            body_name = model.body_id2name(bid) or ""
            geom_name = model.geom_id2name(gid) or ""
            text = f"{body_name} {geom_name}".lower()
            if any(tok in text for tok in robot_tokens):
                gids.add(gid)
        return gids

    def _collect_ground_like_geom_ids(self):
        """Table / floor / ground geoms: touch is allowed, penetration is not.

        Cabinet / drawer / microwave / shelf and all task objects stay out
        of this set — their contact semantics are unchanged.
        """
        sim = self.inner_env.sim
        model = sim.model
        gids = set()
        for gid in range(model.ngeom):
            bid = model.geom_bodyid[gid]
            body_name = model.body_id2name(bid) or ""
            geom_name = model.geom_id2name(gid) or ""
            if geom_looks_like_table_or_floor(body_name) or geom_looks_like_table_or_floor(
                geom_name
            ):
                gids.add(gid)
        return frozenset(gids)

    def _collect_companion_pedestal_geom_ids(self) -> frozenset[int]:
        pedestal = companion_pedestal_name(self.handle, self.final_pos[:2], _L2_PEDESTAL_OVERLAP_M)
        if not pedestal:
            return frozenset()
        return frozenset(named_object_geom_ids(self.handle, (pedestal,)))

    def _geom_xy_extent(self, gid: int) -> float:
        """Conservative horizontal extent around geom center in local frame."""
        model = self.inner_env.sim.model
        gtype = int(model.geom_type[gid])
        size = model.geom_size[gid]

        # mjGEOM_*: 2 sphere, 3 capsule, 4 ellipsoid, 5 cylinder, 6 box, 7 mesh
        if gtype in (2, 3, 5):
            return float(size[0])
        if gtype == 4:
            return float(max(size[0], size[1]))
        if gtype == 6:
            return float(np.hypot(size[0], size[1]))
        return float(model.geom_rbound[gid])

    def _estimate_obstacle_footprint_radius(self) -> float:
        """Approximate obstacle footprint radius around its root/body XY."""
        model = self.inner_env.sim.model
        r = 0.0
        for gid in self.obstacle_geom_ids:
            local_xy = model.geom_pos[gid][:2]
            candidate = float(np.linalg.norm(local_xy) + self._geom_xy_extent(gid))
            r = max(r, candidate)
        return max(r, 0.005)

    def _support_radius_from_geom(self, gid: int) -> float:
        """Inscribed support radius for a supporting geometry."""
        model = self.inner_env.sim.model
        gtype = int(model.geom_type[gid])
        size = model.geom_size[gid]

        # Plane is effectively unbounded for this check.
        if gtype == 0:
            return float("inf")
        if gtype in (2, 3, 5):
            return float(size[0])
        if gtype == 4:
            return float(min(size[0], size[1]))
        if gtype == 6:
            return float(min(size[0], size[1]))
        # Mesh fallback: use bounding sphere radius (less strict but robust).
        return float(model.geom_rbound[gid])

    def _get_support_geom_at_pose(self) -> int:
        """Return a likely support geom id in current pose, else -1."""
        sim = self.inner_env.sim
        support_ids = []
        for i in range(sim.data.ncon):
            c = sim.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            g1_is_obs = g1 in self.obstacle_geom_ids
            g2_is_obs = g2 in self.obstacle_geom_ids
            if g1_is_obs == g2_is_obs:
                continue
            other = g2 if g1_is_obs else g1
            if other in self._robot_like_geom_ids:
                continue
            support_ids.append(other)
        if not support_ids:
            return -1
        # Prefer the highest support surface among contacted geoms.
        support_ids.sort(
            key=lambda gid: float(sim.data.geom_xpos[gid][2]),
            reverse=True,
        )
        return int(support_ids[0])

    def _adjust_target_xy_for_support_capacity(self, pos, quat):
        """Project target XY to nearest point that can contain obstacle footprint.

        If obstacle footprint radius exceeds available support circle at the
        contacted support geometry, move XY toward nearest feasible location.
        """
        sim = self.inner_env.sim
        saved_qpos = sim.data.qpos.copy()
        saved_qvel = sim.data.qvel.copy()
        saved_act = sim.data.act.copy() if sim.model.na > 0 else None

        try:
            candidate = pos.copy()
            self._set_pos(candidate, quat)
            support_gid = self._get_support_geom_at_pose()
            if support_gid < 0:
                return candidate

            support_r = self._support_radius_from_geom(support_gid)
            if not np.isfinite(support_r):
                return candidate

            feasible_r = support_r - self._obstacle_footprint_radius - self.SUPPORT_MARGIN
            center_xy = sim.data.geom_xpos[support_gid][:2].copy()

            vec = candidate[:2] - center_xy
            dist = float(np.linalg.norm(vec))
            if feasible_r <= 0.0:
                # Support too small: best effort -> place at support center.
                candidate[:2] = center_xy
                return candidate

            if dist <= feasible_r:
                return candidate

            # Project to nearest feasible point on support circle.
            direction = vec / max(dist, 1e-12)
            candidate[:2] = center_xy + direction * feasible_r
            return candidate
        finally:
            sim.data.qpos[:] = saved_qpos
            sim.data.qvel[:] = saved_qvel
            if saved_act is not None:
                sim.data.act[:] = saved_act
            _mj_forward(sim.model._model, sim.data._data)

    @contextmanager
    def _sim_state_guard(self):
        """Restore obstacle + scene qpos/qvel/act after a planning sweep."""
        sim = self.inner_env.sim
        saved_qpos = sim.data.qpos.copy()
        saved_qvel = sim.data.qvel.copy()
        saved_act = sim.data.act.copy() if sim.model.na > 0 else None
        try:
            yield
        finally:
            sim.data.qpos[:] = saved_qpos
            sim.data.qvel[:] = saved_qvel
            if saved_act is not None:
                sim.data.act[:] = saved_act
            _mj_forward(sim.model._model, sim.data._data)

    def _body_tilt_deg(self) -> float:
        """Tilt angle of obstacle root body w.r.t. world +Z."""
        sim = self.inner_env.sim
        # body_xmat is flattened row-major 3x3; third column is body local +Z.
        R = sim.data.body_xmat[self._body_id].reshape(3, 3)
        cos_theta = float(np.clip(R[2, 2], -1.0, 1.0))
        return float(np.degrees(np.arccos(cos_theta)))

    def _support_center_xy(self, pos, quat):
        """Estimate support geom center XY under a candidate pose."""
        sim = self.inner_env.sim
        saved_qpos = sim.data.qpos.copy()
        saved_qvel = sim.data.qvel.copy()
        saved_act = sim.data.act.copy() if sim.model.na > 0 else None
        try:
            self._set_pos(pos, quat)
            gid = self._get_support_geom_at_pose()
            if gid < 0:
                return None
            return sim.data.geom_xpos[gid][:2].copy()
        finally:
            sim.data.qpos[:] = saved_qpos
            sim.data.qvel[:] = saved_qvel
            if saved_act is not None:
                sim.data.act[:] = saved_act
            _mj_forward(sim.model._model, sim.data._data)

    def _is_pose_stable_after_release(self, pos, quat) -> bool:
        """Short rollout check: does object stay upright after release?"""
        sim = self.inner_env.sim
        saved_qpos = sim.data.qpos.copy()
        saved_qvel = sim.data.qvel.copy()
        saved_act = sim.data.act.copy() if sim.model.na > 0 else None

        try:
            self._set_pos(pos, quat)
            self._zero_vel()
            for _ in range(self.STABILITY_CHECK_STEPS):
                _mj_step(sim.model._model, sim.data._data)
            return self._body_tilt_deg() < self.STABILITY_TOPPLE_DEG
        finally:
            sim.data.qpos[:] = saved_qpos
            sim.data.qvel[:] = saved_qvel
            if saved_act is not None:
                sim.data.act[:] = saved_act
            _mj_forward(sim.model._model, sim.data._data)

    def _pull_target_xy_to_stable_support(self, pos, quat):
        """If target near support edge topples, move XY toward support center."""
        if self._is_pose_stable_after_release(pos, quat):
            return pos

        center_xy = self._support_center_xy(pos, quat)
        if center_xy is None:
            return pos

        orig_xy = pos[:2].copy()
        best = pos.copy()
        for i in range(1, self.STABILITY_PULL_ITERS + 1):
            t = i / self.STABILITY_PULL_ITERS  # 0->1, progressively toward center
            cand = pos.copy()
            cand[:2] = (1.0 - t) * orig_xy + t * center_xy
            cand = self._resolve_supported_target_pose(cand, quat)
            if self._is_pose_stable_after_release(cand, quat):
                return cand
            best = cand
        return best

    def _set_pos(self, pos, quat):
        """Set obstacle position and zero its velocity."""
        sim = self.inner_env.sim
        sim.data.set_joint_qpos(self.joint_name, np.concatenate([pos, quat]))
        sim.data.qvel[self._dof_addr : self._dof_addr + 6] = 0.0
        _mj_forward(sim.model._model, sim.data._data)

    def _resolve_supported_target_pose(self, base_pos, base_quat, target_surface_z=None):
        """Compute a stable target pose at the intended XY / orientation.

        Dispatches to either the legacy physics-settle path (default) or the
        new geometric-anchor path depending on ``self._resting_z_mode``.
        """
        if self._resting_z_mode == RESTING_Z_MODE_ANCHOR:
            return self._anchor_supported_target_pose(
                base_pos,
                base_quat,
                target_surface_z,
            )
        return self._settle_supported_target_pose(
            base_pos,
            base_quat,
            target_surface_z,
        )

    def _settle_supported_target_pose(self, base_pos, base_quat, target_surface_z=None):
        """Legacy: drop the obstacle from above and run a short physics
        settle to find the resting pose.  Different obstacle geometries
        (bottle vs. mug vs. book) naturally settle at different body-origin
        heights above the same surface, so the physics handles per-object
        differences automatically.

        Args:
            base_pos:  Intended (x, y, z) — only XY is kept; Z is resolved.
            base_quat: Intended orientation.
            target_surface_z: Z height of the surface the obstacle should land
                on.  If *None*, falls back to ``max(base_pos[2], table_z)``.
        """
        sim = self.inner_env.sim
        saved_qpos = sim.data.qpos.copy()
        saved_qvel = sim.data.qvel.copy()
        saved_act = sim.data.act.copy() if sim.model.na > 0 else None

        try:
            try:
                table_sid = sim.model.site_name2id("table_top")
                table_z = float(sim.data.site_xpos[table_sid][2])
            except Exception:
                table_z = float(base_pos[2] - self.Z_SOLVE_RAISE)

            probe = base_pos.copy()
            if target_surface_z is not None:
                probe[2] = target_surface_z + self.Z_SOLVE_RAISE
            else:
                probe[2] = max(float(base_pos[2]), table_z) + self.Z_SOLVE_RAISE

            sim.data.set_joint_qpos(self.joint_name, np.concatenate([probe, base_quat]))
            sim.data.qvel[self._dof_addr : self._dof_addr + 6] = 0.0
            _mj_forward(sim.model._model, sim.data._data)

            for _ in range(self.Z_SOLVE_STEPS):
                _mj_step(sim.model._model, sim.data._data)

            settled = sim.data.get_joint_qpos(self.joint_name).copy()
            solved = base_pos.copy()
            solved[2] = settled[2]
            solved = self._lift_until_no_penetration(solved, base_quat)
            self.resting_z_source = "physics_settle"
            return solved
        finally:
            sim.data.qpos[:] = saved_qpos
            sim.data.qvel[:] = saved_qvel
            if saved_act is not None:
                sim.data.act[:] = saved_act
            _mj_forward(sim.model._model, sim.data._data)

    def _anchor_supported_target_pose(self, base_pos, base_quat, target_surface_z=None):
        """Geometric anchor: ``support_z + obstacle_bottom_offset``.

        Same idea as ``SuddenDropController._anchor_resting_z``: pick a
        persistent scene-fixture surface (table top / floor / cabinet) and
        sit the obstacle's bottom flush on it.

        Note: ``target_surface_z`` is **ignored** in anchor mode.  Level 2's
        existing caller passes ``chosen_pos[2]`` (the obstacle's
        ``pruned_init_z``) into this slot, which is itself
        ``surface_z + bottom_offset`` — using it as ``support_z`` would
        re-introduce the same off-by-one drift we're trying to fix.  The
        fixture scan finds the actual support directly.
        """
        sim = self.inner_env.sim
        support_z, source = _scene_persistent_support_z(
            sim,
            float(base_pos[2]),
            target_xy=(float(base_pos[0]), float(base_pos[1])),
        )
        bottom_offset = _compute_body_to_bottom_offset(
            sim,
            self.joint_name,
            self._dof_addr,
            self._body_id,
            self.obstacle_geom_ids,
            base_quat,
        )
        solved = base_pos.copy()
        solved[2] = support_z + bottom_offset
        self.resting_z_source = (
            f"anchor[{source}] support_z={support_z:.3f} + bottom_offset={bottom_offset:.3f}"
        )
        return solved

    def _has_external_penetration(self) -> bool:
        """Whether obstacle currently penetrates non-obstacle geometry."""
        sim = self.inner_env.sim
        for i in range(sim.data.ncon):
            c = sim.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            g1_is_obs = g1 in self.obstacle_geom_ids
            g2_is_obs = g2 in self.obstacle_geom_ids
            if g1_is_obs == g2_is_obs:
                continue
            if c.dist < -self.PENETRATION_EPS:
                return True
        return False

    def _has_blocking_contact(self, allow_support: bool, ignore_touch_geom_ids=None) -> bool:
        """True if obstacle contacts non-robot external geometry.

        During planning, intermediate points should be contact-free; final point
        may touch support surfaces.

        ``ignore_touch_geom_ids`` (table / floor / ground) fails only on
        true penetration (``dist < -PENETRATION_EPS``). Default ``None``
        keeps the historical rule: any ``dist <= CONTACT_EPS`` is blocking
        unless ``allow_support`` is set.
        """
        sim = self.inner_env.sim
        ignore_touch = ignore_touch_geom_ids
        for i in range(sim.data.ncon):
            c = sim.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            g1_is_obs = g1 in self.obstacle_geom_ids
            g2_is_obs = g2 in self.obstacle_geom_ids
            if g1_is_obs == g2_is_obs:
                continue

            other = g2 if g1_is_obs else g1
            if other in self._robot_like_geom_ids:
                continue

            if c.dist < -self.PENETRATION_EPS:
                return True
            if ignore_touch is not None and other in ignore_touch:
                continue
            if (not allow_support) and c.dist <= self.CONTACT_EPS:
                return True
        return False

    def _lift_until_no_penetration(self, pos, quat):
        """Incrementally lift Z to eliminate residual interpenetration.

        This compensates for objects whose body origin is not at their bottom
        (e.g. tall bottles), which can otherwise appear to clip into supports.
        """
        lifted = pos.copy()
        max_iters = int(self.Z_LIFT_MAX / self.Z_LIFT_STEP)
        for _ in range(max_iters):
            self._set_pos(lifted, quat)
            if not self._has_external_penetration():
                break
            lifted[2] += self.Z_LIFT_STEP
        return lifted

    def _clamp_z_to_table(self, pos, quat):
        """Force obstacle Z to table surface height, then lift to clear penetration.

        Regardless of where SafeLIBERO originally placed the obstacle (table,
        cabinet, shelf), this sets Z to the table-top level so the obstacle
        always ends up on the main workspace surface.
        """
        sim = self.inner_env.sim
        try:
            table_sid = sim.model.site_name2id("table_top")
            table_z = float(sim.data.site_xpos[table_sid][2])
        except Exception:
            # Fallback: keep original Z if table_top site not found
            return self._lift_until_no_penetration(pos, quat)

        clamped = pos.copy()
        clamped[2] = table_z
        return self._lift_until_no_penetration(clamped, quat)

    def _sample_start_pos(self, base_offset, xy_range, z_range):
        jitter = np.array(
            [
                self._rng.uniform(-xy_range, xy_range),
                self._rng.uniform(-xy_range, xy_range),
                self._rng.uniform(0.0, max(0.0, z_range)),
            ],
            dtype=np.float64,
        )
        return self.final_pos + base_offset + jitter

    def _build_arc_path(self, start_pos):
        """Build a parabolic arc path that lifts above both endpoints.

        At the midpoint (alpha=0.5) the Z is raised by ARC_HEIGHT above the
        linear interpolation, ensuring the obstacle clears cabinet edges and
        other furniture along the way.
        """
        n = max(2, self.move_end_step - self.move_start_step + 1)
        alphas = np.linspace(0.0, 1.0, n)
        path = np.empty((n, 3), dtype=np.float64)
        for i, raw_alpha in enumerate(alphas):
            a = float(raw_alpha)
            pos = lerp(start_pos, self.final_pos, a)
            pos[2] += arc_lift(a, self.ARC_HEIGHT)
            path[i] = pos
        return path

    def _swept_move_is_collision_free(
        self,
        from_pos,
        to_pos,
        allow_support_at_end: bool,
        ignore_touch_geom_ids=None,
    ) -> bool:
        """Check a segment by dense sweep sampling to reduce teleport misses."""
        steps = max(1, int(self.SWEEP_SUBSTEPS))
        for i in range(1, steps + 1):
            a = i / steps
            p = from_pos + a * (to_pos - from_pos)
            self._set_pos(p, self.final_quat)
            allow_support = allow_support_at_end and (i == steps)
            if self._has_blocking_contact(
                allow_support=allow_support,
                ignore_touch_geom_ids=ignore_touch_geom_ids,
            ):
                return False
        return True

    def _plan_start_and_trajectory(
        self,
        base_offset,
        enable_random_start: bool,
        random_start_xy_range: float,
        random_start_z_range: float,
        random_start_max_attempts: int,
    ):
        """Plan start position and arc trajectory.

        The arc path lifts above both endpoints, so collision-free sampling
        is no longer needed — the obstacle simply flies over cabinet edges.
        """
        if not self._spawn_diversity_enabled:
            if enable_random_start:
                start = self._sample_start_pos(
                    base_offset=base_offset,
                    xy_range=max(0.0, random_start_xy_range),
                    z_range=max(0.0, random_start_z_range),
                )
            else:
                start = self.final_pos + base_offset

            start = np.array(start, dtype=np.float64, copy=True)
            if self._align_obstacle_start_y_to_eef:
                # Match obstacle start Y to robot EEF Y (world frame) so motion stays
                # in a similar image column; must run after random jitter, before arc.
                sid = self.inner_env.robots[0].eef_site_id
                start[1] = float(self.inner_env.sim.data.site_xpos[sid][1])

            path = self._build_arc_path(start)
            return start, path

        return self._plan_diverse_start_and_trajectory(
            base_offset=base_offset,
            enable_random_start=enable_random_start,
            random_start_xy_range=random_start_xy_range,
            random_start_z_range=random_start_z_range,
        )

    def _legacy_jitter_arc(
        self,
        base_offset,
        enable_random_start: bool,
        random_start_xy_range: float,
        random_start_z_range: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Status-quo jitter + arc used by the ``legacy_jitter`` fallback."""
        if enable_random_start:
            start = self._sample_start_pos(
                base_offset=base_offset,
                xy_range=max(0.0, random_start_xy_range),
                z_range=max(0.0, random_start_z_range),
            )
        else:
            start = self.final_pos + base_offset
        start = np.array(start, dtype=np.float64, copy=True)
        path = self._build_arc_path(start)
        return start, path

    def _fixture_top_fn(self):
        z_land = float(self.final_pos[2])
        sim = self.inner_env.sim

        def _fn(xy):
            return _highest_fixture_top_at_xy(sim, xy, z_land=z_land)

        return _fn

    def _clearance_top_fn(self):
        """Merged fixture + movable-prop ceiling for layer-A solvers.

        ``max(fixture_top(xy), prop_top(xy))``. Arc has no inverse solve
        but reports ``min_clearance_m`` on this same ceiling so styles
        are comparable.
        """
        z_land = float(self.final_pos[2])
        sim = self.inner_env.sim
        exclude = set(self.obstacle_geom_ids)
        exclude.update(getattr(self, "_robot_like_geom_ids", ()) or ())
        exclude.update(getattr(self, "_companion_pedestal_geom_ids", ()) or ())

        def _fn(xy):
            fixture = _highest_fixture_top_at_xy(sim, xy, z_land=z_land)
            prop = _highest_movable_prop_top_at_xy(sim, xy, exclude_geom_ids=exclude)
            return float(max(fixture, prop))

        return _fn

    def _min_clearance_along_path(self, path: np.ndarray) -> float:
        fixture_top = self._clearance_top_fn()
        cfg = self._spawn_diversity
        tail_n = 0 if cfg is None else int(cfg["tail_exclude_waypoints"])
        n = int(path.shape[0])
        end = max(1, n - tail_n)
        best: float | None = None
        bottom = float(getattr(self, "_body_to_bottom_offset", 0.0))
        for p in path[:end]:
            clearance = float(p[2]) - bottom - fixture_top(p[:2])
            best = clearance if best is None else min(best, clearance)
        return float("nan") if best is None else float(best)

    def _scan_spawn_tail_contact(self, path: np.ndarray, judged_end: int) -> bool:
        """Sweep tail waypoints with the same criterion; record only."""
        n = int(path.shape[0])
        if judged_end >= n:
            return False
        ignore = self._ground_like_geom_ids
        prev = path[judged_end - 1] if judged_end > 0 else path[0]
        for i in range(max(judged_end, 1), n):
            allow = i == n - 1
            if not self._swept_move_is_collision_free(
                prev,
                path[i],
                allow_support_at_end=allow,
                ignore_touch_geom_ids=ignore,
            ):
                return True
            prev = path[i]
        return False

    def _validate_spawn_path(self, start, path) -> tuple[bool, bool]:
        """Layer-B accept/reject plus tail recording.

        Planning-time table/floor/ground touch whitelist is intentionally
        **not** the runtime whitelist in ``_try_move``. Resting on the
        table is allowed while planning so a spawn can be accepted; the
        same contact can still roll the obstacle back during execution.
        ``rollback_steps`` is the signal that records that gap.

        Returns ``(accepted, tail_contact)``. Accept/reject covers only the
        first ``n - tail_exclude_waypoints`` points (runtime
        ``near_final = idx >= len(path) - 3``). The start pose is checked
        separately because ``_swept_move_is_collision_free`` starts at
        ``i=1`` and never tests ``from_pos``.
        """
        path = np.asarray(path, dtype=np.float64)
        start_xyz = np.asarray(start, dtype=np.float64).reshape(3)
        n = int(path.shape[0])
        cfg = self._spawn_diversity
        if cfg is None:
            raise RuntimeError("spawn_diversity required for path validation")
        tail_n = int(cfg["tail_exclude_waypoints"])
        judged_end = max(0, n - tail_n)
        ignore = self._ground_like_geom_ids

        self._set_pos(start_xyz, self.final_quat)
        if self._has_blocking_contact(allow_support=False, ignore_touch_geom_ids=ignore):
            return False, self._scan_spawn_tail_contact(path, judged_end)

        if n > 0 and not np.allclose(start_xyz, path[0]):
            self._set_pos(path[0], self.final_quat)
            if self._has_blocking_contact(allow_support=False, ignore_touch_geom_ids=ignore):
                return False, self._scan_spawn_tail_contact(path, judged_end)

        for i in range(1, judged_end):
            if not self._swept_move_is_collision_free(
                path[i - 1],
                path[i],
                allow_support_at_end=False,
                ignore_touch_geom_ids=ignore,
            ):
                return False, self._scan_spawn_tail_contact(path, judged_end)

        return True, self._scan_spawn_tail_contact(path, judged_end)

    def _make_diversity_candidate(
        self,
        mode: str,
        *,
        p0: np.ndarray,
        n_hat: np.ndarray,
        base_offset,
        enable_random_start: bool,
        random_start_xy_range: float,
        random_start_z_range: float,
    ) -> dict[str, Any] | None:
        cfg = self._spawn_diversity
        if cfg is None:
            return None
        n_samples = libero_l2_path_num_samples(self.move_start_step, self.move_end_step)
        landing = np.asarray(self.final_pos, dtype=np.float64)

        if mode == LEGACY_MODE:
            start, path = self._legacy_jitter_arc(
                base_offset,
                enable_random_start,
                random_start_xy_range,
                random_start_z_range,
            )
            return {
                "start": start,
                "path": np.ascontiguousarray(path, dtype=np.float64),
                "linear_solve_mode": "na",
                "min_clearance_m": None,
            }

        slot, sep, style = str(mode).partition("_")
        if not sep or slot not in ("left", "center", "right"):
            return None
        start_xy = libero_l2_candidate_start_xy(p0, n_hat, slot, float(cfg["lateral_width_m"]))

        if style == "arc":
            start = np.array(
                [float(start_xy[0]), float(start_xy[1]), float(p0[2])],
                dtype=np.float64,
            )
            path = libero_l2_build_path(
                start,
                landing,
                "arc",
                n_samples=n_samples,
                arc_height=self.ARC_HEIGHT,
            )
            path = np.ascontiguousarray(path, dtype=np.float64)
            return {
                "start": start,
                "path": path,
                "linear_solve_mode": "na",
                "min_clearance_m": self._min_clearance_along_path(path),
            }

        if style != "linear":
            return None

        fixture_top = self._clearance_top_fn()
        # Solvers constrain body-origin z. Adding bottom_offset makes that
        # equivalent to a bottom-face clearance of linear_z_clearance_m.
        origin_clearance = float(cfg["linear_z_clearance_m"]) + float(
            getattr(self, "_body_to_bottom_offset", 0.0)
        )
        z_start, _lin_reason = libero_l2_solve_linear_start_z(
            start_xy,
            landing,
            fixture_top,
            clearance_m=origin_clearance,
            tail_alpha=float(cfg["linear_tail_alpha"]),
            max_start_rise_m=float(cfg["linear_max_start_z_m"]),
            samples=int(cfg["clearance_samples"]),
            p0_z_hint=float(p0[2]),
        )
        if z_start is not None:
            start = np.array(
                [float(start_xy[0]), float(start_xy[1]), float(z_start)],
                dtype=np.float64,
            )
            path = libero_l2_build_path(
                start,
                landing,
                "linear",
                n_samples=n_samples,
                arc_height=0.0,
            )
            path = np.ascontiguousarray(path, dtype=np.float64)
            return {
                "start": start,
                "path": path,
                "linear_solve_mode": "straight",
                "min_clearance_m": self._min_clearance_along_path(path),
            }

        z_cruise, trap_reason = libero_l2_solve_trapezoid(
            start_xy,
            landing,
            fixture_top,
            clearance_m=origin_clearance,
            max_start_rise_m=float(cfg["linear_max_start_z_m"]),
            descent_alpha=float(cfg["linear_descent_alpha"]),
            samples=int(cfg["clearance_samples"]),
        )
        if z_cruise is None:
            return {"infeasible_reason": str(trap_reason)}

        start = np.array(
            [float(start_xy[0]), float(start_xy[1]), float(p0[2])],
            dtype=np.float64,
        )
        path = libero_l2_build_path(
            start,
            landing,
            "trapezoid",
            n_samples=n_samples,
            arc_height=0.0,
            cruise_z=float(z_cruise),
            descent_alpha=float(cfg["linear_descent_alpha"]),
        )
        path = np.ascontiguousarray(path, dtype=np.float64)
        return {
            "start": start,
            "path": path,
            "linear_solve_mode": "trapezoid",
            "min_clearance_m": self._min_clearance_along_path(path),
        }

    def _plan_diverse_start_and_trajectory(
        self,
        base_offset,
        enable_random_start: bool,
        random_start_xy_range: float,
        random_start_z_range: float,
    ):
        cfg = self._spawn_diversity
        if cfg is None:
            raise RuntimeError("spawn_diversity missing while diversity planning")
        p0 = np.asarray(self.final_pos + base_offset, dtype=np.float64)
        robot_xy = np.asarray(get_robot_base_pos(self.env), dtype=np.float64)[:2]
        table_xy = np.asarray(get_table_center_pos(self.env), dtype=np.float64)[:2]
        n_hat, degenerate = libero_l2_lateral_axis(robot_xy, table_xy)
        forbidden = tuple(cfg["forbidden_slots"])
        requested = libero_l2_select_mode(
            self._mode_seed,
            self._task_id,
            self._episode_index,
            forbidden_slots=forbidden,
        )
        tail_box = {"tail_contact": False}

        def make_candidate(mode: str):
            return self._make_diversity_candidate(
                mode,
                p0=p0,
                n_hat=n_hat,
                base_offset=base_offset,
                enable_random_start=enable_random_start,
                random_start_xy_range=random_start_xy_range,
                random_start_z_range=random_start_z_range,
            )

        def validate(start, path) -> bool:
            ok, tail_hit = self._validate_spawn_path(start, path)
            if ok:
                tail_box["tail_contact"] = bool(tail_hit)
            return bool(ok)

        with self._sim_state_guard():
            result = libero_l2_accept_spawn(
                requested,
                forbidden_slots=forbidden,
                make_candidate=make_candidate,
                validate=validate,
            )

        path = result.get("path")
        if path is None:
            start, path = self._legacy_jitter_arc(
                base_offset,
                enable_random_start,
                random_start_xy_range,
                random_start_z_range,
            )
            result = {
                **result,
                "start": start,
                "path": path,
                "linear_solve_mode": "na",
                "min_clearance_m": None,
                "accepted_mode": LEGACY_MODE,
                "fallback_reason": result.get("fallback_reason") or "all_candidates_rejected",
            }
        path = np.ascontiguousarray(path, dtype=np.float64)
        start = np.array(path[0], dtype=np.float64, copy=True)
        accepted = str(result.get("accepted_mode") or LEGACY_MODE)
        if accepted == LEGACY_MODE:
            slot: str | None = None
            path_style: str | None = "arc"
            jitter_suppressed = False
        else:
            slot, _, path_style = accepted.partition("_")
            if result.get("linear_solve_mode") == "trapezoid":
                path_style = "trapezoid"
            jitter_suppressed = True
        min_c = result.get("min_clearance_m")
        self.spawn_plan = {
            "backend": "libero",
            "enabled": True,
            "requested_mode": result.get("requested_mode"),
            "accepted_mode": accepted,
            "slot": slot,
            "path_style": path_style,
            "fallback_reason": result.get("fallback_reason"),
            "fallback_chain": list(result.get("fallback_chain") or ()),
            "start": [float(v) for v in start],
            "lateral_axis": [float(v) for v in n_hat],
            "lateral_axis_degenerate": bool(degenerate),
            "min_clearance_m": None if min_c is None else float(min_c),
            "linear_solve_mode": result.get("linear_solve_mode") or "na",
            "tail_contact": bool(tail_box["tail_contact"]),
            "validated_path_sha1": _path_sha1(path),
            "rollback_steps": 0,
            "start_settle_rollback": False,
            "jitter_suppressed": bool(jitter_suppressed),
            "body_to_bottom_offset_m": float(self._body_to_bottom_offset),
        }
        return start, path

    def _zero_vel(self):
        """Zero the obstacle velocity so it does not fly away on release."""
        self.inner_env.sim.data.qvel[self._dof_addr : self._dof_addr + 6] = 0.0

    def _has_robot_contact(self) -> bool:
        """True if the obstacle is currently in contact with the robot arm."""
        if not self._robot_geom_ids:
            return False
        sim = self.inner_env.sim
        for i in range(sim.data.ncon):
            c = sim.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 in self._robot_geom_ids and g2 in self.obstacle_geom_ids) or (
                g2 in self._robot_geom_ids and g1 in self.obstacle_geom_ids
            ):
                return True
        return False

    def _runtime_blocking_other_geoms(self, allow_support: bool) -> list[int]:
        """Other-geom IDs that fail the runtime (no-whitelist) contact rule."""
        hits: list[int] = []
        sim = self.inner_env.sim
        for i in range(sim.data.ncon):
            c = sim.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            g1_is_obs = g1 in self.obstacle_geom_ids
            g2_is_obs = g2 in self.obstacle_geom_ids
            if g1_is_obs == g2_is_obs:
                continue
            other = g2 if g1_is_obs else g1
            if other in self._robot_like_geom_ids:
                continue
            if c.dist < -self.PENETRATION_EPS:
                hits.append(int(other))
                continue
            if (not allow_support) and c.dist <= self.CONTACT_EPS:
                hits.append(int(other))
        return hits

    def _is_start_settle_rollback(self, path_idx: int | None, allow_support: bool) -> bool:
        """True for the idx=0 table-settle no-op (metric only; motion unchanged).

        Wait-phase pinning + contact solver can press a table-resting start
        a few hundred µm into the table. The idx=0 near-zero sweep then
        fails the runtime (no-whitelist) rule. That is not a mid-path
        skip. Count it only when diversity is on, idx is 0, and every
        runtime-blocking geom is ground-like.
        """
        if not self._spawn_diversity_enabled:
            return False
        if path_idx != 0:
            return False
        others = self._runtime_blocking_other_geoms(allow_support)
        if not others:
            return False
        ground = getattr(self, "_ground_like_geom_ids", frozenset()) or frozenset()
        return all(gid in ground for gid in others)

    def _try_move(
        self, target_pos, allow_support: bool = False, path_idx: int | None = None
    ) -> bool:
        """Teleport to target_pos; roll back on collision.

        Scene-object collisions cause a per-step rollback (the obstacle retries
        on the next step with the next waypoint) rather than a permanent release,
        so the obstacle can still reach its destination even if an intermediate
        waypoint grazes scene geometry.

        Robot arm contact permanently releases the obstacle (``blocked`` latch).
        Force is measured at the first-touch pose on this segment, not at the
        penetrating waypoint; deep penetration zeros force but still counts
        the contact.

        ``rollback_steps`` counts mid-path skips only. When diversity is
        on, an idx=0 failure whose blocking geoms are all ground-like is
        recorded as ``start_settle_rollback`` instead — the motion still
        rolls back. Runtime contact rules are unchanged (no planning
        whitelist). With diversity off, every scene rollback is counted
        as today.

        Returns True if the move succeeded without blocking contact.
        """
        sim = self.inner_env.sim
        prev_qpos = sim.data.get_joint_qpos(self.joint_name).copy()

        prev_pos = prev_qpos[:3].copy()
        moved_ok = self._swept_move_is_collision_free(
            prev_pos, target_pos, allow_support_at_end=allow_support
        )
        if not moved_ok:
            settle = self._is_start_settle_rollback(path_idx, allow_support)
            sim.data.set_joint_qpos(self.joint_name, prev_qpos)
            self._zero_vel()
            _mj_forward(sim.model._model, sim.data._data)
            if settle:
                if isinstance(self.spawn_plan, dict):
                    self.spawn_plan["start_settle_rollback"] = True
            else:
                self.rollback_steps += 1
                if isinstance(self.spawn_plan, dict):
                    self.spawn_plan["rollback_steps"] = int(self.rollback_steps)
            return False

        if self._has_robot_contact():
            self._seek_first_robot_touch(prev_pos, target_pos)
            self._mark_blocked(stash=True)
            sim.data.set_joint_qpos(self.joint_name, prev_qpos)
            self._zero_vel()
            _mj_forward(sim.model._model, sim.data._data)
            return False
        return True

    def update_before_step(self, task_step: int):
        self._current_task_step = int(task_step)
        if self.released:
            return

        # Check contacts from the previous physics step BEFORE moving
        if not self.blocked and self._has_robot_contact():
            # Pose still overlaps; post-step measure will see it. Do not stash.
            self._mark_blocked(task_step)
            self._zero_vel()
            return

        if task_step < self.move_start_step:
            self._set_pos(self.start_pos, self.final_quat)
            return

        # Stabilisation: hold at target position
        if task_step >= self.move_end_step:
            if task_step < self._settle_end_step:
                self._set_pos(self.final_pos, self.final_quat)
                return
            self._zero_vel()
            self.released = True
            return

        # Controlled movement along pre-planned, validated trajectory.
        idx = task_step - self.move_start_step
        idx = int(np.clip(idx, 0, len(self._planned_path) - 1))
        target = self._planned_path[idx]
        near_final = idx >= len(self._planned_path) - 3
        self._try_move(target, allow_support=near_final, path_idx=idx)

    def check_robot_collision(self, robot_geom_ids) -> bool:
        """Called by episode loop after physics — latches blocked for metrics."""
        if self.blocked:
            return True
        sim = self.inner_env.sim
        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            is_hit = (g1 in robot_geom_ids and g2 in self.obstacle_geom_ids) or (
                g2 in robot_geom_ids and g1 in self.obstacle_geom_ids
            )
            if is_hit:
                self._mark_blocked()
                self._zero_vel()
                return True
        return False
