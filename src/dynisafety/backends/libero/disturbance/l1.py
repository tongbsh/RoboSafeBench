"""Level-1 sudden-drop disturbance (ported from SuddenDropController)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dynisafety.backends.libero.disturbance._support import (
    RESTING_Z_MODE_ANCHOR,
    VALID_RESTING_Z_MODES,
    _compute_body_to_bottom_offset,
    _scene_persistent_support_z,
    detect_visible_obstacles,
    relocate_obstacle_clear_of_task_objects,
)
from dynisafety.backends.libero.env import LiberoEnvHandle


def min_xy_dist(point_xy: np.ndarray, others_xy: Sequence[np.ndarray]) -> float | None:
    """Smallest planar distance from ``point_xy`` to any point in ``others_xy``."""
    if not others_xy:
        return None
    px, py = float(point_xy[0]), float(point_xy[1])
    return min(float(np.hypot(px - float(p[0]), py - float(p[1]))) for p in others_xy)


def min_xyz_dist(point_xyz: np.ndarray, others_xyz: Sequence[np.ndarray]) -> float | None:
    """Smallest 3D distance from ``point_xyz`` to any point in ``others_xyz``."""
    if not others_xyz:
        return None
    src = np.asarray(point_xyz, dtype=np.float64).reshape(-1)[:3]
    return min(
        float(np.linalg.norm(src - np.asarray(p, dtype=np.float64).reshape(-1)[:3]))
        for p in others_xyz
    )


def drop_unreachable(
    drop_xy: np.ndarray,
    task_xys: Sequence[np.ndarray],
    trigger_dist: float,
) -> bool:
    """True when no task object lies inside the EEF→drop trigger circle.

    Those episodes can never fire the standard SuddenDrop check if the policy
    only visits task objects. Workspace-onset is then enabled instead of
    widening ``trigger_dist``.
    """
    nearest = min_xy_dist(drop_xy, task_xys)
    return nearest is not None and nearest > float(trigger_dist)


def eef_inside_drop_trigger(eef_pos: np.ndarray, drop_xy: np.ndarray, trigger_dist: float) -> bool:
    """True when the EEF already sits inside the EEF→drop XY circle."""
    return float(
        np.hypot(
            float(eef_pos[0]) - float(drop_xy[0]),
            float(eef_pos[1]) - float(drop_xy[1]),
        )
    ) <= float(trigger_dist)


def l1_should_drop(
    eef_pos: np.ndarray,
    drop_xy: np.ndarray,
    task_positions: Sequence[np.ndarray],
    trigger_dist: float,
    workspace_onset: bool,
    init_min_task_dist: float | None = None,
    init_drop_xy_dist: float | None = None,
    onset_dist: float | None = None,
) -> bool:
    """L1 drop predicate. Radius is never expanded.

    Default: EEF XY within ``trigger_dist`` of the aligned drop point.

    If ``workspace_onset`` (unreachable drop, or EEF already inside the
    drop circle at bind): drop when the EEF has *approached* a task
    object in 3D — current 3D distance ≤ ``onset_dist`` (or
    ``trigger_dist`` when unset) **and** strictly closer than the
    bind-time clearance. When ``onset_dist`` is unset, the XY circle
    still fires if the EEF later moves closer to the drop than at bind.
    An explicit ``onset_dist`` disables that XY shortcut so a home pose
    already inside the drop circle cannot fire on a 1-step wiggle.
    Home hover that already sits in the circle must not drop at step 0.
    """
    radius = float(trigger_dist)
    approach = float(onset_dist) if onset_dist is not None else radius
    drop_dist = float(
        np.hypot(
            float(eef_pos[0]) - float(drop_xy[0]),
            float(eef_pos[1]) - float(drop_xy[1]),
        )
    )
    xy_hit = drop_dist <= radius
    if not workspace_onset:
        return xy_hit

    nearest_3d = min_xyz_dist(eef_pos, task_positions)
    approached = nearest_3d is not None and nearest_3d <= approach
    if approached:
        if init_min_task_dist is None:
            return True
        if nearest_3d < float(init_min_task_dist) - 1e-4:
            return True
    if onset_dist is not None:
        return False
    if not xy_hit:
        return False
    if init_drop_xy_dist is None:
        return True
    return drop_dist < float(init_drop_xy_dist) - 1e-4


class SuddenDropController:
    """Dynamic level 1: object drops when EEF approaches trigger distance."""

    Z_SOLVE_RAISE = 0.30
    Z_SOLVE_STEPS = 240
    PENETRATION_EPS = 1e-4
    Z_LIFT_STEP = 0.002
    Z_LIFT_MAX = 0.06

    def __init__(
        self,
        handle: LiberoEnvHandle,
        *,
        trigger_dist: float,
        drop_height: float,
        drop_z_bias: float,
        trigger_dist_bias: float,
        resting_z_mode: str,
        drop_normalize_z: bool,
        obstacle_pool: Sequence[str],
        allowlist: Sequence[str],
        seed: int,
        task_object_names: Sequence[str],
        task_clearance_xy: float,
        pedestal_overlap_radius: float,
        home_workspace_onset: bool,
        nudge_on_penetration_only: bool,
        workspace_onset_dist: float | None,
    ) -> None:
        self.handle = handle
        self.env = handle.env
        self.inner_env = self.env.env
        if resting_z_mode not in VALID_RESTING_Z_MODES:
            raise ValueError(
                f"resting_z_mode must be one of {VALID_RESTING_Z_MODES}, got {resting_z_mode!r}"
            )
        self._trigger_dist_cfg = float(trigger_dist)
        self._trigger_dist_bias = float(trigger_dist_bias)
        self._drop_height = float(drop_height)
        self._drop_z_bias = float(drop_z_bias)
        self._resting_z_mode = resting_z_mode
        self._normalize_drop_z = bool(drop_normalize_z)
        self._obstacle_pool = tuple(obstacle_pool)
        self._allowlist = tuple(allowlist)
        self._seed = int(seed)
        self._task_object_names = tuple(task_object_names)
        self._task_clearance_xy = float(task_clearance_xy)
        if self._task_clearance_xy < 0.0:
            raise ValueError(f"task_object_clearance_xy must be >= 0, got {task_clearance_xy!r}")
        self._pedestal_overlap_radius = float(pedestal_overlap_radius)
        if self._pedestal_overlap_radius < 0.0:
            raise ValueError(
                f"pedestal_overlap_radius must be >= 0, got {pedestal_overlap_radius!r}"
            )
        self.obstacle_name: str | None = None
        self.companion_pedestal_name: str | None = None
        self._bound = False
        self.resting_z_source: str | None = None
        self._dropped = False
        self.drop_step = None
        self.blocked = False
        self._home_workspace_onset = bool(home_workspace_onset)
        self._nudge_on_penetration_only = bool(nudge_on_penetration_only)
        if workspace_onset_dist is None:
            self._workspace_onset_dist: float | None = None
        else:
            onset = float(workspace_onset_dist)
            if onset <= 0.0:
                raise ValueError(
                    f"l1_workspace_onset_dist must be > 0, got {workspace_onset_dist!r}"
                )
            self._workspace_onset_dist = onset
        self._workspace_onset = False
        self._onset_baseline_ready = True
        self._init_min_task_3d: float | None = None
        self._init_drop_xy_dist: float | None = None

    def reset(self, episode_index: int) -> None:
        visible_raw = detect_visible_obstacles(self.env)
        if not visible_raw:
            self._bound = False
            self.obstacle_name = None
            self.companion_pedestal_name = None
            self._workspace_onset = False
            self._onset_baseline_ready = True
            self._init_min_task_3d = None
            self._init_drop_xy_dist = None
            return
        visible_lookup = {name: (pos, quat) for name, pos, quat in visible_raw}
        if self._allowlist:
            pool = [n for n in self._allowlist if n in visible_lookup]
        else:
            pool = [n for n in self._obstacle_pool if n in visible_lookup]
        if not pool:
            eef = self.inner_env.sim.data.site_xpos[self.inner_env.robots[0].eef_site_id]
            visible_raw = sorted(visible_raw, key=lambda item: float(np.linalg.norm(item[1] - eef)))
            chosen = visible_raw[0][0]
        else:
            chosen = pool[int(episode_index) % len(pool)]
        target_pos, target_quat = visible_lookup[chosen]
        self._bind(chosen, target_pos, target_quat)

    def _bind(
        self,
        obstacle_name: str,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
    ) -> None:
        self.obstacle_name = obstacle_name
        obj = self.inner_env.objects_dict[obstacle_name]
        self._joint = obj.joints[-1]
        self._body_id = self.inner_env.sim.model.body_name2id(obj.root_body)
        jid = self.inner_env.sim.model.joint_name2id(self._joint)
        self._dof_addr = self.inner_env.sim.model.jnt_dofadr[jid]
        self.target_pos = np.asarray(target_pos, dtype=np.float64).copy()
        self.target_quat = np.asarray(target_quat, dtype=np.float64).copy()
        self._trigger_dist = self._trigger_dist_cfg + self._trigger_dist_bias
        self._eef_site_id = self.inner_env.robots[0].eef_site_id
        self.obstacle_geom_ids = set()
        model = self.inner_env.sim.model
        for gid in range(model.ngeom):
            bid = model.geom_bodyid[gid]
            if bid == self._body_id or model.body_parentid[bid] == self._body_id:
                self.obstacle_geom_ids.add(gid)
        new_xy, pedestal, notes = relocate_obstacle_clear_of_task_objects(
            self.handle,
            obstacle_name,
            self.target_pos[:2],
            self._task_object_names,
            clearance_xy=self._task_clearance_xy,
            pedestal_overlap_radius=self._pedestal_overlap_radius,
            require_penetration=self._nudge_on_penetration_only,
        )
        self.companion_pedestal_name = pedestal
        if notes:
            print(
                f"[SuddenDropController] guard nudged drop xy: "
                f"obstacle={obstacle_name!r} "
                f"orig=({self.target_pos[0]:+.3f}, {self.target_pos[1]:+.3f}) "
                f"-> ({new_xy[0]:+.3f}, {new_xy[1]:+.3f}) " + "; ".join(notes),
                flush=True,
            )
        self.target_pos[0] = float(new_xy[0])
        self.target_pos[1] = float(new_xy[1])
        if self._normalize_drop_z:
            resolved_z = self._resolve_resting_z(self.target_pos, self.target_quat)
            self.target_pos[2] = resolved_z
            self.drop_pos = self.target_pos.copy()
            self.drop_pos[2] = resolved_z + self._drop_height
        else:
            self.drop_pos = self.target_pos.copy()
            self.drop_pos[2] += self._drop_height
        if self._drop_z_bias != 0.0:
            self.drop_pos[2] += self._drop_z_bias
        self._hidden_pos = np.array([10.0, 10.0, 0.0])
        self._hidden_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._dropped = False
        self.drop_step = None
        self.blocked = False
        self._bound = True
        task_xyzs = self._task_object_xyzs()
        task_xys = [p[:2] for p in task_xyzs]
        eef = self.inner_env.sim.data.site_xpos[self._eef_site_id]
        self._capture_onset_baseline(eef, task_xyzs)
        unreachable = drop_unreachable(self.target_pos[:2], task_xys, self._trigger_dist)
        already_inside = eef_inside_drop_trigger(eef, self.target_pos[:2], self._trigger_dist)
        # Home-circle defer is opt-in (long task 0). Other suites keep
        # unreachable-only workspace-onset measured at bind.
        self._workspace_onset = bool(unreachable or (self._home_workspace_onset and already_inside))
        self._onset_baseline_ready = not (self._workspace_onset and self._home_workspace_onset)
        if self._workspace_onset:
            nearest = min_xy_dist(self.target_pos[:2], task_xys)
            init_3d = (
                f"{self._init_min_task_3d:.3f}m" if self._init_min_task_3d is not None else "n/a"
            )
            reason = (
                f"min_task_xy={nearest:.3f}m > trigger={self._trigger_dist:.3f}m"
                if unreachable
                else (
                    "eef already inside drop trigger at bind "
                    f"({self._init_drop_xy_dist:.3f}m ≤ "
                    f"{self._trigger_dist:.3f}m)"
                )
            )
            onset = (
                self._workspace_onset_dist
                if self._workspace_onset_dist is not None
                else self._trigger_dist
            )
            print(
                f"[SuddenDropController] workspace-onset: "
                f"obstacle='{self.obstacle_name}' {reason} "
                f"eef_init_3d={init_3d} onset_3d={onset:.3f}m "
                f"(XY trigger unchanged; drop when EEF approaches a task object in 3D)"
            )
        self._teleport_hidden()

    def step(self, t: int, obs: Mapping[str, Any]) -> None:
        del obs
        if self._bound:
            self.update_before_step(t)

    @property
    def active_obstacle_name(self) -> str | None:
        return self.obstacle_name

    @property
    def disturbance_onset_step(self) -> int | None:
        return self.drop_step

    @property
    def workspace_onset(self) -> bool:
        return bool(self._workspace_onset)

    def _task_object_xyzs(self) -> list[np.ndarray]:
        """Current workspace XYZ of BDDL task objects (hidden bodies skipped)."""
        sim = self.inner_env.sim
        xyzs: list[np.ndarray] = []
        for name in self._task_object_names:
            obj = self.inner_env.objects_dict.get(name)
            if obj is None:
                continue
            try:
                body_id = sim.model.body_name2id(obj.root_body)
            except Exception:
                continue
            pos = sim.data.body_xpos[body_id]
            if abs(float(pos[0])) >= 5.0 or abs(float(pos[1])) >= 5.0:
                continue
            xyzs.append(np.asarray(pos[:3], dtype=np.float64).copy())
        return xyzs

    def _resolve_resting_z(self, base_pos, base_quat):
        """Resolve the obstacle's body-origin Z when at rest.

        Dispatches to either the legacy physics-settle path or the new
        geometric-anchor path depending on ``self._resting_z_mode``.
        """
        if self._resting_z_mode == RESTING_Z_MODE_ANCHOR:
            return self._anchor_resting_z(base_pos, base_quat)
        return self._settle_resting_z(base_pos, base_quat)

    def _settle_resting_z(self, base_pos, base_quat):
        """Legacy: drop from ``Z_SOLVE_RAISE`` above and run a short physics
        settle to find the resting Z.  Sensitive to where other free-jointed
        objects are at episode start, so the resolved Z can drift between
        episodes and may not match the actual support at trigger time.
        """
        import mujoco

        sim = self.inner_env.sim
        saved_qpos = sim.data.qpos.copy()
        saved_qvel = sim.data.qvel.copy()
        saved_act = sim.data.act.copy() if sim.model.na > 0 else None

        try:
            try:
                table_sid = sim.model.site_name2id("table_top")
                table_z = float(sim.data.site_xpos[table_sid][2])
            except Exception:
                table_z = float(base_pos[2])

            probe = base_pos.copy()
            probe[2] = max(float(base_pos[2]), table_z) + self.Z_SOLVE_RAISE

            sim.data.set_joint_qpos(self._joint, np.concatenate([probe, base_quat]))
            sim.data.qvel[self._dof_addr : self._dof_addr + 6] = 0.0
            mujoco.mj_forward(sim.model._model, sim.data._data)

            for _ in range(self.Z_SOLVE_STEPS):
                mujoco.mj_step(sim.model._model, sim.data._data)

            settled = sim.data.get_joint_qpos(self._joint).copy()
            solved_pos = base_pos.copy()
            solved_pos[2] = settled[2]
            solved_pos = self._lift_until_no_penetration(solved_pos, base_quat)
            self.resting_z_source = "physics_settle"
            return float(solved_pos[2])
        finally:
            sim.data.qpos[:] = saved_qpos
            sim.data.qvel[:] = saved_qvel
            if saved_act is not None:
                sim.data.act[:] = saved_act
            mujoco.mj_forward(sim.model._model, sim.data._data)

    def _anchor_resting_z(self, base_pos, base_quat):
        """Geometric anchor: ``support_z + obstacle_bottom_offset``.

        The support is picked from *persistent* scene fixtures (table top,
        floor, cabinet etc.) so the chosen Z does not depend on where other
        movable objects happen to sit at episode init.  After the obstacle
        is teleported to this Z, normal MuJoCo physics takes over for
        contact / collision handling.
        """
        sim = self.inner_env.sim
        support_z, source = _scene_persistent_support_z(
            sim,
            float(base_pos[2]),
            target_xy=(float(base_pos[0]), float(base_pos[1])),
        )
        bottom_offset = _compute_body_to_bottom_offset(
            sim,
            self._joint,
            self._dof_addr,
            self._body_id,
            self.obstacle_geom_ids,
            base_quat,
        )
        self.resting_z_source = (
            f"anchor[{source}] support_z={support_z:.3f} + bottom_offset={bottom_offset:.3f}"
        )
        return support_z + bottom_offset

    def _set_pos(self, pos, quat):
        import mujoco

        sim = self.inner_env.sim
        sim.data.set_joint_qpos(self._joint, np.concatenate([pos, quat]))
        sim.data.qvel[self._dof_addr : self._dof_addr + 6] = 0.0
        mujoco.mj_forward(sim.model._model, sim.data._data)

    def _has_external_penetration(self) -> bool:
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

    def _lift_until_no_penetration(self, pos, quat):
        lifted = pos.copy()
        max_iters = int(self.Z_LIFT_MAX / self.Z_LIFT_STEP)
        for _ in range(max_iters):
            self._set_pos(lifted, quat)
            if not self._has_external_penetration():
                break
            lifted[2] += self.Z_LIFT_STEP
        return lifted

    def _teleport_hidden(self):
        import mujoco

        sim = self.inner_env.sim
        qpos = np.concatenate([self._hidden_pos, self._hidden_quat])
        sim.data.set_joint_qpos(self._joint, qpos)
        sim.data.qvel[self._dof_addr : self._dof_addr + 6] = 0.0
        mujoco.mj_forward(sim.model._model, sim.data._data)

    def _drop_object(self, task_step: int):
        import mujoco

        sim = self.inner_env.sim
        qpos = np.concatenate([self.drop_pos, self.target_quat])
        sim.data.set_joint_qpos(self._joint, qpos)
        sim.data.qvel[self._dof_addr : self._dof_addr + 6] = 0.0
        mujoco.mj_forward(sim.model._model, sim.data._data)
        self._dropped = True
        self.drop_step = task_step

    def _capture_onset_baseline(
        self, eef_pos: np.ndarray, task_xyzs: Sequence[np.ndarray] | None = None
    ) -> None:
        xyzs = list(task_xyzs) if task_xyzs is not None else self._task_object_xyzs()
        self._init_min_task_3d = min_xyz_dist(eef_pos, xyzs)
        self._init_drop_xy_dist = float(
            np.hypot(
                float(eef_pos[0]) - float(self.target_pos[0]),
                float(eef_pos[1]) - float(self.target_pos[1]),
            )
        )

    def _check_trigger(self):
        eef_pos = self.inner_env.sim.data.site_xpos[self._eef_site_id]
        return l1_should_drop(
            eef_pos,
            self.target_pos[:2],
            self._task_object_xyzs(),
            self._trigger_dist,
            self._workspace_onset,
            self._init_min_task_3d,
            self._init_drop_xy_dist,
            onset_dist=self._workspace_onset_dist,
        )

    def update_before_step(self, task_step: int):
        if self._dropped:
            return
        if (
            task_step >= 0
            and self._workspace_onset
            and self._home_workspace_onset
            and not self._onset_baseline_ready
        ):
            eef = self.inner_env.sim.data.site_xpos[self._eef_site_id]
            self._capture_onset_baseline(eef)
            self._onset_baseline_ready = True
        if task_step >= 0 and self._check_trigger():
            self._drop_object(task_step)
        else:
            self._teleport_hidden()

    def check_robot_collision(self, robot_geom_ids) -> bool:
        return False
