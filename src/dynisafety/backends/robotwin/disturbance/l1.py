"""Level-1 sudden-drop disturbance controller."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from dynisafety.backends.robotwin.disturbance._base import BaseDisturbanceController


def point_aabb_distance(point: np.ndarray, aabb_min: np.ndarray, aabb_max: np.ndarray) -> float:
    """Euclidean distance from a point to an axis-aligned box (0 if inside)."""
    closest = np.clip(np.asarray(point, dtype=np.float64), aabb_min, aabb_max)
    return float(np.linalg.norm(np.asarray(point, dtype=np.float64) - closest))


def eef_xy_dist(eef_pos: np.ndarray, drop_xy: np.ndarray) -> float:
    eef = np.asarray(eef_pos, dtype=np.float64).reshape(-1)
    drop = np.asarray(drop_xy, dtype=np.float64).reshape(-1)
    return float(np.hypot(float(eef[0]) - float(drop[0]), float(eef[1]) - float(drop[1])))


def eef_xy_should_latch(
    *,
    xy_dist: float,
    trigger_dist: float,
    init_xy_dist: float,
    home_inside: bool,
    approach_eps: float,
) -> bool:
    """XY latch. Home-inside episodes must get closer than the bind distance."""
    if not home_inside:
        return float(xy_dist) <= float(trigger_dist)
    return float(xy_dist) < float(init_xy_dist) - float(approach_eps)


def object_is_grasped(
    *,
    object_xyz: np.ndarray,
    eef_xyzs: list[np.ndarray],
    gripper_vals: list[float],
    grasp_dist: float,
    gripper_closed: float,
) -> bool:
    """True when any arm is closed and its EEF is within ``grasp_dist`` of the object."""
    if len(eef_xyzs) != len(gripper_vals):
        raise ValueError("eef_xyzs and gripper_vals must have the same length")
    obj = np.asarray(object_xyz, dtype=np.float64).reshape(-1)[:3]
    radius = float(grasp_dist)
    closed = float(gripper_closed)
    for eef, grip in zip(eef_xyzs, gripper_vals, strict=True):
        if float(grip) > closed:
            continue
        pos = np.asarray(eef, dtype=np.float64).reshape(-1)[:3]
        if float(np.linalg.norm(pos - obj)) <= radius:
            return True
    return False


def nudge_xy_away(current_xy: np.ndarray, from_xy: np.ndarray, step: float) -> np.ndarray:
    """Step ``current_xy`` away from ``from_xy``; +X if they coincide."""
    cur = np.asarray(current_xy, dtype=np.float64).reshape(-1)[:2].copy()
    src = np.asarray(from_xy, dtype=np.float64).reshape(-1)[:2]
    delta = cur - src
    norm = float(np.linalg.norm(delta))
    if norm <= 1e-8:
        cur[0] += float(step)
        return cur
    return cur + (delta / norm) * float(step)


class SuddenDropController(BaseDisturbanceController):
    """Spawn an obstacle at ``drop_pos`` once EEF xy latches and the volume is clear.

    Trigger stays horizontal. If the home EEF already sits in the trigger
    circle, latch only after the EEF moves closer than at bind. Spawn hold
    uses robot-link origins and the primary EEF. Release is delayed until
    the manager probes contacts (``needs_spawn_probe``).
    """

    def __init__(
        self,
        actor,
        *,
        target_pos: np.ndarray,
        trigger_dist: float,
        drop_height: float,
        get_eef_pos_fn: Callable[[], np.ndarray],
        quat: np.ndarray,
        obstacle_height: float,
        obstacle_radius_xy: float,
        obstacle_support_offset: float,
        get_robot_link_positions_fn: Callable[[], np.ndarray] | None,
        link_clearance_xy: float,
        z_clearance: float,
        table_surface_z: float,
        eef_clearance_xy: float,
        eef_work_height: float,
        eef_radius: float,
        home_approach_eps: float,
    ) -> None:
        super().__init__(actor, default_quat=quat)
        self.target_pos = np.asarray(target_pos, dtype=np.float64)
        self.trigger_dist = float(trigger_dist)
        self.drop_height = float(drop_height)
        self.get_eef_pos = get_eef_pos_fn
        self.drop_pos = self.target_pos.copy()
        self.drop_pos[2] += self.drop_height

        self.obstacle_height = float(max(obstacle_height, 0.0))
        self.obstacle_radius_xy = float(max(obstacle_radius_xy, 0.0))
        self.obstacle_support_offset = float(max(obstacle_support_offset, 0.0))
        self.get_robot_link_positions = get_robot_link_positions_fn
        self.link_clearance_xy = float(max(link_clearance_xy, 0.0))
        self.z_clearance = float(max(z_clearance, 0.0))
        self.table_surface_z = float(table_surface_z)
        self.eef_clearance_xy = float(max(eef_clearance_xy, 0.0))
        self.eef_work_height = float(max(eef_work_height, 0.0))
        self.eef_radius = float(max(eef_radius, 0.0))
        self.home_approach_eps = float(max(home_approach_eps, 0.0))

        init_eef = self._read_eef_xyz()
        if init_eef is None:
            self.init_xy_dist = float("inf")
            self.home_inside = False
        else:
            self.init_xy_dist = eef_xy_dist(init_eef, self.target_pos[:2])
            self.home_inside = self.init_xy_dist <= self.trigger_dist

        self._dropped = False
        self._xy_latched = False
        self._probe_pending = False
        self._placed_sim_step: int | None = None
        self.drop_step: int | None = None
        self.hide()

    @property
    def disturbance_onset_step(self) -> int | None:
        """Task-step onset. Survives ``on_released_to_physics`` (probe clock)."""
        return self.drop_step

    def lock_drop_task_step(self, task_step: int) -> None:
        """Latch the first task-step sighting of a drop. Never overwrite."""
        if self.drop_step is None:
            self.drop_step = int(task_step)

    def apply_drop_xy(self, xy: np.ndarray) -> None:
        self.drop_pos[0] = float(xy[0])
        self.drop_pos[1] = float(xy[1])
        self.target_pos[0] = float(xy[0])
        self.target_pos[1] = float(xy[1])

    def needs_spawn_probe(self) -> bool:
        return bool(self._probe_pending)

    def just_placed(self, sim_step: int) -> bool:
        return self._placed_sim_step is not None and int(sim_step) == int(self._placed_sim_step)

    def mark_placed(self, sim_step: int) -> None:
        self._placed_sim_step = int(sim_step)
        self._probe_pending = True

    def on_released_to_physics(self) -> None:
        super().on_released_to_physics()
        self._probe_pending = False
        self._placed_sim_step = None

    def _spawn_volume_z_range(self) -> tuple[float, float]:
        bottom_z = self.drop_pos[2] - self.obstacle_support_offset
        top_z = bottom_z + self.obstacle_height
        return bottom_z - self.z_clearance, top_z + self.z_clearance

    def _obstacle_aabb(self) -> tuple[np.ndarray, np.ndarray]:
        radius = self.obstacle_radius_xy
        bottom_z = self.drop_pos[2] - self.obstacle_support_offset
        top_z = bottom_z + self.obstacle_height
        center_xy = self.target_pos[:2]
        mins = np.array(
            [center_xy[0] - radius, center_xy[1] - radius, bottom_z], dtype=np.float64
        )
        maxs = np.array([center_xy[0] + radius, center_xy[1] + radius, top_z], dtype=np.float64)
        return mins, maxs

    def _link_blocks_spawn(self) -> bool:
        if self.get_robot_link_positions is None or self.obstacle_height <= 0.0:
            return False
        try:
            link_xyz = np.asarray(self.get_robot_link_positions(), dtype=np.float64)
        except Exception:
            return False
        if link_xyz.size == 0:
            return False
        if link_xyz.ndim != 2 or link_xyz.shape[1] < 3:
            return False

        radius = self.obstacle_radius_xy + self.link_clearance_xy
        if radius <= 0.0:
            return False
        z_lo, z_hi = self._spawn_volume_z_range()

        dxy = link_xyz[:, :2] - self.target_pos[:2][None, :]
        in_disk = (dxy[:, 0] ** 2 + dxy[:, 1] ** 2) <= (radius * radius)
        z = link_xyz[:, 2]
        in_band = (z >= z_lo) & (z <= z_hi)
        return bool(np.any(in_disk & in_band))

    def _read_eef_xyz(self) -> np.ndarray | None:
        try:
            eef = np.asarray(self.get_eef_pos(), dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if eef.size < 3 or not np.all(np.isfinite(eef[:3])):
            return None
        return eef[:3].copy()

    def _eef_in_work_cylinder(self, eef: np.ndarray) -> bool:
        radius = self.obstacle_radius_xy + self.eef_clearance_xy
        if radius <= 0.0:
            return False
        dxy = float(np.linalg.norm(eef[:2] - self.target_pos[:2]))
        if dxy > radius:
            return False
        return float(eef[2]) <= self.table_surface_z + self.eef_work_height

    def _eef_too_close_to_aabb(self, eef: np.ndarray) -> bool:
        gap = self.eef_radius + self.eef_clearance_xy
        if gap <= 0.0:
            return False
        mins, maxs = self._obstacle_aabb()
        return point_aabb_distance(eef, mins, maxs) <= gap

    def _eef_blocks_spawn(self) -> bool:
        """True when the primary EEF still occupies the drop volume."""
        eef = self._read_eef_xyz()
        if eef is None:
            return True
        return self._eef_in_work_cylinder(eef) or self._eef_too_close_to_aabb(eef)

    def _spawn_blocked(self) -> bool:
        return self._link_blocks_spawn() or self._eef_blocks_spawn()

    def update_before_step(self, sim_step: int) -> None:
        if self._dropped:
            self._active = True
            return

        eef_pos = np.asarray(self.get_eef_pos(), dtype=np.float64)
        xy_dist = eef_xy_dist(eef_pos, self.target_pos[:2])
        if not self._xy_latched:
            if eef_xy_should_latch(
                xy_dist=xy_dist,
                trigger_dist=self.trigger_dist,
                init_xy_dist=self.init_xy_dist,
                home_inside=self.home_inside,
                approach_eps=self.home_approach_eps,
            ):
                self._xy_latched = True
            else:
                self.hide()
                return

        if self._spawn_blocked():
            self.hide()
            return

        self._dropped = True
        self._active = True
        self._set_pose(self.drop_pos)
        self.mark_placed(int(sim_step))
