"""Level-3 dest-apron approaching human (then leave; not table-top Shared Reaching)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dynisafety.backends.robocasa.disturbance._support import (
    UPRIGHT_QUAT_WXYZ,
    Phase,
    actor_has_robot_contact,
    base_before_land,
    cameras_seeing_point,
    collect_body_geom_ids,
    l1_path_spawn_xy,
    l1_spawn_failed_flag,
    l2_lateral_start_xy,
    l3_meta,
    l3_should_appear,
    minimum_jerk,
    path_unit_and_length,
    pos_from_xy_floor,
    probe_floor_placement,
    read_base_pose,
    read_body_pos,
    read_floor_z,
    set_actor_contact_enabled,
    set_free_joint_pose,
    snapshot_actor_geom_contact,
    teleport_actor_hidden,
)
from dynisafety.backends.robocasa.env import RoboCasaEnvHandle, read_nav_dest_xy


def yaw_quat_wxyz(from_xy: Sequence[float], to_xy: Sequence[float]) -> np.ndarray:
    """wxyz yaw so a +X-facing mesh looks toward ``to_xy``."""
    delta = np.asarray(to_xy, dtype=np.float64).reshape(2) - np.asarray(
        from_xy, dtype=np.float64
    ).reshape(2)
    if float(np.linalg.norm(delta)) < 1e-8:
        return UPRIGHT_QUAT_WXYZ.copy()
    yaw = float(np.arctan2(delta[1], delta[0]))
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


class ApproachingHumanController:
    """HIDDEN → ENTER → APPROACH (min-jerk floor XY) → DWELL → WITHDRAW → DONE.

    Pedestrian walks onto the dest apron (p_land) from the side, stands, then
    leaves. Onset requires the chassis still before p_land along u. Collision:
    FLINCH on the floor then hide. toppled stays None (caller skips).
    """

    hides_obstacles = True

    def __init__(
        self,
        handle: RoboCasaEnvHandle,
        *,
        goal_clearance_m: float,
        human_start_lateral: float,
        trigger_dist: float,
        entry_steps: int,
        dwell_steps: int,
        approach_steps: int,
        withdraw_steps: int,
        flinch_dist: float,
        flinch_steps: int,
        retry_step: float,
        n_attempts: int,
        flip_side: bool,
        half_size: Sequence[float],
        min_preview_steps: int,
        close_approach_m: float,
        visibility_cameras: Sequence[str],
        camera_aspect: float,
    ) -> None:
        self.handle = handle
        self.env = handle.env
        self.goal_clearance_m = float(goal_clearance_m)
        self.human_start_lateral = float(human_start_lateral)
        if self.human_start_lateral < 0.0:
            raise ValueError(
                f"human_start_lateral must be >= 0, got {self.human_start_lateral}"
            )
        self.trigger_dist = float(trigger_dist)
        if self.trigger_dist < 0.0:
            raise ValueError(f"trigger_dist must be >= 0, got {self.trigger_dist}")
        self._entry_steps = int(entry_steps)
        self._dwell_steps = int(dwell_steps)
        self._approach_steps = int(approach_steps)
        self._withdraw_steps = int(withdraw_steps)
        if self._entry_steps < 1:
            raise ValueError(f"entry_steps must be >= 1, got {self._entry_steps}")
        if self._dwell_steps < 0:
            raise ValueError(f"dwell_steps must be >= 0, got {self._dwell_steps}")
        if self._approach_steps < 2:
            raise ValueError(f"approach_steps must be >= 2, got {self._approach_steps}")
        if self._withdraw_steps < 2:
            raise ValueError(f"withdraw_steps must be >= 2, got {self._withdraw_steps}")
        self._flinch_dist = float(flinch_dist)
        self._flinch_steps = int(flinch_steps)
        if self._flinch_steps < 1:
            raise ValueError(f"flinch_steps must be >= 1, got {self._flinch_steps}")
        self.retry_step = float(retry_step)
        self.n_attempts = int(n_attempts)
        if self.n_attempts < 1:
            raise ValueError(f"n_attempts must be >= 1, got {self.n_attempts}")
        self.flip_side = bool(flip_side)
        hs = [float(x) for x in half_size]
        if len(hs) != 3:
            raise ValueError(f"half_size must be length 3, got {hs!r}")
        self.half_size = hs
        self._min_start_clearance = 2.0 * max(hs[0], hs[1])
        self._min_preview_steps = int(min_preview_steps)
        if self._min_preview_steps < 0:
            raise ValueError(
                f"min_preview_steps must be >= 0, got {self._min_preview_steps}"
            )
        self._close_approach_m = float(close_approach_m)
        if self._close_approach_m < 0.0:
            raise ValueError(
                f"close_approach_m must be >= 0, got {self._close_approach_m}"
            )
        cams = [str(name).strip() for name in visibility_cameras if str(name).strip()]
        if not cams:
            raise ValueError("visibility_cameras must contain at least one camera name")
        self._visibility_cameras = tuple(dict.fromkeys(cams))
        self._camera_aspect = float(camera_aspect)
        if self._camera_aspect <= 0.0:
            raise ValueError(f"camera_aspect must be > 0, got {self._camera_aspect}")

        meta = l3_meta(self.env)
        self._meta = meta
        self.body_name = str(meta["body_name"])
        self.joint_name = str(meta["joint_name"])
        self.obstacle_name = str(meta.get("object_name") or self.body_name)
        self.obstacle_geom_ids = collect_body_geom_ids(self.env, self.body_name)
        if "geom_contact" not in meta:
            meta_mut = dict(meta)
            meta_mut["geom_contact"] = snapshot_actor_geom_contact(self.env, meta_mut)
            self.env._dynisafety_l3 = meta_mut
            self._meta = meta_mut

        self._phase = Phase.HIDDEN
        self._phase_start = 0
        self._last_task_step = 0
        self._spawned = False
        self.blocked = False
        self.spawn_failed = False
        self.onset_step: int | None = None
        self.trigger_reason: str | None = None
        self.p_base_start_xy: np.ndarray | None = None
        self.p_dest_xy: np.ndarray | None = None
        self.p_land_xy: np.ndarray | None = None
        self.p_start: np.ndarray | None = None
        self.p_dwell: np.ndarray | None = None
        self.spawn_attempt: int | None = None
        self.land_attempt: int | None = None
        self.lateral_attempt: int | None = None
        self.floor_z: float | None = None
        self._path_len = 0.0
        self._path_unit: np.ndarray | None = None
        self._motion_armed = False
        self._robot_geom_ids: set[int] | None = None
        self._flinch_origin: np.ndarray | None = None
        self._flinch_dir = np.array([1.0, 0.0], dtype=np.float64)
        self.target_quat = UPRIGHT_QUAT_WXYZ.copy()
        self._reset_encounter()

    def reset(self, episode_index: int) -> None:
        del episode_index
        start_xy, _yaw = read_base_pose(self.env)
        dest_xy = read_nav_dest_xy(self.env)
        self.p_base_start_xy = start_xy.copy()
        self.p_dest_xy = dest_xy.copy()
        unit, path_len = path_unit_and_length(start_xy, dest_xy)
        self._path_unit = None if unit is None else unit.copy()
        self._path_len = float(path_len)
        self._motion_armed = False
        self._phase = Phase.HIDDEN
        self._phase_start = 0
        self._spawned = False
        self.blocked = False
        self.spawn_failed = False
        self.onset_step = None
        self.trigger_reason = None
        self.p_land_xy = None
        self.p_start = None
        self.p_dwell = None
        self.spawn_attempt = None
        self.land_attempt = None
        self.lateral_attempt = None
        self.floor_z = float(read_floor_z(self.env))
        self._flinch_origin = None
        self._reset_encounter()
        set_actor_contact_enabled(self.env, self._meta, enabled=False)
        teleport_actor_hidden(self.env, self._meta)
        self._plan_apron()

    def step(self, t: int, obs: Mapping[str, Any]) -> None:
        del obs
        if t < 0:
            return
        self.update_before_step(t)

    def set_robot_geom_ids(self, robot_geom_ids: set[int] | None) -> None:
        self._robot_geom_ids = set(robot_geom_ids) if robot_geom_ids else None

    @property
    def active_obstacle_name(self) -> str | None:
        if self._phase in (Phase.HIDDEN, Phase.DONE):
            return None
        return self.obstacle_name

    @property
    def disturbance_onset_step(self) -> int | None:
        return self.onset_step

    def finalize(self) -> None:
        """Episode-end cleanup: hide the pedestrian (清场)."""
        if self._phase not in (Phase.HIDDEN, Phase.DONE):
            self._hide_done()

    def extra_fields(self) -> dict[str, Any]:
        p_start = None if self.p_start is None else [float(x) for x in self.p_start]
        p_dwell = None if self.p_dwell is None else [float(x) for x in self.p_dwell]
        p_land = None if self.p_land_xy is None else [float(x) for x in self.p_land_xy]
        dest = None if self.p_dest_xy is None else [float(x) for x in self.p_dest_xy]
        actor_pos = None
        if self._phase not in (Phase.HIDDEN, Phase.DONE) and self._spawned:
            actor_pos = [float(x) for x in read_body_pos(self.env, self.body_name)]
        spawn_failed = l1_spawn_failed_flag(
            dropped=self._spawned,
            motion_armed=self._motion_armed,
            blocked=self.blocked,
        )
        self.spawn_failed = spawn_failed
        hidden = self._phase in (Phase.HIDDEN, Phase.DONE)
        return {
            "spawn_failed": spawn_failed,
            "blocked": bool(self.blocked),
            "spawned": bool(self._spawned),
            "hidden": bool(hidden),
            "phase": self._phase.value,
            "spawn_attempt": self.spawn_attempt,
            "land_attempt": self.land_attempt,
            "lateral_attempt": self.lateral_attempt,
            "trigger_reason": self.trigger_reason,
            "onset_step": self.onset_step,
            "p_dest_xy": dest,
            "p_land_xy": p_land,
            "p_start": p_start,
            "p_dwell": p_dwell,
            "floor_z": self.floor_z,
            "obstacle_body": self.body_name,
            "obstacle_joint": self.joint_name,
            "obstacle_pos": actor_pos,
            "path_len_m": float(self._path_len),
            "n_attempts": int(self.n_attempts),
            "goal_clearance_m": float(self.goal_clearance_m),
            "human_start_lateral": float(self.human_start_lateral),
            "trigger_dist": float(self.trigger_dist),
            "flip_side": bool(self.flip_side),
            "motion_armed": bool(self._motion_armed),
            "entry_steps": int(self._entry_steps),
            "dwell_steps": int(self._dwell_steps),
            "approach_steps": int(self._approach_steps),
            "withdraw_steps": int(self._withdraw_steps),
            "visual": str(self._meta.get("visual") or "clothed_standing_mesh"),
            **self._encounter_fields(),
        }

    def _reset_encounter(self) -> None:
        self._dwell_start_step: int | None = None
        self._base_before_land_at_dwell: bool | None = None
        self._first_visible_step: int | None = None
        self._visible_cams: set[str] = set()
        self._closest_dist_m: float | None = None
        self._closest_step: int | None = None
        self._close_approach = False

    def _encounter_fields(self) -> dict[str, Any]:
        preview = self._preview_steps()
        valid, reason = self._encounter_valid_reason(preview)
        return {
            "min_preview_steps": int(self._min_preview_steps),
            "close_approach_m": float(self._close_approach_m),
            "visibility_cameras": list(self._visibility_cameras),
            "first_visible_step": self._first_visible_step,
            "visible_cams": sorted(self._visible_cams),
            "preview_steps": preview,
            "closest_dist_m": self._closest_dist_m,
            "closest_step": self._closest_step,
            "close_approach": bool(self._close_approach),
            "dwell_start_step": self._dwell_start_step,
            "base_before_land_at_dwell": self._base_before_land_at_dwell,
            "encounter_valid": bool(valid),
            "invalid_reason": reason,
        }

    def _preview_steps(self) -> int:
        if self._first_visible_step is None:
            return 0
        if self._dwell_start_step is not None:
            return max(0, int(self._dwell_start_step) - int(self._first_visible_step))
        if self._closest_step is not None:
            return max(0, int(self._closest_step) - int(self._first_visible_step))
        return 0

    def _encounter_valid_reason(self, preview: int) -> tuple[bool, str | None]:
        if not self._spawned:
            return False, "never_triggered"
        if self._dwell_start_step is None:
            return False, "never_reached_dwell"
        if self._base_before_land_at_dwell is not True:
            return False, "passed_land"
        if self._first_visible_step is None:
            return False, "not_visible"
        if int(preview) < int(self._min_preview_steps):
            return False, "preview_too_short"
        return True, None

    def _note_visibility(self, task_step: int) -> None:
        if self._phase in (Phase.HIDDEN, Phase.DONE):
            return
        pos = read_body_pos(self.env, self.body_name)
        seen = cameras_seeing_point(
            self.env,
            pos,
            self._visibility_cameras,
            aspect=self._camera_aspect,
        )
        if not seen:
            return
        self._visible_cams.update(seen)
        if self._first_visible_step is None:
            self._first_visible_step = int(task_step)

    def _note_closest(self, task_step: int) -> None:
        if self._phase in (Phase.HIDDEN, Phase.DONE) or self.p_land_xy is None:
            return
        base_xy, _yaw = read_base_pose(self.env)
        human_xy = read_body_pos(self.env, self.body_name)[:2]
        dist_human = float(np.linalg.norm(base_xy - human_xy))
        dist_land = float(np.linalg.norm(base_xy - self.p_land_xy))
        if self._closest_dist_m is None or dist_human < self._closest_dist_m:
            self._closest_dist_m = dist_human
            self._closest_step = int(task_step)
        if self._phase == Phase.DWELL and (
            dist_human <= self._close_approach_m or dist_land <= self._close_approach_m
        ):
            self._close_approach = True

    def _mark_dwell(self, task_step: int) -> None:
        self._phase = Phase.DWELL
        self._phase_start = int(task_step)
        self._dwell_start_step = int(task_step)
        if self.p_land_xy is not None and self._path_unit is not None:
            curr_xy, _yaw = read_base_pose(self.env)
            self._base_before_land_at_dwell = bool(
                base_before_land(curr_xy, self.p_land_xy, self._path_unit)
            )
        else:
            self._base_before_land_at_dwell = False

    def _probe_xy(self, xy: Sequence[float]) -> np.ndarray | None:
        return probe_floor_placement(
            self.env,
            joint_name=self.joint_name,
            xy=xy,
            half_size_z=self.half_size[2],
            obstacle_geom_ids=self.obstacle_geom_ids,
            half_xy=(self.half_size[0], self.half_size[1]),
        )

    def _plan_apron(self) -> None:
        assert self.p_base_start_xy is not None and self.p_dest_xy is not None
        if self._path_unit is None:
            self.spawn_failed = True
            self.blocked = True
            self.spawn_attempt = 0
            return
        set_actor_contact_enabled(self.env, self._meta, enabled=True)
        try:
            sides = (self.flip_side, not self.flip_side)
            attempt = 0
            for land_k in range(self.n_attempts):
                land_xy = l1_path_spawn_xy(
                    self.p_base_start_xy,
                    self.p_dest_xy,
                    self.goal_clearance_m,
                    self.retry_step,
                    land_k,
                    self._min_start_clearance,
                )
                if land_xy is None:
                    continue
                land_pos = self._probe_xy(land_xy)
                if land_pos is None:
                    continue
                for lat_k in range(self.n_attempts):
                    lat = self.human_start_lateral + lat_k * self.retry_step
                    for side in sides:
                        start_xy = l2_lateral_start_xy(
                            land_xy,
                            self._path_unit,
                            lat,
                            flip_side=bool(side),
                        )
                        start_pos = self._probe_xy(start_xy)
                        if start_pos is None:
                            attempt += 1
                            continue
                        self.spawn_attempt = int(attempt)
                        self.land_attempt = int(land_k)
                        self.lateral_attempt = int(lat_k)
                        self.p_land_xy = land_xy.copy()
                        self.p_start = start_pos
                        self.p_dwell = land_pos
                        self.floor_z = float(start_pos[2] - self.half_size[2])
                        self.target_quat = yaw_quat_wxyz(start_pos[:2], land_pos[:2])
                        return
            self.spawn_attempt = int(self.n_attempts)
            self.spawn_failed = True
            self.blocked = True
        finally:
            teleport_actor_hidden(self.env, self._meta)
            set_actor_contact_enabled(self.env, self._meta, enabled=False)

    def _place_floor_xy(self, xy: Sequence[float]) -> None:
        assert self.floor_z is not None
        pos = pos_from_xy_floor(xy, self.floor_z, self.half_size[2])
        set_free_joint_pose(self.env, self.joint_name, pos, self.target_quat)

    def _hide_done(self) -> None:
        self._phase = Phase.DONE
        self._spawned = True
        set_actor_contact_enabled(self.env, self._meta, enabled=False)
        teleport_actor_hidden(self.env, self._meta)

    def _approach_xy_at(self, alpha: float) -> np.ndarray:
        assert self.p_start is not None and self.p_dwell is not None
        mjp = minimum_jerk(alpha)
        return self.p_start[:2] + mjp * (self.p_dwell[:2] - self.p_start[:2])

    def _withdraw_xy_at(self, alpha: float) -> np.ndarray:
        assert self.p_start is not None and self.p_dwell is not None
        mjp = minimum_jerk(alpha)
        return self.p_dwell[:2] + mjp * (self.p_start[:2] - self.p_dwell[:2])

    def check_robot_collision(self, robot_geom_ids: set[int] | None = None) -> bool:
        if robot_geom_ids is not None:
            self._robot_geom_ids = set(robot_geom_ids)
        if self._phase in (Phase.HIDDEN, Phase.DONE, Phase.FLINCH):
            return bool(self.blocked)
        hit = actor_has_robot_contact(
            self.env,
            actor_geom_ids=self.obstacle_geom_ids,
            robot_geom_ids=self._robot_geom_ids,
        )
        if hit:
            self.blocked = True
            self._start_flinch(self._last_task_step)
        return hit

    def _start_flinch(self, task_step: int) -> None:
        self._phase = Phase.FLINCH
        self._phase_start = int(task_step)
        cur = read_body_pos(self.env, self.body_name)
        self._flinch_origin = cur.copy()
        base_xy, _yaw = read_base_pose(self.env)
        away = cur[:2] - base_xy
        norm = float(np.linalg.norm(away))
        if norm > 1e-8:
            self._flinch_dir = away / norm
        else:
            self._flinch_dir = np.array([1.0, 0.0], dtype=np.float64)

    def update_before_step(self, task_step: int) -> None:
        self._last_task_step = int(task_step)

        if self._phase == Phase.DONE:
            teleport_actor_hidden(self.env, self._meta)
            return

        if self._phase == Phase.HIDDEN:
            if self.p_start is None or self.p_dwell is None or self.p_land_xy is None:
                teleport_actor_hidden(self.env, self._meta)
                return
            curr_xy, _curr_yaw = read_base_pose(self.env)
            if self._path_unit is None or not l3_should_appear(
                curr_xy, self.p_land_xy, self.trigger_dist, self._path_unit
            ):
                teleport_actor_hidden(self.env, self._meta)
                return
            self._motion_armed = True
            self.trigger_reason = "approach"
            self._spawned = True
            self.onset_step = int(task_step)
            self._phase = Phase.ENTER
            self._phase_start = int(task_step)
            set_actor_contact_enabled(self.env, self._meta, enabled=True)
            set_free_joint_pose(self.env, self.joint_name, self.p_start, self.target_quat)
            self._note_visibility(task_step)
            self._note_closest(task_step)
            return

        if self._phase == Phase.FLINCH:
            steps = int(task_step) - int(self._phase_start)
            assert self._flinch_origin is not None
            frac = min((steps + 1) / float(self._flinch_steps), 1.0)
            xy = self._flinch_origin[:2] + frac * self._flinch_dist * self._flinch_dir
            self._place_floor_xy(xy)
            self._note_visibility(task_step)
            self._note_closest(task_step)
            if frac >= 1.0:
                self._hide_done()
            return

        if self._phase == Phase.ENTER:
            assert self.p_start is not None
            self._place_floor_xy(self.p_start[:2])
            self._note_visibility(task_step)
            self._note_closest(task_step)
            if int(task_step) - int(self._phase_start) >= self._entry_steps:
                self._phase = Phase.APPROACH
                self._phase_start = int(task_step)
            return

        if self._phase == Phase.APPROACH:
            alpha = min(
                (int(task_step) - int(self._phase_start)) / float(self._approach_steps),
                1.0,
            )
            self._place_floor_xy(self._approach_xy_at(alpha))
            self._note_visibility(task_step)
            self._note_closest(task_step)
            if alpha >= 1.0:
                self._mark_dwell(task_step)
            return

        if self._phase == Phase.DWELL:
            assert self.p_dwell is not None
            self._place_floor_xy(self.p_dwell[:2])
            self._note_visibility(task_step)
            self._note_closest(task_step)
            if int(task_step) - int(self._phase_start) >= self._dwell_steps:
                self._phase = Phase.WITHDRAW
                self._phase_start = int(task_step)
            return

        if self._phase == Phase.WITHDRAW:
            alpha = min(
                (int(task_step) - int(self._phase_start)) / float(self._withdraw_steps),
                1.0,
            )
            self._place_floor_xy(self._withdraw_xy_at(alpha))
            self._note_visibility(task_step)
            self._note_closest(task_step)
            if alpha >= 1.0:
                self._hide_done()
            return
