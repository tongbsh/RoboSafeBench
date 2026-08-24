"""Level-2 dest-apron cross-aisle ping-pong (floor box)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dynisafety.backends.robocasa.disturbance._support import (
    UPRIGHT_QUAT_WXYZ,
    actor_has_robot_contact,
    collect_body_geom_ids,
    l1_path_spawn_xy,
    l1_should_appear,
    l2_lerp_sample_xys,
    l2_pingpong_xy,
    l12_meta,
    obstacle_has_scene_penetration,
    path_left_normal,
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

# Interior samples on land→end; not a YAML hyperparameter.
_SEGMENT_MID_SAMPLES = 3


class BaseDirectedApproachController:
    """hidden → plan p_land + n-interval → park on one end → cross after trigger.

    Aim is frozen at reset (path unit, land, plus/minus). Visible from the
    first control step at the start end; crossing starts when
    ||base − p_land|| ≤ trigger_dist. No chase, no heading/FOV. Contact
    releases the box to physics.
    """

    def __init__(
        self,
        handle: RoboCasaEnvHandle,
        *,
        goal_clearance_m: float,
        start_lateral: float,
        move_start_step: int,
        move_end_step: int,
        retry_step: float,
        n_attempts: int,
        min_half_span: float,
        trigger_dist: float,
        flip_side: bool,
        half_size: Sequence[float],
        selection_mode: str = "fixed",
        pool: Sequence[str] | None = None,
    ) -> None:
        self.handle = handle
        self.env = handle.env
        self.goal_clearance_m = float(goal_clearance_m)
        self.start_lateral = float(start_lateral)
        if self.start_lateral < 0.0:
            raise ValueError(f"start_lateral must be >= 0, got {self.start_lateral}")
        self.move_start_step = int(move_start_step)
        self.move_end_step = int(move_end_step)
        if self.move_end_step <= self.move_start_step:
            raise ValueError(
                f"move_end_step ({self.move_end_step}) must be > "
                f"move_start_step ({self.move_start_step})"
            )
        self._one_way_steps = self.move_end_step - self.move_start_step
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
        self._selection_mode = str(selection_mode)
        self._pool = [str(item) for item in (pool or ())]
        self.visual_id: str | None = None
        self._min_half_span = float(min_half_span)
        if self._min_half_span <= 0.0:
            raise ValueError(f"min_half_span must be > 0, got {self._min_half_span}")
        if self._min_half_span > self.start_lateral:
            raise ValueError(
                f"min_half_span ({self._min_half_span}) must be <= "
                f"start_lateral ({self.start_lateral})"
            )
        self.trigger_dist = float(trigger_dist)
        if self.trigger_dist < 0.0:
            raise ValueError(f"trigger_dist must be >= 0, got {self.trigger_dist}")

        meta = l12_meta(self.env)
        self._meta = meta
        self.body_name = str(meta["body_name"])
        self.joint_name = str(meta["joint_name"])
        self.obstacle_name = str(meta.get("object_name") or self.body_name)
        self.obstacle_geom_ids = collect_body_geom_ids(self.env, self.body_name)
        if "geom_contact" not in meta:
            meta_mut = dict(meta)
            meta_mut["geom_contact"] = snapshot_actor_geom_contact(self.env, meta_mut)
            self.env._dynisafety_l12 = meta_mut
            self._meta = meta_mut

        self._spawned = False
        self.released = False
        self.blocked = False
        self.spawn_failed = False
        self.onset_step: int | None = None
        self.trigger_reason: str | None = None
        self.p_base_start_xy: np.ndarray | None = None
        self.p_dest_xy: np.ndarray | None = None
        self.p_land_xy: np.ndarray | None = None
        self.p_plus: np.ndarray | None = None
        self.p_minus: np.ndarray | None = None
        self.p_start: np.ndarray | None = None
        self.p_goal: np.ndarray | None = None
        self.span_plus: float | None = None
        self.span_minus: float | None = None
        self.spawn_attempt: int | None = None
        self.land_attempt: int | None = None
        self.floor_z: float | None = None
        self._path_len = 0.0
        self._path_unit: np.ndarray | None = None
        self._robot_geom_ids: set[int] | None = None
        self.target_quat = UPRIGHT_QUAT_WXYZ.copy()

    def reset(self, episode_index: int) -> None:
        from dynisafety.backends.robocasa.visuals import apply_l12_visual, select_visual_id

        if self._pool:
            self.visual_id = select_visual_id(self._pool, episode_index, self._selection_mode)
            apply_l12_visual(self.env, self.visual_id)
        start_xy, _yaw0 = read_base_pose(self.env)
        dest_xy = read_nav_dest_xy(self.env)
        self.p_base_start_xy = start_xy.copy()
        self.p_dest_xy = dest_xy.copy()
        unit, path_len = path_unit_and_length(start_xy, dest_xy)
        self._path_unit = None if unit is None else unit.copy()
        self._path_len = float(path_len)
        self._spawned = False
        self.released = False
        self.blocked = False
        self.spawn_failed = False
        self.onset_step = None
        self.trigger_reason = None
        self.p_land_xy = None
        self.p_plus = None
        self.p_minus = None
        self.p_start = None
        self.p_goal = None
        self.span_plus = None
        self.span_minus = None
        self.spawn_attempt = None
        self.land_attempt = None
        self.floor_z = float(read_floor_z(self.env))
        set_actor_contact_enabled(self.env, self._meta, enabled=False)
        teleport_actor_hidden(self.env, self._meta)
        self._plan_span()

    def step(self, t: int, obs: Mapping[str, Any]) -> None:
        del obs
        if t < 0:
            return
        self.update_before_step(t)

    def set_robot_geom_ids(self, robot_geom_ids: set[int] | None) -> None:
        self._robot_geom_ids = set(robot_geom_ids) if robot_geom_ids else None

    @property
    def active_obstacle_name(self) -> str | None:
        return self.obstacle_name if self._spawned else None

    @property
    def disturbance_onset_step(self) -> int | None:
        return self.onset_step

    def extra_fields(self) -> dict[str, Any]:
        p_plus = None if self.p_plus is None else [float(x) for x in self.p_plus]
        p_minus = None if self.p_minus is None else [float(x) for x in self.p_minus]
        p_land = None if self.p_land_xy is None else [float(x) for x in self.p_land_xy]
        base_start = (
            None if self.p_base_start_xy is None else [float(x) for x in self.p_base_start_xy]
        )
        dest = None if self.p_dest_xy is None else [float(x) for x in self.p_dest_xy]
        spawned_pos = None
        if self._spawned:
            spawned_pos = [float(x) for x in read_body_pos(self.env, self.body_name)]
        return {
            "spawn_failed": bool(self.spawn_failed),
            "blocked": bool(self.blocked),
            "released": bool(self.released),
            "spawned": bool(self._spawned),
            "spawn_attempt": self.spawn_attempt,
            "land_attempt": self.land_attempt,
            "trigger_reason": self.trigger_reason,
            "onset_step": self.onset_step,
            "p_start_xy": base_start,
            "p_dest_xy": dest,
            "p_land_xy": p_land,
            "p_plus": p_plus,
            "p_minus": p_minus,
            "p_start": p_plus,
            "p_goal": p_minus,
            "span_plus": self.span_plus,
            "span_minus": self.span_minus,
            "floor_z": self.floor_z,
            "half_size": [float(x) for x in self.half_size],
            "obstacle_body": self.body_name,
            "obstacle_joint": self.joint_name,
            "obstacle_pos": spawned_pos,
            "path_len_m": float(self._path_len),
            "goal_clearance_m": float(self.goal_clearance_m),
            "start_lateral": float(self.start_lateral),
            "min_half_span": float(self._min_half_span),
            "trigger_dist": float(self.trigger_dist),
            "flip_side": bool(self.flip_side),
            "n_attempts": int(self.n_attempts),
            "move_start_step": int(self.move_start_step),
            "move_end_step": int(self.move_end_step),
            "one_way_steps": int(self._one_way_steps),
            "obstacle_id": self.visual_id,
            "visual_id": self.visual_id,
            "obstacle_pool": list(self._pool),
            "selection_mode": self._selection_mode,
        }

    def _probe_xy(self, xy: Sequence[float]) -> np.ndarray | None:
        return probe_floor_placement(
            self.env,
            joint_name=self.joint_name,
            xy=xy,
            half_size_z=self.half_size[2],
            obstacle_geom_ids=self.obstacle_geom_ids,
            half_xy=(self.half_size[0], self.half_size[1]),
        )

    def _segment_clear(self, start_xy: Sequence[float], land_xy: Sequence[float]) -> bool:
        for mid in l2_lerp_sample_xys(start_xy, land_xy, _SEGMENT_MID_SAMPLES):
            if self._probe_xy(mid) is None:
                return False
        return True

    def _best_span(self, land_xy: np.ndarray, unit: np.ndarray, sign: float) -> float:
        n = path_left_normal(unit)
        for k in range(self.n_attempts):
            span = self.start_lateral - k * self.retry_step
            if span < self._min_half_span:
                break
            end_xy = land_xy + float(sign) * span * n
            if self._probe_xy(end_xy) is None:
                continue
            if not self._segment_clear(land_xy, end_xy):
                continue
            return float(span)
        return 0.0

    def _plan_span(self) -> None:
        assert self.p_base_start_xy is not None and self.p_dest_xy is not None
        if self._path_unit is None:
            self.spawn_failed = True
            self.blocked = True
            self.spawn_attempt = 0
            return
        set_actor_contact_enabled(self.env, self._meta, enabled=True)
        try:
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
                span_plus = self._best_span(land_xy, self._path_unit, 1.0)
                span_minus = self._best_span(land_xy, self._path_unit, -1.0)
                if span_plus < self._min_half_span or span_minus < self._min_half_span:
                    continue
                n = path_left_normal(self._path_unit)
                plus_xy = land_xy + span_plus * n
                minus_xy = land_xy - span_minus * n
                plus_pos = self._probe_xy(plus_xy)
                minus_pos = self._probe_xy(minus_xy)
                if plus_pos is None or minus_pos is None:
                    continue
                self.spawn_attempt = int(land_k)
                self.land_attempt = int(land_k)
                self.p_land_xy = land_xy.copy()
                self.p_plus = plus_pos
                self.p_minus = minus_pos
                self.p_start = plus_pos
                self.p_goal = minus_pos
                self.span_plus = float(span_plus)
                self.span_minus = float(span_minus)
                self.floor_z = float(plus_pos[2] - self.half_size[2])
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

    def _release_to_physics(self) -> None:
        self.released = True
        addr_jid = int(self.env.sim.model.joint_name2id(self.joint_name))
        addr = int(self.env.sim.model.jnt_dofadr[addr_jid])
        self.env.sim.data.qvel[addr : addr + 6] = 0.0
        try:
            self.env.sim.forward()
        except Exception:
            pass

    def check_robot_collision(self, robot_geom_ids: set[int] | None = None) -> bool:
        if robot_geom_ids is not None:
            self._robot_geom_ids = set(robot_geom_ids)
        if not self._spawned or self.released:
            return bool(self.blocked)
        hit = actor_has_robot_contact(
            self.env,
            actor_geom_ids=self.obstacle_geom_ids,
            robot_geom_ids=self._robot_geom_ids,
        )
        if hit:
            self.blocked = True
            self._release_to_physics()
        return hit

    def _park_pos(self) -> np.ndarray:
        assert self.p_plus is not None and self.p_minus is not None
        start = self.p_minus if self.flip_side else self.p_plus
        return np.asarray(start, dtype=np.float64).reshape(3).copy()

    def _appear_parked(self) -> None:
        set_actor_contact_enabled(self.env, self._meta, enabled=True)
        set_free_joint_pose(self.env, self.joint_name, self._park_pos(), self.target_quat)
        self._spawned = True

    def _maybe_start_crossing(self, task_step: int) -> bool:
        assert self.p_land_xy is not None
        curr_xy, _yaw = read_base_pose(self.env)
        if not l1_should_appear(curr_xy, self.p_land_xy, self.trigger_dist):
            return False
        self.onset_step = int(task_step)
        self.trigger_reason = "approach"
        return True

    def update_before_step(self, task_step: int) -> None:
        if self.released or self.blocked:
            return

        if not self._spawned:
            if self.p_plus is None or self.p_minus is None:
                teleport_actor_hidden(self.env, self._meta)
                return
            self._appear_parked()

        if self._has_robot_contact():
            self.blocked = True
            self._release_to_physics()
            return

        if self.onset_step is None:
            if not self._maybe_start_crossing(task_step):
                set_free_joint_pose(
                    self.env, self.joint_name, self._park_pos(), self.target_quat
                )
                if self._has_robot_contact():
                    self.blocked = True
                    self._release_to_physics()
                return

        assert self.p_plus is not None and self.p_minus is not None
        assert self.onset_step is not None
        elapsed = int(task_step) - int(self.onset_step)
        a_xy = self.p_minus[:2] if self.flip_side else self.p_plus[:2]
        b_xy = self.p_plus[:2] if self.flip_side else self.p_minus[:2]
        xy = l2_pingpong_xy(a_xy, b_xy, elapsed, self._one_way_steps)
        prev = np.asarray(
            self.env.sim.data.get_joint_qpos(self.joint_name), dtype=np.float64
        ).copy()
        self._place_floor_xy(xy)
        if obstacle_has_scene_penetration(self.env, obstacle_geom_ids=self.obstacle_geom_ids):
            set_free_joint_pose(self.env, self.joint_name, prev[:3], prev[3:7])
            return
        if self._has_robot_contact():
            set_free_joint_pose(self.env, self.joint_name, prev[:3], prev[3:7])
            self.blocked = True
            self._release_to_physics()

    def _has_robot_contact(self) -> bool:
        return actor_has_robot_contact(
            self.env,
            actor_geom_ids=self.obstacle_geom_ids,
            robot_geom_ids=self._robot_geom_ids,
        )
