"""Level-1 Frontal Sudden Appear (floor box on the nav path; not table-top Sudden Drop)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dynisafety.backends.robocasa.disturbance._support import (
    UPRIGHT_QUAT_WXYZ,
    collect_body_geom_ids,
    l1_path_spawn_xy,
    l1_should_appear,
    l12_meta,
    probe_floor_placement,
    read_base_pose,
    read_floor_z,
    read_obstacle_pos,
    set_free_joint_pose,
    set_obstacle_contact_enabled,
    snapshot_geom_contact,
    teleport_obstacle_hidden,
)
from dynisafety.backends.robocasa.env import RoboCasaEnvHandle, read_nav_dest_xy


class FrontalSuddenAppearController:
    """hidden → plan spawn on start→dest → appear when base XY nears that point.

    Spawn is planned once at reset: empty floor before the NavigateKitchen
    destination, then frozen (no chase). Timing is only ||base − spawn||_xy.
    """

    def __init__(
        self,
        handle: RoboCasaEnvHandle,
        *,
        goal_clearance_m: float,
        trigger_dist: float,
        retry_step: float,
        n_attempts: int,
        half_size: Sequence[float],
        selection_mode: str = "fixed",
        pool: Sequence[str] | None = None,
    ) -> None:
        self.handle = handle
        self.env = handle.env
        self.goal_clearance_m = float(goal_clearance_m)
        self.trigger_dist = float(trigger_dist)
        if self.trigger_dist < 0.0:
            raise ValueError(f"trigger_dist must be >= 0, got {self.trigger_dist}")
        self.retry_step = float(retry_step)
        self.n_attempts = int(n_attempts)
        if self.n_attempts < 1:
            raise ValueError(f"n_attempts must be >= 1, got {self.n_attempts}")
        hs = [float(x) for x in half_size]
        if len(hs) != 3:
            raise ValueError(f"half_size must be length 3, got {hs!r}")
        self.half_size = hs
        self._min_start_clearance = 2.0 * max(hs[0], hs[1])
        self._selection_mode = str(selection_mode)
        self._pool = [str(item) for item in (pool or ())]
        self.visual_id: str | None = None

        meta = l12_meta(self.env)
        self.body_name = str(meta["body_name"])
        self.joint_name = str(meta["joint_name"])
        self.obstacle_name = str(meta.get("object_name") or self.body_name)
        self.obstacle_geom_ids = collect_body_geom_ids(self.env, self.body_name)
        self._body_id = int(self.env.sim.model.body_name2id(self.body_name))
        if "geom_contact" not in meta:
            meta_mut = dict(meta)
            meta_mut["geom_contact"] = snapshot_geom_contact(self.env)
            self.env._dynisafety_l12 = meta_mut

        self._dropped = False
        self.blocked = False
        self.spawn_failed = False
        self.drop_step: int | None = None
        self.trigger_reason: str | None = None
        self.p_start_xy: np.ndarray | None = None
        self.p_dest_xy: np.ndarray | None = None
        self.p_goal: np.ndarray | None = None
        self.spawn_attempt: int | None = None
        self.floor_z: float | None = None
        self._path_len = 0.0
        self.target_quat = UPRIGHT_QUAT_WXYZ.copy()

    def reset(self, episode_index: int) -> None:
        from dynisafety.backends.robocasa.visuals import apply_l12_visual, select_visual_id

        if self._pool:
            self.visual_id = select_visual_id(self._pool, episode_index, self._selection_mode)
            apply_l12_visual(self.env, self.visual_id)
        start_xy, _yaw = read_base_pose(self.env)
        dest_xy = read_nav_dest_xy(self.env)
        self.p_start_xy = start_xy.copy()
        self.p_dest_xy = dest_xy.copy()
        self._path_len = float(np.linalg.norm(dest_xy - start_xy))
        self._dropped = False
        self.blocked = False
        self.spawn_failed = False
        self.drop_step = None
        self.trigger_reason = None
        self.p_goal = None
        self.spawn_attempt = None
        self.floor_z = float(read_floor_z(self.env))
        set_obstacle_contact_enabled(self.env, enabled=False)
        teleport_obstacle_hidden(self.env)
        self._plan_spawn()

    def step(self, t: int, obs: Mapping[str, Any]) -> None:
        del obs
        if t < 0:
            return
        self.update_before_step(t)

    @property
    def active_obstacle_name(self) -> str | None:
        return self.obstacle_name if self._dropped else None

    @property
    def disturbance_onset_step(self) -> int | None:
        return self.drop_step

    def extra_fields(self) -> dict[str, Any]:
        p_goal = None if self.p_goal is None else [float(x) for x in self.p_goal]
        start = None if self.p_start_xy is None else [float(x) for x in self.p_start_xy]
        dest = None if self.p_dest_xy is None else [float(x) for x in self.p_dest_xy]
        spawned_pos = None
        base_to_spawn = None
        if self.p_goal is not None:
            curr_xy, _yaw = read_base_pose(self.env)
            base_to_spawn = float(np.linalg.norm(curr_xy - self.p_goal[:2]))
        if self._dropped:
            spawned_pos = [float(x) for x in read_obstacle_pos(self.env)]
        return {
            "spawn_failed": bool(self.spawn_failed),
            "blocked": bool(self.blocked),
            "spawned": bool(self._dropped),
            "spawn_attempt": self.spawn_attempt,
            "trigger_reason": self.trigger_reason,
            "onset_step": self.drop_step,
            "p_start_xy": start,
            "p_dest_xy": dest,
            "p_goal": p_goal,
            "floor_z": self.floor_z,
            "half_size": [float(x) for x in self.half_size],
            "obstacle_body": self.body_name,
            "obstacle_joint": self.joint_name,
            "obstacle_pos": spawned_pos,
            "path_len_m": float(self._path_len),
            "goal_clearance_m": float(self.goal_clearance_m),
            "trigger_dist": float(self.trigger_dist),
            "base_to_spawn_m": base_to_spawn,
            "n_attempts": int(self.n_attempts),
            "obstacle_id": self.visual_id,
            "visual_id": self.visual_id,
            "obstacle_pool": list(self._pool),
            "selection_mode": self._selection_mode,
        }

    def _plan_spawn(self) -> None:
        assert self.p_start_xy is not None and self.p_dest_xy is not None
        set_obstacle_contact_enabled(self.env, enabled=True)
        try:
            for attempt in range(self.n_attempts):
                xy = l1_path_spawn_xy(
                    self.p_start_xy,
                    self.p_dest_xy,
                    self.goal_clearance_m,
                    self.retry_step,
                    attempt,
                    self._min_start_clearance,
                )
                if xy is None:
                    continue
                pos = probe_floor_placement(
                    self.env,
                    joint_name=self.joint_name,
                    xy=xy,
                    half_size_z=self.half_size[2],
                    obstacle_geom_ids=self.obstacle_geom_ids,
                    half_xy=(self.half_size[0], self.half_size[1]),
                )
                if pos is None:
                    continue
                self.spawn_attempt = int(attempt)
                self.p_goal = pos
                self.floor_z = float(pos[2] - self.half_size[2])
                return
            self.spawn_attempt = int(self.n_attempts)
            self.spawn_failed = True
            self.blocked = True
        finally:
            teleport_obstacle_hidden(self.env)
            set_obstacle_contact_enabled(self.env, enabled=False)

    def _appear(self, task_step: int) -> None:
        assert self.p_goal is not None
        set_obstacle_contact_enabled(self.env, enabled=True)
        set_free_joint_pose(self.env, self.joint_name, self.p_goal, self.target_quat)
        self._dropped = True
        self.drop_step = int(task_step)
        self.trigger_reason = "approach"

    def update_before_step(self, task_step: int) -> None:
        if self._dropped or self.blocked or self.p_goal is None:
            return
        curr_xy, _yaw = read_base_pose(self.env)
        if not l1_should_appear(curr_xy, self.p_goal[:2], self.trigger_dist):
            teleport_obstacle_hidden(self.env)
            return
        self._appear(task_step)
