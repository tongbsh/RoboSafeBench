"""Level-1 sudden-drop disturbance (ported from SuddenDropController)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from dynisafety.backends.simplerenv.contact import actor_has_impulse_contact
from dynisafety.backends.simplerenv.disturbance._support import (
    HIDDEN_POS,
    compute_obstacle_quat,
    set_actor_pose,
)


def eef_xy_dist(eef_pos: np.ndarray, drop_xy: np.ndarray) -> float:
    eef = np.asarray(eef_pos, dtype=np.float64).reshape(-1)
    drop = np.asarray(drop_xy, dtype=np.float64).reshape(-1)
    return float(np.hypot(float(eef[0]) - float(drop[0]), float(eef[1]) - float(drop[1])))


def eef_xy_should_drop(
    *,
    xy_dist: float,
    trigger_dist: float,
    init_xy_dist: float,
    home_inside: bool,
    home_onset: bool,
    approach_eps: float,
) -> bool:
    """XY drop latch. Home-inside + home_onset must get closer than bind distance.

    When ``home_onset`` is off, or the home EEF sits outside the circle, this
    is exactly ``xy_dist <= trigger_dist`` (unchanged L1).
    """
    if not home_onset or not home_inside:
        return float(xy_dist) <= float(trigger_dist)
    return float(xy_dist) < float(init_xy_dist) - float(approach_eps)


class SuddenDropController:
    """Level 1 — EEF proximity triggers a kinematic→dynamic object drop."""

    def __init__(
        self,
        actor: Any,
        *,
        target_pos: np.ndarray,
        drop_height: float,
        trigger_dist: float,
        get_eef_pos_fn: Callable[[], np.ndarray],
        home_onset: bool,
        home_approach_eps: float,
    ) -> None:
        self.actor = actor
        self.target_pos = np.asarray(target_pos, dtype=np.float64).copy()
        self.drop_pos = self.target_pos.copy()
        self.drop_pos[2] += float(drop_height)
        self._trigger_dist = float(trigger_dist)
        self._get_eef_pos = get_eef_pos_fn
        self._arm_quat = compute_obstacle_quat(0.0)
        self.home_onset = bool(home_onset)
        self.home_approach_eps = float(home_approach_eps)
        if self.home_approach_eps < 0.0:
            raise ValueError(
                f"extra.simplerenv.l1.home_approach_eps must be >= 0, got {home_approach_eps!r}"
            )

        init_eef = np.asarray(get_eef_pos_fn(), dtype=np.float64).reshape(-1)
        if init_eef.size < 2 or not np.all(np.isfinite(init_eef[:2])):
            self.init_xy_dist = float("inf")
            self.home_inside = False
        else:
            self.init_xy_dist = eef_xy_dist(init_eef, self.target_pos[:2])
            self.home_inside = self.init_xy_dist <= self._trigger_dist

        self.blocked = False
        self._frozen_pos: np.ndarray | None = None
        self._dropped = False
        self.drop_step: int | None = None

        set_actor_pose(self.actor, HIDDEN_POS, self._arm_quat)

    @property
    def needs_physics_release(self) -> bool:
        return bool(self._dropped)

    @property
    def disturbance_onset_step(self) -> int | None:
        return self.drop_step

    def _check_trigger(self) -> bool:
        eef_pos = self._get_eef_pos()
        xy_dist = eef_xy_dist(eef_pos, self.target_pos[:2])
        return eef_xy_should_drop(
            xy_dist=xy_dist,
            trigger_dist=self._trigger_dist,
            init_xy_dist=self.init_xy_dist,
            home_inside=self.home_inside,
            home_onset=self.home_onset,
            approach_eps=self.home_approach_eps,
        )

    def update_before_step(self, task_step: int) -> None:
        if self._dropped:
            return
        if task_step >= 0 and self._check_trigger():
            set_actor_pose(self.actor, self.drop_pos, self._arm_quat)
            self._dropped = True
            self.drop_step = int(task_step)
        else:
            set_actor_pose(self.actor, HIDDEN_POS, self._arm_quat)

    def check_robot_collision(
        self,
        scene: Any,
        robot_link_names: set[str],
        robot_link_positions: np.ndarray,
    ) -> bool:
        del robot_link_positions
        if not self._dropped:
            return False
        hit = actor_has_impulse_contact(
            scene, actor_name=self.actor.name, robot_link_names=robot_link_names
        )
        if hit:
            self.blocked = True
        return hit
