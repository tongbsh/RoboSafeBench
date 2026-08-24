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
    ) -> None:
        self.actor = actor
        self.target_pos = np.asarray(target_pos, dtype=np.float64).copy()
        self.drop_pos = self.target_pos.copy()
        self.drop_pos[2] += float(drop_height)
        self._trigger_dist = float(trigger_dist)
        self._get_eef_pos = get_eef_pos_fn
        self._arm_quat = compute_obstacle_quat(0.0)

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
        xy_dist = float(np.linalg.norm(eef_pos[:2] - self.target_pos[:2]))
        return xy_dist <= self._trigger_dist

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
