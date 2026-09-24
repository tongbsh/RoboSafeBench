"""Shared disturbance controller primitives (lazy SAPIEN import)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dynisafety.core.disturbance import minimum_jerk as minimum_jerk

HIDDEN_POS = np.array([6.0, 6.0, 2.5], dtype=np.float64)
IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


@dataclass
class ControllerPose:
    position: np.ndarray
    quat: np.ndarray


class BaseDisturbanceController:
    def __init__(self, actor: Any, *, default_quat: np.ndarray) -> None:
        self.actor = actor
        self._active = False
        self._release_pending = False
        self.default_quat = np.asarray(default_quat, dtype=np.float64)

    @property
    def entity(self) -> Any:
        return self.actor.actor if hasattr(self.actor, "actor") else self.actor

    @property
    def active(self) -> bool:
        return self._active

    def _set_pose(self, pose: ControllerPose | np.ndarray) -> None:
        import sapien.core as sapien

        if isinstance(pose, ControllerPose):
            pos = pose.position
            quat = pose.quat
        else:
            pos = np.asarray(pose, dtype=np.float64)
            quat = self.default_quat
        self.entity.set_pose(sapien.Pose(pos, quat))

    def hide(self) -> None:
        self._active = False
        if self.actor is None:
            return
        try:
            self._set_pose(HIDDEN_POS)
        except ImportError:
            # Allow constructing controllers in sapien-less unit tests.
            return

    def update_before_step(self, sim_step: int) -> None:
        raise NotImplementedError

    def handle_collision(self, sim_step: int) -> Any:
        return None

    def needs_dynamic_release(self) -> bool:
        return self._release_pending

    def on_released_to_physics(self) -> None:
        self._release_pending = False
