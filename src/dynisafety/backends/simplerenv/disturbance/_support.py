"""Shared helpers for SimplerEnv disturbance controllers (no eval defaults)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from dynisafety.backends.simplerenv.actor import HIDDEN_POS, compute_obstacle_quat
from dynisafety.core.disturbance import (
    Phase,
    arm_quat_from_yaw,
    minimum_jerk,
)
from dynisafety.core.disturbance import (
    compute_approach_steps as _core_compute_approach_steps,
)


class DisturbanceController(Protocol):
    blocked: bool

    def update_before_step(self, task_step: int) -> None: ...

    def check_robot_collision(
        self,
        scene: Any,
        robot_link_names: set[str],
        robot_link_positions: np.ndarray,
    ) -> bool: ...

    @property
    def needs_physics_release(self) -> bool: ...

    @property
    def disturbance_onset_step(self) -> int | None: ...


def compute_arm_quat(yaw: float) -> np.ndarray:
    """Hand/arm orientation used by L3 SharedReaching (wxyz)."""
    return arm_quat_from_yaw(yaw)


def compute_approach_steps(distance: float, peak_velocity: float, control_freq: int) -> int:
    return _core_compute_approach_steps(distance, peak_velocity, control_freq=control_freq)


def set_actor_pose(actor: Any, pos: Sequence[float], quat: np.ndarray) -> None:
    """Lazy-import SAPIEN Pose so dry imports work without the sim package."""
    import sapien.core as sapien

    actor.set_pose(sapien.Pose(np.asarray(pos, dtype=np.float64), quat))


__all__ = [
    "HIDDEN_POS",
    "Phase",
    "DisturbanceController",
    "compute_arm_quat",
    "compute_obstacle_quat",
    "minimum_jerk",
    "compute_approach_steps",
    "set_actor_pose",
]
