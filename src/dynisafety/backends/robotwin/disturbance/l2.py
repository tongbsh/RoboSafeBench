"""Level-2 sliding-obstacle disturbance controller."""

from __future__ import annotations

import numpy as np

from dynisafety.backends.robotwin.disturbance._base import BaseDisturbanceController
from dynisafety.core.disturbance import arc_lift, lerp


class SlidingObstacleController(BaseDisturbanceController):
    def __init__(
        self,
        actor,
        *,
        start_pos: np.ndarray,
        final_pos: np.ndarray,
        move_start_step: int,
        move_end_step: int,
        hold_steps: int,
        arc_height: float,
        quat: np.ndarray,
        path_style: str = "arc",
        cruise_z: float | None = None,
        descent_alpha: float | None = None,
        move_start_task_step: int = 0,
    ) -> None:
        super().__init__(actor, default_quat=quat)
        self.start_pos = np.asarray(start_pos, dtype=np.float64)
        self.final_pos = np.asarray(final_pos, dtype=np.float64)
        self.move_start_step = int(move_start_step)
        self.move_end_step = int(move_end_step)
        self.hold_end_step = int(move_end_step + hold_steps)
        self.move_start_task_step = int(move_start_task_step)
        self.arc_height = float(arc_height)
        self.path_style = str(path_style)
        self.cruise_z = None if cruise_z is None else float(cruise_z)
        self.descent_alpha = None if descent_alpha is None else float(descent_alpha)
        self.spawn_plan: dict | None = None
        self._released = False
        self._frozen = False
        self._release_reason: str | None = None
        self._onset_task_step: int | None = None
        self._path_sim0: int | None = None

    @property
    def disturbance_onset_step(self) -> int | None:
        """Task-step onset (first frame the slide actually starts)."""
        return self._onset_task_step

    @property
    def release_reason(self) -> str | None:
        return self._release_reason

    def _follow_path(self) -> bool:
        return not self._released and not self._frozen

    def _path_pos(self, alpha: float) -> np.ndarray:
        a = float(alpha)
        if self.path_style == "linear":
            return lerp(self.start_pos, self.final_pos, a)
        if self.path_style == "trapezoid":
            if self.cruise_z is None or self.descent_alpha is None:
                raise ValueError("trapezoid path requires cruise_z and descent_alpha")
            descent = float(self.descent_alpha)
            cruise_end = 1.0 - descent
            start_xy = self.start_pos[:2]
            land_xy = self.final_pos[:2]
            z_land = float(self.final_pos[2])
            z_cruise = float(self.cruise_z)
            if a <= cruise_end:
                t = 0.0 if cruise_end <= 1e-12 else a / cruise_end
                xy = (1.0 - t) * start_xy + t * land_xy
                return np.array([float(xy[0]), float(xy[1]), z_cruise], dtype=np.float64)
            t = (a - cruise_end) / descent
            return np.array(
                [float(land_xy[0]), float(land_xy[1]), z_cruise + t * (z_land - z_cruise)],
                dtype=np.float64,
            )
        pos = lerp(self.start_pos, self.final_pos, a)
        pos[2] += arc_lift(a, self.arc_height)
        return pos

    def update_before_step(self, sim_step: int, *, task_step: int | None = None) -> None:
        if not self._follow_path():
            self._active = True
            return
        self._active = True
        t_task = 0 if task_step is None else int(task_step)
        if int(self.move_start_task_step) > 0:
            if t_task < int(self.move_start_task_step):
                self._set_pose(self.start_pos)
                return
            if self._path_sim0 is None:
                self._path_sim0 = int(sim_step)
                self._onset_task_step = t_task
            elapsed = int(sim_step) - int(self._path_sim0)
            duration = max(self.move_end_step - self.move_start_step, 1)
            hold = max(int(self.hold_end_step) - int(self.move_end_step), 0)
            if elapsed <= duration:
                alpha = elapsed / duration
                self._set_pose(self._path_pos(alpha))
                return
            if elapsed <= duration + hold:
                self._set_pose(self.final_pos)
                return
            self._released = True
            self._release_pending = True
            self._release_reason = "settle"
            return
        if self._onset_task_step is None and task_step is not None and int(sim_step) >= int(
            self.move_start_step
        ):
            self._onset_task_step = t_task
        if sim_step < self.move_start_step:
            self._set_pose(self.start_pos)
            return
        if sim_step <= self.move_end_step:
            denom = max(self.move_end_step - self.move_start_step, 1)
            alpha = (sim_step - self.move_start_step) / denom
            self._set_pose(self._path_pos(alpha))
            return
        if sim_step <= self.hold_end_step:
            self._set_pose(self.final_pos)
            return
        self._released = True
        self._release_pending = True
        self._release_reason = "settle"

    def handle_collision(self, sim_step: int) -> None:
        del sim_step
        if not self._follow_path():
            return
        self._released = True
        self._release_pending = True
        self._release_reason = "impulse_hit"

    def cancel_pending_release(self) -> None:
        """Opt-out park: stay kinematic at last free pose (flag off)."""
        self._release_pending = False
        self._released = True
        self._frozen = True
        self._release_reason = "skipped_penetration"

    def on_released_to_physics(self) -> None:
        super().on_released_to_physics()
        self._released = True
        self._frozen = False
