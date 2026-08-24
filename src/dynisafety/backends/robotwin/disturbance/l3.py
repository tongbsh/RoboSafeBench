"""Level-3 shared-reaching (hand) disturbance controller."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from dynisafety.backends.robotwin.disturbance._base import (
    BaseDisturbanceController,
    minimum_jerk,
)
from dynisafety.core.disturbance import Phase, lerp


class SharedReachingController(BaseDisturbanceController):
    def __init__(
        self,
        actor,
        *,
        start_pos: np.ndarray,
        hover_pos: np.ndarray,
        dwell_pos: np.ndarray,
        move_start_step: int,
        entry_steps: int,
        approach_steps: int,
        dwell_steps: int,
        exit_steps: int,
        flinch_dist: float,
        flinch_height: float,
        get_eef_pos_fn: Callable[[], np.ndarray],
        post_hit_hold_steps: int,
        quat: np.ndarray,
        path_style: str = "straight",
        hover_rise_m: float = 0.0,
        via_descent_alpha: float = 0.20,
        pose_start: np.ndarray | None = None,
    ) -> None:
        super().__init__(actor, default_quat=np.asarray(quat, dtype=np.float64))
        self.start_pos = np.asarray(start_pos, dtype=np.float64)
        self.hover_pos = np.asarray(hover_pos, dtype=np.float64)
        self.dwell_pos = np.asarray(dwell_pos, dtype=np.float64)
        self.pose_start = np.asarray(
            start_pos if pose_start is None else pose_start, dtype=np.float64
        )
        self.move_start_step = int(move_start_step)
        self.entry_steps = max(int(entry_steps), 1)
        self.approach_steps = max(int(approach_steps), 1)
        self.dwell_steps = max(int(dwell_steps), 1)
        self.exit_steps = max(int(exit_steps), 1)
        self.post_hit_hold_steps = max(int(post_hit_hold_steps), 0)
        self.flinch_dist = float(flinch_dist)
        self.flinch_height = float(flinch_height)
        self.get_eef_pos = get_eef_pos_fn
        self.path_style = str(path_style)
        self.hover_rise_m = float(hover_rise_m)
        self.via_descent_alpha = float(via_descent_alpha)
        self.reach_plan: dict | None = None

        self.phase = Phase.HIDDEN
        self.phase_start = 0
        self.exit_from = None
        self.exit_to = None
        self.hold_pos = None
        self.hide()

    def _interp(self, start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
        return lerp(start, end, minimum_jerk(alpha))

    def _approach_pos_at(self, alpha: float) -> np.ndarray:
        # ``raised``: hover is already path[0] (lifted); lerp matches validation.
        # ``via``: pose_at from the unlifted pose_start (not hover).
        # ``straight``: lerp hover → dwell (today's interpolate).
        if self.path_style == "via":
            from dynisafety.backends.robotwin.disturbance.l3_reach import robotwin_l3_pose_at

            return robotwin_l3_pose_at(
                self.pose_start,
                self.dwell_pos,
                alpha,
                "via",
                hover_rise_m=self.hover_rise_m,
                via_descent_alpha=self.via_descent_alpha,
            )
        return self._interp(self.hover_pos, self.dwell_pos, alpha)

    def _begin_phase(self, phase: Phase, sim_step: int) -> None:
        self.phase = phase
        self.phase_start = sim_step

    def _begin_exit(self, sim_step: int, from_pos: np.ndarray | None = None) -> None:
        self.exit_from = np.asarray(
            from_pos if from_pos is not None else self.entity.get_pose().p,
            dtype=np.float64,
        )
        self.exit_to = self.start_pos.copy()
        self._begin_phase(Phase.EXIT, sim_step)

    def update_before_step(self, sim_step: int) -> None:
        if self.phase == Phase.DONE:
            self.hide()
            return

        if self.phase == Phase.HIDDEN:
            self.hide()
            if sim_step >= self.move_start_step:
                self._active = True
                self._begin_phase(Phase.ENTER, sim_step)
            return

        self._active = True
        if self.phase == Phase.ENTER:
            alpha = min((sim_step - self.phase_start) / self.entry_steps, 1.0)
            self._set_pose(self._interp(self.start_pos, self.hover_pos, alpha))
            if alpha >= 1.0:
                self._begin_phase(Phase.APPROACH, sim_step)
            return

        if self.phase == Phase.APPROACH:
            alpha = min((sim_step - self.phase_start) / self.approach_steps, 1.0)
            self._set_pose(self._approach_pos_at(alpha))
            if alpha >= 1.0:
                self._begin_phase(Phase.DWELL, sim_step)
            return

        if self.phase == Phase.DWELL:
            self._set_pose(self.dwell_pos)
            if sim_step - self.phase_start >= self.dwell_steps:
                self._begin_exit(sim_step)
            return

        if self.phase == Phase.POST_HIT_HOLD:
            hold_pos = self.hold_pos if self.hold_pos is not None else self.dwell_pos
            self._set_pose(hold_pos)
            if sim_step - self.phase_start >= self.post_hit_hold_steps:
                self._begin_exit(sim_step, from_pos=hold_pos)
            return

        if self.phase in (Phase.EXIT, Phase.FLINCH):
            alpha = min((sim_step - self.phase_start) / self.exit_steps, 1.0)
            self._set_pose(self._interp(self.exit_from, self.exit_to, alpha))
            if alpha >= 1.0:
                self.phase = Phase.DONE
                self.hide()

    def handle_collision(self, sim_step: int) -> None:
        if self.phase in (
            Phase.POST_HIT_HOLD,
            Phase.EXIT,
            Phase.FLINCH,
            Phase.DONE,
            Phase.HIDDEN,
        ):
            return
        self.hold_pos = np.asarray(self.entity.get_pose().p, dtype=np.float64)
        if self.post_hit_hold_steps == 0:
            self._begin_exit(sim_step, from_pos=self.hold_pos)
        else:
            self._begin_phase(Phase.POST_HIT_HOLD, sim_step)
