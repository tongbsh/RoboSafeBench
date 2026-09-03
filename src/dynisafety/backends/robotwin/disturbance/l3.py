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
        release_on_hit: bool = False,
        move_start_task_step: int = 0,
    ) -> None:
        super().__init__(actor, default_quat=np.asarray(quat, dtype=np.float64))
        self.start_pos = np.asarray(start_pos, dtype=np.float64)
        self.hover_pos = np.asarray(hover_pos, dtype=np.float64)
        self.dwell_pos = np.asarray(dwell_pos, dtype=np.float64)
        self.release_on_hit = bool(release_on_hit)
        self.pose_start = np.asarray(
            start_pos if pose_start is None else pose_start, dtype=np.float64
        )
        self.move_start_step = int(move_start_step)
        self.move_start_task_step = int(move_start_task_step)
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
        self._released_to_physics = False
        self._frozen = False
        self._release_reason: str | None = None
        self._onset_task_step: int | None = None
        self.hide()

    @property
    def disturbance_onset_step(self) -> int | None:
        """Task-step onset (first frame ENTER actually starts)."""
        return self._onset_task_step

    @property
    def release_reason(self) -> str | None:
        return self._release_reason

    def _interp(self, start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
        return lerp(start, end, minimum_jerk(alpha))

    def _approach_pos_at(self, alpha: float) -> np.ndarray:
        # Drop only. The inbound is ENTER (start→hover). ``via`` / ``raised``
        # must not replay start→dwell here — that packed the table crossing
        # into ``approach_steps`` (400) after a no-op ENTER.
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

    def request_dynamic_release(self, reason: str = "last_free_hit") -> None:
        """Park at last-free; manager rebuilds a dynamic palm next before_step."""
        if self._released_to_physics or self._frozen:
            return
        self._release_pending = True
        self._release_reason = str(reason)

    def cancel_pending_release(self) -> None:
        """Stay kinematic at the last free pose; do not rebuild as dynamic."""
        self._release_pending = False
        self._frozen = True
        self._release_reason = "skipped_penetration"

    def on_released_to_physics(self) -> None:
        super().on_released_to_physics()
        self._released_to_physics = True
        self._frozen = False

    def update_before_step(self, sim_step: int, *, task_step: int | None = None) -> None:
        if self.phase == Phase.DONE:
            self.hide()
            return
        if self._frozen and not self._released_to_physics:
            self._active = True
            return
        if self._release_pending and not self._released_to_physics:
            # Last-free pose is already restored; do not fly back into overlap.
            self._active = True
            return
        if self._released_to_physics:
            # Palm is dynamic until official force is recorded, then vanish.
            self._active = True
            return
        self._drive_phase(sim_step, write_pose=True, task_step=task_step)

    def _task_gate_open(self, task_step: int | None, sim_step: int) -> bool:
        t_task = 0 if task_step is None else int(task_step)
        if int(self.move_start_task_step) > 0:
            return t_task >= int(self.move_start_task_step)
        return int(sim_step) >= int(self.move_start_step)

    def _lock_onset(self, task_step: int | None) -> None:
        if self._onset_task_step is not None:
            return
        if task_step is not None:
            self._onset_task_step = int(task_step)

    def _drive_phase(
        self, sim_step: int, *, write_pose: bool, task_step: int | None = None
    ) -> None:
        if self.phase == Phase.HIDDEN:
            if write_pose:
                self.hide()
            if not self._task_gate_open(task_step, sim_step):
                return
            self._active = True
            self._lock_onset(task_step)
            self._begin_phase(Phase.ENTER, sim_step)
            return

        self._active = True
        if self.phase == Phase.ENTER:
            alpha = min((sim_step - self.phase_start) / self.entry_steps, 1.0)
            if write_pose:
                self._set_pose(self._interp(self.start_pos, self.hover_pos, alpha))
            if alpha >= 1.0:
                self._begin_phase(Phase.APPROACH, sim_step)
            return

        if self.phase == Phase.APPROACH:
            alpha = min((sim_step - self.phase_start) / self.approach_steps, 1.0)
            if write_pose:
                self._set_pose(self._approach_pos_at(alpha))
            if alpha >= 1.0:
                self._begin_phase(Phase.DWELL, sim_step)
            return

        if self.phase == Phase.DWELL:
            if write_pose:
                self._set_pose(self.dwell_pos)
            if sim_step - self.phase_start >= self.dwell_steps:
                self._begin_exit(sim_step)
            return

        if self.phase == Phase.POST_HIT_HOLD:
            # Legacy hold-then-exit. kinematic_first_hit handle_collision vanishes.
            hold_pos = self.hold_pos if self.hold_pos is not None else self.dwell_pos
            if write_pose:
                self._set_pose(hold_pos)
            if sim_step - self.phase_start >= self.post_hit_hold_steps:
                self._begin_exit(sim_step, from_pos=hold_pos)
            return

        if self.phase in (Phase.EXIT, Phase.FLINCH):
            alpha = min((sim_step - self.phase_start) / self.exit_steps, 1.0)
            if write_pose:
                self._set_pose(self._interp(self.exit_from, self.exit_to, alpha))
            if alpha >= 1.0:
                self._vanish()

    def _vanish(self) -> None:
        self.phase = Phase.DONE
        self.hide()

    def handle_collision(self, sim_step: int) -> None:
        # kinematic_first_hit: official force is already recorded; vanish.
        # dynamic_release (release_on_hit): last-free + palm, like L1/L2.
        # Miss path is still DWELL then EXIT (withdraw to start).
        del sim_step
        if self.phase in (Phase.DONE, Phase.HIDDEN):
            return
        if self.release_on_hit and not self._released_to_physics:
            self.request_dynamic_release("impulse_hit")
            return
        self._vanish()
