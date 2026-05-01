from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
import sapien.core as sapien


HIDDEN_POS = np.array([6.0, 6.0, 2.5], dtype=np.float64)
IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def minimum_jerk(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return 10.0 * alpha**3 - 15.0 * alpha**4 + 6.0 * alpha**5


@dataclass
class ControllerPose:
    position: np.ndarray
    quat: np.ndarray = IDENTITY_QUAT


class BaseDisturbanceController:
    def __init__(self, actor, default_quat=None):
        self.actor = actor
        self._active = False
        self._release_pending = False
        if default_quat is None:
            default_quat = IDENTITY_QUAT
        self.default_quat = np.asarray(default_quat, dtype=np.float64)

    @property
    def entity(self):
        return self.actor.actor if hasattr(self.actor, "actor") else self.actor

    @property
    def active(self) -> bool:
        return self._active

    def _set_pose(self, pose: ControllerPose | np.ndarray):
        if isinstance(pose, ControllerPose):
            pos = pose.position
            quat = pose.quat
        else:
            pos = np.asarray(pose, dtype=np.float64)
            quat = self.default_quat
        self.entity.set_pose(sapien.Pose(pos, quat))

    def hide(self):
        self._active = False
        self._set_pose(HIDDEN_POS)

    def update_before_step(self, sim_step: int):
        raise NotImplementedError

    def handle_collision(self, sim_step: int):
        return None

    def needs_dynamic_release(self) -> bool:
        return self._release_pending

    def on_released_to_physics(self):
        self._release_pending = False


class SuddenDropController(BaseDisturbanceController):
    """Level-1 disturbance: spawn an obstacle at ``drop_pos`` once it is safe.

    The trigger is a two-stage state machine:

    1. ``waiting_xy`` -- wait until the EEF xy comes within ``trigger_dist`` of
       the planned drop xy. Once satisfied this condition is *latched*: even
       if the EEF later wanders outside the trigger radius we will still
       proceed to step 2.
    2. ``waiting_z`` -- wait until no robot link occupies the cylindrical
       spawn volume above the drop point. The volume is defined as

           xy: a disk of radius ``obstacle_radius_xy + link_clearance_xy``
               centered at ``target_pos.xy``,
           z:  ``[bottom_z - z_clearance, top_z + z_clearance]``

       where ``bottom_z = drop_pos.z - obstacle_support_offset`` and
       ``top_z = bottom_z + obstacle_height``. As soon as no link overlaps,
       teleport the actor to ``drop_pos`` and request dynamic release.

    There is no soft / hard fallback: if step 2 is never satisfied (e.g. the
    arm hovers over the drop point until the end of the episode) the
    obstacle simply never appears for this episode. This is the desired
    behaviour -- spawning into the arm causes large, unrealistic impulses
    that flatten the whole table state.
    """

    def __init__(
        self,
        actor,
        target_pos: np.ndarray,
        trigger_dist: float,
        drop_height: float,
        get_eef_pos_fn: Callable[[], np.ndarray],
        quat: np.ndarray | None = None,
        obstacle_height: float = 0.0,
        obstacle_radius_xy: float = 0.0,
        obstacle_support_offset: float = 0.0,
        get_robot_link_positions_fn: Callable[[], np.ndarray] | None = None,
        link_clearance_xy: float = 0.05,
        z_clearance: float = 0.03,
    ):
        super().__init__(actor, default_quat=quat)
        self.target_pos = np.asarray(target_pos, dtype=np.float64)
        self.trigger_dist = float(trigger_dist)
        self.drop_height = float(drop_height)
        self.get_eef_pos = get_eef_pos_fn
        self.drop_pos = self.target_pos.copy()
        self.drop_pos[2] += self.drop_height

        self.obstacle_height = float(max(obstacle_height, 0.0))
        self.obstacle_radius_xy = float(max(obstacle_radius_xy, 0.0))
        self.obstacle_support_offset = float(max(obstacle_support_offset, 0.0))
        self.get_robot_link_positions = get_robot_link_positions_fn
        self.link_clearance_xy = float(max(link_clearance_xy, 0.0))
        self.z_clearance = float(max(z_clearance, 0.0))

        self._dropped = False
        self._xy_latched = False
        self.hide()

    # ------------------------------------------------------------------ utils
    def _spawn_volume_z_range(self) -> tuple[float, float]:
        bottom_z = self.drop_pos[2] - self.obstacle_support_offset
        top_z = bottom_z + self.obstacle_height
        return bottom_z - self.z_clearance, top_z + self.z_clearance

    def _link_blocks_spawn(self) -> bool:
        """True iff any robot link currently sits inside the spawn cylinder."""
        if self.get_robot_link_positions is None or self.obstacle_height <= 0.0:
            # No way to tell -> err on the side of "no block" so behaviour
            # falls back to the original immediate drop.
            return False
        try:
            link_xyz = np.asarray(self.get_robot_link_positions(), dtype=np.float64)
        except Exception:
            return False
        if link_xyz.size == 0:
            return False
        if link_xyz.ndim != 2 or link_xyz.shape[1] < 3:
            return False

        radius = self.obstacle_radius_xy + self.link_clearance_xy
        if radius <= 0.0:
            return False
        z_lo, z_hi = self._spawn_volume_z_range()

        dxy = link_xyz[:, :2] - self.target_pos[:2][None, :]
        in_disk = (dxy[:, 0] ** 2 + dxy[:, 1] ** 2) <= (radius * radius)
        z = link_xyz[:, 2]
        in_band = (z >= z_lo) & (z <= z_hi)
        return bool(np.any(in_disk & in_band))

    # ------------------------------------------------------------- core hook
    def update_before_step(self, sim_step: int):
        if self._dropped:
            self._active = True
            return

        if not self._xy_latched:
            eef_pos = np.asarray(self.get_eef_pos(), dtype=np.float64)
            xy_dist = float(np.linalg.norm(eef_pos[:2] - self.target_pos[:2]))
            if xy_dist <= self.trigger_dist:
                self._xy_latched = True
            else:
                self.hide()
                return

        if self._link_blocks_spawn():
            self.hide()
            return

        self._dropped = True
        self._active = True
        self._set_pose(self.drop_pos)
        self._release_pending = True


class SlidingObstacleController(BaseDisturbanceController):
    def __init__(
        self,
        actor,
        start_pos: np.ndarray,
        final_pos: np.ndarray,
        move_start_step: int,
        move_end_step: int,
        hold_steps: int,
        arc_height: float,
        quat: np.ndarray | None = None,
    ):
        super().__init__(actor, default_quat=quat)
        self.start_pos = np.asarray(start_pos, dtype=np.float64)
        self.final_pos = np.asarray(final_pos, dtype=np.float64)
        self.move_start_step = int(move_start_step)
        self.move_end_step = int(move_end_step)
        self.hold_end_step = int(move_end_step + hold_steps)
        self.arc_height = float(arc_height)
        self._released = False

    def _path_pos(self, alpha: float) -> np.ndarray:
        pos = self.start_pos + alpha * (self.final_pos - self.start_pos)
        pos[2] += 4.0 * self.arc_height * alpha * (1.0 - alpha)
        return pos

    def update_before_step(self, sim_step: int):
        if self._released:
            self._active = True
            return
        self._active = True
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

    def handle_collision(self, sim_step: int):
        # Soft-release on first contact: instead of continuing to kinematically
        # push the obstacle through the arm along the scripted arc, mark the
        # actor for an immediate kinematic -> dynamic swap. The manager's
        # before_step() picks this up next tick via needs_dynamic_release()
        # and recreates the actor at its current pose with physics enabled,
        # so any further interaction is governed by gravity + contact forces
        # rather than a forced set_pose().
        if self._released:
            return
        self._released = True
        self._release_pending = True

    def on_released_to_physics(self):
        super().on_released_to_physics()
        self._released = True


class ReachPhase(str, Enum):
    HIDDEN = "hidden"
    ENTER = "enter"
    APPROACH = "approach"
    DWELL = "dwell"
    POST_HIT_HOLD = "post_hit_hold"  # frozen at the contact pose for a few sim steps
    EXIT = "exit"
    FLINCH = "flinch"  # legacy small dodge; kept for backward compat but unused by default
    DONE = "done"


class SharedReachingController(BaseDisturbanceController):
    def __init__(
        self,
        actor,
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
        post_hit_hold_steps: int = 3,
    ):
        super().__init__(actor)
        self.start_pos = np.asarray(start_pos, dtype=np.float64)
        self.hover_pos = np.asarray(hover_pos, dtype=np.float64)
        self.dwell_pos = np.asarray(dwell_pos, dtype=np.float64)
        self.move_start_step = int(move_start_step)
        self.entry_steps = max(int(entry_steps), 1)
        self.approach_steps = max(int(approach_steps), 1)
        self.dwell_steps = max(int(dwell_steps), 1)
        self.exit_steps = max(int(exit_steps), 1)
        # Post-collision behaviour: hold the contact pose for ``post_hit_hold_steps``
        # sim steps, then retreat to ``start_pos`` over ``exit_steps`` -- the
        # SAME duration as the natural DWELL->EXIT path, so the user only has
        # one knob to tune the retreat speed (l3_exit_steps) regardless of
        # whether the retreat was triggered by a collision or by completing
        # the dwell normally.
        self.post_hit_hold_steps = max(int(post_hit_hold_steps), 0)
        # Legacy small-dodge knobs kept on the constructor for backward compat;
        # the FLINCH path is no longer triggered by handle_collision.
        self.flinch_dist = float(flinch_dist)
        self.flinch_height = float(flinch_height)
        self.get_eef_pos = get_eef_pos_fn

        self.phase = ReachPhase.HIDDEN
        self.phase_start = 0
        self.exit_from = None
        self.exit_to = None
        self.hold_pos = None  # frozen contact pose during POST_HIT_HOLD
        self.hide()

    def _interp(self, start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
        return start + minimum_jerk(alpha) * (end - start)

    def _begin_phase(self, phase: ReachPhase, sim_step: int):
        self.phase = phase
        self.phase_start = sim_step

    def _begin_exit(self, sim_step: int, from_pos: np.ndarray | None = None):
        self.exit_from = np.asarray(from_pos if from_pos is not None else self.entity.get_pose().p, dtype=np.float64)
        self.exit_to = self.start_pos.copy()
        self._begin_phase(ReachPhase.EXIT, sim_step)

    def update_before_step(self, sim_step: int):
        if self.phase == ReachPhase.DONE:
            self.hide()
            return

        if self.phase == ReachPhase.HIDDEN:
            self.hide()
            if sim_step >= self.move_start_step:
                self._active = True
                self._begin_phase(ReachPhase.ENTER, sim_step)
            return

        self._active = True
        if self.phase == ReachPhase.ENTER:
            alpha = min((sim_step - self.phase_start) / self.entry_steps, 1.0)
            self._set_pose(self._interp(self.start_pos, self.hover_pos, alpha))
            if alpha >= 1.0:
                self._begin_phase(ReachPhase.APPROACH, sim_step)
            return

        if self.phase == ReachPhase.APPROACH:
            alpha = min((sim_step - self.phase_start) / self.approach_steps, 1.0)
            self._set_pose(self._interp(self.hover_pos, self.dwell_pos, alpha))
            if alpha >= 1.0:
                self._begin_phase(ReachPhase.DWELL, sim_step)
            return

        if self.phase == ReachPhase.DWELL:
            self._set_pose(self.dwell_pos)
            if sim_step - self.phase_start >= self.dwell_steps:
                self._begin_exit(sim_step)
            return

        if self.phase == ReachPhase.POST_HIT_HOLD:
            # Freeze the hand at the contact pose so it stops "fighting" the
            # robot arm; advance to the home-bound retreat once the hold timer
            # elapses.
            hold_pos = self.hold_pos if self.hold_pos is not None else self.dwell_pos
            self._set_pose(hold_pos)
            if sim_step - self.phase_start >= self.post_hit_hold_steps:
                self._begin_exit(sim_step, from_pos=hold_pos)
            return

        if self.phase in (ReachPhase.EXIT, ReachPhase.FLINCH):
            alpha = min((sim_step - self.phase_start) / self.exit_steps, 1.0)
            self._set_pose(self._interp(self.exit_from, self.exit_to, alpha))
            if alpha >= 1.0:
                self.phase = ReachPhase.DONE
                self.hide()

    def handle_collision(self, sim_step: int):
        # Only react once: after first contact, go HOLD -> EXIT and ignore any
        # further collision callbacks until the next episode.
        if self.phase in (
            ReachPhase.POST_HIT_HOLD,
            ReachPhase.EXIT,
            ReachPhase.FLINCH,
            ReachPhase.DONE,
            ReachPhase.HIDDEN,
        ):
            return
        self.hold_pos = np.asarray(self.entity.get_pose().p, dtype=np.float64)
        if self.post_hit_hold_steps == 0:
            # Skip the freeze and go straight to retreat.
            self._begin_exit(sim_step, from_pos=self.hold_pos)
        else:
            self._begin_phase(ReachPhase.POST_HIT_HOLD, sim_step)
