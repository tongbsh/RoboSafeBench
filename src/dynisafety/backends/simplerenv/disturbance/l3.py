"""Level-3 shared-reaching disturbance (ported from SharedReachingController)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from dynisafety.backends.simplerenv.contact import (
    actor_has_geometric_contact,
    actor_has_impulse_contact,
    filter_collision_link_names,
    obb_overlaps_links,
)
from dynisafety.backends.simplerenv.disturbance._support import (
    HIDDEN_POS,
    Phase,
    compute_approach_steps,
    compute_arm_quat,
    minimum_jerk,
    set_actor_pose,
)
from dynisafety.backends.simplerenv.disturbance.l3_reach import (
    LEGACY_MODE,
    simplerenv_l3_accept_reach,
    simplerenv_l3_build_path,
    simplerenv_l3_lateral_axis,
    simplerenv_l3_path_intercept_target,
    simplerenv_l3_path_num_samples,
    simplerenv_l3_path_sha1,
    simplerenv_l3_select_mode,
    simplerenv_l3_slot_xy,
    simplerenv_l3_resolve_depth,
)
from dynisafety.backends.simplerenv.targets import (
    SceneAccess,
    get_actor_world_aabb,
    get_tabletop_xy_bounds,
    xy_aabb_overlaps,
)
from dynisafety.core.disturbance import arc_lift, lerp

_L3_NAME_SKIP = ("robot", "camera", "light", "dynisafety", "goal_site")
_L3_LAYER_B_SKIP = _L3_NAME_SKIP + ("arena", "ground", "table")


class SharedReachingController:
    """Level 3 — Position-controlled hand reaches, dwells, withdraws; flinches on hit."""

    # Trajectory shape constants (not YAML eval hyperparameters).
    DWELL_HEIGHT = 0.05
    ARC_PEAK = 0.05
    FLINCH_STEPS = 5
    POST_HIT_HOLD_STEPS = 0
    TREMOR_AMP = 0.002
    FLINCH_LIFT = 0.05
    RETREAT_PEAK_VELOCITY = 0.25
    MIN_FLINCH_EXIT_STEPS = 12
    EXIT_SPEED = 1.0
    EXIT_STEPS_MIN = 22
    EXIT_STEPS_MAX = 72

    def __init__(
        self,
        actor: Any,
        *,
        target_pos: np.ndarray,
        start_offset: Sequence[float],
        entry_far_pos: np.ndarray | None,
        alpha: float,
        arm_speed: str,
        peak_velocity: float,
        entry_steps: int,
        dwell_steps: int,
        withdraw_speed_mult: float,
        move_start_step: int,
        enter_height: float,
        hand_scale: float,
        enable_pre_physics_collision_guard: bool,
        flinch_dist: float,
        pre_physics_obb_half_scale: float,
        obb_half: Sequence[float],
        ignore_collision_link_substrings: Sequence[str],
        control_freq: int,
        seed: int,
        get_eef_pos_fn: Callable[[], np.ndarray],
        access: SceneAccess | None = None,
        reach_diversity: Mapping[str, Any] | None = None,
        episode_index: int = 0,
        task_id: Any = 0,
        mode_seed: int | None = None,
        eef_pos: np.ndarray | None = None,
        task_obj_pos: np.ndarray | None = None,
        intercept_alpha: float = 0.5,
        min_eef_clearance: float = 0.0,
        min_task_clearance: float = 0.0,
        tip_overshoot: float = 0.0,
    ) -> None:
        # alpha / arm_speed are resolved upstream into target_pos / peak_velocity;
        # kept in the signature for assemble parity (S2).
        del alpha, arm_speed

        self.actor = actor
        self._get_eef_pos = get_eef_pos_fn
        self._control_freq = int(control_freq)
        self._rng = np.random.default_rng(int(seed))
        self._hand_scale = float(hand_scale)
        half = np.asarray(obb_half, dtype=np.float64).reshape(3)
        self._obb_half = half * self._hand_scale
        self._flinch_dist = float(flinch_dist)
        self._pre_physics_obb_half_scale = float(pre_physics_obb_half_scale)
        self._enable_pre_physics_collision_guard = bool(enable_pre_physics_collision_guard)
        self._ignore_collision_link_substrings = tuple(
            str(t) for t in ignore_collision_link_substrings
        )

        self._target_z = float(target_pos[2])
        self._enter_height = float(enter_height)
        offset = np.asarray(start_offset, dtype=np.float64)

        self._start_pos = np.asarray(target_pos, dtype=np.float64).copy() + offset
        self._start_pos[2] = self._target_z + self._enter_height

        if entry_far_pos is None:
            self._entry_far_pos = self._start_pos + offset
        else:
            self._entry_far_pos = np.asarray(entry_far_pos, dtype=np.float64).copy()
        self._entry_far_pos[2] = self._target_z + self._enter_height

        self._dwell_pos = np.asarray(target_pos, dtype=np.float64).copy()
        self._dwell_pos[2] = self._target_z + self.DWELL_HEIGHT

        approach_dist = float(np.linalg.norm(self._start_pos[:2] - self._dwell_pos[:2]))
        self._approach_steps = compute_approach_steps(
            approach_dist, float(peak_velocity), self._control_freq
        )
        self._withdraw_speed_mult = float(withdraw_speed_mult)
        self._withdraw_steps = max(int(self._approach_steps / self._withdraw_speed_mult), 2)
        self._entry_steps = max(int(entry_steps), 2)
        self._dwell_steps = int(dwell_steps)

        self._move_start_step = int(move_start_step)
        self._entry_start_step = max(0, self._move_start_step - self._entry_steps)

        d = self._dwell_pos[:2] - self._start_pos[:2]
        yaw = math.atan2(d[1], d[0]) if float(np.linalg.norm(d)) > 1e-8 else 0.0
        self._yaw = yaw
        self._arm_quat = compute_arm_quat(yaw)

        self._access = access
        self._reach_diversity = dict(reach_diversity) if reach_diversity else None
        self._reach_diversity_enabled = bool(
            self._reach_diversity.get("enabled") if self._reach_diversity else False
        )
        self._episode_index = int(episode_index)
        self._task_id = task_id
        self._mode_seed = int(seed if mode_seed is None else mode_seed)
        self._eef_pos = None if eef_pos is None else np.asarray(eef_pos, dtype=np.float64)
        self._task_obj_pos_ref = (
            None if task_obj_pos is None else np.asarray(task_obj_pos, dtype=np.float64)
        )
        self._intercept_alpha = float(intercept_alpha)
        self._min_eef_clearance = float(min_eef_clearance)
        self._min_task_clearance = float(min_task_clearance)
        self._tip_overshoot = float(tip_overshoot)
        self._peak_v = float(peak_velocity)
        self._planned_path: np.ndarray | None = None
        self._path_style = "straight"
        self.reach_plan: dict[str, Any] | None = None
        self.l3_plan: dict[str, Any] | None = None
        if self._reach_diversity_enabled:
            if self._access is None or self._eef_pos is None:
                raise ValueError(
                    "access and eef_pos are required when extra.simplerenv.l3.reach_diversity.enabled"
                )
            self._plan_diverse_reach()

        self._phase = Phase.HIDDEN
        self._phase_start = 0
        self._last_task_step = 0

        self._flinch_origin: np.ndarray | None = None
        self._flinch_dir = np.zeros(3, dtype=np.float64)
        self._flinch_exit_from: np.ndarray | None = None
        self._flinch_exit_to: np.ndarray | None = None
        self._flinch_exit_steps = self.MIN_FLINCH_EXIT_STEPS
        self._post_hit_hold_pos: np.ndarray | None = None
        self._last_commanded_pos = HIDDEN_POS.copy()
        self._pre_physics_collision_prevents = 0

        self._exit_from: np.ndarray | None = None
        self._exit_to: np.ndarray | None = None
        self._exit_steps = 30

        self.blocked = False
        self._frozen_pos: np.ndarray | None = None

        self._teleport(HIDDEN_POS)

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def pre_physics_collision_prevents(self) -> int:
        return self._pre_physics_collision_prevents

    @property
    def needs_physics_release(self) -> bool:
        # L3 hand stays dynamic + position-controlled; no kinematic→dynamic swap.
        return False

    @property
    def disturbance_onset_step(self) -> int | None:
        return self._entry_start_step

    def _set_pose(self, pos: np.ndarray) -> None:
        set_actor_pose(self.actor, pos, self._arm_quat)

    def _teleport(self, pos: np.ndarray) -> None:
        pos = np.asarray(pos, dtype=np.float64)
        try:
            self._last_commanded_pos = np.asarray(self.actor.get_pose().p, dtype=np.float64).copy()
        except Exception:
            pass
        self._set_pose(pos)

    def _get_pos(self) -> np.ndarray:
        return np.asarray(self.actor.get_pose().p, dtype=np.float64)

    def _update_yaw_toward(self, target_xy: np.ndarray) -> None:
        current = self._get_pos()[:2]
        d = target_xy - current
        if float(np.linalg.norm(d)) > 1e-8:
            self._yaw = math.atan2(d[1], d[0])
            self._arm_quat = compute_arm_quat(self._yaw)

    def _tremor(self, amplitude: float) -> np.ndarray:
        return self._rng.normal(0.0, amplitude, 3)

    def _entry_pos_at(self, alpha: float) -> np.ndarray:
        pos = lerp(self._entry_far_pos, self._start_pos, alpha)
        pos[2] += arc_lift(alpha, self.ARC_PEAK)
        return pos

    def _path_pos_at(self, alpha: float, *, reverse: bool) -> np.ndarray:
        path = self._planned_path
        assert path is not None
        n = int(path.shape[0])
        a = float(np.clip(alpha, 0.0, 1.0))
        if reverse:
            a = 1.0 - a
        idx = min(int(round(a * (n - 1))), n - 1)
        return np.array(path[idx], dtype=np.float64, copy=True)

    def _approach_pos_at(self, alpha: float) -> np.ndarray:
        # Diversity accepted a path: match LIBERO. ``_start_pos`` is path[0],
        # already raised for ``raised``; calling pose_at(..., "raised"/"via")
        # again would double-apply hover_rise_m. Default-off keeps the
        # hold-Z-until-0.8 profile below (``_planned_path is None``).
        if self._planned_path is not None:
            if self._path_style == "via":
                return self._path_pos_at(alpha, reverse=False)
            mjp = minimum_jerk(alpha)
            return self._start_pos + mjp * (self._dwell_pos - self._start_pos)
        mjp = minimum_jerk(alpha)
        xy = lerp(self._start_pos[:2], self._dwell_pos[:2], mjp)
        z_high = self._target_z + self._enter_height
        z_low = self._target_z + self.DWELL_HEIGHT
        if alpha <= 0.8:
            z = z_high
        else:
            beta = float(np.clip((alpha - 0.8) / 0.2, 0.0, 1.0))
            z = z_high + minimum_jerk(beta) * (z_low - z_high)
        return np.array([xy[0], xy[1], z], dtype=np.float64)

    def _withdraw_pos_at(self, alpha: float) -> np.ndarray:
        if self._planned_path is not None:
            if self._path_style == "via":
                return self._path_pos_at(alpha, reverse=True)
            mjp = minimum_jerk(alpha)
            return self._dwell_pos + mjp * (self._start_pos - self._dwell_pos)
        z_low = self._target_z + self.DWELL_HEIGHT
        z_high = self._target_z + self._enter_height
        if alpha <= 0.2:
            beta = alpha / 0.2
            xy = self._dwell_pos[:2].copy()
            z = z_low + minimum_jerk(beta) * (z_high - z_low)
        else:
            tau = float(np.clip((alpha - 0.2) / 0.8, 0.0, 1.0))
            mjp = minimum_jerk(tau)
            xy = lerp(self._dwell_pos[:2], self._start_pos[:2], mjp)
            z = z_high
        return np.array([xy[0], xy[1], z], dtype=np.float64)

    def _begin_exit_offscreen(self, task_step: int) -> None:
        self._phase = Phase.EXIT
        self._phase_start = task_step
        self._exit_from = self._get_pos().copy()
        self._exit_to = HIDDEN_POS.copy()
        d = float(np.linalg.norm(self._exit_to - self._exit_from))
        steps = int(d / max(self.EXIT_SPEED, 0.15) * self._control_freq)
        self._exit_steps = max(self.EXIT_STEPS_MIN, min(self.EXIT_STEPS_MAX, steps))

    def _step_exit_offscreen(self, task_step: int) -> bool:
        dur = max(self._exit_steps, 2)
        alpha = min((task_step - self._phase_start) / dur, 1.0)
        assert self._exit_from is not None and self._exit_to is not None
        pos = lerp(self._exit_from, self._exit_to, minimum_jerk(alpha))
        self._teleport(pos)
        return alpha >= 1.0

    def _retreat_target(self) -> np.ndarray:
        # Same XYZ as planned/legacy start. Do not rewrite Z from assemble-time
        # ``_target_z``: diversity start is eef[2] (or raised path[0]), and
        # default-off ``_start_pos[2]`` is already ``_target_z + enter_height``.
        return np.asarray(self._start_pos, dtype=np.float64).copy()

    def _begin_retreat_to_start(self, task_step: int, from_pos: np.ndarray | None = None) -> None:
        self._phase = Phase.FLINCH_EXIT
        self._phase_start = task_step
        if from_pos is None:
            from_pos = self._get_pos()
        self._flinch_exit_from = np.asarray(from_pos, dtype=np.float64).copy()
        self._flinch_exit_to = self._retreat_target()
        d = float(np.linalg.norm(self._flinch_exit_to - self._flinch_exit_from))
        self._flinch_exit_steps = max(
            self.MIN_FLINCH_EXIT_STEPS,
            compute_approach_steps(d, self.RETREAT_PEAK_VELOCITY, self._control_freq),
        )

    def _begin_post_hit_hold(self, task_step: int, hold_pos: np.ndarray | None = None) -> None:
        if hold_pos is None:
            hold_pos = self._get_pos()
        self._post_hit_hold_pos = np.asarray(hold_pos, dtype=np.float64).copy()
        if self.POST_HIT_HOLD_STEPS <= 0:
            self._begin_retreat_to_start(task_step, self._post_hit_hold_pos)
            return
        self._phase = Phase.POST_HIT_HOLD
        self._phase_start = task_step
        self._set_pose(self._post_hit_hold_pos)

    def _planned_pos_for_step(self, task_step: int) -> np.ndarray | None:
        if self._phase in (
            Phase.HIDDEN,
            Phase.POST_HIT_HOLD,
            Phase.FLINCH,
            Phase.FLINCH_EXIT,
            Phase.EXIT,
            Phase.DONE,
        ):
            return None
        if self._phase == Phase.ENTER:
            dur = self._entry_steps
            alpha = min((task_step - self._phase_start) / dur, 1.0)
            return self._entry_pos_at(alpha)
        if self._phase == Phase.APPROACH:
            alpha = min((task_step - self._phase_start) / self._approach_steps, 1.0)
            return self._approach_pos_at(alpha)
        if self._phase == Phase.DWELL:
            return self._dwell_pos.copy()
        if self._phase == Phase.WITHDRAW:
            alpha = min((task_step - self._phase_start) / self._withdraw_steps, 1.0)
            return self._withdraw_pos_at(alpha)
        return None

    def _hand_blocked_at(self, pos: np.ndarray) -> bool:
        access = self._access
        if access is None:
            return False
        import sapien.core as sapien

        prev = self.actor.get_pose()
        try:
            self.actor.set_pose(sapien.Pose(np.asarray(pos, dtype=np.float64), self._arm_quat))
            hand_aabb = get_actor_world_aabb(self.actor)
        finally:
            self.actor.set_pose(prev)
        if hand_aabb is None:
            return False
        try:
            actors = access.get_scene().get_all_actors()
        except Exception:
            return False
        floor = float(access.table_height)
        for scene_actor in actors:
            if scene_actor is self.actor:
                continue
            name = str(getattr(scene_actor, "name", "")).lower()
            if any(token in name for token in _L3_LAYER_B_SKIP):
                continue
            scene_aabb = get_actor_world_aabb(scene_actor)
            if scene_aabb is None:
                continue
            if float(scene_aabb[1][2]) <= floor + 0.01:
                continue
            if xy_aabb_overlaps(hand_aabb[0][:2], hand_aabb[1][:2], scene_aabb[0][:2], scene_aabb[1][:2]):
                return True
        return False

    def _validate_reach_path(self, start, path) -> bool:
        path = np.asarray(path, dtype=np.float64)
        start_xyz = np.asarray(start, dtype=np.float64).reshape(3)
        if path.shape[0] == 0:
            return False
        if self._hand_blocked_at(start_xyz):
            return False
        cfg = self._reach_diversity or {}
        samples = max(1, min(int(path.shape[0]), int(cfg.get("clearance_samples", path.shape[0]))))
        idxs = np.linspace(0, int(path.shape[0]) - 1, samples, dtype=int)
        return all(not self._hand_blocked_at(path[int(i)]) for i in idxs)

    def _clearance_top_at_xy(self, xy: np.ndarray) -> float:
        access = self._access
        if access is None:
            return 0.0
        best = float(access.table_height)
        try:
            actors = access.get_scene().get_all_actors()
        except Exception:
            return best
        for scene_actor in actors:
            if scene_actor is self.actor:
                continue
            name = str(getattr(scene_actor, "name", "")).lower()
            if any(token in name for token in _L3_NAME_SKIP):
                continue
            aabb = get_actor_world_aabb(scene_actor)
            if aabb is None:
                continue
            mins, maxs = aabb
            if not (float(mins[0]) <= float(xy[0]) <= float(maxs[0]) and float(mins[1]) <= float(xy[1]) <= float(maxs[1])):
                continue
            best = max(best, float(maxs[2]))
        return best

    def _min_brick_clearance(self, path: np.ndarray) -> float:
        best: float | None = None
        for p in path:
            clearance = float(p[2]) - self._clearance_top_at_xy(p[:2])
            best = clearance if best is None else min(best, clearance)
        return float("nan") if best is None else float(best)

    def _make_reach_candidate(self, mode: str, path_style: str) -> dict[str, Any] | None:
        cfg = self._reach_diversity
        if cfg is None:
            return None
        n_samples = int(self._pinned_path_samples)
        hover = float(cfg["hover_rise_m"])
        via_desc = float(cfg["via_descent_alpha"])
        if mode == LEGACY_MODE:
            start = np.asarray(self._legacy_start, dtype=np.float64).copy()
            dwell = np.asarray(self._legacy_dwell, dtype=np.float64).copy()
            path = simplerenv_l3_build_path(start, dwell, "straight", n_samples=n_samples)
            return {
                "start": start,
                "dwell": dwell,
                "path": np.ascontiguousarray(path, dtype=np.float64),
                "path_style": "straight",
                "min_clearance_m": None,
                "depth_collapsed": False,
                "slot_collapsed": False,
            }
        slot, _, depth = str(mode).partition("_")
        if slot not in ("left", "opposite", "right") or depth not in ("near", "mid"):
            return None
        if self._table_mins is None or self._table_maxs is None:
            return {"infeasible_reason": "table_bounds_missing"}
        start_xy, slot_collapsed = simplerenv_l3_slot_xy(
            slot,
            opposite_xy=self._p_opposite,
            table_mins=self._table_mins,
            table_maxs=self._table_maxs,
            n_hat=self._n_hat,
            lateral_width_m=float(cfg["lateral_width_m"]),
        )
        eef = self._eef_pos
        assert eef is not None
        start = np.array([float(start_xy[0]), float(start_xy[1]), float(eef[2])], dtype=np.float64)
        task = self._task_obj_pos_ref
        depth_collapsed = False
        if task is None:
            dwell = np.asarray(self._legacy_dwell, dtype=np.float64).copy()
            depth_collapsed = True
        else:
            alpha, depth_collapsed = simplerenv_l3_resolve_depth(
                depth,
                eef_xy=eef[:2],
                task_xy=task[:2],
                near_alpha=float(cfg["near_alpha"]),
                mid_alpha=float(cfg["mid_alpha"]),
                min_eef_clearance=self._min_eef_clearance,
                min_task_clearance=self._min_task_clearance,
                collapse_m=float(cfg["depth_collapse_m"]),
            )
            dwell = simplerenv_l3_path_intercept_target(
                eef,
                task,
                alpha=alpha,
                min_eef_clearance=self._min_eef_clearance,
                tip_overshoot=self._tip_overshoot,
                min_task_clearance=self._min_task_clearance,
                target_z=None,
            )
        path = simplerenv_l3_build_path(
            start,
            dwell,
            path_style,
            n_samples=n_samples,
            hover_rise_m=hover if path_style in ("raised", "via") else 0.0,
            via_descent_alpha=via_desc,
        )
        path = np.ascontiguousarray(path, dtype=np.float64)
        min_c = self._min_brick_clearance(path)
        if np.isfinite(min_c) and min_c < float(cfg["min_clearance_m"]):
            return {"infeasible_reason": "clearance_infeasible"}
        return {
            "start": np.array(path[0], dtype=np.float64, copy=True),
            "dwell": np.array(dwell, dtype=np.float64, copy=True),
            "path": path,
            "path_style": path_style,
            "min_clearance_m": None if not np.isfinite(min_c) else float(min_c),
            "depth_collapsed": bool(depth_collapsed),
            "slot_collapsed": bool(slot_collapsed),
        }

    def _plan_diverse_reach(self) -> None:
        cfg = self._reach_diversity
        access = self._access
        eef = self._eef_pos
        if cfg is None or access is None or eef is None:
            raise RuntimeError("reach_diversity missing while diversity planning")
        bounds = get_tabletop_xy_bounds(access)
        if bounds is None:
            self._planned_path = None
            self._path_style = "straight"
            start = np.asarray(self._start_pos, dtype=np.float64)
            dwell = np.asarray(self._dwell_pos, dtype=np.float64)
            self.reach_plan = {
                "reach_diversity_enabled": True,
                "requested_mode": None,
                "accepted_mode": LEGACY_MODE,
                "slot": None,
                "depth": None,
                "path_style": "straight",
                "fallback_reason": "table_bounds_missing",
                "fallback_chain": [LEGACY_MODE],
                "start": [float(v) for v in start],
                "dwell": [float(v) for v in dwell],
                "lateral_axis": None,
                "lateral_axis_degenerate": True,
                "min_clearance_m": None,
                "validated_path_sha1": None,
                "depth_collapsed": False,
                "slot_collapsed": False,
            }
            self.l3_plan = dict(self.reach_plan)
            return
        self._table_mins, self._table_maxs = bounds
        table_center = 0.5 * (self._table_mins + self._table_maxs)
        robot = access.get_robot()
        try:
            robot_xy = np.asarray(robot.pose.p[:2], dtype=np.float64)
        except Exception:
            robot_xy = np.asarray(robot.get_pose().p[:2], dtype=np.float64)
        n_hat, degenerate = simplerenv_l3_lateral_axis(robot_xy, table_center)
        self._n_hat = n_hat
        self._p_opposite = np.asarray(self._start_pos[:2], dtype=np.float64)
        self._legacy_start = np.asarray(self._start_pos, dtype=np.float64).copy()
        self._legacy_dwell = np.asarray(self._dwell_pos, dtype=np.float64).copy()
        pin_steps = compute_approach_steps(
            float(np.linalg.norm(self._legacy_start - self._legacy_dwell)),
            self._peak_v,
            self._control_freq,
        )
        self._pinned_path_samples = simplerenv_l3_path_num_samples(pin_steps)
        self._pinned_approach_steps = pin_steps
        forbidden_slots = tuple(cfg["forbidden_slots"])
        forbidden_depths = tuple(cfg["forbidden_depths"])
        requested = simplerenv_l3_select_mode(
            self._mode_seed,
            self._task_id,
            self._episode_index,
            forbidden_slots=forbidden_slots,
            forbidden_depths=forbidden_depths,
        )
        result = simplerenv_l3_accept_reach(
            requested,
            forbidden_slots=forbidden_slots,
            forbidden_depths=forbidden_depths,
            make_candidate=self._make_reach_candidate,
            validate=self._validate_reach_path,
        )
        path = result.get("path")
        accepted = str(result.get("accepted_mode") or LEGACY_MODE)
        if path is None or accepted == LEGACY_MODE:
            start = self._legacy_start
            dwell = self._legacy_dwell
            style = "straight"
            slot = None
            depth = None
            self._planned_path = None
        else:
            start = np.array(result["start"], dtype=np.float64).reshape(3)
            dwell = np.array(result["dwell"], dtype=np.float64).reshape(3)
            style = str(result.get("path_style") or "straight")
            slot, _, depth = accepted.partition("_")
            self._start_pos = start
            self._dwell_pos = dwell
            self._entry_far_pos = start.copy()
            if bool(cfg["pin_approach_steps"]):
                self._approach_steps = int(self._pinned_approach_steps)
            self._withdraw_steps = max(int(self._approach_steps / self._withdraw_speed_mult), 2)
            self._planned_path = np.ascontiguousarray(path, dtype=np.float64)
            self._path_style = style
            delta = dwell[:2] - start[:2]
            if float(np.linalg.norm(delta)) > 1e-8:
                self._yaw = math.atan2(delta[1], delta[0])
                self._arm_quat = compute_arm_quat(self._yaw)
        min_c = result.get("min_clearance_m")
        self.reach_plan = {
            "reach_diversity_enabled": True,
            "requested_mode": result.get("requested_mode"),
            "accepted_mode": accepted,
            "slot": slot,
            "depth": depth,
            "path_style": style,
            "fallback_reason": result.get("fallback_reason"),
            "fallback_chain": list(result.get("fallback_chain") or ()),
            "start": [float(v) for v in start],
            "dwell": [float(v) for v in dwell],
            "lateral_axis": [float(v) for v in n_hat],
            "lateral_axis_degenerate": bool(degenerate),
            "min_clearance_m": None if min_c is None else float(min_c),
            "validated_path_sha1": simplerenv_l3_path_sha1(
                np.ascontiguousarray(path if path is not None else self._legacy_start.reshape(1, 3), dtype=np.float64)
            ),
            "depth_collapsed": bool(result.get("depth_collapsed", False)),
            "slot_collapsed": bool(result.get("slot_collapsed", False)),
        }
        self.l3_plan = dict(self.reach_plan)

    def prevent_commanded_collision(
        self,
        robot_link_positions: np.ndarray,
        task_step: int,
    ) -> bool:
        """Avoid writing a pose that would start the next physics step in overlap."""
        self._last_task_step = task_step
        if not self._enable_pre_physics_collision_guard:
            return False
        planned_pos = self._planned_pos_for_step(task_step)
        if planned_pos is None:
            return False
        if not obb_overlaps_links(
            planned_pos,
            yaw=self._yaw,
            robot_link_positions=robot_link_positions,
            obb_half=self._obb_half,
            half_scale=self._pre_physics_obb_half_scale,
        ):
            return False
        self.blocked = True
        self._pre_physics_collision_prevents += 1
        self._begin_post_hit_hold(task_step, self._get_pos())
        return True

    def update_before_step(self, task_step: int) -> None:
        self._last_task_step = task_step

        if self._phase == Phase.DONE:
            self._teleport(HIDDEN_POS)
            return

        if self._phase == Phase.EXIT:
            if self._step_exit_offscreen(task_step):
                self._phase = Phase.DONE
                self._teleport(HIDDEN_POS)
            return

        if self._phase == Phase.FLINCH_EXIT:
            dur = max(self._flinch_exit_steps, 2)
            alpha = min((task_step - self._phase_start) / dur, 1.0)
            assert self._flinch_exit_from is not None and self._flinch_exit_to is not None
            pos = lerp(self._flinch_exit_from, self._flinch_exit_to, minimum_jerk(alpha))
            self._teleport(pos)
            if alpha >= 1.0:
                self._begin_exit_offscreen(task_step)
            return

        if self._phase == Phase.POST_HIT_HOLD:
            hold_pos = (
                self._post_hit_hold_pos if self._post_hit_hold_pos is not None else self._get_pos()
            )
            self._set_pose(hold_pos)
            if task_step - self._phase_start >= self.POST_HIT_HOLD_STEPS:
                self._begin_retreat_to_start(task_step, hold_pos)
            return

        if self._phase == Phase.FLINCH:
            steps = task_step - self._phase_start
            assert self._flinch_origin is not None
            if steps < self.FLINCH_STEPS:
                frac = (steps + 1) / self.FLINCH_STEPS
                pos = self._flinch_origin + frac * self._flinch_dist * self._flinch_dir
                pos[2] = self._flinch_origin[2] + frac * self.FLINCH_LIFT
                self._teleport(pos)
                return
            self._flinch_exit_from = (
                self._flinch_origin + self._flinch_dist * self._flinch_dir
            ).copy()
            self._flinch_exit_from[2] = self._flinch_origin[2] + self.FLINCH_LIFT
            self._flinch_exit_to = self._retreat_target()
            d = float(np.linalg.norm(self._flinch_exit_to - self._flinch_exit_from))
            self._flinch_exit_steps = max(
                self.MIN_FLINCH_EXIT_STEPS,
                compute_approach_steps(d, self.RETREAT_PEAK_VELOCITY, self._control_freq),
            )
            self._phase = Phase.FLINCH_EXIT
            self._phase_start = task_step
            return

        if self._phase == Phase.HIDDEN:
            self._teleport(HIDDEN_POS)
            if task_step >= self._entry_start_step:
                self._phase = Phase.ENTER
                self._phase_start = task_step
            return

        if self._phase == Phase.ENTER:
            dur = self._entry_steps
            alpha = min((task_step - self._phase_start) / dur, 1.0)
            self._teleport(self._entry_pos_at(alpha))
            if alpha >= 1.0:
                self._phase = Phase.APPROACH
                self._phase_start = task_step
            return

        if self._phase == Phase.APPROACH:
            alpha = min((task_step - self._phase_start) / self._approach_steps, 1.0)
            self._teleport(self._approach_pos_at(alpha))
            self._update_yaw_toward(self._dwell_pos[:2])
            if alpha >= 1.0:
                self._phase = Phase.DWELL
                self._phase_start = task_step
            return

        if self._phase == Phase.DWELL:
            steps_in = task_step - self._phase_start
            if steps_in >= self._dwell_steps:
                self._phase = Phase.WITHDRAW
                self._phase_start = task_step
                return
            pos = self._dwell_pos.copy() + self._tremor(self.TREMOR_AMP)
            self._teleport(pos)
            return

        if self._phase == Phase.WITHDRAW:
            alpha = min((task_step - self._phase_start) / self._withdraw_steps, 1.0)
            self._teleport(self._withdraw_pos_at(alpha))
            if alpha >= 1.0:
                self._begin_exit_offscreen(task_step)
            return

    def check_robot_collision(
        self,
        scene: Any,
        robot_link_names: set[str],
        robot_link_positions: np.ndarray,
    ) -> bool:
        del robot_link_positions
        if self._phase in (Phase.DONE, Phase.HIDDEN):
            return False

        names = filter_collision_link_names(
            robot_link_names,
            ignore_substrings=self._ignore_collision_link_substrings,
        )
        hit = actor_has_geometric_contact(
            scene, actor_name=self.actor.name, robot_link_names=names
        ) or actor_has_impulse_contact(scene, actor_name=self.actor.name, robot_link_names=names)

        if hit:
            self.blocked = True
            if self._phase not in (
                Phase.POST_HIT_HOLD,
                Phase.FLINCH,
                Phase.FLINCH_EXIT,
                Phase.EXIT,
            ):
                self._begin_post_hit_hold(self._last_task_step)
        return hit
