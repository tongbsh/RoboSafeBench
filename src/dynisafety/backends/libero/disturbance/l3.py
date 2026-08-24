"""Level-3 shared-reaching disturbance (ported from SharedReachingController)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np

from dynisafety.backends.libero.disturbance._support import (
    HandBase,
    HandPhysicsParams,
    Phase,
    _body_name_for_geom,
    _geom_is_table_or_floor,
    arm_quat_from_yaw,
    compute_approach_steps,
    compute_l3_path_intercept_target,
    geom_looks_like_table_or_floor,
    get_robot_base_pos,
    get_table_center_pos,
    minimum_jerk,
)
from dynisafety.backends.libero.disturbance.l2 import (
    _highest_fixture_top_at_xy,
    _highest_movable_prop_top_at_xy,
)
from dynisafety.backends.libero.disturbance.l3_reach import (
    LEGACY_MODE,
    libero_l3_accept_reach,
    libero_l3_build_path,
    libero_l3_lateral_axis,
    libero_l3_opposite_xy,
    libero_l3_path_num_samples,
    libero_l3_path_sha1,
    libero_l3_resolve_depth,
    libero_l3_select_mode,
    libero_l3_slot_xy,
)
from dynisafety.backends.libero.env import LiberoEnvHandle
from dynisafety.core.disturbance import arc_lift, lerp


def _mj_forward(model: Any, data: Any) -> None:
    import mujoco

    mujoco.mj_forward(model, data)


def l3_collision_retreat_pos(
    current: np.ndarray, start_pos: np.ndarray, alpha: float
) -> np.ndarray:
    """Straight line from the hit pose back to start; linear in time (no jerk ease-in)."""
    cur = np.asarray(current, dtype=np.float64).reshape(3)
    start = np.asarray(start_pos, dtype=np.float64).reshape(3)
    tau = float(np.clip(alpha, 0.0, 1.0))
    return cur + tau * (start - cur)


def l3_body_looks_like_robot(name: str) -> bool:
    """True for robosuite arm / gripper bodies (never a vanish prop)."""
    n = str(name or "")
    return n.startswith("robot0_") or n.startswith("gripper0_")


def l3_hand_contact_is_scene_prop(
    *,
    hand_gids: set[int],
    robot_gids: set[int],
    prop_gids: set[int],
    geom_a: int,
    geom_b: int,
    other_is_table: bool,
    other_is_robot_body: bool,
) -> bool:
    """True when a contact pairs the hand with a free task prop.

    Robot / table / fixture contacts do not count — those must not vanish the hand.
    """
    hand = {int(g) for g in hand_gids}
    robot = {int(g) for g in robot_gids}
    props = {int(g) for g in prop_gids}
    pair = {int(geom_a), int(geom_b)}
    if not (pair & hand):
        return False
    others = pair - hand
    if not others:
        return False
    other = next(iter(others))
    if other in robot or bool(other_is_robot_body):
        return False
    if other not in props:
        return False
    return not bool(other_is_table)


def l3_prop_was_nudged(
    prev: Mapping[str, np.ndarray],
    curr: Mapping[str, np.ndarray],
    hand_xy: np.ndarray,
    *,
    move_m: float,
    near_m: float,
) -> bool:
    """True if a nearby free prop translated by at least ``move_m``."""
    hand = np.asarray(hand_xy, dtype=np.float64).reshape(2)
    thresh = float(move_m)
    radius = float(near_m)
    for name, pos in curr.items():
        old = prev.get(name)
        if old is None:
            continue
        now = np.asarray(pos, dtype=np.float64).reshape(3)
        was = np.asarray(old, dtype=np.float64).reshape(3)
        if float(np.linalg.norm(now - was)) < thresh:
            continue
        if float(np.linalg.norm(now[:2] - hand)) <= radius:
            return True
    return False


# Vanish only on a real hand↔free-prop touch or a 2 mm nudge. Do not use a
# padded AABB: the 22 cm collision brick already sweeps past nearby cans
# while the visible hand is still only meeting the robot.
_VANISH_NUDGE_M = 0.002
_VANISH_NEAR_M = 0.12
_VANISH_CONTACT_DIST_M = 0.002


class SharedReachingController(HandBase):
    """Level 3 — co-worker reaches for a nearby object, dwells, withdraws."""

    hides_obstacles = True

    # Trajectory shape numerics (not YAML hyperparameters).
    ARC_PEAK = 0.05
    FLINCH_DIST = 0.06
    FLINCH_STEPS = 5
    TREMOR_AMP = 0.002
    CONTACT_EPS = 1e-6
    PENETRATION_EPS = 1e-4
    SWEEP_SUBSTEPS = 8

    def __init__(
        self,
        handle: LiberoEnvHandle,
        *,
        alpha: float,
        dwell_steps: int,
        entry_steps: int,
        withdraw_speed_mult: float,
        arm_speed: str,
        peak_velocity: float,
        start_mode: str | int,
        start_offset: Sequence[float],
        move_start_step: int,
        target_mode: str | int,
        interp_z: float | None,
        start_z: float | None,
        seed: int,
        target_pos: np.ndarray,
        hand_physics: HandPhysicsParams,
        control_freq: int,
        start_pos_override: np.ndarray | None,
        enter_height: float,
        dwell_height: float,
        withdraw_height: float,
        skip_offscreen_exit: bool,
        reach_diversity: Mapping[str, Any] | None = None,
        episode_index: int = 0,
        task_id: Any = None,
        mode_seed: int | None = None,
        eef_pos: np.ndarray | None = None,
        task_obj_pos: np.ndarray | None = None,
        start_xy_half: Sequence[float] | None = None,
        min_eef_clearance: float | None = None,
        min_task_clearance: float | None = None,
        tip_overshoot: float | None = None,
        intercept_alpha: float | None = None,
        hand_half_size: Sequence[float] | None = None,
        hand_collision_offset: Sequence[float] | None = None,
    ) -> None:
        # alpha / target_mode / start_mode / interp_z / start_z / arm_speed:
        # target & start pose are resolved by the caller (backend assembly);
        # retained for API parity with RunConfig fields.
        del alpha, target_mode, start_mode, interp_z, start_z, arm_speed
        super().__init__(
            handle,
            hand_name="hand_pusher_1",
            seed=int(seed),
            hand_physics=hand_physics,
            control_freq=int(control_freq),
        )
        self._target_obj_pos = np.asarray(target_pos, dtype=np.float64).copy()
        self._target_z = float(self._target_obj_pos[2])
        self._peak_v = float(peak_velocity)
        self._enter_height = float(enter_height)
        self._dwell_height = float(dwell_height)
        self._withdraw_height = float(withdraw_height)
        if self._enter_height <= 0.0 or self._dwell_height <= 0.0 or self._withdraw_height <= 0.0:
            raise ValueError(
                "enter_height / dwell_height / withdraw_height must be > 0, "
                f"got {enter_height!r}, {dwell_height!r}, {withdraw_height!r}"
            )

        if start_pos_override is not None:
            self._start_pos = np.asarray(start_pos_override, dtype=np.float64).copy()
        else:
            offset = np.asarray(start_offset, dtype=np.float64)
            self._start_pos = self._target_obj_pos.copy() + offset
            self._start_pos[2] = self._target_z + self._enter_height
        self._entry_far_pos = self._start_pos.copy()
        self._skip_offscreen_exit = bool(skip_offscreen_exit)

        self._dwell_pos = self._target_obj_pos.copy()
        self._withdraw_speed_mult = float(withdraw_speed_mult)
        approach_dist = float(np.linalg.norm(self._start_pos - self._dwell_pos))
        self._approach_steps = compute_approach_steps(
            approach_dist, self._peak_v, control_freq=self._control_freq
        )
        self._withdraw_steps = max(int(self._approach_steps / self._withdraw_speed_mult), 2)
        self._entry_steps = max(int(entry_steps), 0)
        self._dwell_steps = int(dwell_steps)
        self._move_start_step = int(move_start_step)
        self._entry_start_step = max(0, self._move_start_step - self._entry_steps)
        d = self._dwell_pos[:2] - self._start_pos[:2]
        yaw = math.atan2(d[1], d[0]) if np.linalg.norm(d) > 1e-8 else 0.0
        self._arm_quat = arm_quat_from_yaw(yaw)
        self._phase = Phase.HIDDEN
        self._phase_start = 0
        self._flinch_origin = None
        self._flinch_dir = np.array([0.0, 0.0, 0.0])
        self._flinch_exit_from = None
        self._flinch_exit_to = None
        self._flinch_exit_steps = 20
        self._prop_pose_snap: dict[str, np.ndarray] | None = None
        self.reach_plan: dict[str, Any] | None = None
        self._planned_path: np.ndarray | None = None
        self._path_style = "straight"
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
        self._start_xy_half = None if start_xy_half is None else [float(v) for v in start_xy_half]
        self._min_eef_clearance = min_eef_clearance
        self._min_task_clearance = min_task_clearance
        self._tip_overshoot = tip_overshoot
        self._intercept_alpha = intercept_alpha
        self._hand_half_size = (
            None
            if hand_half_size is None
            else np.asarray(hand_half_size, dtype=np.float64).reshape(3)
        )
        self._hand_collision_offset = (
            None
            if hand_collision_offset is None
            else np.asarray(hand_collision_offset, dtype=np.float64).reshape(3)
        )
        self._robot_like_geom_ids = self._collect_l3_robot_like_geom_ids()
        self._ground_like_geom_ids = self._collect_l3_ground_like_geom_ids()
        if self._reach_diversity_enabled:
            if self._hand_half_size is None or self._hand_collision_offset is None:
                raise ValueError(
                    "hand.half_size and hand.collision_offset are required "
                    "when extra.libero.l3.reach_diversity.enabled"
                )
            if self._eef_pos is None or self._start_xy_half is None:
                raise ValueError(
                    "eef_pos and start_xy_half are required when "
                    "extra.libero.l3.reach_diversity.enabled"
                )
            if self._min_eef_clearance is None or self._min_task_clearance is None:
                raise ValueError(
                    "min_eef_clearance / min_task_clearance required when "
                    "reach_diversity.enabled"
                )
            if self._tip_overshoot is None or self._intercept_alpha is None:
                raise ValueError(
                    "tip_overshoot / intercept_alpha required when reach_diversity.enabled"
                )
            self._plan_diverse_reach()
        self._teleport(self._hidden_pos)

    def reset(self, episode_index: int) -> None:
        del episode_index

    def step(self, t: int, obs: Mapping[str, Any]) -> None:
        del obs
        self.update_before_step(t)

    @property
    def active_obstacle_name(self) -> str | None:
        return self.hand_name

    @property
    def disturbance_onset_step(self) -> int | None:
        return self._entry_start_step

    # --- trajectory helpers ---

    def _entry_pos_at(self, alpha):
        """Position along the entry arc (alpha ∈ [0, 1])."""
        pos = lerp(self._entry_far_pos, self._start_pos, alpha)
        pos[2] += arc_lift(alpha, self.ARC_PEAK)
        return pos

    def _approach_pos_at(self, alpha):
        """Straight 3D line: advance and descend together from start to dwell."""
        if self._path_style == "via" and self._planned_path is not None:
            return self._path_pos_at(alpha, reverse=False)
        mjp = minimum_jerk(alpha)
        return self._start_pos + mjp * (self._dwell_pos - self._start_pos)

    def _withdraw_pos_at(self, alpha):
        """Same 3D line as approach, reversed (dwell → start)."""
        if self._path_style == "via" and self._planned_path is not None:
            return self._path_pos_at(alpha, reverse=True)
        mjp = minimum_jerk(alpha)
        return self._dwell_pos + mjp * (self._start_pos - self._dwell_pos)

    def _path_pos_at(self, alpha, *, reverse: bool) -> np.ndarray:
        path = self._planned_path
        assert path is not None
        n = int(path.shape[0])
        a = float(np.clip(alpha, 0.0, 1.0))
        if reverse:
            a = 1.0 - a
        idx = min(int(round(a * (n - 1))), n - 1)
        return np.array(path[idx], dtype=np.float64, copy=True)

    def _collect_l3_robot_like_geom_ids(self) -> set[int]:
        sim = self.inner_env.sim
        gids: set[int] = set()
        for gid in range(int(sim.model.ngeom)):
            if l3_body_looks_like_robot(_body_name_for_geom(sim, gid)):
                gids.add(int(gid))
        return gids

    def _collect_l3_ground_like_geom_ids(self) -> frozenset[int]:
        sim = self.inner_env.sim
        model = sim.model
        gids: set[int] = set()
        for gid in range(int(model.ngeom)):
            bid = int(model.geom_bodyid[gid])
            body_name = model.body_id2name(bid) or ""
            geom_name = model.geom_id2name(gid) or ""
            if geom_looks_like_table_or_floor(body_name) or geom_looks_like_table_or_floor(
                geom_name
            ):
                gids.add(int(gid))
        return frozenset(gids)

    @contextmanager
    def _sim_state_guard(self):
        sim = self.inner_env.sim
        saved_qpos = sim.data.qpos.copy()
        saved_qvel = sim.data.qvel.copy()
        saved_act = sim.data.act.copy() if sim.model.na > 0 else None
        try:
            yield
        finally:
            sim.data.qpos[:] = saved_qpos
            sim.data.qvel[:] = saved_qvel
            if saved_act is not None:
                sim.data.act[:] = saved_act
            _mj_forward(sim.model._model, sim.data._data)

    def _hand_yaw(self, start: np.ndarray, dwell: np.ndarray) -> float:
        delta = np.asarray(dwell, dtype=np.float64).reshape(3)[:2] - np.asarray(
            start, dtype=np.float64
        ).reshape(3)[:2]
        if float(np.linalg.norm(delta)) <= 1e-8:
            return 0.0
        return float(math.atan2(delta[1], delta[0]))

    def _palm_xy(self, pos: np.ndarray, yaw: float) -> np.ndarray:
        off = self._hand_collision_offset
        assert off is not None
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array(
            [float(pos[0] + c * off[0] - s * off[1]), float(pos[1] + s * off[0] + c * off[1])],
            dtype=np.float64,
        )

    def _hand_bottom_z(self, pos: np.ndarray) -> float:
        half = self._hand_half_size
        off = self._hand_collision_offset
        assert half is not None and off is not None
        return float(pos[2]) + float(off[2]) - float(half[2])

    def _clearance_top_fn(self):
        z_land = float(self._dwell_pos[2])
        sim = self.inner_env.sim
        exclude = set(self.obstacle_geom_ids)
        exclude.update(self._robot_like_geom_ids)

        def _fn(xy):
            fixture = _highest_fixture_top_at_xy(sim, xy, z_land=z_land)
            prop = _highest_movable_prop_top_at_xy(sim, xy, exclude_geom_ids=exclude)
            return float(max(fixture, prop))

        return _fn

    def _min_brick_clearance(self, path: np.ndarray, yaw: float) -> float:
        top_fn = self._clearance_top_fn()
        cfg = self._reach_diversity
        n = int(path.shape[0])
        samples = n if cfg is None else max(1, min(n, int(cfg["clearance_samples"])))
        if samples >= n:
            picks = path
        else:
            idxs = np.linspace(0, n - 1, samples, dtype=int)
            picks = path[idxs]
        best: float | None = None
        for p in picks:
            bottom = self._hand_bottom_z(p)
            for xy in (p[:2], self._palm_xy(p, yaw)):
                clearance = bottom - top_fn(xy)
                best = clearance if best is None else min(best, clearance)
        return float("nan") if best is None else float(best)

    def _hand_has_blocking_contact(self, *, allow_table_touch: bool) -> bool:
        sim = self.inner_env.sim
        hand = set(self.obstacle_geom_ids)
        robot = set(self._robot_like_geom_ids)
        ground = set(self._ground_like_geom_ids)
        props = self._prop_geom_ids()
        for i in range(int(sim.data.ncon)):
            c = sim.data.contact[i]
            pair = {int(c.geom1), int(c.geom2)}
            if not (pair & hand):
                continue
            others = pair - hand
            if len(others) != 1:
                continue
            other = next(iter(others))
            if other in robot or l3_body_looks_like_robot(_body_name_for_geom(sim, other)):
                continue
            dist = float(c.dist)
            if dist < -self.PENETRATION_EPS:
                return True
            if allow_table_touch and (other in ground or _geom_is_table_or_floor(sim, other)):
                continue
            if other in props or other not in ground:
                if dist <= self.CONTACT_EPS:
                    return True
        return False

    def _swept_hand_free(self, from_pos: np.ndarray, to_pos: np.ndarray, yaw: float) -> bool:
        steps = max(1, int(self.SWEEP_SUBSTEPS))
        quat = arm_quat_from_yaw(yaw)
        saved_quat = self._arm_quat
        self._arm_quat = quat
        try:
            for i in range(1, steps + 1):
                a = i / steps
                pos = from_pos + a * (to_pos - from_pos)
                self._teleport(pos, relax=False)
                if self._hand_has_blocking_contact(allow_table_touch=True):
                    return False
        finally:
            self._arm_quat = saved_quat
        return True

    def _validate_reach_path(self, start, path) -> bool:
        path = np.asarray(path, dtype=np.float64)
        start_xyz = np.asarray(start, dtype=np.float64).reshape(3)
        if path.shape[0] == 0:
            return False
        dwell = path[-1]
        yaw = self._hand_yaw(start_xyz, dwell)
        self._arm_quat = arm_quat_from_yaw(yaw)
        self._teleport(start_xyz, relax=False)
        if self._hand_has_blocking_contact(allow_table_touch=True):
            return False
        if not np.allclose(start_xyz, path[0]):
            self._teleport(path[0], relax=False)
            if self._hand_has_blocking_contact(allow_table_touch=True):
                return False
        for i in range(1, int(path.shape[0])):
            if not self._swept_hand_free(path[i - 1], path[i], yaw):
                return False
        return True

    def _legacy_start_dwell(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self._legacy_start, dtype=np.float64).copy(),
            np.asarray(self._legacy_dwell, dtype=np.float64).copy(),
        )

    def _make_reach_candidate(self, mode: str, path_style: str) -> dict[str, Any] | None:
        cfg = self._reach_diversity
        if cfg is None:
            return None
        n_samples = int(self._pinned_path_samples)
        hover = float(cfg["hover_rise_m"])
        via_desc = float(cfg["via_descent_alpha"])
        if mode == LEGACY_MODE:
            start, dwell = self._legacy_start_dwell()
            path = libero_l3_build_path(
                start, dwell, "straight", n_samples=n_samples, hover_rise_m=0.0
            )
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
        start_xy, slot_collapsed = libero_l3_slot_xy(
            slot,
            opposite_xy=self._p_opposite,
            table_center_xy=self._table_xy,
            start_xy_half=self._start_xy_half,
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
            alpha, depth_collapsed = libero_l3_resolve_depth(
                depth,
                eef_xy=eef[:2],
                task_xy=task[:2],
                near_alpha=float(cfg["near_alpha"]),
                mid_alpha=float(cfg["mid_alpha"]),
                min_eef_clearance=float(self._min_eef_clearance),
                min_task_clearance=float(self._min_task_clearance),
                collapse_m=float(cfg["depth_collapse_m"]),
            )
            dwell = compute_l3_path_intercept_target(
                eef,
                task,
                alpha=alpha,
                min_eef_clearance=float(self._min_eef_clearance),
                tip_overshoot=float(self._tip_overshoot),
                min_task_clearance=float(self._min_task_clearance),
                target_z=None,
            )
        path = libero_l3_build_path(
            start,
            dwell,
            path_style,
            n_samples=n_samples,
            hover_rise_m=hover if path_style in ("raised", "via") else 0.0,
            via_descent_alpha=via_desc,
        )
        path = np.ascontiguousarray(path, dtype=np.float64)
        yaw = self._hand_yaw(path[0], path[-1])
        min_c = self._min_brick_clearance(path, yaw)
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
        if cfg is None:
            raise RuntimeError("reach_diversity missing while diversity planning")
        eef = self._eef_pos
        assert eef is not None and self._start_xy_half is not None
        robot_xy = np.asarray(get_robot_base_pos(self.env), dtype=np.float64)[:2]
        table_xy = np.asarray(get_table_center_pos(self.env), dtype=np.float64)[:2]
        n_hat, degenerate = libero_l3_lateral_axis(robot_xy, table_xy)
        p_opp = libero_l3_opposite_xy(robot_xy, table_xy, self._start_xy_half)
        self._n_hat = n_hat
        self._table_xy = table_xy
        self._p_opposite = p_opp
        self._legacy_start = np.asarray(self._start_pos, dtype=np.float64).copy()
        self._legacy_dwell = np.asarray(self._dwell_pos, dtype=np.float64).copy()

        mid_start = np.array([float(p_opp[0]), float(p_opp[1]), float(eef[2])], dtype=np.float64)
        if self._task_obj_pos_ref is None:
            mid_dwell = self._legacy_dwell.copy()
        else:
            mid_dwell = compute_l3_path_intercept_target(
                eef,
                self._task_obj_pos_ref,
                alpha=float(cfg["mid_alpha"]),
                min_eef_clearance=float(self._min_eef_clearance),
                tip_overshoot=float(self._tip_overshoot),
                min_task_clearance=float(self._min_task_clearance),
                target_z=None,
            )
        pin_steps = compute_approach_steps(
            float(np.linalg.norm(mid_start - mid_dwell)),
            self._peak_v,
            control_freq=self._control_freq,
        )
        if bool(cfg["pin_approach_steps"]):
            self._pinned_path_samples = libero_l3_path_num_samples(pin_steps)
            self._pinned_approach_steps = pin_steps
        else:
            own = compute_approach_steps(
                float(np.linalg.norm(self._legacy_start - self._legacy_dwell)),
                self._peak_v,
                control_freq=self._control_freq,
            )
            self._pinned_path_samples = libero_l3_path_num_samples(own)
            self._pinned_approach_steps = own

        forbidden_slots = tuple(cfg["forbidden_slots"])
        forbidden_depths = tuple(cfg["forbidden_depths"])
        requested = libero_l3_select_mode(
            self._mode_seed,
            self._task_id,
            self._episode_index,
            forbidden_slots=forbidden_slots,
            forbidden_depths=forbidden_depths,
        )

        def make_candidate(mode: str, path_style: str):
            return self._make_reach_candidate(mode, path_style)

        def validate(start, path) -> bool:
            return bool(self._validate_reach_path(start, path))

        with self._sim_state_guard():
            result = libero_l3_accept_reach(
                requested,
                forbidden_slots=forbidden_slots,
                forbidden_depths=forbidden_depths,
                make_candidate=make_candidate,
                validate=validate,
            )

        path = result.get("path")
        if path is None:
            start, dwell = self._legacy_start_dwell()
            path = libero_l3_build_path(
                start,
                dwell,
                "straight",
                n_samples=int(self._pinned_path_samples),
                hover_rise_m=0.0,
            )
            result = {
                **result,
                "start": start,
                "dwell": dwell,
                "path": path,
                "path_style": "straight",
                "accepted_mode": LEGACY_MODE,
                "fallback_reason": result.get("fallback_reason") or "all_candidates_rejected",
            }
        path = np.ascontiguousarray(path, dtype=np.float64)
        start = np.array(path[0], dtype=np.float64, copy=True)
        dwell = np.asarray(result.get("dwell") if result.get("dwell") is not None else path[-1])
        dwell = np.array(dwell, dtype=np.float64).reshape(3)
        accepted = str(result.get("accepted_mode") or LEGACY_MODE)
        style = result.get("path_style")
        if accepted == LEGACY_MODE:
            start, dwell = self._legacy_start_dwell()
            path = libero_l3_build_path(
                start,
                dwell,
                "straight",
                n_samples=libero_l3_path_num_samples(self._approach_steps),
                hover_rise_m=0.0,
            )
            path = np.ascontiguousarray(path, dtype=np.float64)
            start = np.array(path[0], dtype=np.float64, copy=True)
            style = "straight"
            slot: str | None = None
            depth: str | None = None
            self._planned_path = None
        else:
            slot, _, depth = accepted.partition("_")
            self._start_pos = start
            self._dwell_pos = dwell
            self._entry_far_pos = start.copy()
            if bool(cfg["pin_approach_steps"]):
                self._approach_steps = int(self._pinned_approach_steps)
            else:
                self._approach_steps = compute_approach_steps(
                    float(np.linalg.norm(start - dwell)),
                    self._peak_v,
                    control_freq=self._control_freq,
                )
            self._withdraw_steps = max(
                int(self._approach_steps / self._withdraw_speed_mult), 2
            )
            self._planned_path = path
            self._path_style = str(style or "straight")
            delta = dwell[:2] - start[:2]
            yaw = math.atan2(delta[1], delta[0]) if float(np.linalg.norm(delta)) > 1e-8 else 0.0
            self._arm_quat = arm_quat_from_yaw(yaw)

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
            "validated_path_sha1": libero_l3_path_sha1(path),
            "approach_steps": int(self._approach_steps),
            "pin_approach_steps": bool(cfg["pin_approach_steps"]),
            "depth_collapsed": bool(result.get("depth_collapsed", False)),
            "slot_collapsed": bool(result.get("slot_collapsed", False)),
            "hover_rise_m": float(cfg["hover_rise_m"]),
            "vanish_step": None,
            "vanish_reason": None,
            "retreat_step": None,
            "hand_visible_through_dwell": False,
        }

    def _finish_hidden(self) -> None:
        self._phase = Phase.DONE
        self._teleport(self._hidden_pos)

    def _begin_high_exit_offscreen(self, task_step: int) -> None:
        """Leave the table at withdraw height, or hide immediately if off-screen."""
        if self._skip_offscreen_exit:
            self._finish_hidden()
            return
        self._begin_exit_offscreen(task_step)
        self._exit_to[2] = self._target_z + self._withdraw_height

    # --- collision retreat (LIBERO L3 only) ---

    def _note_reach_event(self, kind: str) -> None:
        plan = self.reach_plan if isinstance(self.reach_plan, dict) else None
        extra = getattr(self, "l3_plan", None)
        if not isinstance(extra, dict):
            extra = plan
        if not isinstance(extra, dict):
            return
        step = int(getattr(self, "_last_task_step", 0) or 0)
        if kind == "vanish":
            extra["vanish_step"] = step
            extra["vanish_reason"] = "prop_contact"
            extra["hand_visible_through_dwell"] = False
            if plan is not None and plan is not extra:
                plan["vanish_step"] = step
                plan["vanish_reason"] = "prop_contact"
                plan["hand_visible_through_dwell"] = False
        elif kind == "retreat":
            extra.setdefault("retreat_step", step)
            if plan is not None and plan is not extra:
                plan.setdefault("retreat_step", step)

    def _begin_collision_retreat(self, task_step: int) -> None:
        """Reverse along current → start at approach speed; no 5 mm contact cap."""
        self._note_reach_event("retreat")
        cur = self._get_pos()
        self._flinch_exit_from = np.asarray(cur, dtype=np.float64).copy()
        self._flinch_exit_to = self._start_pos.copy()
        d = float(np.linalg.norm(self._flinch_exit_to - self._flinch_exit_from))
        if d < 1e-4:
            self._begin_high_exit_offscreen(task_step)
            return
        self._flinch_exit_steps = max(
            2, compute_approach_steps(d, self._peak_v, control_freq=self._control_freq)
        )
        self._phase = Phase.FLINCH_EXIT
        self._phase_start = task_step

    def _start_flinch(self, task_step):
        self._begin_collision_retreat(task_step)

    def _vanish_in_place(self) -> None:
        """Drop the hand off-screen immediately (no retreat sweep)."""
        self._note_reach_event("vanish")
        self._finish_hidden()

    def _is_free_joint_prop(self, obj: Any) -> bool:
        import mujoco

        joints = getattr(obj, "joints", None) or ()
        if not joints:
            return False
        sim = self.inner_env.sim
        try:
            jid = int(sim.model.joint_name2id(joints[-1]))
        except Exception:
            return False
        return int(sim.model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE)

    def _prop_geom_ids(self) -> set[int]:
        sim = self.inner_env.sim
        model = sim.model
        gids: set[int] = set()
        for name, obj in self.inner_env.objects_dict.items():
            nl = str(name).lower()
            if "hand" in nl or geom_looks_like_table_or_floor(str(name)):
                continue
            if "obstacle" in nl or "box_base" in nl:
                continue
            if not self._is_free_joint_prop(obj):
                continue
            try:
                root = int(model.body_name2id(obj.root_body))
            except Exception:
                continue
            root_xy = np.asarray(sim.data.body_xpos[root][:2], dtype=np.float64)
            if abs(float(root_xy[0])) >= 5.0 or abs(float(root_xy[1])) >= 5.0:
                continue
            for gid in range(int(model.ngeom)):
                bid = int(model.geom_bodyid[gid])
                if bid == root or int(model.body_parentid[bid]) == root:
                    gids.add(int(gid))
        return gids

    def _hand_contacts_scene_prop(self) -> bool:
        sim = self.inner_env.sim
        hand = set(self.obstacle_geom_ids)
        robot = set(self._robot_geom_ids or ())
        props = self._prop_geom_ids()
        for i in range(int(sim.data.ncon)):
            c = sim.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            pair = {g1, g2}
            others = pair - hand
            if len(others) != 1:
                continue
            other = next(iter(others))
            if not l3_hand_contact_is_scene_prop(
                hand_gids=hand,
                robot_gids=robot,
                prop_gids=props,
                geom_a=g1,
                geom_b=g2,
                other_is_table=_geom_is_table_or_floor(sim, other),
                other_is_robot_body=l3_body_looks_like_robot(_body_name_for_geom(sim, other)),
            ):
                continue
            if float(c.dist) <= _VANISH_CONTACT_DIST_M:
                return True
        return False

    def _snapshot_prop_poses(self) -> dict[str, np.ndarray]:
        sim = self.inner_env.sim
        out: dict[str, np.ndarray] = {}
        for name, obj in self.inner_env.objects_dict.items():
            nl = str(name).lower()
            if "hand" in nl or geom_looks_like_table_or_floor(str(name)):
                continue
            if not self._is_free_joint_prop(obj):
                continue
            try:
                bid = sim.model.body_name2id(obj.root_body)
                pos = np.asarray(sim.data.body_xpos[bid], dtype=np.float64).copy()
            except Exception:
                continue
            if abs(float(pos[0])) >= 5.0 or abs(float(pos[1])) >= 5.0:
                continue
            out[str(name)] = pos
        return out

    def _maybe_vanish_for_prop(self) -> bool:
        """Vanish on first overlap / contact, or a nearby prop starting to move."""
        if self._phase in (Phase.HIDDEN, Phase.DONE, Phase.EXIT):
            return False
        curr = self._snapshot_prop_poses()
        nudged = False
        if self._prop_pose_snap is not None:
            nudged = l3_prop_was_nudged(
                self._prop_pose_snap,
                curr,
                self._get_pos()[:2],
                move_m=_VANISH_NUDGE_M,
                near_m=_VANISH_NEAR_M,
            )
        self._prop_pose_snap = curr
        if self._hand_contacts_scene_prop() or nudged:
            self._vanish_in_place()
            return True
        return False

    # --- main step ---

    def update_before_step(self, task_step: int):
        self._last_task_step = task_step

        if self._phase == Phase.DONE:
            self._teleport(self._hidden_pos)
            return

        if self._phase == Phase.EXIT:
            if self._step_exit_offscreen(task_step):
                self._phase = Phase.DONE
                self._teleport(self._hidden_pos)
            return

        if self._phase == Phase.FLINCH_EXIT:
            dur = max(self._flinch_exit_steps, 2)
            alpha = min((task_step - self._phase_start) / dur, 1.0)
            pos = l3_collision_retreat_pos(self._flinch_exit_from, self._flinch_exit_to, alpha)
            # relax=False: do not clamp to in_contact_max_step_m (5 mm).
            self._teleport(pos, relax=False)
            if self._maybe_vanish_for_prop():
                return
            if alpha >= 1.0:
                self._begin_high_exit_offscreen(task_step)
            return

        if self._phase == Phase.FLINCH:
            self._begin_collision_retreat(task_step)
            return

        if self._phase == Phase.HIDDEN:
            self._teleport(self._hidden_pos)
            if task_step >= self._entry_start_step:
                if self._entry_steps <= 0:
                    self._phase = Phase.APPROACH
                else:
                    self._phase = Phase.ENTER
                self._phase_start = task_step
            return

        if self._phase == Phase.ENTER:
            dur = max(self._entry_steps, 1)
            alpha = min((task_step - self._phase_start) / dur, 1.0)
            self._teleport(self._entry_pos_at(alpha))
            if self._maybe_vanish_for_prop():
                return
            if alpha >= 1.0:
                self._phase = Phase.APPROACH
                self._phase_start = task_step
            return

        if self._phase == Phase.APPROACH:
            alpha = min((task_step - self._phase_start) / self._approach_steps, 1.0)
            self._teleport(self._approach_pos_at(alpha))
            if self._maybe_vanish_for_prop():
                return
            if alpha >= 1.0:
                self._phase = Phase.DWELL
                self._phase_start = task_step
            return

        if self._phase == Phase.DWELL:
            steps_in = task_step - self._phase_start
            if steps_in >= self._dwell_steps:
                if isinstance(self.reach_plan, dict):
                    self.reach_plan["hand_visible_through_dwell"] = True
                extra = getattr(self, "l3_plan", None)
                if isinstance(extra, dict):
                    extra["hand_visible_through_dwell"] = True
                self._phase = Phase.WITHDRAW
                self._phase_start = task_step
                return
            pos = self._dwell_pos.copy() + self._tremor(self.TREMOR_AMP)
            self._teleport(pos)
            if self._maybe_vanish_for_prop():
                return
            return

        if self._phase == Phase.WITHDRAW:
            alpha = min((task_step - self._phase_start) / self._withdraw_steps, 1.0)
            self._teleport(self._withdraw_pos_at(alpha))
            if self._maybe_vanish_for_prop():
                return
            if alpha >= 1.0:
                self._begin_high_exit_offscreen(task_step)
            return

    def check_robot_collision(self, robot_geom_ids) -> bool:
        hit_now = super().check_robot_collision(robot_geom_ids)
        if self._phase in (Phase.EXIT, Phase.DONE, Phase.HIDDEN):
            return hit_now
        # After physics: vanish only if a free prop was touched or nudged.
        # Arm / gripper contact alone still retreats along the approach line.
        if self._maybe_vanish_for_prop():
            return hit_now
        if hit_now and self._phase not in (Phase.FLINCH, Phase.FLINCH_EXIT):
            self._start_flinch(self._last_task_step)
        return hit_now
