"""Level-2 sliding-obstacle disturbance (ported from SlidingObstacleController)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from dynisafety.backends.simplerenv.contact import (
    actor_has_geometric_contact,
    actor_has_impulse_contact,
)
from dynisafety.backends.simplerenv.disturbance._support import (
    compute_obstacle_quat,
    set_actor_pose,
)
from dynisafety.backends.simplerenv.disturbance.l2_spawn import (
    LEGACY_MODE,
    simplerenv_l2_accept_spawn,
    simplerenv_l2_build_path,
    simplerenv_l2_candidate_start_xy,
    simplerenv_l2_lateral_axis,
    simplerenv_l2_path_num_samples,
    simplerenv_l2_path_sha1,
    simplerenv_l2_select_mode,
    simplerenv_l2_solve_linear_start_z,
    simplerenv_l2_solve_trapezoid,
)
from dynisafety.backends.simplerenv.targets import (
    SceneAccess,
    get_actor_world_aabb,
    get_tabletop_xy_bounds,
    xy_aabb_overlaps,
)
from dynisafety.core.disturbance import arc_lift, lerp

_L2_NAME_SKIP = ("robot", "camera", "light", "dynisafety", "goal_site")
_L2_LAYER_B_SKIP = _L2_NAME_SKIP + ("arena", "ground", "table")
_BLOCKING_HEIGHT_MARGIN_Z = 0.02


class SlidingObstacleController:
    """Level 2 — Arc-path slide into workspace; optional physics release."""

    def __init__(
        self,
        actor: Any,
        *,
        final_pos: np.ndarray,
        start_offset: Sequence[float],
        move_start_step: int,
        move_end_step: int,
        settle_steps: int,
        arc_height: float,
        enable_random_start: bool,
        random_start_xy_range: float,
        random_start_z_range: float,
        align_start_y_to_eef: bool,
        release_after_settle: bool,
        get_eef_pos_fn: Callable[[], np.ndarray],
        seed: int,
        access: SceneAccess | None = None,
        spawn_diversity: Mapping[str, Any] | None = None,
        episode_index: int = 0,
        task_id: Any = 0,
        mode_seed: int | None = None,
    ) -> None:
        self.actor = actor
        self.move_start_step = int(move_start_step)
        self.move_end_step = int(move_end_step)
        self._settle_end_step = self.move_end_step + int(settle_steps)
        self._arc_height = float(arc_height)
        self._release_after_settle = bool(release_after_settle)
        self._access = access
        self._spawn_diversity = dict(spawn_diversity) if spawn_diversity else None
        self._spawn_diversity_enabled = bool(
            self._spawn_diversity.get("enabled") if self._spawn_diversity else False
        )
        self._episode_index = int(episode_index)
        self._task_id = task_id
        self._mode_seed = int(seed if mode_seed is None else mode_seed)
        self.spawn_plan: dict[str, Any] | None = None

        self.final_pos = np.asarray(final_pos, dtype=np.float64).copy()
        base_offset = np.asarray(start_offset, dtype=np.float64)

        rng = np.random.default_rng(int(seed))
        if enable_random_start:
            jitter = np.array(
                [
                    rng.uniform(-random_start_xy_range, random_start_xy_range),
                    rng.uniform(-random_start_xy_range, random_start_xy_range),
                    rng.uniform(0.0, max(0.0, float(random_start_z_range))),
                ],
                dtype=np.float64,
            )
            self.start_pos = self.final_pos + base_offset + jitter
        else:
            self.start_pos = self.final_pos + base_offset

        if align_start_y_to_eef:
            eef_pos = get_eef_pos_fn()
            self.start_pos[1] = float(eef_pos[1])

        self._p0 = np.asarray(self.final_pos + base_offset, dtype=np.float64)
        self._legacy_start = np.asarray(self.start_pos, dtype=np.float64).copy()
        self._enable_random_start = bool(enable_random_start)

        push_dir = self.final_pos - self.start_pos
        push_dir[2] = 0.0
        norm = float(np.linalg.norm(push_dir))
        if norm > 1e-8:
            push_dir /= norm
        self._yaw = math.atan2(push_dir[1], push_dir[0])
        self._arm_quat = compute_obstacle_quat(self._yaw)
        self._path = self._build_arc_path()
        self._body_to_bottom_offset = self._probe_body_to_bottom_offset()

        if self._spawn_diversity_enabled:
            if self._access is None:
                raise ValueError("SceneAccess is required when spawn_diversity.enabled")
            self._plan_diverse_start_and_trajectory()

        self.blocked = False
        self._frozen_pos: np.ndarray | None = None
        self._released = False

        set_actor_pose(self.actor, self.start_pos, self._arm_quat)

    @property
    def needs_physics_release(self) -> bool:
        return bool(self._released)

    @property
    def disturbance_onset_step(self) -> int | None:
        return self.move_start_step

    def _build_arc_path(self) -> np.ndarray:
        n = max(2, self.move_end_step - self.move_start_step + 1)
        alphas = np.linspace(0.0, 1.0, n)
        path = np.empty((n, 3), dtype=np.float64)
        for i, a in enumerate(alphas):
            pos = lerp(self.start_pos, self.final_pos, float(a))
            pos[2] += arc_lift(float(a), self._arc_height)
            path[i] = pos
        return path

    def _probe_body_to_bottom_offset(self) -> float:
        aabb = get_actor_world_aabb(self.actor)
        if aabb is None:
            return 0.0
        return float(np.asarray(self.actor.get_pose().p, dtype=np.float64)[2] - aabb[0][2])

    def _clearance_top_fn(self):
        access = self._access
        assert access is not None
        exclude = self.actor
        table_z = float(access.table_height)

        def _fn(xy):
            best = table_z
            try:
                actors = access.get_scene().get_all_actors()
            except Exception:
                return best
            for scene_actor in actors:
                if scene_actor is exclude:
                    continue
                name = str(getattr(scene_actor, "name", "")).lower()
                if any(token in name for token in _L2_NAME_SKIP):
                    continue
                aabb = get_actor_world_aabb(scene_actor)
                if aabb is None:
                    continue
                mins, maxs = aabb
                if not (float(mins[0]) <= float(xy[0]) <= float(maxs[0]) and float(mins[1]) <= float(xy[1]) <= float(maxs[1])):
                    continue
                best = max(best, float(maxs[2]))
            return best

        return _fn

    def _min_clearance_along_path(self, path: np.ndarray) -> float:
        top_fn = self._clearance_top_fn()
        cfg = self._spawn_diversity
        tail_n = 0 if cfg is None else int(cfg["tail_exclude_waypoints"])
        n = int(path.shape[0])
        end = max(1, n - tail_n)
        bottom = float(self._body_to_bottom_offset)
        best: float | None = None
        for p in path[:end]:
            clearance = float(p[2]) - bottom - top_fn(p[:2])
            best = clearance if best is None else min(best, clearance)
        return float("nan") if best is None else float(best)

    def _xy_blocked_at(self, pos: np.ndarray) -> bool:
        access = self._access
        if access is None:
            return False
        import sapien.core as sapien

        prev = self.actor.get_pose()
        try:
            self.actor.set_pose(sapien.Pose(np.asarray(pos, dtype=np.float64), self._arm_quat))
            obstacle_aabb = get_actor_world_aabb(self.actor)
        finally:
            self.actor.set_pose(prev)
        if obstacle_aabb is None:
            return False
        floor = float(access.table_height) + _BLOCKING_HEIGHT_MARGIN_Z
        try:
            actors = access.get_scene().get_all_actors()
        except Exception:
            return False
        for scene_actor in actors:
            if scene_actor is self.actor:
                continue
            name = str(getattr(scene_actor, "name", "")).lower()
            if any(token in name for token in _L2_LAYER_B_SKIP):
                continue
            scene_aabb = get_actor_world_aabb(scene_actor)
            if scene_aabb is None:
                continue
            if float(scene_aabb[1][2]) <= floor:
                continue
            if xy_aabb_overlaps(obstacle_aabb[0][:2], obstacle_aabb[1][:2], scene_aabb[0][:2], scene_aabb[1][:2]):
                return True
        return False

    def _validate_spawn_path(self, start, path) -> bool:
        path = np.asarray(path, dtype=np.float64)
        start_xyz = np.asarray(start, dtype=np.float64).reshape(3)
        n = int(path.shape[0])
        cfg = self._spawn_diversity
        if cfg is None:
            raise RuntimeError("spawn_diversity required for path validation")
        judged_end = max(0, n - int(cfg["tail_exclude_waypoints"]))
        if self._xy_blocked_at(start_xyz):
            return False
        if n > 0 and not np.allclose(start_xyz, path[0]) and self._xy_blocked_at(path[0]):
            return False
        for i in range(0, judged_end):
            if self._xy_blocked_at(path[i]):
                return False
        return True

    def _legacy_jitter_arc(self) -> tuple[np.ndarray, np.ndarray]:
        start = np.asarray(self._legacy_start, dtype=np.float64).copy()
        path = simplerenv_l2_build_path(
            start,
            self.final_pos,
            "arc",
            n_samples=simplerenv_l2_path_num_samples(self.move_start_step, self.move_end_step),
            arc_height=self._arc_height,
        )
        return start, np.ascontiguousarray(path, dtype=np.float64)

    def _make_diversity_candidate(self, mode: str, *, p0: np.ndarray, n_hat: np.ndarray) -> dict[str, Any] | None:
        cfg = self._spawn_diversity
        if cfg is None:
            return None
        n_samples = simplerenv_l2_path_num_samples(self.move_start_step, self.move_end_step)
        landing = np.asarray(self.final_pos, dtype=np.float64)
        if mode == LEGACY_MODE:
            start, path = self._legacy_jitter_arc()
            return {
                "start": start,
                "path": path,
                "linear_solve_mode": "na",
                "min_clearance_m": None,
            }
        slot, sep, style = str(mode).partition("_")
        if not sep or slot not in ("left", "center", "right"):
            return None
        start_xy = simplerenv_l2_candidate_start_xy(p0, n_hat, slot, float(cfg["lateral_width_m"]))
        if style == "arc":
            start = np.array([float(start_xy[0]), float(start_xy[1]), float(p0[2])], dtype=np.float64)
            path = simplerenv_l2_build_path(
                start, landing, "arc", n_samples=n_samples, arc_height=self._arc_height
            )
            path = np.ascontiguousarray(path, dtype=np.float64)
            return {
                "start": start,
                "path": path,
                "linear_solve_mode": "na",
                "min_clearance_m": self._min_clearance_along_path(path),
            }
        if style != "linear":
            return None
        origin_clearance = float(cfg["linear_z_clearance_m"]) + float(self._body_to_bottom_offset)
        z_start, _reason = simplerenv_l2_solve_linear_start_z(
            start_xy,
            landing,
            self._clearance_top_fn(),
            clearance_m=origin_clearance,
            tail_alpha=float(cfg["linear_tail_alpha"]),
            max_start_rise_m=float(cfg["linear_max_start_z_m"]),
            samples=int(cfg["clearance_samples"]),
            p0_z_hint=float(p0[2]),
        )
        if z_start is not None:
            start = np.array([float(start_xy[0]), float(start_xy[1]), float(z_start)], dtype=np.float64)
            path = simplerenv_l2_build_path(
                start, landing, "linear", n_samples=n_samples, arc_height=0.0
            )
            path = np.ascontiguousarray(path, dtype=np.float64)
            return {
                "start": start,
                "path": path,
                "linear_solve_mode": "straight",
                "min_clearance_m": self._min_clearance_along_path(path),
            }
        z_cruise, trap_reason = simplerenv_l2_solve_trapezoid(
            start_xy,
            landing,
            self._clearance_top_fn(),
            clearance_m=origin_clearance,
            max_start_rise_m=float(cfg["linear_max_start_z_m"]),
            descent_alpha=float(cfg["linear_descent_alpha"]),
            samples=int(cfg["clearance_samples"]),
        )
        if z_cruise is None:
            return {"infeasible_reason": str(trap_reason)}
        start = np.array([float(start_xy[0]), float(start_xy[1]), float(p0[2])], dtype=np.float64)
        path = simplerenv_l2_build_path(
            start,
            landing,
            "trapezoid",
            n_samples=n_samples,
            arc_height=0.0,
            cruise_z=float(z_cruise),
            descent_alpha=float(cfg["linear_descent_alpha"]),
        )
        path = np.ascontiguousarray(path, dtype=np.float64)
        return {
            "start": start,
            "path": path,
            "linear_solve_mode": "trapezoid",
            "min_clearance_m": self._min_clearance_along_path(path),
        }

    def _plan_diverse_start_and_trajectory(self) -> None:
        cfg = self._spawn_diversity
        if cfg is None or self._access is None:
            raise RuntimeError("spawn_diversity missing while diversity planning")
        p0 = np.asarray(self._p0, dtype=np.float64)
        bounds = get_tabletop_xy_bounds(self._access)
        mins = None if bounds is None else bounds[0]
        maxs = None if bounds is None else bounds[1]
        n_hat, degenerate = simplerenv_l2_lateral_axis(mins, maxs, self.final_pos[:2], p0[:2])
        forbidden = tuple(cfg["forbidden_slots"])
        requested = simplerenv_l2_select_mode(
            self._mode_seed, self._task_id, self._episode_index, forbidden_slots=forbidden
        )

        def make_candidate(mode: str):
            return self._make_diversity_candidate(mode, p0=p0, n_hat=n_hat)

        def validate(start, path) -> bool:
            return bool(self._validate_spawn_path(start, path))

        result = simplerenv_l2_accept_spawn(
            requested,
            forbidden_slots=forbidden,
            make_candidate=make_candidate,
            validate=validate,
        )
        path = result.get("path")
        if path is None:
            start, path = self._legacy_jitter_arc()
            result = {
                **result,
                "start": start,
                "path": path,
                "linear_solve_mode": "na",
                "min_clearance_m": None,
                "accepted_mode": LEGACY_MODE,
                "fallback_reason": result.get("fallback_reason") or "all_candidates_rejected",
            }
        path = np.ascontiguousarray(path, dtype=np.float64)
        start = np.array(path[0], dtype=np.float64, copy=True)
        accepted = str(result.get("accepted_mode") or LEGACY_MODE)
        if accepted == LEGACY_MODE:
            slot = None
            path_style: str | None = "arc"
            jitter_suppressed = False
        else:
            slot, _, path_style = accepted.partition("_")
            if result.get("linear_solve_mode") == "trapezoid":
                path_style = "trapezoid"
            jitter_suppressed = True
        self.start_pos = start
        self._path = path
        push_dir = self.final_pos - self.start_pos
        push_dir[2] = 0.0
        norm = float(np.linalg.norm(push_dir))
        if norm > 1e-8:
            push_dir /= norm
        self._yaw = math.atan2(push_dir[1], push_dir[0])
        self._arm_quat = compute_obstacle_quat(self._yaw)
        min_c = result.get("min_clearance_m")
        self.spawn_plan = {
            "backend": "simplerenv",
            "enabled": True,
            "requested_mode": result.get("requested_mode"),
            "accepted_mode": accepted,
            "slot": slot,
            "path_style": path_style,
            "fallback_reason": result.get("fallback_reason"),
            "fallback_chain": list(result.get("fallback_chain") or ()),
            "start": [float(v) for v in start],
            "lateral_axis": [float(v) for v in n_hat],
            "lateral_axis_degenerate": bool(degenerate),
            "min_clearance_m": None if min_c is None else float(min_c),
            "linear_solve_mode": result.get("linear_solve_mode") or "na",
            "validated_path_sha1": simplerenv_l2_path_sha1(path),
            "rollback_steps": 0,
            "jitter_suppressed": bool(jitter_suppressed),
            "body_to_bottom_offset_m": float(self._body_to_bottom_offset),
        }

    def update_before_step(self, task_step: int) -> None:
        if self._released:
            return

        if task_step < self.move_start_step:
            set_actor_pose(self.actor, self.start_pos, self._arm_quat)
        elif task_step < self.move_end_step:
            idx = min(task_step - self.move_start_step, len(self._path) - 1)
            set_actor_pose(self.actor, self._path[idx], self._arm_quat)
        elif task_step < self._settle_end_step:
            set_actor_pose(self.actor, self.final_pos, self._arm_quat)
        else:
            if self._release_after_settle:
                self._released = True
            else:
                set_actor_pose(self.actor, self.final_pos, self._arm_quat)

    def check_robot_collision(
        self,
        scene: Any,
        robot_link_names: set[str],
        robot_link_positions: np.ndarray,
    ) -> bool:
        del robot_link_positions
        # Scheme B: kinematic settle may report ~0 impulse; geometric contact
        # still counts so we release to physics and record CR.
        hit = actor_has_geometric_contact(
            scene, actor_name=self.actor.name, robot_link_names=robot_link_names
        ) or actor_has_impulse_contact(
            scene, actor_name=self.actor.name, robot_link_names=robot_link_names
        )
        if hit:
            self.blocked = True
            # Contact always hands the obstacle to physics; release_after_settle
            # only controls clean settle without contact.
            if not self._released:
                self._released = True
        return hit
