"""RoboTwin disturbance manager (migrated from envs/dynisafety/manager.py).

SAPIEN / RoboTwin actor helpers are imported lazily so importing this module
does not require a live sim env. ``config`` must expose all required fields
(no ``RoboTwinDyniSafetyConfig.from_mapping`` dead defaults).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from dynisafety.backends._engine.sapien_contact import (
    read_pair_body_entity,
    scan_actor_contacts,
    scene_timestep,
)
from dynisafety.backends.robotwin import task_api
from dynisafety.backends.robotwin.avoid import (
    nudge_xy_aabb,
    nudge_xy_disk,
    push_xy_away,
    world_xy_aabb_from_obb,
)
from dynisafety.backends.robotwin.contact import (
    SAPIEN_CONTACT_IMPULSE_EPS,
    classify_topple,
    contact_point_impulse_scalar,
    filter_collision_names,
    force_equiv_timestep,
    is_ghost_overlap,
    is_toppled,
    measure_tilt_deg,
    official_l12_contact,
    rising_edge_contact,
    should_skip_dynamic_release,
    xyz_list,
)
from dynisafety.backends.robotwin.disturbance._base import ControllerPose, HIDDEN_POS
from dynisafety.backends.robotwin.disturbance.l1 import (
    SuddenDropController,
    nudge_xy_away,
    object_is_grasped,
)
from dynisafety.backends.robotwin.disturbance.l2 import SlidingObstacleController
from dynisafety.backends.robotwin.disturbance.l2_spawn import (
    LEGACY_MODE as L2_LEGACY_MODE,
    robotwin_l2_accept_spawn,
    robotwin_l2_build_path,
    robotwin_l2_candidate_start_xy,
    robotwin_l2_lateral_axis,
    robotwin_l2_origin_is_valid,
    robotwin_l2_path_num_samples,
    robotwin_l2_path_sha1,
    robotwin_l2_select_mode,
    robotwin_l2_solve_linear_start_z,
    robotwin_l2_solve_trapezoid,
)
from dynisafety.backends.robotwin.disturbance.l3 import SharedReachingController
from dynisafety.backends.robotwin.disturbance.l3_reach import (
    LEGACY_MODE as L3_LEGACY_MODE,
    robotwin_l3_accept_reach,
    robotwin_l3_build_path,
    robotwin_l3_execution_anchors,
    robotwin_l3_lateral_axis,
    robotwin_l3_origin_is_valid,
    robotwin_l3_path_intercept_target,
    robotwin_l3_path_num_samples,
    robotwin_l3_path_sha1,
    robotwin_l3_resolve_depth,
    robotwin_l3_select_mode,
    robotwin_l3_slot_xy,
)
from dynisafety.backends.robotwin.obstacles import get_semantic_family
from dynisafety.core.disturbance import interpolate_toward, perpendicular_2d, quat_mul

# Repo root: .../src/dynisafety/backends/robotwin/disturbance/manager.py → parents[5]
_REPO_ROOT = Path(__file__).resolve().parents[5]


def require_hand_mesh_path(raw: str, *, repo_root: Path | None = None) -> Path:
    """Resolve a required L3 hand mesh. Empty/missing paths fail (no box fallback)."""
    text = str(raw or "").strip()
    if not text:
        raise FileNotFoundError("hand.mesh_path is empty; cannot load mesh/mano_mesh geometry")
    path = Path(text)
    if not path.is_absolute():
        root = _REPO_ROOT if repo_root is None else Path(repo_root)
        path = (root / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"hand mesh not found: {path}")
    return path


def _sapien():
    import sapien.core as sapien

    return sapien


def _create_actor_api():
    from envs.utils.create_actor import create_actor, create_box

    return create_actor, create_box


def _actor_cls():
    from envs.utils.actor_utils import Actor

    return Actor


def _require_attr(config: Any, name: str) -> Any:
    if not hasattr(config, name):
        raise ValueError(
            f"DisturbanceManager config missing required attribute {name!r} "
            "(set via YAML / RunConfig; no Python defaults)"
        )
    return getattr(config, name)


def _matrix_to_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a SAPIEN-style (w, x, y, z) quaternion."""
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    n = float(np.linalg.norm(q))
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _hand_quat_from_motion(
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    base_quat: np.ndarray | None = None,
) -> np.ndarray:
    """Build a (w, x, y, z) quaternion that orients the Level 3 hand mesh so:

      * fingertips (local +x) point along the horizontal start -> end direction
        (i.e. the forearm trails behind the motion);
      * the palm-normal axis (local +y for ``assets/dynisafety/hand_with_arm.stl``)
        points to world +z, i.e. the palm faces the sky and the palm plane is
        parallel to the table top;
      * no extra roll about the motion axis (the spread axis local +z lies
        horizontally, perpendicular to the motion).

    The optional ``base_quat`` is applied as a *local* pre-rotation, so the
    auto-orient frame above can be flipped or further rolled without rewriting
    this function. Common values:

      * ``(1, 0, 0, 0)``   - identity, default palm-up behavior;
      * ``(0, 1, 0, 0)``   - 180 deg about local +x, flips palm-up to palm-down
        (back of the hand faces the sky);
      * ``(c, 0, 0, s)``   - yaw about local +z (s = sin(theta/2),
        c = cos(theta/2)), useful if a custom mesh has fingertips along a
        different axis.

    If start/end are essentially co-located, fall back to ``base_quat``
    (or identity) so the controller still has a valid orientation.
    """
    delta = np.asarray(end_pos, dtype=np.float64) - np.asarray(start_pos, dtype=np.float64)
    horiz = delta.copy()
    horiz[2] = 0.0
    norm = float(np.linalg.norm(horiz))
    if norm < 1e-6:
        if base_quat is not None:
            return np.asarray(base_quat, dtype=np.float64)
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    fwd = horiz / norm  # local +x in world
    palm_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # local +y in world (palm faces up)
    spread = np.cross(fwd, palm_normal)  # local +z in world (right-handed)
    spread_norm = float(np.linalg.norm(spread))
    if spread_norm < 1e-6:
        return (
            base_quat if base_quat is not None else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        )
    spread /= spread_norm
    R_auto = np.column_stack([fwd, palm_normal, spread])
    q_auto = _matrix_to_wxyz(R_auto)

    if base_quat is None:
        return q_auto
    return quat_mul(q_auto, np.asarray(base_quat, dtype=np.float64))


class DisturbanceManager:
    def __init__(self, task: Any, config: Any) -> None:
        if config is None:
            raise ValueError(
                "DisturbanceManager requires an explicit config object "
                "(YAML-resolved fields; no RoboTwinDyniSafetyConfig dead defaults)"
            )
        self.task = task
        self.config = config

        self.actor = None
        self.controller = None
        self.primary_arm_tag = None
        self._selected_obstacle_spec = None
        self._actor_is_kinematic = False
        self._active_obstacle_name = "box_fallback"
        self._obstacle_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._obstacle_support_offset = (
            float(self.config.obstacle_half_size[2]) * self.obstacle_scale
        )
        self._obstacle_radius_xy = (
            float(max(self.config.obstacle_half_size[0], self.config.obstacle_half_size[1]))
            * self.obstacle_scale
        )
        self._obstacle_height = 2.0 * float(self.config.obstacle_half_size[2]) * self.obstacle_scale
        self._robot_link_names = self._get_robot_link_names()
        self._collision_target_names: set[str] = set(
            self._robot_link_names
        )  # robot links + held objects
        self._collision_steps = 0
        self._collision_events = 0
        self._first_collision_step: int | None = None
        self._max_collision_impulse: float = 0.0  # N*s, peak per-step impulse magnitude
        self._max_collision_force: float = 0.0  # N, max_impulse / force_equiv_dt
        # Names of the two entities that produced the peak-force contact
        # (disturbance_actor_name, robot_link_or_held_object_name).
        self._max_collision_pair: tuple[str, str] | None = None
        self._max_collision_step: int | None = None
        # Per-partner step counters: maps "robot_link_or_held_object_name"
        # -> number of sim steps that link was in contact with the
        # disturbance actor (a single step can increment multiple keys).
        self._collision_partner_steps: dict[str, int] = {}
        self._max_tilt_deg: float = 0.0
        self._obstacle_toppled: bool = False
        self._landing_toppled: bool = False
        self._obstacle_half_xy: tuple[float, float] = (0.0, 0.0)
        # Episode cumulative impulse (sum of per-sim-step manifold peaks on contact).
        self._cumulative_impulse_ns: float = 0.0
        # Peaks since last ``reset_action_contact_acc`` (one take_action window).
        self._action_peak_impulse: float = 0.0
        self._action_peak_force: float = 0.0
        self._action_contact_without_force: bool = False
        self._contact_without_force_steps: int = 0
        self._held_candidates: list[Any] = []
        self._l1_nudge_count = 0
        self._prev_official_contact = False
        self._obstacle_catalog_index: int | None = None
        self._obstacle_skipped_conflicts: list[str] = []
        self._disturbance_record: dict[str, Any] | None = None
        self._ghost_contact_sim_steps = 0
        self._rollback_steps = 0
        self._release_reason: str | None = None
        self._last_free_pos: np.ndarray | None = None
        self._last_free_quat: np.ndarray | None = None
        # L3: geometric overlaps after the first official event (not in CR/force).
        self._l3_post_first_overlap_sim_steps = 0
        # L2: official overlap after skipped_penetration freeze (not in CR/force).
        self._l2_post_freeze_overlap_sim_steps = 0
        self._l2_post_freeze_peak_force_n = 0.0
        self._l2_post_freeze_peak_impulse_ns = 0.0
        self._spawn_plan: dict[str, Any] | None = None
        self._reach_plan: dict[str, Any] | None = None

    @property
    def obstacle_scale(self) -> float:
        return float(self.config.obstacle_scale)

    @property
    def dynamic_level(self) -> int:
        return int(self.config.dynamic_level)

    @property
    def collision_metrics(
        self,
    ) -> dict[str, int | float | str | tuple[str, str] | dict[str, int] | None]:
        return {
            "collision_steps": int(self._collision_steps),
            "collision_events": int(self._collision_events),
            "first_collision_step": (
                None if self._first_collision_step is None else int(self._first_collision_step)
            ),
            "max_collision_impulse": float(self._max_collision_impulse),
            "max_collision_force": float(self._max_collision_force),
            "max_collision_pair": (
                None if self._max_collision_pair is None else tuple(self._max_collision_pair)
            ),
            "max_collision_step": (
                None if self._max_collision_step is None else int(self._max_collision_step)
            ),
            "collision_partner_steps": dict(self._collision_partner_steps),
            "max_tilt_deg": float(self._max_tilt_deg),
            "obstacle_toppled": bool(self._obstacle_toppled),
            "landing_toppled": bool(self._landing_toppled),
            "topple_threshold_deg": float(self.config.topple_threshold_deg),
            "collision_force_equiv_dt": str(self.config.collision_force_equiv_dt),
            "collision_impulse_point_reduce": str(self.config.collision_impulse_point_reduce),
            "collision_force_report_window_s": float(self.config.collision_force_report_window_s),
            "collision_force_control_freq_hz": float(self.config.collision_force_control_freq_hz),
            "collision_force_equiv_dt_s": float(self._force_equiv_timestep()),
            "cumulative_impulse_ns": float(self._cumulative_impulse_ns),
            "step_force_n": float(self._action_peak_force),
            "step_impulse_ns": float(self._action_peak_impulse),
            "contact_without_force": bool(self._action_contact_without_force),
            "contact_without_force_steps": int(self._contact_without_force_steps),
        }

    def reset_action_contact_acc(self) -> None:
        """Clear per-take_action peaks after they have been snapshotted into info."""
        self._action_peak_impulse = 0.0
        self._action_peak_force = 0.0
        self._action_contact_without_force = False

    def destroy(self):
        if self.actor is None:
            return
        try:
            self.task.scene.remove_actor(self.actor.actor)
        except Exception:
            pass
        self.actor = None
        self.controller = None
        self._selected_obstacle_spec = None
        self._actor_is_kinematic = False
        self._active_obstacle_name = "box_fallback"
        self._obstacle_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._obstacle_support_offset = (
            float(self.config.obstacle_half_size[2]) * self.obstacle_scale
        )
        self._obstacle_radius_xy = (
            float(max(self.config.obstacle_half_size[0], self.config.obstacle_half_size[1]))
            * self.obstacle_scale
        )
        self._obstacle_height = 2.0 * float(self.config.obstacle_half_size[2]) * self.obstacle_scale
        self._collision_target_names = set(self._robot_link_names)
        self._collision_steps = 0
        self._collision_events = 0
        self._first_collision_step = None
        self._max_collision_impulse = 0.0
        self._max_collision_force = 0.0
        self._max_collision_pair = None
        self._max_collision_step = None
        self._collision_partner_steps = {}
        self._max_tilt_deg = 0.0
        self._obstacle_toppled = False
        self._landing_toppled = False
        self._obstacle_half_xy = (0.0, 0.0)
        self._cumulative_impulse_ns = 0.0
        self.reset_action_contact_acc()
        self._contact_without_force_steps = 0
        self._held_candidates: list[Any] = []
        self._l1_nudge_count = 0
        self._prev_official_contact = False
        self._obstacle_catalog_index = None
        self._obstacle_skipped_conflicts = []
        self._disturbance_record = None
        self._ghost_contact_sim_steps = 0
        self._rollback_steps = 0
        self._release_reason = None
        self._last_free_pos = None
        self._last_free_quat = None
        self._l3_post_first_overlap_sim_steps = 0
        self._l2_post_freeze_overlap_sim_steps = 0
        self._l2_post_freeze_peak_force_n = 0.0
        self._l2_post_freeze_peak_impulse_ns = 0.0
        self._spawn_plan = None
        self._reach_plan = None

    def arm(self):
        if self.dynamic_level <= 0:
            return

        self.primary_arm_tag = self._get_primary_arm_tag()
        target_pos = self._resolve_target_pos()
        eef_pos = self._get_primary_eef_pos()
        self._held_candidates = self._get_held_actors()
        self._l1_nudge_count = 0
        self._collision_target_names = set(self._robot_link_names)
        self._refresh_grasped_collision_targets()

        if self.dynamic_level in (1, 2):
            self._ghost_contact_sim_steps = 0
            self._rollback_steps = 0
            self._release_reason = None
            self._last_free_pos = None
            self._last_free_quat = None
            self._landing_toppled = False
            self._obstacle_toppled = False
            self._l2_post_freeze_overlap_sim_steps = 0
            self._l2_post_freeze_peak_force_n = 0.0
            self._l2_post_freeze_peak_impulse_ns = 0.0
            self.actor = self._create_level12_actor()
        else:
            self._l3_post_first_overlap_sim_steps = 0
            self.actor = self._create_level3_hand_actor()
            try:
                self.actor.set_mass(self.config.hand_mass)
            except Exception:
                pass

        if self.dynamic_level == 1:
            final_pos = self._interpolate_target(eef_pos, target_pos, self.config.l1_alpha)
            final_pos = self._apply_landing_push(final_pos, target_pos)
            final_pos = self._resolve_target_xy_overlap(final_pos)
            final_pos[2] = self._obstacle_support_z()
            self._record_disturbance_spawn(
                eef_pos=eef_pos,
                target_pos=target_pos,
                start_pos=None,
                final_pos=final_pos,
                alpha=float(self.config.l1_alpha),
                start_mode=None,
                travel_dist=None,
            )
            self.controller = SuddenDropController(
                actor=self.actor,
                target_pos=final_pos,
                trigger_dist=self.config.l1_trigger_dist,
                drop_height=self.config.l1_drop_height,
                get_eef_pos_fn=self._get_primary_eef_pos,
                quat=self._obstacle_quat,
                obstacle_height=self._obstacle_height,
                obstacle_radius_xy=self._obstacle_radius_xy,
                obstacle_support_offset=self._obstacle_support_offset,
                get_robot_link_positions_fn=self._get_robot_link_positions,
                link_clearance_xy=self.config.l1_spawn_link_clearance_xy,
                z_clearance=self.config.l1_spawn_z_clearance,
                table_surface_z=self._table_surface_z(),
                eef_clearance_xy=self.config.l1_spawn_eef_clearance_xy,
                eef_work_height=self.config.l1_spawn_eef_work_height,
                eef_radius=self.config.l1_spawn_eef_radius,
                home_approach_eps=self.config.l1_home_approach_eps,
            )
        elif self.dynamic_level == 2:
            final_pos = self._interpolate_target(eef_pos, target_pos, self.config.l2_alpha)
            final_pos = self._apply_landing_push(final_pos, target_pos)
            final_pos = self._resolve_target_xy_overlap(final_pos)
            final_pos[2] = self._obstacle_support_z()
            start_pos = self._compute_approach_start(
                eef_pos,
                final_pos,
                self.config.l2_start_dist_mult,
                self.config.l2_start_offset,
                mode=self.config.l2_start_mode,
                flip_side=self.config.l2_flip_side,
                min_dist=self.config.l2_start_min_distance,
            )
            self._record_disturbance_spawn(
                eef_pos=eef_pos,
                target_pos=target_pos,
                start_pos=start_pos,
                final_pos=final_pos,
                alpha=float(self.config.l2_alpha),
                start_mode=str(self.config.l2_start_mode),
                travel_dist=self._approach_travel_dist(
                    eef_pos,
                    final_pos,
                    self.config.l2_start_dist_mult,
                    self.config.l2_start_min_distance,
                ),
            )
            path_style = "arc"
            cruise_z = None
            descent_alpha = None
            spawn_cfg = getattr(self.config, "spawn_diversity", None)
            if isinstance(spawn_cfg, dict) and spawn_cfg.get("enabled"):
                planned = self._plan_l2_spawn(start_pos, final_pos, eef_pos, spawn_cfg)
                start_pos = planned["start"]
                path_style = str(planned["path_style"] or "arc")
                cruise_z = planned.get("cruise_z")
                descent_alpha = planned.get("descent_alpha")
                self._spawn_plan = planned.get("spawn_plan")
            else:
                self._spawn_plan = None
            self.controller = SlidingObstacleController(
                actor=self.actor,
                start_pos=start_pos,
                final_pos=final_pos,
                move_start_step=self.config.l2_move_start_step,
                move_end_step=self.config.l2_move_end_step,
                hold_steps=self.config.l2_hold_steps,
                arc_height=self.config.l2_arc_height,
                quat=self._obstacle_quat,
                path_style=path_style,
                cruise_z=cruise_z,
                descent_alpha=descent_alpha,
            )
            self.controller.spawn_plan = self._spawn_plan
            self._last_free_pos = np.asarray(start_pos, dtype=np.float64).copy()
            self._last_free_quat = np.asarray(self._obstacle_quat, dtype=np.float64).copy()
        elif self.dynamic_level == 3:
            # Hover/dwell anchor: where the hand actually lingers in the workspace.
            if str(getattr(self.config, "l3_target_mode", "current")) == "path_intercept":
                reach_target = robotwin_l3_path_intercept_target(
                    eef_pos,
                    target_pos,
                    alpha=float(_require_attr(self.config, "l3_intercept_alpha")),
                    min_eef_clearance=float(_require_attr(self.config, "l3_min_eef_clearance")),
                    tip_overshoot=float(_require_attr(self.config, "l3_tip_overshoot")),
                    min_task_clearance=float(_require_attr(self.config, "l3_min_task_clearance")),
                    target_z=None,
                )
            else:
                reach_target = self._interpolate_target(eef_pos, target_pos, self.config.l3_alpha)
            reach_target[2] = max(
                reach_target[2], self._table_surface_z() + self.config.l3_dwell_height
            )

            # Spawn xy mirrors the Level 2 obstacle exactly: same alpha-based
            # placement anchor and same head-on entry formula. This guarantees
            # the hand enters from the same off-table xy as the L2 obstacle.
            spawn_anchor = self._interpolate_target(eef_pos, target_pos, self.config.l2_alpha)
            spawn_anchor = self._apply_landing_push(spawn_anchor, target_pos)
            spawn_anchor = self._resolve_target_xy_overlap(spawn_anchor)
            spawn_anchor[2] = self._obstacle_support_z()
            start_pos = self._compute_approach_start(
                eef_pos,
                spawn_anchor,
                self.config.l2_start_dist_mult,
                self.config.l2_start_offset,
                mode=self.config.l2_start_mode,
                flip_side=self.config.l2_flip_side,
                min_dist=self.config.l2_start_min_distance,
            )
            # Per-axis blend of the spawn xy toward the arms' midpoint, so the
            # entry direction can be made SYMMETRIC across primary-arm choice
            # (typically by snapping x fully and keeping a partial forward
            # push on y). ``l3_spawn_center_source`` selects the reference:
            #   * "arm_base" -> midpoint of the two MOUNTED bases (default,
            #                   strictly symmetric across primary-arm choice);
            #   * "arm_eef"  -> midpoint of the two EEFs at episode reset
            #                   (legacy; can drift when idle arm pose is not a
            #                   mirror of the active arm and cause the hand to
            #                   collide with the idle arm).
            raw_start_xy = start_pos[:2].copy()
            midpoint_xy = self._get_arms_midpoint_xy(
                source=_require_attr(self.config, "l3_spawn_center_source"),
            )
            pull_x = float(_require_attr(self.config, "l3_spawn_pull_x"))
            pull_y = float(_require_attr(self.config, "l3_spawn_pull_y"))
            if (pull_x > 0.0 or pull_y > 0.0) and midpoint_xy is not None:
                start_pos[0] = (1.0 - pull_x) * start_pos[0] + pull_x * float(midpoint_xy[0])
                start_pos[1] = (1.0 - pull_y) * start_pos[1] + pull_y * float(midpoint_xy[1])
            blended_start_xy = start_pos[:2].copy()
            xy_offset = _require_attr(self.config, "l3_spawn_xy_offset")
            start_pos[0] += float(xy_offset[0])
            start_pos[1] += float(xy_offset[1])
            # Hard override: when both components are finite, the user has
            # pinned the spawn xy explicitly in world frame; ignore the
            # head_on push, the per-axis blend and the constant offset above.
            world_xy = _require_attr(self.config, "l3_spawn_world_xy")
            try:
                wx, wy = float(world_xy[0]), float(world_xy[1])
                if wx == wx and wy == wy:  # both finite (NaN != NaN)
                    start_pos[0] = wx
                    start_pos[1] = wy
            except Exception:
                pass
            # Lift the spawn well above the idle arm so the kinematic hand
            # flies OVER it on its way to the dwell pose, instead of slicing
            # through it horizontally at workspace height.
            start_pos[2] += float(_require_attr(self.config, "l3_spawn_extra_z_lift"))

            hover_pos = reach_target.copy()
            hover_pos[2] = reach_target[2] + self.config.l3_enter_height
            dwell_pos = reach_target.copy()
            dwell_pos[2] = reach_target[2] + self.config.l3_dwell_height

            if bool(_require_attr(self.config, "debug_l3_spawn")):
                mid_str = (
                    f"({midpoint_xy[0]:+.3f},{midpoint_xy[1]:+.3f})"
                    if midpoint_xy is not None
                    else "(n/a)"
                )
                print(
                    "[dynisafety/L3] "
                    f"primary={getattr(self.primary_arm_tag, 'arm', self.primary_arm_tag)} | "
                    f"eef.xy=({eef_pos[0]:+.3f},{eef_pos[1]:+.3f}) | "
                    f"target.xy=({target_pos[0]:+.3f},{target_pos[1]:+.3f}) | "
                    f"midpoint.xy={mid_str} (source="
                    f"{_require_attr(self.config, 'l3_spawn_center_source')}) | "
                    f"raw_start.xy=({raw_start_xy[0]:+.3f},{raw_start_xy[1]:+.3f}) | "
                    f"blended.xy=({blended_start_xy[0]:+.3f},{blended_start_xy[1]:+.3f}) | "
                    f"final.xyz=({start_pos[0]:+.3f},{start_pos[1]:+.3f},{start_pos[2]:+.3f}) | "
                    f"dwell.xyz=({dwell_pos[0]:+.3f},{dwell_pos[1]:+.3f},{dwell_pos[2]:+.3f})"
                )

            from dynisafety.backends.robotwin.disturbance._base import IDENTITY_QUAT

            path_style = "straight"
            hover_rise_m = 0.0
            via_descent_alpha = 0.20
            reach_cfg = getattr(self.config, "reach_diversity", None)
            if isinstance(reach_cfg, dict) and reach_cfg.get("enabled"):
                planned = self._plan_l3_reach(
                    start_pos, hover_pos, dwell_pos, eef_pos, target_pos, reach_cfg
                )
                start_pos = planned["start"]
                hover_pos = planned["hover"]
                dwell_pos = planned["dwell"]
                path_style = str(planned.get("path_style") or "straight")
                hover_rise_m = float(planned.get("hover_rise_m") or 0.0)
                via_descent_alpha = float(planned.get("via_descent_alpha") or 0.20)
                pose_start = planned.get("pose_start")
                self._reach_plan = planned.get("reach_plan")
            else:
                self._reach_plan = None
                pose_start = None

            self.controller = SharedReachingController(
                actor=self.actor,
                start_pos=start_pos,
                hover_pos=hover_pos,
                dwell_pos=dwell_pos,
                move_start_step=self.config.l3_move_start_step,
                entry_steps=self.config.l3_entry_steps,
                approach_steps=self.config.l3_approach_steps,
                dwell_steps=self.config.l3_dwell_steps,
                exit_steps=self.config.l3_exit_steps,
                flinch_dist=self.config.l3_flinch_dist,
                flinch_height=self.config.l3_flinch_height,
                get_eef_pos_fn=self._get_primary_eef_pos,
                post_hit_hold_steps=self.config.l3_post_hit_hold_steps,
                quat=IDENTITY_QUAT,
                path_style=path_style,
                hover_rise_m=hover_rise_m,
                via_descent_alpha=via_descent_alpha,
                pose_start=pose_start,
            )
            self.controller.reach_plan = self._reach_plan
            # When using a mesh-based hand the object's local frame
            # (+x = fingertips, -z = palm; for hand_with_arm.stl the elbow
            # extends along -x) needs reorienting to look natural in the
            # scene. SharedReachingController writes pose every step via
            # _set_pose(), which falls back to ``default_quat`` whenever the
            # phase does not specify its own quaternion -- so overriding
            # default_quat once here propagates to every entry/approach/
            # dwell/exit pose.
            if str(_require_attr(self.config, "hand_geometry")).lower() in ("mano_mesh", "mesh"):
                base_quat = np.asarray(self.config.hand_mesh_base_quat, dtype=np.float64)
                if bool(_require_attr(self.config, "hand_mesh_auto_orient")):
                    base_quat = _hand_quat_from_motion(start_pos, dwell_pos, base_quat)
                self.controller.default_quat = base_quat
        else:
            raise ValueError(f"Unsupported dynamic_level={self.dynamic_level}")

    def before_step(self, sim_step: int):
        if self.controller is None:
            return
        self.controller.update_before_step(sim_step)
        if self.dynamic_level == 1 and self._actor_is_kinematic:
            self._maybe_probe_and_release_l1(int(sim_step))
            return
        if (
            self.dynamic_level == 2
            and self._actor_is_kinematic
            and self.controller.needs_dynamic_release()
        ):
            self._maybe_release_level2_actor()
            return
        if self._actor_is_kinematic and self.controller.needs_dynamic_release():
            self._release_level12_actor_to_dynamic()
            self.controller.on_released_to_physics()

    def _update_topple_metrics(self) -> None:
        """Passively record Level-1/2 obstacle tilt without changing simulation."""
        if self.dynamic_level not in (1, 2) or self.actor is None:
            return
        try:
            pose = self.actor.get_pose()
            current_quat = np.asarray(pose.q, dtype=np.float64)
        except Exception:
            return
        tilt_deg = measure_tilt_deg(ref_quat=self._obstacle_quat, current_quat=current_quat)
        if tilt_deg > self._max_tilt_deg:
            self._max_tilt_deg = float(tilt_deg)
        official, landing = classify_topple(
            tilt_exceeds=is_toppled(
                tilt_deg, threshold_deg=float(self.config.topple_threshold_deg)
            ),
            released_to_physics=not bool(self._actor_is_kinematic),
        )
        if official:
            self._obstacle_toppled = True
        if landing:
            self._landing_toppled = True

    def after_step(self, sim_step: int) -> bool:
        if self.controller is None or not self.controller.active:
            return False
        self._refresh_grasped_collision_targets()
        self._update_topple_metrics()
        collided, impulse, peak_partner, step_partners = self._detect_obstacle_collision()
        official = self._official_contact_this_step(collided, impulse)
        ghost = is_ghost_overlap(geometric=collided, official=official)
        is_event, self._prev_official_contact = rising_edge_contact(
            self._prev_official_contact, official
        )
        # L3 metrics are first-official-event only (CR + force/impulse snapshot).
        # Later geometric overlap still drives hold/exit via handle_collision, but
        # must not raise peak_force or contact_steps (EXIT scrape / re-entry).
        if (
            official
            and self.dynamic_level == 3
            and self._first_collision_step is not None
        ):
            self._l3_post_first_overlap_sim_steps += 1
            self.controller.handle_collision(sim_step)
            return False
        # L2 metrics freeze after skipped_penetration (kinematic park).
        # Approach-hit frame still counts (reason is still impulse_hit);
        # later grind against the frozen actor is extra only, not CR/force.
        if (
            official
            and self.dynamic_level == 2
            and self._release_reason == "skipped_penetration"
            and bool(self._actor_is_kinematic)
        ):
            self._l2_post_freeze_overlap_sim_steps += 1
            dt = self._force_equiv_timestep()
            force = float(impulse / dt) if dt > 0 else 0.0
            if impulse > self._l2_post_freeze_peak_impulse_ns:
                self._l2_post_freeze_peak_impulse_ns = float(impulse)
            if force > self._l2_post_freeze_peak_force_n:
                self._l2_post_freeze_peak_force_n = float(force)
            return False
        if official:
            self._collision_steps += 1
            if is_event:
                self._collision_events += 1
            dt = self._force_equiv_timestep()
            force = float(impulse / dt) if dt > 0 else 0.0
            without_force = float(impulse) <= SAPIEN_CONTACT_IMPULSE_EPS
            if without_force:
                self._contact_without_force_steps += 1
                self._action_contact_without_force = True
            else:
                self._cumulative_impulse_ns += float(impulse)
            self._action_peak_impulse = max(self._action_peak_impulse, float(impulse))
            self._action_peak_force = max(self._action_peak_force, float(force))
            if self._first_collision_step is None:
                self._first_collision_step = int(sim_step)
            if impulse > self._max_collision_impulse:
                self._max_collision_impulse = float(impulse)
                if peak_partner is not None:
                    actor_name = self.actor.get_name() if self.actor is not None else ""
                    self._max_collision_pair = (str(actor_name), str(peak_partner))
                    self._max_collision_step = int(sim_step)
            if force > self._max_collision_force:
                self._max_collision_force = force
            for partner in step_partners:
                self._collision_partner_steps[partner] = (
                    self._collision_partner_steps.get(partner, 0) + 1
                )
            self.controller.handle_collision(sim_step)
        elif ghost:
            self._ghost_contact_sim_steps += 1
            if self._should_rollback_ghost():
                if self._restore_last_free_pose():
                    self._rollback_steps += 1
        elif self._should_snapshot_last_free():
            self._snapshot_last_free_pose()
        return official

    def build_info(
        self, step_collided: bool
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            "step_collided": bool(step_collided),
            "collision_steps": int(self._collision_steps),
            "collision_events": int(self._collision_events),
            "first_collision_step": (
                None if self._first_collision_step is None else int(self._first_collision_step)
            ),
            "max_collision_impulse": float(self._max_collision_impulse),
            "max_collision_force": float(self._max_collision_force),
            "max_collision_pair": (
                None if self._max_collision_pair is None else tuple(self._max_collision_pair)
            ),
            "max_collision_step": (
                None if self._max_collision_step is None else int(self._max_collision_step)
            ),
            "collision_partner_steps": dict(self._collision_partner_steps),
            "max_tilt_deg": float(self._max_tilt_deg),
            "obstacle_toppled": bool(self._obstacle_toppled),
            "landing_toppled": bool(self._landing_toppled),
            "topple_threshold_deg": float(self.config.topple_threshold_deg),
            "dynamic_level": int(self.dynamic_level),
            "obstacle_name": self._active_obstacle_name if self.dynamic_level in (1, 2) else None,
            "collision_force_equiv_dt": str(self.config.collision_force_equiv_dt),
            "collision_impulse_point_reduce": str(self.config.collision_impulse_point_reduce),
            "collision_force_report_window_s": float(self.config.collision_force_report_window_s),
            "collision_force_control_freq_hz": float(self.config.collision_force_control_freq_hz),
            "collision_force_equiv_dt_s": float(self._force_equiv_timestep()),
            "cumulative_impulse_ns": float(self._cumulative_impulse_ns),
            # Per take_action window (reset via reset_action_contact_acc).
            "step_force_n": float(self._action_peak_force),
            "step_impulse_ns": float(self._action_peak_impulse),
            "contact_without_force": bool(self._action_contact_without_force),
            "contact_without_force_steps": int(self._contact_without_force_steps),
        }
        if self.dynamic_level == 3:
            info["l3_post_first_overlap_sim_steps"] = int(
                self._l3_post_first_overlap_sim_steps
            )
            info["l3_ignore_collision_link_substrings"] = list(
                self._l3_exclude_substrings()
            )
        if self.dynamic_level in (1, 2) and self._disturbance_record is not None:
            record = dict(self._disturbance_record)
            record["released_to_physics"] = not bool(self._actor_is_kinematic)
            record["ghost_contact_sim_steps"] = int(self._ghost_contact_sim_steps)
            record["rollback_steps"] = int(self._rollback_steps)
            record["release_reason"] = self._release_reason
            record["landing_toppled"] = bool(self._landing_toppled)
            if self.dynamic_level == 2:
                record["l2_post_freeze_overlap_sim_steps"] = int(
                    self._l2_post_freeze_overlap_sim_steps
                )
                record["l2_post_freeze_peak_force_n"] = float(
                    self._l2_post_freeze_peak_force_n
                )
                record["l2_post_freeze_peak_impulse_ns"] = float(
                    self._l2_post_freeze_peak_impulse_ns
                )
            info["robotwin_disturbance"] = record
        ctrl = getattr(self, "controller", None)
        spawn_plan = getattr(ctrl, "spawn_plan", None) if ctrl is not None else None
        if spawn_plan is None:
            spawn_plan = getattr(self, "_spawn_plan", None)
        if isinstance(spawn_plan, dict) and spawn_plan:
            info["spawn_plan"] = dict(spawn_plan)
        reach_plan = getattr(ctrl, "reach_plan", None) if ctrl is not None else None
        if reach_plan is None:
            reach_plan = getattr(self, "_reach_plan", None)
        if isinstance(reach_plan, dict) and reach_plan:
            info["l3_plan"] = dict(reach_plan)
        return info

    def _create_level12_actor(self, kinematic=True):
        self._actor_is_kinematic = bool(kinematic)
        self._obstacle_catalog_index = None
        self._obstacle_skipped_conflicts = []
        self._obstacle_support_offset = (
            float(self.config.obstacle_half_size[2]) * self.obstacle_scale
        )
        self._active_obstacle_name = "box_fallback"
        self._obstacle_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._obstacle_radius_xy = (
            float(max(self.config.obstacle_half_size[0], self.config.obstacle_half_size[1]))
            * self.obstacle_scale
        )
        self._obstacle_half_xy = (
            float(self.config.obstacle_half_size[0]) * self.obstacle_scale,
            float(self.config.obstacle_half_size[1]) * self.obstacle_scale,
        )
        # Total vertical extent of the obstacle (used by the L1 spawn-clearance
        # check). For the box fallback we use 2 * half_size_z; for mesh
        # candidates we overwrite this with the spec ``height`` below.
        self._obstacle_height = 2.0 * float(self.config.obstacle_half_size[2]) * self.obstacle_scale
        self._selected_obstacle_spec = None

        try:
            candidates = list(
                task_api.get_obstacle_candidates(
                    self.task,
                    candidate_modelnames=_require_attr(self.config, "obstacle_pool"),
                    upright_quats=_require_attr(self.config, "obstacle_upright_quats"),
                    model_scales=_require_attr(self.config, "obstacle_rescale"),
                    semantic_families=self._semantic_families(),
                    default_upright_quat=_require_attr(
                        self.config, "obstacle_default_upright_quat"
                    ),
                )
                or []
            )
        except Exception:
            candidates = []

        if candidates:
            selected = self._select_obstacle_candidate(candidates)
            if selected is not None:
                actor_name = (
                    f"dynisafety_obstacle_l{self.dynamic_level}_"
                    f"{selected['modelname']}_{selected['model_id']}"
                )
                extra_scale = float(max(selected.get("extra_scale", 1.0), 0.0))
                effective_scale = self.obstacle_scale * extra_scale
                actor = _create_actor_api()[0](
                    scene=self.task,
                    pose=_sapien().Pose(HIDDEN_POS),
                    modelname=selected["modelname"],
                    model_id=selected["model_id"],
                    scale_multiplier=effective_scale,
                    convex=bool(selected.get("convex", True)),
                    is_kinematic=kinematic,
                )
                if actor is not None:
                    actor.set_name(actor_name)
                    actor.set_mass(self.config.obstacle_mass)
                    self._selected_obstacle_spec = dict(selected)
                    self._active_obstacle_name = (
                        f"{selected['modelname']}/base{selected['model_id']}"
                    )
                    self._obstacle_quat = np.asarray(
                        selected.get("spawn_quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64
                    )
                    # ``selected`` already has its dimensions pre-multiplied by
                    # ``extra_scale`` (see ``build_obstacle_spec``), so here we only
                    # apply the global ``obstacle_scale``.
                    self._obstacle_support_offset = (
                        float(max(selected.get("support_offset", 0.0), 0.0)) * self.obstacle_scale
                    )
                    self._obstacle_radius_xy = (
                        float(max(selected.get("radius_xy", 0.0), 0.0)) * self.obstacle_scale
                    )
                    half = selected.get("half_xy")
                    if isinstance(half, (list, tuple)) and len(half) >= 2:
                        self._obstacle_half_xy = (
                            float(half[0]) * self.obstacle_scale,
                            float(half[1]) * self.obstacle_scale,
                        )
                    else:
                        self._obstacle_half_xy = (
                            self._obstacle_radius_xy,
                            self._obstacle_radius_xy,
                        )
                    self._obstacle_height = (
                        float(max(selected.get("height", 0.0), 0.0)) * self.obstacle_scale
                    )
                    return actor

        actor_name = f"dynisafety_obstacle_l{self.dynamic_level}"
        scaled_half_size = tuple(
            float(v) * self.obstacle_scale for v in self.config.obstacle_half_size
        )
        actor = _create_actor_api()[1](
            scene=self.task,
            pose=_sapien().Pose(HIDDEN_POS),
            half_size=scaled_half_size,
            color=self.config.obstacle_color,
            name=actor_name,
            is_kinematic=kinematic,
        )
        actor.set_mass(self.config.obstacle_mass)
        return actor

    def _get_existing_obstacle_modelnames(self) -> set[str]:
        """Best-effort exact model names already present in the current task scene.

        Exact-name reuse is always treated as a conflict. Family-level fallback
        is layered on top via ``_get_existing_obstacle_semantic_families()`` and
        only activates for model names explicitly mapped in
        ``envs.dynisafety.obstacle_candidates``.
        """
        existing_names: set[str] = set()

        def _add_name(raw_name):
            name = str(raw_name or "").strip()
            if not name or name.startswith("dynisafety_"):
                return
            existing_names.add(name)

        scene = getattr(self.task, "scene", None)
        if scene is not None and hasattr(scene, "get_all_actors"):
            try:
                for actor in scene.get_all_actors():
                    if hasattr(actor, "get_name"):
                        _add_name(actor.get_name())
            except Exception:
                pass

        def _scan(value):
            if isinstance(value, _actor_cls()):
                try:
                    _add_name(value.get_name())
                except Exception:
                    pass
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    _scan(item)
            elif isinstance(value, dict):
                for item in value.values():
                    _scan(item)

        for value in getattr(self.task, "__dict__", {}).values():
            _scan(value)
        return existing_names

    def _get_existing_obstacle_semantic_families(
        self,
        existing_modelnames: set[str] | None = None,
    ) -> set[str]:
        names = (
            existing_modelnames
            if existing_modelnames is not None
            else self._get_existing_obstacle_modelnames()
        )
        families: set[str] = set()
        table = self._semantic_families()
        for name in names:
            family = get_semantic_family(name, families=table)
            if family:
                families.add(family)
        return families

    def _candidate_conflicts_with_scene(
        self,
        candidate: dict,
        existing_modelnames: set[str] | None = None,
        existing_semantic_families: set[str] | None = None,
    ) -> bool:
        modelname = str(candidate.get("modelname", "") or "").strip()
        if not modelname:
            return False
        names = (
            existing_modelnames
            if existing_modelnames is not None
            else self._get_existing_obstacle_modelnames()
        )
        if modelname in names:
            return True

        family = candidate.get("semantic_family")
        if not family:
            family = get_semantic_family(modelname, families=self._semantic_families())
        if not family:
            return False

        families = (
            existing_semantic_families
            if existing_semantic_families is not None
            else self._get_existing_obstacle_semantic_families(names)
        )
        return family in families

    def _select_obstacle_candidate(self, candidates: list[dict]) -> dict | None:
        """Choose one obstacle candidate according to config.

        The nominal pick is made on the FULL candidate pool. If the selected
        obstacle's exact ``modelname`` OR curated ``semantic_family`` already
        exists in the current task scene, we perform a deterministic forward
        scan to the next non-conflicting candidate. This preserves the original
        ``trajectory_order`` schedule as much as possible while still avoiding
        redundant disturbance semantics.
        """
        if not candidates:
            raise ValueError("candidates must be non-empty")

        mode = str(_require_attr(self.config, "obstacle_selection_mode")).lower()
        if mode == "trajectory_order":
            ep_idx = int(getattr(self.task, "ep_num", 0))
            base_idx = ep_idx % len(candidates)
        else:
            base_idx = int(np.random.randint(len(candidates)))

        existing_modelnames = self._get_existing_obstacle_modelnames()
        existing_semantic_families = self._get_existing_obstacle_semantic_families(
            existing_modelnames
        )
        skipped: list[str] = []
        for offset in range(len(candidates)):
            idx = (base_idx + offset) % len(candidates)
            candidate = candidates[idx]
            if not self._candidate_conflicts_with_scene(
                candidate,
                existing_modelnames,
                existing_semantic_families,
            ):
                self._obstacle_catalog_index = int(idx)
                self._obstacle_skipped_conflicts = skipped
                return candidate
            name = str(candidate.get("modelname") or "").strip()
            if name:
                skipped.append(name)
        self._obstacle_catalog_index = None
        self._obstacle_skipped_conflicts = skipped
        return None

    def _create_level3_hand_actor(self):
        """Build the Level 3 'shared workspace hand'.

        ``box`` uses the YAML half-size brick. ``mesh`` / ``mano_mesh`` must
        load the configured file; missing assets fail instead of silently
        changing contact geometry.
        """
        actor_name = "dynisafety_shared_hand"
        geometry = str(_require_attr(self.config, "hand_geometry")).lower()

        if geometry in ("mano_mesh", "mesh"):
            return self._build_hand_mesh_actor(actor_name)
        if geometry != "box":
            raise ValueError(
                f"hand.geometry must be 'box', 'mesh', or 'mano_mesh', got {geometry!r}"
            )

        return _create_actor_api()[1](
            scene=self.task,
            pose=_sapien().Pose(HIDDEN_POS),
            half_size=self.config.hand_half_size,
            color=self.config.hand_color,
            name=actor_name,
        )

    def _build_hand_mesh_actor(self, actor_name: str):
        mesh_path = require_hand_mesh_path(str(_require_attr(self.config, "hand_mesh_path") or ""))

        try:
            scene = self.task.scene
            builder = scene.create_actor_builder()
            scale_val = float(_require_attr(self.config, "hand_mesh_scale"))
            scale = (scale_val, scale_val, scale_val)

            builder.add_multiple_convex_collisions_from_file(
                filename=str(mesh_path),
                scale=scale,
            )
            skin_color = list(self.config.hand_color) + [1.0]
            sapien = _sapien()
            material = sapien.render.RenderMaterial(base_color=skin_color)
            builder.add_visual_from_file(
                filename=str(mesh_path),
                scale=scale,
                material=material,
            )

            if hasattr(builder, "build_kinematic"):
                entity = builder.build_kinematic(name=actor_name)
            else:
                entity = builder.build(name=actor_name)
            entity.set_name(actor_name)
            entity.set_pose(sapien.Pose(HIDDEN_POS))
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise RuntimeError(f"failed to build hand mesh actor from {mesh_path}: {exc}") from exc

        # Minimal model_data so downstream code that probes Actor.config (e.g.
        # the obstacle-overlap helper) does not crash. Only Level 1/2 actually
        # consults this; Level 3 mostly just calls set_pose(). Try to derive
        # extents from the mesh bbox so a long "hand+arm" mesh advertises a
        # reasonable footprint instead of the legacy MANO hand-only size.
        extents = self._estimate_mesh_extents(mesh_path, scale_val)
        model_data = {
            "extents": list(extents),
            "scale": [1.0, 1.0, 1.0],
            "center": [0.0, 0.0, 0.0],
        }
        return _actor_cls()(entity, model_data)

    @staticmethod
    def _estimate_mesh_extents(mesh_path: Path, scale_val: float) -> tuple[float, float, float]:
        """Best-effort axis-aligned bbox extents of ``mesh_path`` (binary STL or
        ASCII OBJ), multiplied by ``scale_val``. Falls back to the historical
        MANO hand-only size if the file cannot be parsed."""
        fallback = (0.475 * scale_val, 0.100 * scale_val, 0.175 * scale_val)
        try:
            ext = str(mesh_path).lower()
            mins = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
            maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
            if ext.endswith(".stl"):
                import struct

                with open(mesh_path, "rb") as fh:
                    head = fh.read(80)
                    is_ascii = head.lstrip().lower().startswith(b"solid") and b"facet" in fh.read(
                        256
                    )
                if is_ascii:
                    with open(mesh_path, errors="ignore") as fh:
                        for line in fh:
                            parts = line.strip().split()
                            if len(parts) == 4 and parts[0] == "vertex":
                                v = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                                mins = np.minimum(mins, v)
                                maxs = np.maximum(maxs, v)
                else:
                    with open(mesh_path, "rb") as fh:
                        fh.read(80)
                        n = struct.unpack("<I", fh.read(4))[0]
                        for _ in range(n):
                            fh.read(12)  # normal
                            for _ in range(3):
                                v = np.array(struct.unpack("<fff", fh.read(12)), dtype=np.float64)
                                mins = np.minimum(mins, v)
                                maxs = np.maximum(maxs, v)
                            fh.read(2)
            elif ext.endswith(".obj"):
                with open(mesh_path, errors="ignore") as fh:
                    for line in fh:
                        if line.startswith("v "):
                            parts = line.split()
                            if len(parts) >= 4:
                                v = np.array(
                                    [float(parts[1]), float(parts[2]), float(parts[3])],
                                    dtype=np.float64,
                                )
                                mins = np.minimum(mins, v)
                                maxs = np.maximum(maxs, v)
            else:
                return fallback

            if np.all(np.isfinite(mins)) and np.all(np.isfinite(maxs)):
                size = (maxs - mins) * float(scale_val)
                return float(size[0]), float(size[1]), float(size[2])
        except Exception:
            pass
        return fallback

    def _release_level12_actor_to_dynamic(self):
        if self.actor is None or not self._actor_is_kinematic:
            return

        current_pose = self.actor.get_pose()
        try:
            self.task.scene.remove_actor(self.actor.actor)
        except Exception:
            pass

        if self._selected_obstacle_spec is not None:
            selected = self._selected_obstacle_spec
            actor_name = f"dynisafety_obstacle_l{self.dynamic_level}_{selected['modelname']}_{selected['model_id']}"
            extra_scale = float(max(selected.get("extra_scale", 1.0), 0.0))
            effective_scale = self.obstacle_scale * extra_scale
            actor = _create_actor_api()[0](
                scene=self.task,
                pose=current_pose,
                modelname=selected["modelname"],
                model_id=selected["model_id"],
                scale_multiplier=effective_scale,
                convex=bool(selected.get("convex", True)),
                is_kinematic=False,
            )
            if actor is None:
                scaled_half_size = tuple(
                    float(v) * self.obstacle_scale for v in self.config.obstacle_half_size
                )
                actor = _create_actor_api()[1](
                    scene=self.task,
                    pose=current_pose,
                    half_size=scaled_half_size,
                    color=self.config.obstacle_color,
                    name=f"dynisafety_obstacle_l{self.dynamic_level}",
                    is_kinematic=False,
                )
                self._selected_obstacle_spec = None
                self._active_obstacle_name = "box_fallback"
                self._obstacle_support_offset = (
                    float(self.config.obstacle_half_size[2]) * self.obstacle_scale
                )
                self._obstacle_radius_xy = (
                    float(max(self.config.obstacle_half_size[0], self.config.obstacle_half_size[1]))
                    * self.obstacle_scale
                )
                self._obstacle_height = (
                    2.0 * float(self.config.obstacle_half_size[2]) * self.obstacle_scale
                )
            else:
                actor.set_name(actor_name)
        else:
            scaled_half_size = tuple(
                float(v) * self.obstacle_scale for v in self.config.obstacle_half_size
            )
            actor = _create_actor_api()[1](
                scene=self.task,
                pose=current_pose,
                half_size=scaled_half_size,
                color=self.config.obstacle_color,
                name=f"dynisafety_obstacle_l{self.dynamic_level}",
                is_kinematic=False,
            )

        actor.set_mass(self.config.obstacle_mass)
        try:
            actor.actor.set_velocity(np.zeros(3))
        except Exception:
            pass
        try:
            actor.actor.set_angular_velocity(np.zeros(3))
        except Exception:
            pass

        self.actor = actor
        self._actor_is_kinematic = False
        if self.controller is not None:
            self.controller.actor = actor

    def _get_primary_arm_tag(self):
        from dynisafety.backends.robotwin.primary_arm import resolve_primary_arm_tag

        return resolve_primary_arm_tag(self.task)

    def _resolve_target_pos(self) -> np.ndarray:
        mode = str(_require_attr(self.config, "obstacle_target_mode")).lower()
        return task_api.resolve_target_pose_xyz(self.task, obstacle_target_mode=mode)

    def _get_primary_eef_pos(self) -> np.ndarray:
        arm_pose = np.asarray(self.task.get_arm_pose(self.primary_arm_tag), dtype=np.float64)
        return arm_pose[:3].copy()

    def _get_arms_midpoint_xy(self, source: str = "arm_base") -> np.ndarray | None:
        """Best-effort xy midpoint of the two robot arms.

        ``source`` selects the reference frame:

        * ``"arm_base"`` (default): use the two arms' MOUNTED BASE poses.
          Bases are static and symmetric across primary-arm choice, so the
          midpoint does not drift when the idle arm's home pose is not a
          mirror of the active arm.
        * ``"arm_eef"``: use the two arms' EEFs at episode reset. Legacy
          behaviour that can bias the spawn toward the idle arm.

        Returns None when the task does not expose enough information so
        callers can skip the blend.
        """
        src = str(source).lower()
        if src == "arm_base":
            base = self._get_arms_base_midpoint_xy()
            if base is not None:
                return base
            # Fallback: degrade to EEF midpoint if base lookup failed.
        return self._get_arms_eef_midpoint_xy()

    def _get_arms_eef_midpoint_xy(self) -> np.ndarray | None:
        primary_tag = self.primary_arm_tag
        opposite_tag = getattr(primary_tag, "opposite", None)
        try:
            primary_pose = np.asarray(self.task.get_arm_pose(primary_tag), dtype=np.float64)[:2]
        except Exception:
            return None
        if opposite_tag is None:
            return primary_pose.copy()
        try:
            other_pose = np.asarray(self.task.get_arm_pose(opposite_tag), dtype=np.float64)[:2]
        except Exception:
            return primary_pose.copy()
        return ((primary_pose + other_pose) * 0.5).copy()

    def _get_arms_base_midpoint_xy(self) -> np.ndarray | None:
        """Midpoint xy of the two robot arms' BASE / root link world poses.

        Reads from ``task.robot.left_entity`` / ``right_entity`` (used by
        ``_get_robot_link_names``). Falls back gracefully when either entity
        or its pose is unavailable; callers degrade to the EEF midpoint.
        """
        robot = getattr(self.task, "robot", None)
        if robot is None:
            return None

        def _entity_xy(entity) -> np.ndarray | None:
            if entity is None:
                return None
            for getter in ("get_pose", "get_root_pose"):
                fn = getattr(entity, getter, None)
                if not callable(fn):
                    continue
                try:
                    pose = fn()
                    p = getattr(pose, "p", None)
                    if p is None:
                        continue
                    return np.asarray(p, dtype=np.float64)[:2].copy()
                except Exception:
                    continue
            # Last resort: try root link.
            links = getattr(entity, "get_links", None)
            if callable(links):
                try:
                    link_list = list(links())
                    if link_list:
                        pose = link_list[0].get_pose()
                        return np.asarray(pose.p, dtype=np.float64)[:2].copy()
                except Exception:
                    pass
            return None

        left_xy = _entity_xy(getattr(robot, "left_entity", None))
        right_xy = _entity_xy(getattr(robot, "right_entity", None))
        if left_xy is None and right_xy is None:
            return None
        if left_xy is None:
            return right_xy
        if right_xy is None:
            return left_xy
        return ((left_xy + right_xy) * 0.5).copy()

    def _get_robot_link_names(self) -> set[str]:
        names = set()
        for link in self.task.robot.left_entity.get_links():
            names.add(link.get_name())
        for link in self.task.robot.right_entity.get_links():
            names.add(link.get_name())
        return names

    def _get_robot_link_positions(self) -> np.ndarray:
        """World-frame xyz of every robot link, shape (N, 3).

        Used by the Level-1 spawn-clearance check so that the obstacle is not
        teleported onto a link of the upper arm. Returns an empty (0, 3) array
        if the robot is not yet built or any link query fails, in which case
        the controller skips the clearance check and behaves as before.
        """
        positions: list[list[float]] = []
        robot = getattr(self.task, "robot", None)
        if robot is None:
            return np.zeros((0, 3), dtype=np.float64)
        for entity_attr in ("left_entity", "right_entity"):
            entity = getattr(robot, entity_attr, None)
            if entity is None or not hasattr(entity, "get_links"):
                continue
            try:
                for link in entity.get_links():
                    try:
                        p = link.get_pose().p
                    except Exception:
                        continue
                    positions.append([float(p[0]), float(p[1]), float(p[2])])
            except Exception:
                continue
        if not positions:
            return np.zeros((0, 3), dtype=np.float64)
        return np.asarray(positions, dtype=np.float64)

    def _l12_exclude_substrings(self) -> tuple[str, ...]:
        raw = _require_attr(self.config, "l12_collision_exclude_substrings")
        if isinstance(raw, str):
            raise ValueError("l12_collision_exclude_substrings must be a list of strings")
        return tuple(str(item) for item in raw)

    def _l3_exclude_substrings(self) -> tuple[str, ...]:
        raw = _require_attr(self.config, "l3_ignore_collision_link_substrings")
        if isinstance(raw, str):
            raise ValueError("l3_ignore_collision_link_substrings must be a list of strings")
        return tuple(str(item) for item in raw)

    def _official_collision_target_names(self) -> set[str]:
        names = set(self._collision_target_names)
        if self.dynamic_level in (1, 2):
            return filter_collision_names(names, self._l12_exclude_substrings())
        if self.dynamic_level == 3:
            return filter_collision_names(names, self._l3_exclude_substrings())
        return names

    def _official_contact_this_step(self, geometric: bool, impulse: float) -> bool:
        if self.dynamic_level not in (1, 2):
            return bool(geometric)
        return official_l12_contact(
            geometric=geometric,
            impulse=impulse,
            require_impulse=bool(_require_attr(self.config, "l12_contact_require_impulse")),
            impulse_eps=SAPIEN_CONTACT_IMPULSE_EPS,
        )

    def _should_rollback_ghost(self) -> bool:
        return (
            self.dynamic_level == 2
            and bool(self._actor_is_kinematic)
            and bool(_require_attr(self.config, "l2_keep_last_free_pose"))
        )

    def _should_snapshot_last_free(self) -> bool:
        return self._should_rollback_ghost()

    def _snapshot_last_free_pose(self) -> None:
        if self.actor is None:
            return
        try:
            pose = self.actor.get_pose()
        except Exception:
            return
        self._last_free_pos = np.asarray(pose.p, dtype=np.float64).copy()
        self._last_free_quat = np.asarray(pose.q, dtype=np.float64).copy()

    def _restore_last_free_pose(self) -> bool:
        if self._last_free_pos is None or self.controller is None:
            return False
        quat = (
            self._last_free_quat
            if self._last_free_quat is not None
            else np.asarray(self.controller.default_quat, dtype=np.float64)
        )
        self.controller._set_pose(ControllerPose(self._last_free_pos, quat))
        return True

    def _l12_penetration_blocker_names(self) -> set[str]:
        """Robot (no cameras) + grasped objects + task objects. Table/ground excluded."""
        names = set(self._official_collision_target_names())
        try:
            avoid = list(task_api.get_obstacle_avoid_actors(self.task) or [])
        except Exception:
            avoid = []
        for actor in avoid:
            try:
                name = actor.get_name() if hasattr(actor, "get_name") else None
            except Exception:
                name = None
            if name:
                names.add(str(name))
        return filter_collision_names(names, self._l12_exclude_substrings())

    def _maybe_release_level2_actor(self) -> None:
        if self.controller is None or not self._actor_is_kinematic:
            return
        penetrating = self._obstacle_contacts_names(self._l12_penetration_blocker_names())
        if should_skip_dynamic_release(
            penetrating=penetrating,
            release_on_penetration=bool(_require_attr(self.config, "l2_release_on_penetration")),
        ):
            if bool(_require_attr(self.config, "l2_keep_last_free_pose")):
                if self._restore_last_free_pose():
                    self._rollback_steps += 1
            if hasattr(self.controller, "cancel_pending_release"):
                self.controller.cancel_pending_release()
            self._release_reason = "skipped_penetration"
            return
        reason = getattr(self.controller, "release_reason", None)
        self._release_level12_actor_to_dynamic()
        self.controller.on_released_to_physics()
        self._release_reason = str(reason) if reason else "settle"

    def _approach_travel_dist(
        self,
        eef_pos: np.ndarray,
        target_pos: np.ndarray,
        dist_mult: float,
        min_dist: float,
    ) -> float:
        return max(
            float(np.linalg.norm(target_pos[:2] - eef_pos[:2])) * float(dist_mult),
            float(min_dist),
        )

    def _record_disturbance_spawn(
        self,
        *,
        eef_pos: np.ndarray,
        target_pos: np.ndarray,
        start_pos: np.ndarray | None,
        final_pos: np.ndarray,
        alpha: float,
        start_mode: str | None,
        travel_dist: float | None,
    ) -> None:
        spec = self._selected_obstacle_spec
        box_fallback = spec is None
        modelname = None if spec is None else str(spec.get("modelname") or "") or None
        model_id = None if spec is None else spec.get("model_id")
        if model_id is not None:
            model_id = int(model_id)
        self._disturbance_record = {
            "obstacle_name": str(self._active_obstacle_name),
            "modelname": modelname,
            "model_id": model_id,
            "box_fallback": bool(box_fallback),
            "selection_mode": str(_require_attr(self.config, "obstacle_selection_mode")),
            "catalog_index": self._obstacle_catalog_index,
            "skipped_conflicts": list(self._obstacle_skipped_conflicts),
            "start": None if start_pos is None else xyz_list(start_pos),
            "final": xyz_list(final_pos),
            "source": xyz_list(target_pos),
            "eef": xyz_list(eef_pos),
            "alpha": float(alpha),
            "start_mode": None if start_mode is None else str(start_mode),
            "travel_dist": None if travel_dist is None else float(travel_dist),
            "released_to_physics": not bool(self._actor_is_kinematic),
        }

    def _detect_obstacle_collision(self) -> tuple[bool, float, str | None, set[str]]:
        """Scan all SAPIEN contacts touching the disturbance actor.

        Returns:
            any_hit: whether the actor touched any robot link / held object this step
            max_impulse: peak per-contact manifold impulse magnitude (N*s)
            peak_partner: name of the entity that produced ``max_impulse`` (None if no
                impulse data was available for any matching contact)
            step_partners: set of all partner entity names that produced a contact
                with the disturbance actor this step (used for per-partner step counters)
        """
        actor_name = self.actor.get_name()
        scan = scan_actor_contacts(
            self.task.scene,
            actor_name=actor_name,
            target_names=self._official_collision_target_names(),
            pair_reader=read_pair_body_entity,
            impulse_reduce=str(self.config.collision_impulse_point_reduce),
        )
        return scan.in_contact, scan.max_impulse_ns, scan.peak_partner, set(scan.partners)

    def _get_held_actors(self) -> list[Any]:
        try:
            held = task_api.get_held_objects(self.task) or []
        except Exception:
            return []
        return [item for item in held if item is not None]

    def _get_held_object_names(self) -> set[str]:
        names: set[str] = set()
        for actor in self._held_candidates or self._get_held_actors():
            try:
                name = actor.get_name() if hasattr(actor, "get_name") else None
            except Exception:
                name = None
            if name:
                names.add(str(name))
        return names

    def _arm_eef_and_gripper(self) -> tuple[list[np.ndarray], list[float]]:
        eefs: list[np.ndarray] = []
        grips: list[float] = []
        robot = getattr(self.task, "robot", None)
        for tag, grip_attr in (("left", "get_left_gripper_val"), ("right", "get_right_gripper_val")):
            try:
                from envs.utils.action import ArmTag

                pose = np.asarray(self.task.get_arm_pose(ArmTag(tag)), dtype=np.float64)[:3]
            except Exception:
                continue
            grip = 1.0
            if robot is not None:
                fn = getattr(robot, grip_attr, None)
                if callable(fn):
                    try:
                        grip = float(fn())
                    except Exception:
                        grip = 1.0
            eefs.append(pose)
            grips.append(grip)
        return eefs, grips

    def _actor_is_grasped(self, actor: Any) -> bool:
        try:
            obj_xyz = np.asarray(actor.get_pose().p, dtype=np.float64)[:3]
        except Exception:
            return False
        eefs, grips = self._arm_eef_and_gripper()
        if not eefs:
            return False
        return object_is_grasped(
            object_xyz=obj_xyz,
            eef_xyzs=eefs,
            gripper_vals=grips,
            grasp_dist=float(_require_attr(self.config, "held_grasp_dist")),
            gripper_closed=float(_require_attr(self.config, "held_gripper_closed")),
        )

    def _refresh_grasped_collision_targets(self) -> None:
        names = set(self._robot_link_names)
        for actor in self._held_candidates:
            if not self._actor_is_grasped(actor):
                continue
            try:
                name = actor.get_name() if hasattr(actor, "get_name") else None
            except Exception:
                name = None
            if name:
                names.add(str(name))
        self._collision_target_names = names

    def _spawn_blocker_names(self) -> set[str]:
        names = set(self._robot_link_names)
        try:
            avoid = list(task_api.get_obstacle_avoid_actors(self.task) or [])
        except Exception:
            avoid = []
        for actor in list(avoid) + list(self._held_candidates):
            try:
                name = actor.get_name() if hasattr(actor, "get_name") else None
            except Exception:
                name = None
            if name:
                names.add(str(name))
        return names

    def _obstacle_contacts_names(self, target_names: set[str]) -> bool:
        if self.actor is None or not target_names:
            return False
        try:
            actor_name = self.actor.get_name()
        except Exception:
            return False
        scan = scan_actor_contacts(
            self.task.scene,
            actor_name=str(actor_name),
            target_names=target_names,
            pair_reader=read_pair_body_entity,
            impulse_reduce=str(self.config.collision_impulse_point_reduce),
        )
        return bool(scan.in_contact)

    def _nudge_l1_drop_xy(self) -> bool:
        ctrl = self.controller
        if ctrl is None or not hasattr(ctrl, "drop_pos"):
            return False
        max_tries = int(_require_attr(self.config, "l1_spawn_nudge_max_tries"))
        if self._l1_nudge_count >= max_tries:
            return False
        current = np.asarray(ctrl.drop_pos, dtype=np.float64).copy()
        adjusted = self._resolve_target_xy_overlap(current)
        if float(np.linalg.norm(adjusted[:2] - current[:2])) <= 1e-8:
            eef = self._get_primary_eef_pos()
            step = float(_require_attr(self.config, "l1_spawn_nudge_step"))
            new_xy = nudge_xy_away(current[:2], eef[:2], step)
            adjusted = current.copy()
            adjusted[0] = float(new_xy[0])
            adjusted[1] = float(new_xy[1])
        ctrl.apply_drop_xy(adjusted[:2])
        self._l1_nudge_count += 1
        return True

    def _maybe_probe_and_release_l1(self, sim_step: int) -> None:
        ctrl = self.controller
        if ctrl is None or not ctrl.needs_spawn_probe():
            return
        if ctrl.just_placed(sim_step):
            return
        if self._obstacle_contacts_names(self._spawn_blocker_names()):
            if self._nudge_l1_drop_xy():
                ctrl._set_pose(ctrl.drop_pos)
                ctrl.mark_placed(sim_step)
            return
        self._release_level12_actor_to_dynamic()
        ctrl.on_released_to_physics()

    def _scene_timestep(self) -> float:
        scene = getattr(self.task, "scene", None)
        if scene is None:
            raise ValueError("task.scene is required to read simulation timestep")
        return scene_timestep(scene)

    def _force_equiv_timestep(self) -> float:
        """Denominator for reported force = impulse / dt (see collision_force_equiv_dt)."""
        return force_equiv_timestep(
            self._scene_timestep(),
            force_equiv_dt=str(self.config.collision_force_equiv_dt),
            force_report_window_s=float(self.config.collision_force_report_window_s),
            control_freq_hz=float(self.config.collision_force_control_freq_hz),
        )

    def _contact_point_impulse_scalar(self, points: Sequence[Any]) -> float:
        return contact_point_impulse_scalar(
            points,
            impulse_reduce=str(self.config.collision_impulse_point_reduce),
        )

    def _table_surface_z(self) -> float:
        base = float(self.config.table_surface_z)
        return base + float(getattr(self.task, "table_z_bias", 0.0))

    def _obstacle_support_z(self) -> float:
        return (
            self._table_surface_z()
            + self._obstacle_support_offset
            + float(self.config.support_clearance_z)
        )

    def _semantic_families(self) -> dict[str, str]:
        raw = _require_attr(self.config, "obstacle_semantic_families")
        if not isinstance(raw, dict):
            raise ValueError("obstacle_semantic_families must be a string map")
        return {str(key): str(value) for key, value in raw.items()}

    def _apply_landing_push(self, target_pos: np.ndarray, source_pos: np.ndarray) -> np.ndarray:
        extra = float(_require_attr(self.config, "l2_landing_push_m"))
        out = np.asarray(target_pos, dtype=np.float64).copy()
        out[:2] = push_xy_away(out[:2], np.asarray(source_pos, dtype=np.float64)[:2], extra)
        return out

    def _resolve_target_xy_overlap(self, target_pos: np.ndarray) -> np.ndarray:
        """Nudge landing XY off task objects (disk or world-XY AABB)."""
        adjusted = np.asarray(target_pos, dtype=np.float64).copy()
        try:
            avoid_actors = list(task_api.get_obstacle_avoid_actors(self.task) or [])
        except Exception:
            avoid_actors = []

        own_entity = (
            self.actor.actor
            if (self.actor is not None and hasattr(self.actor, "actor"))
            else self.actor
        )
        clearance_margin = float(_require_attr(self.config, "obstacle_avoid_margin"))
        shape = str(_require_attr(self.config, "obstacle_avoid_shape")).strip().lower()

        for actor in avoid_actors:
            actor_entity = actor.actor if hasattr(actor, "actor") else actor
            if own_entity is not None and actor_entity is own_entity:
                continue
            actor_pose = actor.get_pose() if hasattr(actor, "get_pose") else None
            if actor_pose is None:
                continue
            if shape == "aabb":
                aabb = self._get_actor_world_xy_aabb(actor)
                if aabb is None:
                    continue
                mins, maxs = aabb
                adjusted[:2] = nudge_xy_aabb(
                    adjusted[:2],
                    self._obstacle_half_xy,
                    mins,
                    maxs,
                    clearance_margin,
                )
                continue
            actor_radius = self._get_actor_radius_xy(actor)
            if actor_radius is None:
                continue
            actor_xy = np.asarray(actor_pose.p[:2], dtype=np.float64)
            min_dist = float(actor_radius + self._obstacle_radius_xy + clearance_margin)
            adjusted[:2] = nudge_xy_disk(adjusted[:2], actor_xy, min_dist)

        return adjusted

    def _get_actor_world_xy_aabb(self, actor) -> tuple[np.ndarray, np.ndarray] | None:
        half = self._get_actor_half_xyz(actor)
        if half is None:
            return None
        pose = actor.get_pose() if hasattr(actor, "get_pose") else None
        if pose is None:
            return None
        return world_xy_aabb_from_obb(
            center=np.asarray(pose.p, dtype=np.float64),
            quat_wxyz=np.asarray(pose.q, dtype=np.float64),
            half_xyz=half,
        )

    @staticmethod
    def _get_actor_half_xyz(actor) -> np.ndarray | None:
        config = getattr(actor, "config", None)
        if not isinstance(config, dict):
            return None
        extents = np.asarray(config.get("extents", []), dtype=np.float64).reshape(-1)
        if extents.size < 3:
            return None
        scale = np.asarray(config.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64).reshape(-1)
        if scale.size == 0:
            return None
        if scale.size == 1:
            scale = np.repeat(scale.item(), 3)
        elif scale.size < 3:
            return None
        half = extents[:3] * scale[:3] * 0.5
        if not np.all(np.isfinite(half)) or np.any(half <= 0.0):
            return None
        return half

    @staticmethod
    def _get_actor_radius_xy(actor) -> float | None:
        half = DisturbanceManager._get_actor_half_xyz(actor)
        if half is None:
            return None
        return float(max(half[0], half[2]))

    @staticmethod
    def _interpolate_target(
        eef_pos: np.ndarray, target_pos: np.ndarray, alpha: float
    ) -> np.ndarray:
        return interpolate_toward(eef_pos, target_pos, alpha)

    @staticmethod
    def _safe_xy_direction(vec: np.ndarray) -> np.ndarray:
        direction = np.asarray(vec, dtype=np.float64).copy()
        direction[2] = 0.0
        norm = float(np.linalg.norm(direction[:2]))
        if norm <= 1e-8:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return direction / norm

    def _episode_task_keys(self) -> tuple[int, Any, int]:
        episode_index = int(getattr(self.task, "now_ep_num", 0) or 0)
        task_id = type(self.task).__name__
        mode_seed = int(getattr(self.config, "mode_seed", self.config.seed))
        return episode_index, task_id, mode_seed

    def _support_top_at_xy(self, xy: np.ndarray) -> float:
        best = float(self._table_surface_z())
        try:
            avoid_actors = list(task_api.get_obstacle_avoid_actors(self.task) or [])
        except Exception:
            return best
        for actor in avoid_actors:
            pose = actor.get_pose() if hasattr(actor, "get_pose") else None
            if pose is None:
                continue
            half = self._get_actor_half_xyz(actor)
            if half is None:
                continue
            center = np.asarray(pose.p[:2], dtype=np.float64)
            if float(np.linalg.norm(np.asarray(xy[:2], dtype=np.float64) - center)) > max(
                float(half[0]), float(half[2])
            ):
                continue
            best = max(best, float(pose.p[2]) + float(half[1]))
        return best

    def _xy_overlaps_avoid(self, xy: np.ndarray, *, radius: float) -> bool:
        try:
            avoid_actors = list(task_api.get_obstacle_avoid_actors(self.task) or [])
        except Exception:
            return False
        margin = float(_require_attr(self.config, "obstacle_avoid_margin"))
        shape = str(_require_attr(self.config, "obstacle_avoid_shape")).strip().lower()
        for actor in avoid_actors:
            pose = actor.get_pose() if hasattr(actor, "get_pose") else None
            if pose is None:
                continue
            if shape == "aabb":
                aabb = self._get_actor_world_xy_aabb(actor)
                if aabb is None:
                    continue
                mins, maxs = aabb
                px, py = float(xy[0]), float(xy[1])
                if mins[0] - margin <= px <= maxs[0] + margin and mins[1] - margin <= py <= maxs[1] + margin:
                    return True
                continue
            actor_radius = self._get_actor_radius_xy(actor)
            if actor_radius is None:
                continue
            actor_xy = np.asarray(pose.p[:2], dtype=np.float64)
            if float(np.linalg.norm(np.asarray(xy[:2], dtype=np.float64) - actor_xy)) < float(
                actor_radius + radius + margin
            ):
                return True
        return False

    def _plan_l2_spawn(
        self,
        start_pos: np.ndarray,
        final_pos: np.ndarray,
        eef_pos: np.ndarray,
        cfg: dict[str, Any],
    ) -> dict[str, Any]:
        p0 = np.asarray(start_pos, dtype=np.float64)
        landing = np.asarray(final_pos, dtype=np.float64)
        n_hat, degenerate = robotwin_l2_lateral_axis(eef_pos[:2], landing[:2])
        episode_index, task_id, mode_seed = self._episode_task_keys()
        forbidden = tuple(cfg["forbidden_slots"])
        requested = robotwin_l2_select_mode(
            mode_seed, task_id, episode_index, forbidden_slots=forbidden
        )
        n_samples = robotwin_l2_path_num_samples(int(cfg["path_num_samples"]))
        bottom = float(self._obstacle_support_offset)
        origin_clearance = float(cfg["linear_z_clearance_m"]) + bottom
        radius = float(self._obstacle_radius_xy)
        tail_n = int(cfg["tail_exclude_waypoints"])

        def fixture_top(xy):
            return self._support_top_at_xy(xy)

        def min_clearance(path: np.ndarray) -> float:
            end = max(1, int(path.shape[0]) - tail_n)
            best: float | None = None
            for p in path[:end]:
                c = float(p[2]) - bottom - fixture_top(p[:2])
                best = c if best is None else min(best, c)
            return float("nan") if best is None else float(best)

        def validate(start, path) -> bool:
            return robotwin_l2_origin_is_valid(
                start,
                path,
                origin_blocked=lambda xy: self._xy_overlaps_avoid(xy, radius=radius),
            )

        def make_candidate(mode: str) -> dict[str, Any] | None:
            if mode == L2_LEGACY_MODE:
                path = robotwin_l2_build_path(
                    p0, landing, "arc", n_samples=n_samples, arc_height=float(self.config.l2_arc_height)
                )
                return {
                    "start": p0.copy(),
                    "path": np.ascontiguousarray(path, dtype=np.float64),
                    "linear_solve_mode": "na",
                    "min_clearance_m": None,
                }
            slot, sep, style = str(mode).partition("_")
            if not sep:
                return None
            start_xy = robotwin_l2_candidate_start_xy(
                p0, n_hat, slot, float(cfg["lateral_width_m"])
            )
            if style == "arc":
                start = np.array([float(start_xy[0]), float(start_xy[1]), float(p0[2])], dtype=np.float64)
                path = robotwin_l2_build_path(
                    start, landing, "arc", n_samples=n_samples, arc_height=float(self.config.l2_arc_height)
                )
                path = np.ascontiguousarray(path, dtype=np.float64)
                return {
                    "start": start,
                    "path": path,
                    "linear_solve_mode": "na",
                    "min_clearance_m": min_clearance(path),
                }
            if style != "linear":
                return None
            z_start, _reason = robotwin_l2_solve_linear_start_z(
                start_xy,
                landing,
                fixture_top,
                clearance_m=origin_clearance,
                tail_alpha=float(cfg["linear_tail_alpha"]),
                max_start_rise_m=float(cfg["linear_max_start_z_m"]),
                samples=int(cfg["clearance_samples"]),
                p0_z_hint=float(p0[2]),
            )
            if z_start is not None:
                start = np.array([float(start_xy[0]), float(start_xy[1]), float(z_start)], dtype=np.float64)
                path = robotwin_l2_build_path(start, landing, "linear", n_samples=n_samples, arc_height=0.0)
                path = np.ascontiguousarray(path, dtype=np.float64)
                return {
                    "start": start,
                    "path": path,
                    "linear_solve_mode": "straight",
                    "min_clearance_m": min_clearance(path),
                }
            z_cruise, trap_reason = robotwin_l2_solve_trapezoid(
                start_xy,
                landing,
                fixture_top,
                clearance_m=origin_clearance,
                max_start_rise_m=float(cfg["linear_max_start_z_m"]),
                descent_alpha=float(cfg["linear_descent_alpha"]),
                samples=int(cfg["clearance_samples"]),
            )
            if z_cruise is None:
                return {"infeasible_reason": str(trap_reason)}
            start = np.array([float(start_xy[0]), float(start_xy[1]), float(p0[2])], dtype=np.float64)
            path = robotwin_l2_build_path(
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
                "min_clearance_m": min_clearance(path),
            }

        result = robotwin_l2_accept_spawn(
            requested, forbidden_slots=forbidden, make_candidate=make_candidate, validate=validate
        )
        path = result.get("path")
        if path is None:
            start = p0.copy()
            path = robotwin_l2_build_path(
                start, landing, "arc", n_samples=n_samples, arc_height=float(self.config.l2_arc_height)
            )
            accepted = L2_LEGACY_MODE
            path_style: str | None = "arc"
            slot = None
            cruise_z = None
            descent_alpha = None
        else:
            start = np.array(path[0], dtype=np.float64, copy=True)
            accepted = str(result.get("accepted_mode") or L2_LEGACY_MODE)
            if accepted == L2_LEGACY_MODE:
                slot = None
                path_style = "arc"
                cruise_z = None
                descent_alpha = None
            else:
                slot, _, path_style = accepted.partition("_")
                if result.get("linear_solve_mode") == "trapezoid":
                    path_style = "trapezoid"
                    cruise_z = float(np.max(np.asarray(path)[:, 2]))
                    descent_alpha = float(cfg["linear_descent_alpha"])
                else:
                    cruise_z = None
                    descent_alpha = None
        min_c = result.get("min_clearance_m")
        spawn_plan = {
            "backend": "robotwin",
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
            "validated_path_sha1": robotwin_l2_path_sha1(np.ascontiguousarray(path, dtype=np.float64)),
            "rollback_steps": 0,
        }
        return {
            "start": start,
            "path_style": path_style or "arc",
            "cruise_z": cruise_z,
            "descent_alpha": descent_alpha,
            "spawn_plan": spawn_plan,
        }

    def _plan_l3_reach(
        self,
        start_pos: np.ndarray,
        hover_pos: np.ndarray,
        dwell_pos: np.ndarray,
        eef_pos: np.ndarray,
        task_pos: np.ndarray,
        cfg: dict[str, Any],
    ) -> dict[str, Any]:
        opposite = np.asarray(start_pos, dtype=np.float64)
        n_hat, degenerate = robotwin_l3_lateral_axis(eef_pos[:2], opposite[:2])
        episode_index, task_id, mode_seed = self._episode_task_keys()
        forbidden_slots = tuple(cfg["forbidden_slots"])
        forbidden_depths = tuple(cfg["forbidden_depths"])
        requested = robotwin_l3_select_mode(
            mode_seed,
            task_id,
            episode_index,
            forbidden_slots=forbidden_slots,
            forbidden_depths=forbidden_depths,
        )
        n_samples = robotwin_l3_path_num_samples(int(cfg["path_num_samples"]))
        radius = float(max(self.config.hand_half_size[0], self.config.hand_half_size[1]))

        def validate(start, path) -> bool:
            return robotwin_l3_origin_is_valid(
                start,
                path,
                origin_blocked=lambda xy: self._xy_overlaps_avoid(xy, radius=radius),
            )

        def make_candidate(mode: str, path_style: str) -> dict[str, Any] | None:
            if mode == L3_LEGACY_MODE:
                path = robotwin_l3_build_path(opposite, dwell_pos, "straight", n_samples=n_samples)
                return {
                    "start": opposite.copy(),
                    "dwell": np.asarray(dwell_pos, dtype=np.float64).copy(),
                    "path": np.ascontiguousarray(path, dtype=np.float64),
                    "path_style": "straight",
                    "min_clearance_m": None,
                    "depth_collapsed": False,
                    "slot_collapsed": False,
                }
            slot, _, depth = str(mode).partition("_")
            if slot not in ("left", "opposite", "right") or depth not in ("near", "mid"):
                return None
            start_xy, slot_collapsed = robotwin_l3_slot_xy(
                slot,
                opposite_xy=opposite[:2],
                n_hat=n_hat,
                lateral_width_m=float(cfg["lateral_width_m"]),
            )
            start = np.array([float(start_xy[0]), float(start_xy[1]), float(opposite[2])], dtype=np.float64)
            alpha, depth_collapsed = robotwin_l3_resolve_depth(
                depth,
                eef_xy=eef_pos[:2],
                task_xy=np.asarray(task_pos, dtype=np.float64)[:2],
                near_alpha=float(cfg["near_alpha"]),
                mid_alpha=float(cfg["mid_alpha"]),
                min_eef_clearance=float(_require_attr(self.config, "l3_min_eef_clearance")),
                min_task_clearance=float(_require_attr(self.config, "l3_min_task_clearance")),
                collapse_m=float(cfg["depth_collapse_m"]),
            )
            dwell = robotwin_l3_path_intercept_target(
                eef_pos,
                task_pos,
                alpha=alpha,
                min_eef_clearance=float(_require_attr(self.config, "l3_min_eef_clearance")),
                tip_overshoot=float(_require_attr(self.config, "l3_tip_overshoot")),
                min_task_clearance=float(_require_attr(self.config, "l3_min_task_clearance")),
                target_z=float(dwell_pos[2]),
            )
            path = robotwin_l3_build_path(
                start,
                dwell,
                path_style,
                n_samples=n_samples,
                hover_rise_m=float(cfg["hover_rise_m"]) if path_style in ("raised", "via") else 0.0,
                via_descent_alpha=float(cfg["via_descent_alpha"]),
            )
            path = np.ascontiguousarray(path, dtype=np.float64)
            tops = [self._support_top_at_xy(p[:2]) for p in path]
            min_c = min(float(p[2]) - t for p, t in zip(path, tops))
            if min_c < float(cfg["min_clearance_m"]):
                return {"infeasible_reason": "clearance_infeasible"}
            return {
                "start": np.array(path[0], dtype=np.float64, copy=True),
                "dwell": np.array(dwell, dtype=np.float64, copy=True),
                "path": path,
                "path_style": path_style,
                "min_clearance_m": float(min_c),
                "depth_collapsed": bool(depth_collapsed),
                "slot_collapsed": bool(slot_collapsed),
            }

        result = robotwin_l3_accept_reach(
            requested,
            forbidden_slots=forbidden_slots,
            forbidden_depths=forbidden_depths,
            make_candidate=make_candidate,
            validate=validate,
        )
        accepted = str(result.get("accepted_mode") or L3_LEGACY_MODE)
        if result.get("path") is None or accepted == L3_LEGACY_MODE:
            start = opposite.copy()
            dwell = np.asarray(dwell_pos, dtype=np.float64).copy()
            hover = np.asarray(hover_pos, dtype=np.float64).copy()
            pose_start = start.copy()
            style = "straight"
            slot = None
            depth = None
        else:
            start = np.array(result["start"], dtype=np.float64).reshape(3)
            dwell = np.array(result["dwell"], dtype=np.float64).reshape(3)
            style = str(result.get("path_style") or "straight")
            hover, pose_start = robotwin_l3_execution_anchors(
                start,
                dwell,
                style,
                enter_height=float(self.config.l3_enter_height),
                hover_rise_m=float(cfg["hover_rise_m"]),
            )
            slot, _, depth = accepted.partition("_")
        min_c = result.get("min_clearance_m")
        path = result.get("path")
        reach_plan = {
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
            "validated_path_sha1": robotwin_l3_path_sha1(
                np.ascontiguousarray(
                    path if path is not None else start.reshape(1, 3), dtype=np.float64
                )
            ),
            "depth_collapsed": bool(result.get("depth_collapsed", False)),
            "slot_collapsed": bool(result.get("slot_collapsed", False)),
        }
        return {
            "start": start,
            "hover": hover,
            "dwell": dwell,
            "path_style": style,
            "pose_start": pose_start,
            "hover_rise_m": float(cfg["hover_rise_m"]),
            "via_descent_alpha": float(cfg["via_descent_alpha"]),
            "reach_plan": reach_plan,
        }

    def _compute_approach_start(
        self,
        eef_pos: np.ndarray,
        target_pos: np.ndarray,
        dist_mult: float,
        offset: tuple[float, float, float],
        mode: str = "perpendicular",
        flip_side: bool = False,
        min_dist: float = 0.18,
    ) -> np.ndarray:
        """
        Pick a starting xy for a sliding obstacle.

        - mode='head_on'      : start beyond target along the EEF->target axis
                                (target + (target-eef).normalize() * travel_dist).
                                The obstacle then slides back toward EEF, visually
                                approaching the arm "from the front".
        - mode='perpendicular': start on the side perpendicular to EEF->target
                                line (legacy behavior, used by Level 3 reaching).

        ``min_dist`` is the floor for the xy travel distance from target -> start.
        Use a larger value (e.g. >= 0.7 m) when you want the start to be off
        the table / outside the camera frame.
        """
        direction = self._safe_xy_direction(target_pos - eef_pos)
        mode_norm = str(mode).lower() if mode is not None else "perpendicular"
        if mode_norm == "head_on":
            shift = direction.copy()
        else:
            shift = perpendicular_2d(direction, flip=flip_side)
        travel_dist = max(
            float(np.linalg.norm(target_pos[:2] - eef_pos[:2])) * float(dist_mult),
            float(min_dist),
        )
        start = target_pos.copy()
        start += shift * travel_dist
        start += np.asarray(offset, dtype=np.float64)
        return start

    # Backwards-compatible alias retained for Level 3 reaching controller.
    def _compute_perpendicular_start(
        self,
        eef_pos: np.ndarray,
        target_pos: np.ndarray,
        dist_mult: float,
        offset: tuple[float, float, float],
        flip_side: bool = False,
    ) -> np.ndarray:
        return self._compute_approach_start(
            eef_pos,
            target_pos,
            dist_mult,
            offset,
            mode="perpendicular",
            flip_side=flip_side,
            min_dist=0.18,
        )


# Backward-compatible name for callers still expecting the old symbol.
RoboTwinDyniSafetyManager = DisturbanceManager
