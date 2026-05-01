from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import sapien.core as sapien

from envs.dynisafety.config import RoboTwinDyniSafetyConfig
from envs.dynisafety.contact_metrics import (
    contact_point_impulse_scalar,
    force_equiv_timestep,
)
from envs.dynisafety.obstacle_candidates import get_semantic_family
from envs.dynisafety.controllers import (
    SharedReachingController,
    SlidingObstacleController,
    SuddenDropController,
)
from envs.utils.actor_utils import Actor
from envs.utils.create_actor import create_actor, create_box

# Repo root used to resolve relative mesh paths (e.g. hand_mesh_path).
# manager.py lives at <repo>/dynamic_RoboTwin/RoboTwin/envs/dynisafety/manager.py
# parents[4] -> <repo> (dynamic_safty_bench/)
_REPO_ROOT = Path(__file__).resolve().parents[4]


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


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two (w, x, y, z) quaternions: ``q1 * q2``.

    With our convention this means "first rotate by ``q2`` in the mesh's local
    frame, then rotate by ``q1`` in the world frame", so passing a non-identity
    ``q2`` lets a caller pre-rotate the local axes (e.g. flip palm-up to
    palm-down with 180 deg about local +x: ``(0, 1, 0, 0)``).
    """
    w1, x1, y1, z1 = (float(v) for v in q1)
    w2, x2, y2, z2 = (float(v) for v in q2)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert a (w, x, y, z) quaternion to a 3x3 rotation matrix."""
    q = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = (float(v) for v in (q / norm))
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quat_tilt_deg(ref_quat: np.ndarray, current_quat: np.ndarray) -> float:
    """Tilt of the reference upright axis, ignoring pure yaw about world z."""
    ref_rot = _quat_to_matrix(ref_quat)
    cur_rot = _quat_to_matrix(current_quat)
    local_up = ref_rot.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    current_up = cur_rot @ local_up
    up_norm = float(np.linalg.norm(current_up))
    if up_norm < 1e-12:
        return 0.0
    cos_angle = float(np.clip(current_up[2] / up_norm, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


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

    fwd = horiz / norm                                    # local +x in world
    palm_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # local +y in world (palm faces up)
    spread = np.cross(fwd, palm_normal)                   # local +z in world (right-handed)
    spread_norm = float(np.linalg.norm(spread))
    if spread_norm < 1e-6:
        return base_quat if base_quat is not None else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    spread /= spread_norm
    R_auto = np.column_stack([fwd, palm_normal, spread])
    q_auto = _matrix_to_wxyz(R_auto)

    if base_quat is None:
        return q_auto
    return _quat_mul(q_auto, np.asarray(base_quat, dtype=np.float64))


class RoboTwinDyniSafetyManager:
    def __init__(self, task, config: RoboTwinDyniSafetyConfig | Mapping[str, Any] | None = None):
        self.task = task
        if isinstance(config, RoboTwinDyniSafetyConfig):
            self.config = config
        else:
            self.config = RoboTwinDyniSafetyConfig.from_mapping(config)

        self.actor = None
        self.controller = None
        self.primary_arm_tag = None
        self._selected_obstacle_spec = None
        self._actor_is_kinematic = False
        self._active_obstacle_name = "box_fallback"
        self._obstacle_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._obstacle_support_offset = float(self.config.obstacle_half_size[2]) * self.obstacle_scale
        self._obstacle_radius_xy = float(max(self.config.obstacle_half_size[0], self.config.obstacle_half_size[1])) * self.obstacle_scale
        self._obstacle_height = 2.0 * float(self.config.obstacle_half_size[2]) * self.obstacle_scale
        self._robot_link_names = self._get_robot_link_names()
        self._collision_target_names: set[str] = set(self._robot_link_names)  # robot links + held objects
        self._collision_steps = 0
        self._collision_events = 0
        self._first_collision_step: int | None = None
        self._max_collision_impulse: float = 0.0  # N*s, peak per-step impulse magnitude
        self._max_collision_force: float = 0.0    # N, max_impulse / force_equiv_dt
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

    @property
    def obstacle_scale(self) -> float:
        return float(getattr(self.config, "obstacle_scale", 1.0))

    @property
    def dynamic_level(self) -> int:
        return int(self.config.dynamic_level)

    @property
    def collision_metrics(self) -> dict[str, int | float | str | tuple[str, str] | dict[str, int] | None]:
        return {
            "collision_steps": int(self._collision_steps),
            "collision_events": int(self._collision_events),
            "first_collision_step": (None if self._first_collision_step is None
                                     else int(self._first_collision_step)),
            "max_collision_impulse": float(self._max_collision_impulse),
            "max_collision_force": float(self._max_collision_force),
            "max_collision_pair": (None if self._max_collision_pair is None
                                   else tuple(self._max_collision_pair)),
            "max_collision_step": (None if self._max_collision_step is None
                                   else int(self._max_collision_step)),
            "collision_partner_steps": dict(self._collision_partner_steps),
            "max_tilt_deg": float(self._max_tilt_deg),
            "obstacle_toppled": bool(self._obstacle_toppled),
            "topple_threshold_deg": float(self.config.topple_threshold_deg),
            "collision_force_equiv_dt": str(self.config.collision_force_equiv_dt),
            "collision_impulse_point_reduce": str(self.config.collision_impulse_point_reduce),
            "collision_force_report_window_s": float(
                getattr(self.config, "collision_force_report_window_s", 0.05) or 0.05
            ),
            "collision_force_control_freq_hz": float(
                getattr(self.config, "collision_force_control_freq_hz", 20.0) or 20.0
            ),
            "collision_force_equiv_dt_s": float(self._force_equiv_timestep()),
        }

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
        self._obstacle_support_offset = float(self.config.obstacle_half_size[2]) * self.obstacle_scale
        self._obstacle_radius_xy = float(max(self.config.obstacle_half_size[0], self.config.obstacle_half_size[1])) * self.obstacle_scale
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

    def arm(self):
        if self.dynamic_level <= 0:
            return

        self.primary_arm_tag = self._get_primary_arm_tag()
        target_pos = self._resolve_target_pos()
        eef_pos = self._get_primary_eef_pos()
        self._collision_target_names = set(self._robot_link_names) | self._get_held_object_names()

        if self.dynamic_level in (1, 2):
            self.actor = self._create_level12_actor()
        else:
            self.actor = self._create_level3_hand_actor()
            try:
                self.actor.set_mass(self.config.hand_mass)
            except Exception:
                pass

        if self.dynamic_level == 1:
            final_pos = self._interpolate_target(eef_pos, target_pos, self.config.l1_alpha)
            final_pos = self._resolve_target_xy_overlap(final_pos)
            final_pos[2] = self._obstacle_support_z()
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
            )
        elif self.dynamic_level == 2:
            final_pos = self._interpolate_target(eef_pos, target_pos, self.config.l2_alpha)
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
            self.controller = SlidingObstacleController(
                actor=self.actor,
                start_pos=start_pos,
                final_pos=final_pos,
                move_start_step=self.config.l2_move_start_step,
                move_end_step=self.config.l2_move_end_step,
                hold_steps=self.config.l2_hold_steps,
                arc_height=self.config.l2_arc_height,
                quat=self._obstacle_quat,
            )
        elif self.dynamic_level == 3:
            # Hover/dwell anchor: where the hand actually lingers in the workspace.
            reach_target = self._interpolate_target(eef_pos, target_pos, self.config.l3_alpha)
            reach_target[2] = max(reach_target[2], self._table_surface_z() + self.config.l3_dwell_height)

            # Spawn xy mirrors the Level 2 obstacle exactly: same alpha-based
            # placement anchor and same head-on entry formula. This guarantees
            # the hand enters from the same off-table xy as the L2 obstacle.
            spawn_anchor = self._interpolate_target(eef_pos, target_pos, self.config.l2_alpha)
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
                source=getattr(self.config, "l3_spawn_center_source", "arm_base"),
            )
            pull_x = float(getattr(self.config, "l3_spawn_pull_x", 0.0))
            pull_y = float(getattr(self.config, "l3_spawn_pull_y", 0.0))
            if (pull_x > 0.0 or pull_y > 0.0) and midpoint_xy is not None:
                start_pos[0] = (1.0 - pull_x) * start_pos[0] + pull_x * float(midpoint_xy[0])
                start_pos[1] = (1.0 - pull_y) * start_pos[1] + pull_y * float(midpoint_xy[1])
            blended_start_xy = start_pos[:2].copy()
            xy_offset = getattr(self.config, "l3_spawn_xy_offset", (0.0, 0.0))
            start_pos[0] += float(xy_offset[0])
            start_pos[1] += float(xy_offset[1])
            # Hard override: when both components are finite, the user has
            # pinned the spawn xy explicitly in world frame; ignore the
            # head_on push, the per-axis blend and the constant offset above.
            world_xy = getattr(
                self.config, "l3_spawn_world_xy", (float("nan"), float("nan"))
            )
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
            start_pos[2] += float(getattr(self.config, "l3_spawn_extra_z_lift", 0.0))

            if bool(getattr(self.config, "debug_l3_spawn", False)):
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
                    f"{getattr(self.config, 'l3_spawn_center_source', 'arm_base')}) | "
                    f"raw_start.xy=({raw_start_xy[0]:+.3f},{raw_start_xy[1]:+.3f}) | "
                    f"blended.xy=({blended_start_xy[0]:+.3f},{blended_start_xy[1]:+.3f}) | "
                    f"final.xyz=({start_pos[0]:+.3f},{start_pos[1]:+.3f},{start_pos[2]:+.3f}) | "
                    f"dwell.xyz=({dwell_pos[0]:+.3f},{dwell_pos[1]:+.3f},{dwell_pos[2]:+.3f})"
                )

            hover_pos = reach_target.copy()
            hover_pos[2] = reach_target[2] + self.config.l3_enter_height
            dwell_pos = reach_target.copy()
            dwell_pos[2] = reach_target[2] + self.config.l3_dwell_height
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
            )
            # When using a mesh-based hand the object's local frame
            # (+x = fingertips, -z = palm; for hand_with_arm.stl the elbow
            # extends along -x) needs reorienting to look natural in the
            # scene. SharedReachingController writes pose every step via
            # _set_pose(), which falls back to ``default_quat`` whenever the
            # phase does not specify its own quaternion -- so overriding
            # default_quat once here propagates to every entry/approach/
            # dwell/exit pose.
            if str(getattr(self.config, "hand_geometry", "box")).lower() == "mano_mesh":
                base_quat = np.asarray(self.config.hand_mesh_base_quat, dtype=np.float64)
                if bool(getattr(self.config, "hand_mesh_auto_orient", True)):
                    base_quat = _hand_quat_from_motion(start_pos, dwell_pos, base_quat)
                self.controller.default_quat = base_quat
        else:
            raise ValueError(f"Unsupported dynamic_level={self.dynamic_level}")

    def before_step(self, sim_step: int):
        if self.controller is None:
            return
        self.controller.update_before_step(sim_step)
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
        tilt_deg = _quat_tilt_deg(self._obstacle_quat, current_quat)
        if tilt_deg > self._max_tilt_deg:
            self._max_tilt_deg = float(tilt_deg)
        if tilt_deg >= float(self.config.topple_threshold_deg):
            self._obstacle_toppled = True

    def after_step(self, sim_step: int) -> bool:
        if self.controller is None or not self.controller.active:
            return False
        self._update_topple_metrics()
        collided, impulse, peak_partner, step_partners = self._detect_obstacle_collision()
        if collided:
            self._collision_steps += 1
            self._collision_events += 1
            dt = self._force_equiv_timestep()
            force = float(impulse / dt) if dt > 0 else 0.0
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
        return collided

    def build_info(
        self, step_collided: bool
    ) -> dict[str, int | float | bool | str | tuple[str, str] | dict[str, int] | None]:
        return {
            "step_collided": bool(step_collided),
            "collision_steps": int(self._collision_steps),
            "collision_events": int(self._collision_events),
            "first_collision_step": (None if self._first_collision_step is None
                                     else int(self._first_collision_step)),
            "max_collision_impulse": float(self._max_collision_impulse),
            "max_collision_force": float(self._max_collision_force),
            "max_collision_pair": (None if self._max_collision_pair is None
                                   else tuple(self._max_collision_pair)),
            "max_collision_step": (None if self._max_collision_step is None
                                   else int(self._max_collision_step)),
            "collision_partner_steps": dict(self._collision_partner_steps),
            "max_tilt_deg": float(self._max_tilt_deg),
            "obstacle_toppled": bool(self._obstacle_toppled),
            "topple_threshold_deg": float(self.config.topple_threshold_deg),
            "dynamic_level": int(self.dynamic_level),
            "obstacle_name": self._active_obstacle_name if self.dynamic_level in (1, 2) else None,
            "collision_force_equiv_dt": str(self.config.collision_force_equiv_dt),
            "collision_impulse_point_reduce": str(self.config.collision_impulse_point_reduce),
            "collision_force_report_window_s": float(
                getattr(self.config, "collision_force_report_window_s", 0.05) or 0.05
            ),
            "collision_force_control_freq_hz": float(
                getattr(self.config, "collision_force_control_freq_hz", 20.0) or 20.0
            ),
            "collision_force_equiv_dt_s": float(self._force_equiv_timestep()),
        }

    def _create_level12_actor(self, kinematic=True):
        self._actor_is_kinematic = bool(kinematic)
        self._obstacle_support_offset = float(self.config.obstacle_half_size[2]) * self.obstacle_scale
        self._active_obstacle_name = "box_fallback"
        self._obstacle_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._obstacle_radius_xy = float(max(self.config.obstacle_half_size[0], self.config.obstacle_half_size[1])) * self.obstacle_scale
        # Total vertical extent of the obstacle (used by the L1 spawn-clearance
        # check). For the box fallback we use 2 * half_size_z; for mesh
        # candidates we overwrite this with the spec ``height`` below.
        self._obstacle_height = 2.0 * float(self.config.obstacle_half_size[2]) * self.obstacle_scale
        self._selected_obstacle_spec = None

        candidates = []
        if hasattr(self.task, "get_dynisafety_obstacle_candidates"):
            try:
                candidates = list(self.task.get_dynisafety_obstacle_candidates() or [])
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
                actor = create_actor(
                    scene=self.task,
                    pose=sapien.Pose([6.0, 6.0, 2.5]),
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
                    self._active_obstacle_name = f"{selected['modelname']}/base{selected['model_id']}"
                    self._obstacle_quat = np.asarray(selected.get("spawn_quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
                    # ``selected`` already has its dimensions pre-multiplied by
                    # ``extra_scale`` (see ``build_obstacle_spec``), so here we only
                    # apply the global ``obstacle_scale``.
                    self._obstacle_support_offset = float(max(selected.get("support_offset", 0.0), 0.0)) * self.obstacle_scale
                    self._obstacle_radius_xy = float(max(selected.get("radius_xy", 0.0), 0.0)) * self.obstacle_scale
                    self._obstacle_height = float(max(selected.get("height", 0.0), 0.0)) * self.obstacle_scale
                    return actor

        actor_name = f"dynisafety_obstacle_l{self.dynamic_level}"
        scaled_half_size = tuple(float(v) * self.obstacle_scale for v in self.config.obstacle_half_size)
        actor = create_box(
            scene=self.task,
            pose=sapien.Pose([6.0, 6.0, 2.5]),
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
            if isinstance(value, Actor):
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
        for name in names:
            family = get_semantic_family(name)
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
            family = get_semantic_family(modelname)
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

        mode = str(getattr(self.config, "obstacle_selection_mode", "random")).lower()
        if mode == "trajectory_order":
            ep_idx = int(getattr(self.task, "ep_num", 0))
            base_idx = ep_idx % len(candidates)
        else:
            base_idx = int(np.random.randint(len(candidates)))

        existing_modelnames = self._get_existing_obstacle_modelnames()
        existing_semantic_families = self._get_existing_obstacle_semantic_families(
            existing_modelnames
        )
        for offset in range(len(candidates)):
            candidate = candidates[(base_idx + offset) % len(candidates)]
            if not self._candidate_conflicts_with_scene(
                candidate,
                existing_modelnames,
                existing_semantic_families,
            ):
                return candidate
        return None

    def _create_level3_hand_actor(self):
        """Build the Level 3 'shared workspace hand'.

        Returns an Actor regardless of geometry choice; falls back to the
        legacy box if mesh loading fails so an episode never crashes due
        to a missing asset.
        """
        actor_name = "dynisafety_shared_hand"
        geometry = str(getattr(self.config, "hand_geometry", "box")).lower()

        if geometry == "mano_mesh":
            mesh_actor = self._build_hand_mesh_actor(actor_name)
            if mesh_actor is not None:
                return mesh_actor
            # Asset missing or load failed: silently fall back to box so the
            # benchmark still runs end-to-end.

        return create_box(
            scene=self.task,
            pose=sapien.Pose([6.0, 6.0, 2.5]),
            half_size=self.config.hand_half_size,
            color=self.config.hand_color,
            name=actor_name,
        )

    def _build_hand_mesh_actor(self, actor_name: str):
        mesh_path_str = str(getattr(self.config, "hand_mesh_path", "") or "")
        if not mesh_path_str:
            return None

        mesh_path = Path(mesh_path_str)
        if not mesh_path.is_absolute():
            mesh_path = (_REPO_ROOT / mesh_path).resolve()
        if not mesh_path.exists():
            print(f"[dynisafety] hand mesh not found: {mesh_path}; falling back to box")
            return None

        try:
            scene = self.task.scene
            builder = scene.create_actor_builder()
            scale_val = float(getattr(self.config, "hand_mesh_scale", 1.0))
            scale = (scale_val, scale_val, scale_val)

            builder.add_multiple_convex_collisions_from_file(
                filename=str(mesh_path), scale=scale,
            )
            skin_color = list(self.config.hand_color) + [1.0]
            material = sapien.render.RenderMaterial(base_color=skin_color)
            builder.add_visual_from_file(
                filename=str(mesh_path), scale=scale, material=material,
            )

            if hasattr(builder, "build_kinematic"):
                entity = builder.build_kinematic(name=actor_name)
            else:
                entity = builder.build(name=actor_name)
            entity.set_name(actor_name)
            entity.set_pose(sapien.Pose([6.0, 6.0, 2.5]))
        except Exception as exc:
            print(f"[dynisafety] failed to build hand mesh actor: {exc}; falling back to box")
            return None

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
        return Actor(entity, model_data)

    @staticmethod
    def _estimate_mesh_extents(
        mesh_path: Path, scale_val: float
    ) -> tuple[float, float, float]:
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
                    is_ascii = head.lstrip().lower().startswith(b"solid") and b"facet" in fh.read(256)
                if is_ascii:
                    with open(mesh_path, "r", errors="ignore") as fh:
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
                with open(mesh_path, "r", errors="ignore") as fh:
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
            actor = create_actor(
                scene=self.task,
                pose=current_pose,
                modelname=selected["modelname"],
                model_id=selected["model_id"],
                scale_multiplier=effective_scale,
                convex=bool(selected.get("convex", True)),
                is_kinematic=False,
            )
            if actor is None:
                scaled_half_size = tuple(float(v) * self.obstacle_scale for v in self.config.obstacle_half_size)
                actor = create_box(
                    scene=self.task,
                    pose=current_pose,
                    half_size=scaled_half_size,
                    color=self.config.obstacle_color,
                    name=f"dynisafety_obstacle_l{self.dynamic_level}",
                    is_kinematic=False,
                )
                self._selected_obstacle_spec = None
                self._active_obstacle_name = "box_fallback"
                self._obstacle_support_offset = float(self.config.obstacle_half_size[2]) * self.obstacle_scale
                self._obstacle_radius_xy = float(max(self.config.obstacle_half_size[0], self.config.obstacle_half_size[1])) * self.obstacle_scale
                self._obstacle_height = 2.0 * float(self.config.obstacle_half_size[2]) * self.obstacle_scale
            else:
                actor.set_name(actor_name)
        else:
            scaled_half_size = tuple(float(v) * self.obstacle_scale for v in self.config.obstacle_half_size)
            actor = create_box(
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
        if not hasattr(self.task, "get_dynisafety_primary_arm_tag"):
            raise NotImplementedError(
                f"{self.task.__class__.__name__} must implement get_dynisafety_primary_arm_tag() "
                "before dynamic disturbance can be enabled."
            )
        return self.task.get_dynisafety_primary_arm_tag()

    def _resolve_target_pos(self) -> np.ndarray:
        mode = str(getattr(self.config, "obstacle_target_mode", "place")).lower()

        if mode == "object":
            getter_name = "get_dynisafety_target_object_pose"
        elif mode == "source_object":
            getter_name = "get_dynisafety_source_object_pose"
        else:
            getter_name = "get_dynisafety_target_pose"

        getter = getattr(self.task, getter_name, None)
        if getter is None:
            raise NotImplementedError(
                f"{self.task.__class__.__name__} must implement {getter_name}() "
                f"before dynamic disturbance with obstacle_target_mode='{mode}' can be enabled."
            )
        target = getter()
        if isinstance(target, sapien.Pose):
            return np.asarray(target.p, dtype=np.float64)
        target_arr = np.asarray(target, dtype=np.float64)
        if target_arr.shape[0] < 3:
            raise ValueError("Dynamic safety target pose must have at least 3 values.")
        return target_arr[:3].copy()

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
            primary_pose = np.asarray(
                self.task.get_arm_pose(primary_tag), dtype=np.float64
            )[:2]
        except Exception:
            return None
        if opposite_tag is None:
            return primary_pose.copy()
        try:
            other_pose = np.asarray(
                self.task.get_arm_pose(opposite_tag), dtype=np.float64
            )[:2]
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
        max_impulse = 0.0
        peak_partner: str | None = None
        step_partners: set[str] = set()
        any_hit = False
        target_names = self._collision_target_names
        for contact in self.task.scene.get_contacts():
            body0 = getattr(contact, "bodies", [None, None])[0]
            body1 = getattr(contact, "bodies", [None, None])[1]
            if body0 is None or body1 is None:
                continue
            entity0 = getattr(body0, "entity", None)
            entity1 = getattr(body1, "entity", None)
            if entity0 is None or entity1 is None:
                continue
            names = (entity0.name, entity1.name)
            if names[0] == actor_name and names[1] in target_names:
                partner = names[1]
            elif names[1] == actor_name and names[0] in target_names:
                partner = names[0]
            else:
                continue
            points = getattr(contact, "points", [])
            if not points:
                any_hit = True
                step_partners.add(partner)
                continue
            scalar = self._contact_point_impulse_scalar(points)
            any_hit = True
            step_partners.add(partner)
            if scalar > max_impulse:
                max_impulse = scalar
                peak_partner = partner
        return any_hit, max_impulse, peak_partner, step_partners

    def _get_held_object_names(self) -> set[str]:
        """Resolve actors that the EEF will be carrying / interacting with at episode start.
        Collisions between obstacle and these are functionally robot collisions."""
        names: set[str] = set()
        if not hasattr(self.task, "get_dynisafety_held_objects"):
            return names
        try:
            held = self.task.get_dynisafety_held_objects() or []
        except Exception:
            return names
        for h in held:
            try:
                name = h.get_name() if hasattr(h, "get_name") else None
            except Exception:
                name = None
            if name:
                names.add(name)
        return names

    def _scene_timestep(self) -> float:
        scene = getattr(self.task, "scene", None)
        if scene is None:
            return 1.0 / 250.0
        ts = None
        for attr in ("get_timestep", "timestep"):
            obj = getattr(scene, attr, None)
            if callable(obj):
                try:
                    ts = float(obj())
                    break
                except Exception:
                    pass
            elif obj is not None:
                try:
                    ts = float(obj)
                    break
                except Exception:
                    pass
        return ts if (ts is not None and ts > 0) else (1.0 / 250.0)

    def _force_equiv_timestep(self) -> float:
        """Denominator for reported force = impulse / dt (see collision_force_equiv_dt)."""
        return force_equiv_timestep(
            self._scene_timestep(),
            collision_force_equiv_dt=str(self.config.collision_force_equiv_dt),
            collision_force_report_window_s=float(
                getattr(self.config, "collision_force_report_window_s", 0.05) or 0.05
            ),
            collision_force_control_freq_hz=float(
                getattr(self.config, "collision_force_control_freq_hz", 20.0) or 20.0
            ),
        )

    def _contact_point_impulse_scalar(self, points: Sequence[Any]) -> float:
        return contact_point_impulse_scalar(
            points,
            str(self.config.collision_impulse_point_reduce),
        )

    def _table_surface_z(self) -> float:
        return 0.74 + float(getattr(self.task, "table_z_bias", 0.0))

    def _obstacle_support_z(self) -> float:
        return self._table_surface_z() + self._obstacle_support_offset + float(self.config.support_clearance_z)

    def _resolve_target_xy_overlap(self, target_pos: np.ndarray) -> np.ndarray:
        """
        Conditional anti-clipping.

        Iterates over the task's known scene objects and only nudges the
        obstacle xy when its xy footprint would overlap an existing actor's
        xy footprint. Otherwise the input position is returned unchanged.

        Overlap is defined as
            ``|center_xy - actor_center_xy| < actor_radius_xy + obstacle_radius_xy + margin``
        with ``margin = config.obstacle_avoid_margin`` (default 0.015 m).
        """
        adjusted = np.asarray(target_pos, dtype=np.float64).copy()
        avoid_actors = []
        if hasattr(self.task, "get_dynisafety_obstacle_avoid_actors"):
            try:
                avoid_actors = list(self.task.get_dynisafety_obstacle_avoid_actors() or [])
            except Exception:
                avoid_actors = []

        own_entity = self.actor.actor if (self.actor is not None and hasattr(self.actor, "actor")) else self.actor
        clearance_margin = float(getattr(self.config, "obstacle_avoid_margin", 0.015))

        for actor in avoid_actors:
            actor_entity = actor.actor if hasattr(actor, "actor") else actor
            if own_entity is not None and actor_entity is own_entity:
                continue

            actor_radius = self._get_actor_radius_xy(actor)
            if actor_radius is None:
                continue

            actor_pose = actor.get_pose() if hasattr(actor, "get_pose") else None
            if actor_pose is None:
                continue
            actor_xy = np.asarray(actor_pose.p[:2], dtype=np.float64)
            obstacle_xy = adjusted[:2]
            diff = obstacle_xy - actor_xy
            dist = float(np.linalg.norm(diff))
            min_dist = float(actor_radius + self._obstacle_radius_xy + clearance_margin)
            if dist >= min_dist:
                continue  # no overlap, leave position untouched

            if dist <= 1e-8:
                diff = np.array([1.0, 0.0], dtype=np.float64)
                dist = 1.0
            diff /= dist
            adjusted[:2] = actor_xy + diff * min_dist

        return adjusted

    @staticmethod
    def _get_actor_radius_xy(actor) -> float | None:
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

        radius_xy = float(max(extents[0] * scale[0], extents[2] * scale[2]) * 0.5)
        if not np.isfinite(radius_xy) or radius_xy <= 0.0:
            return None
        return radius_xy

    @staticmethod
    def _interpolate_target(eef_pos: np.ndarray, target_pos: np.ndarray, alpha: float) -> np.ndarray:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return eef_pos + alpha * (target_pos - eef_pos)

    @staticmethod
    def _safe_xy_direction(vec: np.ndarray) -> np.ndarray:
        direction = np.asarray(vec, dtype=np.float64).copy()
        direction[2] = 0.0
        norm = float(np.linalg.norm(direction[:2]))
        if norm <= 1e-8:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return direction / norm

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
            shift = np.array([-direction[1], direction[0], 0.0], dtype=np.float64)
            if flip_side:
                shift *= -1.0
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
