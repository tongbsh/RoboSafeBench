"""RoboTwin L1: EEF spawn hold, home-inside latch, grasp gate, nudge (no sapien)."""

from __future__ import annotations

import numpy as np

from dynisafety.backends.robotwin.disturbance.l1 import (
    SuddenDropController,
    eef_xy_should_latch,
    nudge_xy_away,
    object_is_grasped,
    point_aabb_distance,
)


class _DummyActor:
    def set_pose(self, pose) -> None:
        self.last_pose = pose


def _ctrl(
    *,
    eef: np.ndarray,
    links: np.ndarray | None = None,
    home_approach_eps: float = 0.02,
) -> SuddenDropController:
    eef = np.asarray(eef, dtype=np.float64)
    state = {"eef": eef.copy()}

    def _eef() -> np.ndarray:
        return state["eef"].copy()

    def _links() -> np.ndarray:
        if links is None:
            return np.zeros((0, 3), dtype=np.float64)
        return np.asarray(links, dtype=np.float64)

    ctrl = SuddenDropController(
        actor=_DummyActor(),
        target_pos=np.array([0.0, 0.0, 0.785], dtype=np.float64),
        trigger_dist=0.23,
        drop_height=0.01,
        get_eef_pos_fn=_eef,
        quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        obstacle_height=0.10,
        obstacle_radius_xy=0.04,
        obstacle_support_offset=0.04,
        get_robot_link_positions_fn=_links,
        link_clearance_xy=0.05,
        z_clearance=0.03,
        table_surface_z=0.74,
        eef_clearance_xy=0.05,
        eef_work_height=0.25,
        eef_radius=0.06,
        home_approach_eps=home_approach_eps,
    )
    ctrl._set_pose = lambda pose: setattr(ctrl, "_last_set", np.asarray(pose, dtype=np.float64))
    ctrl._eef_state = state
    return ctrl


def test_point_aabb_distance_inside_is_zero() -> None:
    mins = np.array([-1.0, -1.0, -1.0])
    maxs = np.array([1.0, 1.0, 1.0])
    assert point_aabb_distance(np.zeros(3), mins, maxs) == 0.0
    assert point_aabb_distance(np.array([2.0, 0.0, 0.0]), mins, maxs) == 1.0


def test_hovering_eef_in_work_cylinder_blocks() -> None:
    ctrl = _ctrl(eef=np.array([0.02, 0.0, 0.90]))
    assert ctrl._eef_in_work_cylinder(ctrl._read_eef_xyz())
    assert ctrl._eef_blocks_spawn()


def test_high_eef_outside_work_height_does_not_use_cylinder() -> None:
    eef = np.array([0.0, 0.0, 1.10])
    ctrl = _ctrl(eef=eef)
    assert not ctrl._eef_in_work_cylinder(eef)
    assert not ctrl._eef_too_close_to_aabb(eef)
    assert not ctrl._eef_blocks_spawn()


def test_side_eef_blocks_via_aabb_gap() -> None:
    eef = np.array([0.12, 0.0, 0.80])
    ctrl = _ctrl(eef=eef)
    assert not ctrl._eef_in_work_cylinder(eef)
    assert ctrl._eef_too_close_to_aabb(eef)
    assert ctrl._eef_blocks_spawn()


def test_far_eef_does_not_block() -> None:
    eef = np.array([0.40, 0.0, 0.90])
    ctrl = _ctrl(eef=eef)
    assert not ctrl._eef_blocks_spawn()


def test_unreadable_eef_fail_closed() -> None:
    ctrl = _ctrl(eef=np.array([0.40, 0.0, 0.90]))
    ctrl.get_eef_pos = lambda: (_ for _ in ()).throw(RuntimeError("no eef"))
    assert ctrl._eef_blocks_spawn() is True


def test_home_inside_does_not_latch_at_step_zero() -> None:
    ctrl = _ctrl(eef=np.array([0.10, 0.0, 1.10]))
    assert ctrl.home_inside is True
    ctrl.update_before_step(0)
    assert ctrl._xy_latched is False
    assert ctrl._dropped is False


def test_home_inside_latches_only_after_closer_than_start() -> None:
    ctrl = _ctrl(eef=np.array([0.10, 0.0, 1.10]))
    ctrl.update_before_step(0)
    assert ctrl._xy_latched is False

    ctrl._eef_state["eef"] = np.array([0.07, 0.0, 1.10], dtype=np.float64)
    ctrl.update_before_step(1)
    assert ctrl._xy_latched is True
    assert ctrl._dropped is True
    assert ctrl.needs_spawn_probe() is True
    assert ctrl.just_placed(1) is True
    assert ctrl.needs_dynamic_release() is False


def test_home_outside_latches_when_entering_circle() -> None:
    ctrl = _ctrl(eef=np.array([0.30, 0.0, 0.90]))
    assert ctrl.home_inside is False
    ctrl.update_before_step(0)
    assert ctrl._xy_latched is False

    ctrl._eef_state["eef"] = np.array([0.20, 0.0, 0.90], dtype=np.float64)
    ctrl.update_before_step(1)
    assert ctrl._xy_latched is True
    assert ctrl._dropped is True
    assert ctrl.needs_spawn_probe() is True


def test_eef_hold_after_latch_then_probe_when_clear() -> None:
    ctrl = _ctrl(eef=np.array([0.30, 0.0, 0.90]))
    ctrl.update_before_step(0)
    ctrl._eef_state["eef"] = np.array([0.02, 0.0, 0.90], dtype=np.float64)
    ctrl.update_before_step(1)
    assert ctrl._xy_latched is True
    assert ctrl._dropped is False

    ctrl._eef_state["eef"] = np.array([0.40, 0.0, 0.90], dtype=np.float64)
    ctrl.update_before_step(2)
    assert ctrl._dropped is True
    assert ctrl.needs_spawn_probe() is True


def test_eef_xy_should_latch_predicates() -> None:
    assert eef_xy_should_latch(
        xy_dist=0.20,
        trigger_dist=0.23,
        init_xy_dist=0.40,
        home_inside=False,
        approach_eps=0.02,
    )
    assert not eef_xy_should_latch(
        xy_dist=0.18,
        trigger_dist=0.23,
        init_xy_dist=0.18,
        home_inside=True,
        approach_eps=0.02,
    )
    assert eef_xy_should_latch(
        xy_dist=0.15,
        trigger_dist=0.23,
        init_xy_dist=0.18,
        home_inside=True,
        approach_eps=0.02,
    )


def test_object_is_grasped_requires_closed_and_near() -> None:
    obj = np.array([0.0, 0.0, 0.8])
    near = np.array([0.02, 0.0, 0.8])
    far = np.array([0.40, 0.0, 0.8])
    assert object_is_grasped(
        object_xyz=obj,
        eef_xyzs=[near],
        gripper_vals=[0.1],
        grasp_dist=0.08,
        gripper_closed=0.2,
    )
    assert not object_is_grasped(
        object_xyz=obj,
        eef_xyzs=[near],
        gripper_vals=[0.9],
        grasp_dist=0.08,
        gripper_closed=0.2,
    )
    assert not object_is_grasped(
        object_xyz=obj,
        eef_xyzs=[far],
        gripper_vals=[0.1],
        grasp_dist=0.08,
        gripper_closed=0.2,
    )


def test_nudge_xy_away_from_eef() -> None:
    out = nudge_xy_away(np.array([0.10, 0.0]), np.array([0.0, 0.0]), 0.02)
    assert abs(float(out[0]) - 0.12) < 1e-9
    coincide = nudge_xy_away(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0.02)
    assert abs(float(coincide[0]) - 0.02) < 1e-9


def test_l1_drop_step_survives_physics_release() -> None:
    ctrl = _ctrl(eef=np.array([0.07, 0.0, 1.10]))
    ctrl._dropped = True
    ctrl.mark_placed(1501)
    assert ctrl.drop_step is None
    ctrl.lock_drop_task_step(42)
    assert ctrl.drop_step == 42
    assert ctrl.disturbance_onset_step == 42
    ctrl.on_released_to_physics()
    assert ctrl._placed_sim_step is None
    assert ctrl.drop_step == 42
    assert ctrl.disturbance_onset_step == 42
    ctrl.lock_drop_task_step(99)
    assert ctrl.drop_step == 42
