"""P2: L1/L2 impulse gate, last-free-pose rollback, L3 force schemes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dynisafety.backends.robotwin.contact import (
    L3_RELEASE_MIN_SPEED_MPS,
    SAPIEN_CONTACT_IMPULSE_EPS,
    is_ghost_overlap,
    l12_episode_extra_from_info,
    l3_closing_speed_mps,
    l3_episode_extra_from_info,
    l3_freeze_after_first_event,
    l3_newton_normal_impulse_ns,
    l3_release_linear_velocity,
    l3_release_momentum_impulse_ns,
    official_l12_contact,
    official_l12_scheme_contact,
    official_l3_contact,
    orient_normal_from_actor,
    should_skip_dynamic_release,
)
from dynisafety.backends.robotwin.disturbance.l2 import SlidingObstacleController
from dynisafety.backends.robotwin.disturbance.l3 import SharedReachingController
from dynisafety.backends.robotwin.disturbance.manager import (
    DisturbanceManager,
    _get_sapien_linear_velocity,
    _set_sapien_linear_velocity,
)
from dynisafety.core.disturbance import Phase


def test_official_l12_requires_impulse_when_gated() -> None:
    assert official_l12_contact(
        geometric=True,
        impulse=0.0,
        require_impulse=True,
        impulse_eps=SAPIEN_CONTACT_IMPULSE_EPS,
    ) is False
    assert official_l12_contact(
        geometric=True,
        impulse=SAPIEN_CONTACT_IMPULSE_EPS,
        require_impulse=True,
        impulse_eps=SAPIEN_CONTACT_IMPULSE_EPS,
    ) is False
    assert official_l12_contact(
        geometric=True,
        impulse=SAPIEN_CONTACT_IMPULSE_EPS + 1e-9,
        require_impulse=True,
        impulse_eps=SAPIEN_CONTACT_IMPULSE_EPS,
    ) is True
    assert official_l12_contact(
        geometric=False,
        impulse=1.0,
        require_impulse=True,
        impulse_eps=SAPIEN_CONTACT_IMPULSE_EPS,
    ) is False


def test_l3_scheme_b_is_geometric_only() -> None:
    """L3 does not apply the L1/L2 impulse gate (require_impulse=False ≡ Scheme B)."""
    assert official_l12_contact(
        geometric=True,
        impulse=0.0,
        require_impulse=False,
        impulse_eps=SAPIEN_CONTACT_IMPULSE_EPS,
    ) is True
    assert official_l12_contact(
        geometric=False,
        impulse=0.0,
        require_impulse=False,
        impulse_eps=SAPIEN_CONTACT_IMPULSE_EPS,
    ) is False


def test_official_l3_schemes_are_mutually_exclusive() -> None:
    assert official_l3_contact(
        geometric=True,
        impulse=0.0,
        force_scheme="kinematic_first_hit",
        released_to_physics=False,
        require_impulse=True,
        impulse_eps=SAPIEN_CONTACT_IMPULSE_EPS,
    ) is True
    assert official_l3_contact(
        geometric=True,
        impulse=0.02,
        force_scheme="dynamic_release",
        released_to_physics=False,
        require_impulse=True,
        impulse_eps=SAPIEN_CONTACT_IMPULSE_EPS,
    ) is False
    assert official_l3_contact(
        geometric=True,
        impulse=0.0,
        force_scheme="dynamic_release",
        released_to_physics=True,
        require_impulse=True,
        impulse_eps=SAPIEN_CONTACT_IMPULSE_EPS,
    ) is False
    assert official_l3_contact(
        geometric=True,
        impulse=0.02,
        force_scheme="dynamic_release",
        released_to_physics=True,
        require_impulse=True,
        impulse_eps=SAPIEN_CONTACT_IMPULSE_EPS,
    ) is True
    assert l3_freeze_after_first_event("kinematic_first_hit") is True
    assert l3_freeze_after_first_event("dynamic_release") is False
    with pytest.raises(ValueError, match="l3_force_scheme"):
        official_l3_contact(
            geometric=True,
            impulse=0.02,
            force_scheme="both",
            released_to_physics=True,
            require_impulse=True,
            impulse_eps=SAPIEN_CONTACT_IMPULSE_EPS,
        )


def test_official_l12_schemes_are_mutually_exclusive() -> None:
    assert official_l12_scheme_contact(
        geometric=True,
        impulse=0.02,
        force_scheme="kinematic_first_hit",
        released_to_physics=False,
        require_impulse=True,
        impulse_eps=SAPIEN_CONTACT_IMPULSE_EPS,
    ) is True
    assert official_l12_scheme_contact(
        geometric=True,
        impulse=0.02,
        force_scheme="newton_relative",
        released_to_physics=False,
        require_impulse=True,
        impulse_eps=SAPIEN_CONTACT_IMPULSE_EPS,
    ) is False
    assert official_l12_scheme_contact(
        geometric=True,
        impulse=0.0,
        force_scheme="newton_relative",
        released_to_physics=True,
        require_impulse=True,
        impulse_eps=SAPIEN_CONTACT_IMPULSE_EPS,
    ) is False
    assert official_l12_scheme_contact(
        geometric=True,
        impulse=0.02,
        force_scheme="newton_relative",
        released_to_physics=True,
        require_impulse=True,
        impulse_eps=SAPIEN_CONTACT_IMPULSE_EPS,
    ) is True
    with pytest.raises(ValueError, match="l12_force_scheme"):
        official_l12_scheme_contact(
            geometric=True,
            impulse=0.02,
            force_scheme="dynamic_release",
            released_to_physics=True,
            require_impulse=True,
            impulse_eps=SAPIEN_CONTACT_IMPULSE_EPS,
        )


def test_l3_release_velocity_continues_inbound_and_floors_speed() -> None:
    last_free = np.array([0.0, 0.0, 1.0])
    overlap = np.array([0.001, 0.0, 1.0])
    vel = l3_release_linear_velocity(
        overlap_pos=overlap,
        last_free_pos=last_free,
        path_velocity=np.array([0.01, 0.0, 0.0]),
        sim_dt=0.004,
        min_speed_mps=L3_RELEASE_MIN_SPEED_MPS,
    )
    # 0.001 m / 0.004 s = 0.25 m/s, already at the floor.
    np.testing.assert_allclose(vel, np.array([0.25, 0.0, 0.0]))
    slow = l3_release_linear_velocity(
        overlap_pos=np.array([1e-4, 0.0, 1.0]),
        last_free_pos=last_free,
        path_velocity=np.zeros(3),
        sim_dt=0.004,
        min_speed_mps=0.25,
    )
    np.testing.assert_allclose(slow, np.array([0.25, 0.0, 0.0]))
    path_only = l3_release_linear_velocity(
        overlap_pos=last_free,
        last_free_pos=last_free,
        path_velocity=np.array([0.0, 0.4, 0.0]),
        sim_dt=0.004,
        min_speed_mps=0.25,
    )
    np.testing.assert_allclose(path_only, np.array([0.0, 0.4, 0.0]))


def test_set_sapien_linear_velocity_writes_physx_component() -> None:
    written: list[np.ndarray] = []

    class _Comp:
        def set_linear_velocity(self, vel):
            written.append(np.asarray(vel, dtype=np.float64))

    entity = SimpleNamespace(get_components=lambda: [_Comp()])
    actor = SimpleNamespace(actor=entity)
    assert _set_sapien_linear_velocity(actor, np.array([0.25, 0.0, 0.0])) is True
    np.testing.assert_allclose(written[0], np.array([0.25, 0.0, 0.0]))
    bare = SimpleNamespace()
    assert _set_sapien_linear_velocity(bare, np.zeros(3)) is False


def test_get_sapien_linear_velocity_reads_physx_component() -> None:
    class _Comp:
        def get_linear_velocity(self):
            return np.array([0.04, -0.08, 0.0])

    entity = SimpleNamespace(get_components=lambda: [_Comp()])
    actor = SimpleNamespace(actor=entity)
    np.testing.assert_allclose(
        _get_sapien_linear_velocity(actor), np.array([0.04, -0.08, 0.0])
    )
    assert _get_sapien_linear_velocity(SimpleNamespace()) is None


def test_ghost_overlap_is_geometric_minus_official() -> None:
    assert is_ghost_overlap(geometric=True, official=False) is True
    assert is_ghost_overlap(geometric=True, official=True) is False
    assert is_ghost_overlap(geometric=False, official=False) is False


def test_skip_dynamic_release_only_when_penetrating_and_disallowed() -> None:
    assert should_skip_dynamic_release(penetrating=True, release_on_penetration=False) is True
    assert should_skip_dynamic_release(penetrating=True, release_on_penetration=True) is False
    assert should_skip_dynamic_release(penetrating=False, release_on_penetration=False) is False


def _sliding_ctrl() -> SlidingObstacleController:
    return SlidingObstacleController(
        actor=object(),
        start_pos=np.array([0.0, 0.0, 1.0]),
        final_pos=np.array([1.0, 0.0, 0.8]),
        move_start_step=1,
        move_end_step=10,
        hold_steps=2,
        arc_height=0.12,
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
    )


def test_l2_settle_requests_release_without_pose_write() -> None:
    ctrl = _sliding_ctrl()
    # hold_end = 12; past hold does not call _set_pose (sapien-free).
    ctrl.update_before_step(13)
    assert ctrl.needs_dynamic_release() is True
    assert ctrl.release_reason == "settle"
    assert ctrl.disturbance_onset_step is None


def test_l2_onset_locks_to_task_clock_on_default_sim_path() -> None:
    poses: list[np.ndarray] = []
    ctrl = _sliding_ctrl()
    ctrl._set_pose = lambda pose: poses.append(  # type: ignore[method-assign]
        np.asarray(getattr(pose, "position", pose), dtype=np.float64).copy()
    )
    ctrl.update_before_step(0, task_step=5)
    assert ctrl.disturbance_onset_step is None
    ctrl.update_before_step(1, task_step=5)
    assert ctrl.disturbance_onset_step == 5
    ctrl.update_before_step(2, task_step=6)
    assert ctrl.disturbance_onset_step == 5


def test_l2_move_start_task_step_holds_then_runs_full_arc() -> None:
    poses: list[np.ndarray] = []
    ctrl = SlidingObstacleController(
        actor=object(),
        start_pos=np.array([0.0, 0.0, 1.0]),
        final_pos=np.array([1.0, 0.0, 0.8]),
        move_start_step=1,
        move_end_step=10,
        hold_steps=2,
        arc_height=0.12,
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        move_start_task_step=20,
    )
    ctrl._set_pose = lambda pose: poses.append(  # type: ignore[method-assign]
        np.asarray(getattr(pose, "position", pose), dtype=np.float64).copy()
    )
    ctrl.update_before_step(100, task_step=19)
    assert ctrl.disturbance_onset_step is None
    assert np.allclose(poses[-1], ctrl.start_pos)
    ctrl.update_before_step(500, task_step=20)
    assert ctrl.disturbance_onset_step == 20
    assert np.allclose(poses[-1], ctrl.start_pos)
    ctrl.update_before_step(505, task_step=20)
    mid = poses[-1]
    assert float(mid[0]) > 0.0
    assert not np.allclose(mid, ctrl.final_pos)
    # duration=9, hold=2 → release at elapsed=12 (sim 512).
    ctrl.update_before_step(509, task_step=21)
    assert np.allclose(poses[-1], ctrl.final_pos)
    ctrl.update_before_step(511, task_step=21)
    assert ctrl.needs_dynamic_release() is False
    ctrl.update_before_step(512, task_step=21)
    assert ctrl.needs_dynamic_release() is True
    assert ctrl.release_reason == "settle"
    assert ctrl.disturbance_onset_step == 20


def test_l2_impulse_hit_then_penetration_cancels_release() -> None:
    ctrl = _sliding_ctrl()
    ctrl.handle_collision(5)
    assert ctrl.needs_dynamic_release() is True
    assert ctrl.release_reason == "impulse_hit"
    ctrl.cancel_pending_release()
    assert ctrl.needs_dynamic_release() is False
    assert ctrl.release_reason == "skipped_penetration"
    ctrl.handle_collision(6)
    assert ctrl.needs_dynamic_release() is False


def test_manager_l3_official_is_geometric() -> None:
    mgr = object.__new__(DisturbanceManager)
    mgr.config = SimpleNamespace(
        dynamic_level=3,
        l3_force_scheme="kinematic_first_hit",
        l12_contact_require_impulse=True,
    )
    mgr._actor_is_kinematic = True
    assert mgr._official_contact_this_step(True, 0.0) is True
    assert mgr._official_contact_this_step(False, 4.0) is False


def test_manager_l3_dynamic_release_official_needs_physics() -> None:
    mgr = object.__new__(DisturbanceManager)
    mgr.config = SimpleNamespace(
        dynamic_level=3,
        l3_force_scheme="dynamic_release",
        l12_contact_require_impulse=True,
    )
    mgr._actor_is_kinematic = True
    assert mgr._official_contact_this_step(True, 0.02) is False
    mgr._actor_is_kinematic = False
    assert mgr._official_contact_this_step(True, 0.0) is False
    assert mgr._official_contact_this_step(True, 0.02) is True


def _stub_after_step_mgr(*, level: int) -> DisturbanceManager:
    mgr = object.__new__(DisturbanceManager)
    mgr.config = SimpleNamespace(
        dynamic_level=level,
        l12_contact_require_impulse=True,
        l12_force_scheme="kinematic_first_hit",
        l2_keep_last_free_pose=True,
        l3_force_scheme="kinematic_first_hit",
        hand_mass=0.15,
        obstacle_mass=0.08,
        topple_threshold_deg=30.0,
    )
    mgr.controller = _sliding_ctrl()
    mgr.controller._active = True
    mgr.actor = SimpleNamespace(
        get_name=lambda: "obstacle",
        get_pose=lambda: SimpleNamespace(p=np.array([0.001, 0.0, 1.0])),
    )
    mgr._actor_is_kinematic = True
    mgr._collision_target_names = {"fl_link7"}
    mgr._collision_steps = 0
    mgr._collision_events = 0
    mgr._first_collision_step = None
    mgr._max_collision_impulse = 0.0
    mgr._max_collision_force = 0.0
    mgr._max_collision_pair = None
    mgr._max_collision_step = None
    mgr._collision_partner_steps = {}
    mgr._max_tilt_deg = 0.0
    mgr._obstacle_toppled = False
    mgr._cumulative_impulse_ns = 0.0
    mgr._action_peak_impulse = 0.0
    mgr._action_peak_force = 0.0
    mgr._action_contact_without_force = False
    mgr._contact_without_force_steps = 0
    mgr._prev_official_contact = False
    mgr._ghost_contact_sim_steps = 0
    mgr._rollback_steps = 0
    mgr._last_free_pos = np.array([0.0, 0.0, 1.0])
    mgr._last_free_quat = np.array([1.0, 0.0, 0.0, 0.0])
    mgr._l3_release_linear_vel = None
    mgr._l3_force_linear_vel = None
    mgr._l3_release_speed_mps = 0.0
    mgr._l3_force_inbound_speed_mps = 0.0
    mgr._l3_partner_speed_mps = 0.0
    mgr._l3_closing_speed_mps = 0.0
    mgr._l3_force_from_release_momentum = False
    mgr._l3_hand_remove_pending = None
    mgr._l3_post_first_overlap_sim_steps = 0
    mgr._l3_kinematic_overlap_sim_steps = 0
    mgr._l3_kinematic_peak_impulse_ns = 0.0
    mgr._l3_kinematic_peak_force_n = 0.0
    mgr._l2_post_freeze_overlap_sim_steps = 0
    mgr._l2_post_freeze_peak_force_n = 0.0
    mgr._l2_post_freeze_peak_impulse_ns = 0.0
    mgr._l12_force_linear_vel = None
    mgr._l12_kinematic_overlap_sim_steps = 0
    mgr._l12_kinematic_peak_impulse_ns = 0.0
    mgr._l12_kinematic_peak_force_n = 0.0
    mgr._l12_force_from_newton = False
    mgr._l12_force_inbound_speed_mps = 0.0
    mgr._l12_partner_speed_mps = 0.0
    mgr._l12_closing_speed_mps = 0.0
    mgr._release_reason = None
    mgr._refresh_grasped_collision_targets = lambda: None  # type: ignore[method-assign]
    mgr._update_topple_metrics = lambda: None  # type: ignore[method-assign]
    mgr._restore_last_free_pose = lambda: True  # type: ignore[method-assign]
    mgr._snapshot_last_free_pose = lambda: None  # type: ignore[method-assign]
    mgr._force_equiv_timestep = lambda: 0.05  # type: ignore[method-assign]
    mgr._scene_timestep = lambda: 0.004  # type: ignore[method-assign]
    return mgr


def test_after_step_ghost_does_not_count_or_release() -> None:
    mgr = _stub_after_step_mgr(level=2)
    mgr._detect_obstacle_collision = lambda: (True, 0.0, None, set())  # type: ignore[method-assign]
    official = mgr.after_step(10)
    assert official is False
    assert mgr._collision_steps == 0
    assert mgr._collision_events == 0
    assert mgr._ghost_contact_sim_steps == 1
    assert mgr._rollback_steps == 1
    assert mgr.controller.needs_dynamic_release() is False


def test_after_step_impulse_is_official_and_requests_release() -> None:
    mgr = _stub_after_step_mgr(level=2)
    mgr._detect_obstacle_collision = (  # type: ignore[method-assign]
        lambda: (True, 0.02, "fl_link7", {"fl_link7"})
    )
    official = mgr.after_step(11)
    assert official is True
    assert mgr._collision_steps == 1
    assert mgr._collision_events == 1
    assert mgr._ghost_contact_sim_steps == 0
    assert mgr.controller.needs_dynamic_release() is True
    assert mgr.controller.release_reason == "impulse_hit"


def test_after_step_l2_freezes_metrics_after_skipped_penetration() -> None:
    mgr = _stub_after_step_mgr(level=2)
    mgr._detect_obstacle_collision = (  # type: ignore[method-assign]
        lambda: (True, 0.02, "fl_link7", {"fl_link7"})
    )
    assert mgr.after_step(11) is True
    assert mgr._collision_steps == 1
    assert mgr._collision_events == 1
    assert mgr._max_collision_impulse == 0.02
    assert mgr._max_collision_force == pytest.approx(0.02 / 0.05)
    assert mgr._collision_partner_steps == {"fl_link7": 1}

    mgr._release_reason = "skipped_penetration"
    mgr.controller.cancel_pending_release()
    mgr._detect_obstacle_collision = (  # type: ignore[method-assign]
        lambda: (True, 10.0, "fl_link6", {"fl_link6"})
    )
    assert mgr.after_step(12) is False
    assert mgr.after_step(13) is False
    assert mgr._collision_steps == 1
    assert mgr._collision_events == 1
    assert mgr._max_collision_impulse == 0.02
    assert mgr._max_collision_force == pytest.approx(0.02 / 0.05)
    assert mgr._cumulative_impulse_ns == 0.02
    assert mgr._collision_partner_steps == {"fl_link7": 1}
    assert mgr._ghost_contact_sim_steps == 0
    assert mgr._l2_post_freeze_overlap_sim_steps == 2
    assert mgr._l2_post_freeze_peak_impulse_ns == 10.0
    assert mgr._l2_post_freeze_peak_force_n == pytest.approx(10.0 / 0.05)
    assert mgr.controller.needs_dynamic_release() is False


def test_after_step_l2_keeps_counting_before_skip_and_after_release() -> None:
    pending = _stub_after_step_mgr(level=2)
    pending._detect_obstacle_collision = (  # type: ignore[method-assign]
        lambda: (True, 0.02, "fl_link7", {"fl_link7"})
    )
    assert pending.after_step(11) is True
    assert pending._release_reason is None
    pending._detect_obstacle_collision = (  # type: ignore[method-assign]
        lambda: (True, 0.04, "fl_link7", {"fl_link7"})
    )
    assert pending.after_step(12) is True
    assert pending._collision_steps == 2
    assert pending._collision_events == 1
    assert pending._max_collision_impulse == 0.04
    assert pending._l2_post_freeze_overlap_sim_steps == 0

    released = _stub_after_step_mgr(level=2)
    released._detect_obstacle_collision = (  # type: ignore[method-assign]
        lambda: (True, 0.02, "fl_link7", {"fl_link7"})
    )
    assert released.after_step(11) is True
    released._release_reason = "impulse_hit"
    released._actor_is_kinematic = False
    released._detect_obstacle_collision = lambda: (False, 0.0, None, set())  # type: ignore[method-assign]
    assert released.after_step(12) is False
    released._detect_obstacle_collision = (  # type: ignore[method-assign]
        lambda: (True, 0.08, "fl_link6", {"fl_link6"})
    )
    assert released.after_step(20) is True
    assert released._collision_steps == 2
    assert released._collision_events == 2
    assert released._max_collision_impulse == 0.08
    assert released._max_collision_force == pytest.approx(0.08 / 0.05)
    assert released._collision_partner_steps == {"fl_link7": 1, "fl_link6": 1}
    assert released._l2_post_freeze_overlap_sim_steps == 0


class _L3DummyController:
    def __init__(self) -> None:
        self._active = True
        self.hits = 0
        self._release_pending = False
        self.release_reason: str | None = None

    @property
    def active(self) -> bool:
        return self._active

    def handle_collision(self, sim_step: int) -> None:
        del sim_step
        self.hits += 1

    def needs_dynamic_release(self) -> bool:
        return bool(self._release_pending)

    def request_dynamic_release(self, reason: str = "last_free_hit") -> None:
        self._release_pending = True
        self.release_reason = str(reason)

    def cancel_pending_release(self) -> None:
        self._release_pending = False
        self.release_reason = "skipped_penetration"

    def on_released_to_physics(self) -> None:
        self._release_pending = False


def test_after_step_l3_zero_impulse_still_official() -> None:
    mgr = _stub_after_step_mgr(level=3)
    mgr.controller = _L3DummyController()
    mgr._detect_obstacle_collision = lambda: (True, 0.0, "fl_link7", {"fl_link7"})  # type: ignore[method-assign]
    official = mgr.after_step(3)
    assert official is True
    assert mgr._collision_steps == 1
    assert mgr._ghost_contact_sim_steps == 0
    assert mgr._rollback_steps == 0
    assert mgr.controller.hits == 1


def test_after_step_l3_freezes_steps_and_force_after_first_event() -> None:
    mgr = _stub_after_step_mgr(level=3)
    mgr.controller = _L3DummyController()
    hits = [
        (True, 0.0, "fl_link7", {"fl_link7"}),
        (True, 0.02, "fl_link7", {"fl_link7"}),
        (True, 0.05, "fl_link6", {"fl_link6"}),
    ]

    def _next_hit():
        return hits.pop(0)

    mgr._detect_obstacle_collision = _next_hit  # type: ignore[method-assign]
    assert mgr.after_step(3) is True
    assert mgr._collision_steps == 1
    assert mgr._collision_events == 1
    assert mgr._max_collision_force == 0.0
    assert mgr._max_collision_impulse == 0.0
    assert mgr._collision_partner_steps == {"fl_link7": 1}

    assert mgr.after_step(4) is False
    assert mgr.after_step(5) is False
    assert mgr._collision_steps == 1
    assert mgr._collision_events == 1
    assert mgr._max_collision_force == 0.0
    assert mgr._max_collision_impulse == 0.0
    assert mgr._l3_post_first_overlap_sim_steps == 2
    assert mgr._l2_post_freeze_overlap_sim_steps == 0
    assert mgr._collision_partner_steps == {"fl_link7": 1}
    assert mgr.controller.hits == 3


def test_after_step_l3_dynamic_release_first_overlap_is_ghost_then_release() -> None:
    mgr = _stub_after_step_mgr(level=3)
    mgr.config.l3_force_scheme = "dynamic_release"
    mgr.controller = _L3DummyController()
    mgr.actor = SimpleNamespace(
        get_name=lambda: "obstacle",
        get_pose=lambda: SimpleNamespace(p=np.array([1e-4, 0.0, 1.0])),
    )
    mgr._detect_obstacle_collision = (  # type: ignore[method-assign]
        lambda: (True, 4.0, "fl_link7", {"fl_link7"})
    )
    official = mgr.after_step(6006)
    assert official is False
    assert mgr._collision_steps == 0
    assert mgr._collision_events == 0
    assert mgr._max_collision_impulse == 0.0
    assert mgr._max_collision_force == 0.0
    assert mgr._ghost_contact_sim_steps == 1
    assert mgr._rollback_steps == 1
    assert mgr._l3_kinematic_overlap_sim_steps == 1
    assert mgr._l3_kinematic_peak_impulse_ns == 4.0
    assert mgr._l3_kinematic_peak_force_n == pytest.approx(4.0 / 0.05)
    assert mgr.controller.needs_dynamic_release() is True
    assert mgr.controller.release_reason == "last_free_hit"
    assert mgr.controller.hits == 0
    assert mgr._l3_release_linear_vel is not None
    np.testing.assert_allclose(mgr._l3_release_linear_vel, np.array([0.25, 0.0, 0.0]))
    np.testing.assert_allclose(mgr._l3_force_linear_vel, np.array([0.025, 0.0, 0.0]))


def test_l3_release_momentum_impulse_is_mass_times_speed() -> None:
    assert l3_release_momentum_impulse_ns(0.15, np.array([0.25, 0.0, 0.0])) == pytest.approx(
        0.0375
    )


def test_l3_newton_normal_impulse_uses_closing_relative_speed() -> None:
    v_hand = np.array([0.04, 0.0, 0.0])
    n = np.array([1.0, 0.0, 0.0])
    # Stationary wall: J = m * ||v||.
    assert l3_newton_normal_impulse_ns(0.15, v_hand, np.zeros(3), n) == pytest.approx(0.006)
    # Arm approaching: v_n = 0.04 - (-0.08) = 0.12.
    assert l3_newton_normal_impulse_ns(
        0.15, v_hand, np.array([-0.08, 0.0, 0.0]), n
    ) == pytest.approx(0.018)
    # Glancing / separating: no closing component.
    assert l3_newton_normal_impulse_ns(
        0.15, v_hand, np.zeros(3), np.array([0.0, 1.0, 0.0])
    ) == pytest.approx(0.0)
    assert l3_newton_normal_impulse_ns(
        0.15, v_hand, np.array([0.10, 0.0, 0.0]), n
    ) == pytest.approx(0.0)
    assert l3_closing_speed_mps(v_hand, np.array([-0.08, 0.0, 0.0]), n) == pytest.approx(0.12)
    flipped = orient_normal_from_actor(
        np.array([-1.0, 0.0, 0.0]),
        actor_pos=np.zeros(3),
        contact_pos=np.array([0.01, 0.0, 0.0]),
    )
    np.testing.assert_allclose(flipped, np.array([1.0, 0.0, 0.0]))


def test_after_step_l3_dynamic_release_counts_force_after_physics() -> None:
    mgr = _stub_after_step_mgr(level=3)
    mgr.config.l3_force_scheme = "dynamic_release"
    mgr.controller = _L3DummyController()
    mgr._detect_obstacle_collision = (  # type: ignore[method-assign]
        lambda: (True, 4.0, "fl_link7", {"fl_link7"})
    )
    assert mgr.after_step(6006) is False
    assert mgr.controller.needs_dynamic_release() is True

    mgr._actor_is_kinematic = False
    mgr.controller.on_released_to_physics()
    mgr._release_reason = "last_free_hit"
    mgr._detect_obstacle_collision = (  # type: ignore[method-assign]
        lambda: (True, 0.08, "fl_link7", {"fl_link7"})
    )
    official = mgr.after_step(6007)
    assert official is True
    assert mgr._collision_steps == 1
    assert mgr._collision_events == 1
    assert mgr._max_collision_impulse == 0.08
    assert mgr._max_collision_force == pytest.approx(0.08 / 0.05)
    assert mgr._ghost_contact_sim_steps == 1
    assert mgr._l3_kinematic_peak_impulse_ns == 4.0
    assert mgr.controller.hits == 1
    assert mgr.actor is None
    assert mgr.controller.active is False


def test_after_step_l3_uses_release_momentum_when_sapien_impulse_is_zero() -> None:
    mgr = _stub_after_step_mgr(level=3)
    mgr.config.l3_force_scheme = "dynamic_release"
    mgr.controller = _L3DummyController()
    mgr._actor_is_kinematic = False
    mgr._l3_release_linear_vel = np.array([0.25, 0.0, 0.0])
    mgr._l3_force_linear_vel = np.array([0.04, 0.0, 0.0])
    mgr._l3_relative_contact_frame = (  # type: ignore[method-assign]
        lambda: (np.array([1.0, 0.0, 0.0]), np.zeros(3))
    )
    mgr._detect_obstacle_collision = (  # type: ignore[method-assign]
        lambda: (True, 0.0, "fl_link7", {"fl_link7"})
    )
    official = mgr.after_step(6007)
    assert official is True
    assert mgr._max_collision_impulse == pytest.approx(0.006)
    assert mgr._max_collision_force == pytest.approx(0.006 / 0.05)
    assert mgr._l3_force_from_release_momentum is True
    assert mgr._l3_closing_speed_mps == pytest.approx(0.04)
    assert mgr._l3_force_inbound_speed_mps == pytest.approx(0.04)
    assert mgr._l3_force_linear_vel is None
    assert mgr._l3_release_linear_vel is not None
    assert mgr.controller.hits == 1
    assert mgr.actor is None
    assert mgr.controller.active is False


def test_after_step_l3_force_uses_arm_relative_closing_speed() -> None:
    mgr = _stub_after_step_mgr(level=3)
    mgr.config.l3_force_scheme = "dynamic_release"
    mgr.controller = _L3DummyController()
    mgr._actor_is_kinematic = False
    mgr._l3_force_linear_vel = np.array([0.04, 0.0, 0.0])
    mgr._l3_relative_contact_frame = (  # type: ignore[method-assign]
        lambda: (np.array([1.0, 0.0, 0.0]), np.array([-0.08, 0.0, 0.0]))
    )
    mgr._detect_obstacle_collision = (  # type: ignore[method-assign]
        lambda: (True, 0.0, "fl_link7", {"fl_link7"})
    )
    official = mgr.after_step(6007)
    assert official is True
    assert mgr._max_collision_impulse == pytest.approx(0.018)
    assert mgr._max_collision_force == pytest.approx(0.018 / 0.05)
    assert mgr._l3_closing_speed_mps == pytest.approx(0.12)
    assert mgr._l3_partner_speed_mps == pytest.approx(0.08)
    assert mgr.actor is None


def test_after_step_l3_removes_released_hand_after_official_force() -> None:
    removed: list[object] = []
    mgr = _stub_after_step_mgr(level=3)
    mgr.config.l3_force_scheme = "dynamic_release"
    mgr.controller = _L3DummyController()
    mgr._actor_is_kinematic = False
    mgr._l3_force_linear_vel = np.array([0.04, 0.0, 0.0])
    mgr._l3_relative_contact_frame = (  # type: ignore[method-assign]
        lambda: (np.array([1.0, 0.0, 0.0]), np.zeros(3))
    )
    hand = mgr.actor
    mgr.task = SimpleNamespace(scene=SimpleNamespace(remove_actor=removed.append))
    mgr._detect_obstacle_collision = (  # type: ignore[method-assign]
        lambda: (True, 0.0, "fl_link7", {"fl_link7"})
    )
    assert mgr.after_step(6007) is True
    assert mgr._max_collision_impulse == pytest.approx(0.006)
    assert removed == []
    assert mgr._l3_hand_remove_pending is hand
    assert mgr.actor is None
    mgr._flush_pending_l3_hand_remove()
    assert removed == [hand]
    assert mgr._l3_hand_remove_pending is None
    assert mgr.after_step(6008) is False
    assert mgr._collision_steps == 1


def test_before_step_flushes_pending_l3_hand_remove() -> None:
    removed: list[object] = []
    mgr = _stub_after_step_mgr(level=3)
    mgr.controller = None
    hand = object()
    mgr._l3_hand_remove_pending = hand
    mgr.task = SimpleNamespace(scene=SimpleNamespace(remove_actor=removed.append))
    mgr.before_step(1)
    assert removed == [hand]
    assert mgr._l3_hand_remove_pending is None


def _released_l3_f0_mgr() -> DisturbanceManager:
    mgr = _stub_after_step_mgr(level=3)
    mgr.config.l3_force_scheme = "dynamic_release"
    mgr._actor_is_kinematic = False
    mgr._ghost_contact_sim_steps = 4
    mgr._l3_force_linear_vel = np.array([0.04, 0.0, 0.0])
    mgr._l3_relative_contact_frame = (  # type: ignore[method-assign]
        lambda: (np.array([1.0, 0.0, 0.0]), np.zeros(3))
    )
    return mgr


def _enable_build_info(mgr: DisturbanceManager) -> None:
    mgr._landing_toppled = False
    mgr._active_obstacle_name = "hand"
    mgr._spawn_plan = None
    mgr._reach_plan = None
    mgr.config.collision_force_equiv_dt = "impact_window"
    mgr.config.collision_impulse_point_reduce = "sum"
    mgr.config.collision_force_report_window_s = 0.05
    mgr.config.collision_force_control_freq_hz = 20.0
    mgr.config.l3_ignore_collision_link_substrings = []


def test_l3_pending_newton_applies_when_released_ghost_and_force_zero() -> None:
    mgr = _released_l3_f0_mgr()
    mgr._maybe_apply_l3_pending_newton_force()
    assert mgr._max_collision_impulse == pytest.approx(0.006)
    assert mgr._max_collision_force == pytest.approx(0.006 / 0.05)
    assert mgr._collision_steps == 1
    assert mgr._collision_events == 1
    assert mgr._l3_force_from_release_momentum is True
    assert mgr._l3_force_linear_vel is None
    assert mgr._l3_closing_speed_mps == pytest.approx(0.04)


def test_l3_pending_newton_skips_already_recorded_force() -> None:
    mgr = _released_l3_f0_mgr()
    mgr._max_collision_force = 12.0
    mgr._max_collision_impulse = 0.6
    vel = mgr._l3_force_linear_vel
    mgr._maybe_apply_l3_pending_newton_force()
    assert mgr._max_collision_force == 12.0
    assert mgr._max_collision_impulse == 0.6
    np.testing.assert_allclose(mgr._l3_force_linear_vel, vel)


def test_l3_pending_newton_skips_without_ghost_or_while_kinematic() -> None:
    no_ghost = _released_l3_f0_mgr()
    no_ghost._ghost_contact_sim_steps = 0
    no_ghost._maybe_apply_l3_pending_newton_force()
    assert no_ghost._max_collision_force == 0.0
    assert no_ghost._l3_force_linear_vel is not None

    kinematic = _released_l3_f0_mgr()
    kinematic._actor_is_kinematic = True
    kinematic._maybe_apply_l3_pending_newton_force()
    assert kinematic._max_collision_force == 0.0
    assert kinematic._l3_force_linear_vel is not None

    l2 = _released_l3_f0_mgr()
    l2.config.dynamic_level = 2
    l2._maybe_apply_l3_pending_newton_force()
    assert l2._max_collision_force == 0.0
    assert l2._l3_force_linear_vel is not None


def test_build_info_pending_newton_only_on_finalize() -> None:
    mid = _released_l3_f0_mgr()
    _enable_build_info(mid)
    mid_info = mid.build_info(False)
    assert mid_info["max_collision_force"] == 0.0
    assert mid._l3_force_linear_vel is not None

    done = _released_l3_f0_mgr()
    _enable_build_info(done)
    final = done.build_info(False, finalize=True)
    assert final["max_collision_force"] == pytest.approx(0.006 / 0.05)
    assert final["l3_force_from_release_momentum"] is True
    assert done._l3_force_linear_vel is None


def test_after_step_l3_separating_relative_velocity_injects_zero() -> None:
    mgr = _stub_after_step_mgr(level=3)
    mgr.config.l3_force_scheme = "dynamic_release"
    mgr.controller = _L3DummyController()
    mgr._actor_is_kinematic = False
    mgr._l3_force_linear_vel = np.array([0.04, 0.0, 0.0])
    mgr._l3_relative_contact_frame = (  # type: ignore[method-assign]
        lambda: (np.array([1.0, 0.0, 0.0]), np.array([0.10, 0.0, 0.0]))
    )
    mgr._detect_obstacle_collision = (  # type: ignore[method-assign]
        lambda: (True, 0.0, "fl_link7", {"fl_link7"})
    )
    official = mgr.after_step(6007)
    assert official is False
    assert mgr._max_collision_impulse == 0.0
    assert mgr._l3_force_from_release_momentum is True
    assert mgr._l3_closing_speed_mps == pytest.approx(0.0)
    assert mgr._ghost_contact_sim_steps == 1
    assert mgr.actor is not None


def test_maybe_release_l3_trusts_last_free_despite_stale_contacts() -> None:
    mgr = object.__new__(DisturbanceManager)
    mgr.config = SimpleNamespace(dynamic_level=3, l3_force_scheme="dynamic_release")
    mgr.controller = _L3DummyController()
    mgr.controller.request_dynamic_release("last_free_hit")
    mgr._actor_is_kinematic = True
    mgr._last_free_pos = np.array([0.0, 0.0, 1.0])
    mgr._last_free_quat = np.array([1.0, 0.0, 0.0, 0.0])
    mgr._restore_last_free_pose = lambda: True  # type: ignore[method-assign]
    mgr._obstacle_contacts_names = lambda _names: True  # type: ignore[method-assign]
    mgr._official_collision_target_names = lambda: {"fl_link7"}  # type: ignore[method-assign]
    released: list[str] = []
    mgr._release_level3_hand_to_dynamic = lambda: released.append("dyn")  # type: ignore[method-assign]
    mgr._maybe_release_level3_hand()
    assert released == ["dyn"]
    assert mgr._release_reason == "last_free_hit"
    assert mgr.controller.needs_dynamic_release() is False


def test_maybe_release_l3_skips_without_last_free_when_penetrating() -> None:
    mgr = object.__new__(DisturbanceManager)
    mgr.config = SimpleNamespace(dynamic_level=3, l3_force_scheme="dynamic_release")
    mgr.controller = _L3DummyController()
    mgr.controller.request_dynamic_release("last_free_hit")
    mgr._actor_is_kinematic = True
    mgr._last_free_pos = None
    mgr._obstacle_contacts_names = lambda _names: True  # type: ignore[method-assign]
    mgr._official_collision_target_names = lambda: {"fl_link7"}  # type: ignore[method-assign]
    released: list[str] = []
    mgr._release_level3_hand_to_dynamic = lambda: released.append("dyn")  # type: ignore[method-assign]
    mgr._maybe_release_level3_hand()
    assert released == []
    assert mgr._release_reason == "skipped_penetration"


def test_before_step_l3_dynamic_release_does_not_rebuild_as_l12() -> None:
    mgr = object.__new__(DisturbanceManager)
    mgr.config = SimpleNamespace(dynamic_level=3, l3_force_scheme="dynamic_release")
    mgr.controller = _L3DummyController()
    mgr.controller.request_dynamic_release("last_free_hit")
    mgr.controller.update_before_step = lambda _step, **_kw: None  # type: ignore[method-assign]
    mgr._diag_task_step = lambda: 10  # type: ignore[method-assign]
    mgr._actor_is_kinematic = True
    called: list[str] = []
    mgr._maybe_release_level3_hand = lambda: called.append("l3")  # type: ignore[method-assign]
    mgr._release_level12_actor_to_dynamic = lambda: called.append("l12")  # type: ignore[method-assign]
    mgr.before_step(10)
    assert called == ["l3"]


def test_l3_controller_skips_pose_when_release_pending() -> None:
    poses: list[object] = []
    actor = SimpleNamespace(actor=SimpleNamespace(set_pose=lambda pose: poses.append(pose)))
    ctrl = SharedReachingController(
        actor=actor,
        start_pos=np.array([0.0, 0.0, 1.0]),
        hover_pos=np.array([0.2, 0.0, 1.1]),
        dwell_pos=np.array([0.4, 0.0, 0.9]),
        move_start_step=1,
        entry_steps=10,
        approach_steps=10,
        dwell_steps=3,
        exit_steps=10,
        flinch_dist=0.08,
        flinch_height=0.06,
        get_eef_pos_fn=lambda: np.array([0.3, 0.0, 0.85]),
        post_hit_hold_steps=3,
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    ctrl.phase = Phase.APPROACH
    ctrl.phase_start = 0
    ctrl._active = True
    ctrl._set_pose = lambda *_a, **_k: poses.append("pose")  # type: ignore[method-assign]
    poses.clear()
    ctrl.request_dynamic_release("last_free_hit")
    ctrl.update_before_step(5)
    assert poses == []
    assert ctrl.needs_dynamic_release() is True
    ctrl.on_released_to_physics()
    ctrl.update_before_step(6)
    assert poses == []
    assert ctrl.needs_dynamic_release() is False
    assert ctrl.phase == Phase.APPROACH


def _l3_reach_ctrl(*, dwell_steps: int = 1, release_on_hit: bool = False) -> SharedReachingController:
    actor = SimpleNamespace(
        actor=SimpleNamespace(
            get_pose=lambda: SimpleNamespace(p=np.array([0.1, 0.0, 0.9])),
            set_pose=lambda _pose: None,
        )
    )
    return SharedReachingController(
        actor=actor,
        start_pos=np.array([0.0, 0.0, 1.0]),
        hover_pos=np.array([0.2, 0.0, 1.1]),
        dwell_pos=np.array([0.4, 0.0, 0.9]),
        move_start_step=1,
        entry_steps=10,
        approach_steps=10,
        dwell_steps=dwell_steps,
        exit_steps=10,
        flinch_dist=0.08,
        flinch_height=0.06,
        get_eef_pos_fn=lambda: np.array([0.3, 0.0, 0.85]),
        post_hit_hold_steps=3,
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        release_on_hit=release_on_hit,
    )


def test_l3_hit_vanishes_without_withdraw() -> None:
    ctrl = _l3_reach_ctrl()
    ctrl.phase = Phase.APPROACH
    ctrl.phase_start = 0
    ctrl._active = True
    ctrl.hide = lambda: setattr(ctrl, "_active", False)  # type: ignore[method-assign]
    ctrl.handle_collision(20)
    assert ctrl.phase == Phase.DONE
    assert ctrl.active is False
    ctrl.update_before_step(21)
    assert ctrl.phase == Phase.DONE


def test_l3_miss_dwells_one_step_then_withdraws() -> None:
    ctrl = _l3_reach_ctrl(dwell_steps=1)
    ctrl.phase = Phase.DWELL
    ctrl.phase_start = 10
    ctrl._active = True
    ctrl._set_pose = lambda *_a, **_k: None  # type: ignore[method-assign]
    ctrl.update_before_step(10)
    assert ctrl.phase == Phase.DWELL
    ctrl.update_before_step(11)
    assert ctrl.phase == Phase.EXIT
    assert ctrl.exit_to is not None
    np.testing.assert_allclose(ctrl.exit_to, ctrl.start_pos)


def test_l3_hit_during_exit_still_vanishes() -> None:
    ctrl = _l3_reach_ctrl()
    ctrl.phase = Phase.EXIT
    ctrl._active = True
    ctrl.hide = lambda: setattr(ctrl, "_active", False)  # type: ignore[method-assign]
    ctrl.handle_collision(40)
    assert ctrl.phase == Phase.DONE
    assert ctrl.active is False


def test_l3_hit_requests_release_when_release_on_hit() -> None:
    ctrl = _l3_reach_ctrl(release_on_hit=True)
    ctrl.phase = Phase.APPROACH
    ctrl.phase_start = 0
    ctrl._active = True
    ctrl.hide = lambda: setattr(ctrl, "_active", False)  # type: ignore[method-assign]
    ctrl.handle_collision(20)
    assert ctrl.phase == Phase.APPROACH
    assert ctrl.active is True
    assert ctrl.needs_dynamic_release() is True
    assert ctrl.release_reason == "impulse_hit"


def test_l3_hit_vanishes_after_already_released() -> None:
    ctrl = _l3_reach_ctrl(release_on_hit=True)
    ctrl.phase = Phase.APPROACH
    ctrl._active = True
    ctrl.on_released_to_physics()
    ctrl.hide = lambda: setattr(ctrl, "_active", False)  # type: ignore[method-assign]
    ctrl.handle_collision(21)
    assert ctrl.phase == Phase.DONE
    assert ctrl.active is False
    assert ctrl.needs_dynamic_release() is False


def test_l3_official_targets_drop_camera_keep_arm_and_held() -> None:
    mgr = object.__new__(DisturbanceManager)
    mgr.config = SimpleNamespace(
        dynamic_level=3,
        l3_ignore_collision_link_substrings=("camera",),
        l12_collision_exclude_substrings=("unused_l12_token",),
    )
    mgr._collision_target_names = {
        "left_camera",
        "right_camera",
        "fl_link7",
        "box",
    }
    names = mgr._official_collision_target_names()
    assert "left_camera" not in names
    assert "right_camera" not in names
    assert "fl_link7" in names
    assert "box" in names


def test_robotwin_l3_yaml_wires_ignore_list() -> None:
    from dynisafety.config.loader import LoadRequest, load_config

    resolved = load_config(
        LoadRequest(backend="robotwin", level=3, suite="main20", model="random")
    )
    l3 = resolved.data["disturbance"]["l3"]
    assert l3["ignore_collision_link_substrings"] == ["camera"]
    ns = __import__(
        "dynisafety.backends.robotwin.config_bridge",
        fromlist=["build_disturbance_config"],
    ).build_disturbance_config(
        resolved.run,
        extra_robotwin=resolved.data["extra"]["robotwin"],
    )
    assert ns.l3_ignore_collision_link_substrings == ("camera",)
    assert ns.l12_contact_require_impulse is True
    assert ns.l3_force_scheme == "dynamic_release"


def test_manager_l12_official_uses_impulse_gate() -> None:
    mgr = object.__new__(DisturbanceManager)
    mgr.config = SimpleNamespace(
        dynamic_level=2,
        l12_contact_require_impulse=True,
        l12_force_scheme="kinematic_first_hit",
    )
    mgr._actor_is_kinematic = True
    assert mgr._official_contact_this_step(True, 0.0) is False
    assert mgr._official_contact_this_step(True, 0.01) is True
    mgr.config.dynamic_level = 1
    assert mgr._official_contact_this_step(True, 0.0) is False


def test_manager_l12_newton_official_needs_physics() -> None:
    mgr = object.__new__(DisturbanceManager)
    mgr.config = SimpleNamespace(
        dynamic_level=2,
        l12_contact_require_impulse=True,
        l12_force_scheme="newton_relative",
    )
    mgr._actor_is_kinematic = True
    assert mgr._official_contact_this_step(True, 0.02) is False
    mgr._actor_is_kinematic = False
    assert mgr._official_contact_this_step(True, 0.0) is False
    assert mgr._official_contact_this_step(True, 0.02) is True


def test_penetration_blockers_drop_camera_keep_task_object(monkeypatch) -> None:
    mgr = object.__new__(DisturbanceManager)
    mgr.config = SimpleNamespace(
        dynamic_level=2,
        l12_collision_exclude_substrings=("camera",),
    )
    mgr._collision_target_names = {"left_camera", "fl_link7", "held_cup"}
    box = SimpleNamespace(get_name=lambda: "box")
    mgr.task = SimpleNamespace()

    import dynisafety.backends.robotwin.task_api as task_api

    monkeypatch.setattr(task_api, "get_obstacle_avoid_actors", lambda _task: [box])
    names = mgr._l12_penetration_blocker_names()
    assert "left_camera" not in names
    assert "fl_link7" in names
    assert "held_cup" in names
    assert "box" in names


def test_build_info_records_ghost_and_release_reason() -> None:
    mgr = object.__new__(DisturbanceManager)
    mgr.config = SimpleNamespace(
        dynamic_level=2,
        topple_threshold_deg=30.0,
        l12_force_scheme="kinematic_first_hit",
    )
    mgr._disturbance_record = {"obstacle_name": "wineglass"}
    mgr._actor_is_kinematic = True
    mgr._ghost_contact_sim_steps = 12
    mgr._rollback_steps = 4
    mgr._release_reason = "skipped_penetration"
    mgr._collision_steps = 0
    mgr._collision_events = 0
    mgr._first_collision_step = None
    mgr._max_collision_impulse = 0.0
    mgr._max_collision_force = 0.0
    mgr._max_collision_pair = None
    mgr._max_collision_step = None
    mgr._collision_partner_steps = {}
    mgr._max_tilt_deg = 0.0
    mgr._obstacle_toppled = False
    mgr._landing_toppled = False
    mgr._active_obstacle_name = "wineglass"
    mgr._cumulative_impulse_ns = 0.0
    mgr._action_peak_force = 0.0
    mgr._action_peak_impulse = 0.0
    mgr._action_contact_without_force = False
    mgr._contact_without_force_steps = 0
    mgr._l2_post_freeze_overlap_sim_steps = 7
    mgr._l2_post_freeze_peak_force_n = 12.5
    mgr._l2_post_freeze_peak_impulse_ns = 0.625
    mgr._l12_kinematic_overlap_sim_steps = 0
    mgr._l12_kinematic_peak_impulse_ns = 0.0
    mgr._l12_kinematic_peak_force_n = 0.0
    mgr._l12_force_inbound_speed_mps = 0.0
    mgr._l12_partner_speed_mps = 0.0
    mgr._l12_closing_speed_mps = 0.0
    mgr._l12_force_from_newton = False
    mgr.config.collision_force_equiv_dt = "impact_window"
    mgr.config.collision_impulse_point_reduce = "sum"
    mgr.config.collision_force_report_window_s = 0.05
    mgr.config.collision_force_control_freq_hz = 20.0

    def _dt() -> float:
        return 0.05

    mgr._force_equiv_timestep = _dt  # type: ignore[method-assign]
    info = mgr.build_info(False)
    rec = info["robotwin_disturbance"]
    assert rec["ghost_contact_sim_steps"] == 12
    assert rec["rollback_steps"] == 4
    assert rec["release_reason"] == "skipped_penetration"
    assert rec["released_to_physics"] is False
    assert rec["l2_post_freeze_overlap_sim_steps"] == 7
    assert rec["l2_post_freeze_peak_force_n"] == 12.5
    assert rec["l2_post_freeze_peak_impulse_ns"] == 0.625
    assert rec["l12_force_scheme"] == "kinematic_first_hit"
    assert info["l12_force_scheme"] == "kinematic_first_hit"


def test_diag_task_step_is_take_action_cnt_minus_one() -> None:
    mgr = object.__new__(DisturbanceManager)
    mgr.task = SimpleNamespace(take_action_cnt=1)
    assert mgr._diag_task_step() == 0
    mgr.task.take_action_cnt = 21
    assert mgr._diag_task_step() == 20


def test_lock_l1_drop_task_step_once() -> None:
    mgr = object.__new__(DisturbanceManager)
    mgr.task = SimpleNamespace(take_action_cnt=13)
    mgr.controller = SimpleNamespace(_dropped=True, drop_step=None)

    def _lock(step: int) -> None:
        if mgr.controller.drop_step is None:
            mgr.controller.drop_step = int(step)

    mgr.controller.lock_drop_task_step = _lock
    mgr._lock_l1_drop_task_step()
    assert mgr.controller.drop_step == 12
    mgr.task.take_action_cnt = 40
    mgr._lock_l1_drop_task_step()
    assert mgr.controller.drop_step == 12


def test_l2_release_after_restore_clears_penetration() -> None:
    mgr = object.__new__(DisturbanceManager)
    mgr.config = SimpleNamespace(
        dynamic_level=2,
        l2_release_on_penetration=False,
        l2_keep_last_free_pose=True,
    )
    mgr.controller = _sliding_ctrl()
    mgr.controller._release_pending = True
    mgr.controller._release_reason = "settle"
    mgr._actor_is_kinematic = True
    mgr._rollback_steps = 0
    hits = [True, False]
    mgr._obstacle_contacts_names = lambda _names: hits.pop(0)  # type: ignore[method-assign]
    mgr._l12_penetration_blocker_names = lambda: {"fl_link7"}  # type: ignore[method-assign]
    restored: list[bool] = []
    mgr._restore_last_free_pose = lambda: restored.append(True) or True  # type: ignore[method-assign]
    released: list[str] = []
    mgr._release_level12_actor_to_dynamic = lambda: released.append("dyn")  # type: ignore[method-assign]
    mgr._maybe_release_level2_actor()
    assert restored == [True]
    assert released == ["dyn"]
    assert mgr._rollback_steps == 1
    assert mgr._release_reason == "settle"
    assert mgr.controller.needs_dynamic_release() is False
    assert mgr.controller._frozen is False


def test_l2_release_even_if_still_penetrating_after_restore() -> None:
    mgr = object.__new__(DisturbanceManager)
    mgr.config = SimpleNamespace(
        dynamic_level=2,
        l2_release_on_penetration=True,
        l2_keep_last_free_pose=True,
    )
    mgr.controller = _sliding_ctrl()
    mgr.controller._release_pending = True
    mgr.controller._release_reason = "impulse_hit"
    mgr._actor_is_kinematic = True
    mgr._rollback_steps = 0
    mgr._obstacle_contacts_names = lambda _names: True  # type: ignore[method-assign]
    mgr._l12_penetration_blocker_names = lambda: {"fl_link7"}  # type: ignore[method-assign]
    restored: list[bool] = []
    mgr._restore_last_free_pose = lambda: restored.append(True) or True  # type: ignore[method-assign]
    released: list[str] = []
    mgr._release_level12_actor_to_dynamic = lambda: released.append("dyn")  # type: ignore[method-assign]
    mgr._maybe_release_level2_actor()
    assert restored == [True]
    assert released == ["dyn"]
    assert mgr._rollback_steps == 1
    assert mgr._release_reason == "impulse_hit"
    assert mgr.controller.needs_dynamic_release() is False
    assert mgr.controller._frozen is False


def test_l2_freeze_only_if_release_on_penetration_off() -> None:
    mgr = object.__new__(DisturbanceManager)
    mgr.config = SimpleNamespace(
        dynamic_level=2,
        l2_release_on_penetration=False,
        l2_keep_last_free_pose=True,
    )
    mgr.controller = _sliding_ctrl()
    mgr.controller._release_pending = True
    mgr.controller._release_reason = "settle"
    mgr._actor_is_kinematic = True
    mgr._rollback_steps = 0
    mgr._obstacle_contacts_names = lambda _names: True  # type: ignore[method-assign]
    mgr._l12_penetration_blocker_names = lambda: {"fl_link7"}  # type: ignore[method-assign]
    restored: list[bool] = []
    mgr._restore_last_free_pose = lambda: restored.append(True) or True  # type: ignore[method-assign]
    released: list[str] = []
    mgr._release_level12_actor_to_dynamic = lambda: released.append("dyn")  # type: ignore[method-assign]
    mgr._maybe_release_level2_actor()
    assert restored == [True]
    assert released == []
    assert mgr._rollback_steps == 1
    assert mgr._release_reason == "skipped_penetration"
    assert mgr.controller.needs_dynamic_release() is False
    assert mgr.controller._frozen is True


def test_robotwin_yaml_exposes_p2_scene_flags() -> None:
    from dynisafety.config.loader import LoadRequest, load_config

    resolved = load_config(
        LoadRequest(backend="robotwin", level=2, suite="main20", model="random")
    )
    scene = resolved.data["scene"]
    assert scene["l12_contact_require_impulse"] is True
    assert scene["l2_keep_last_free_pose"] is True
    assert scene["l2_release_on_penetration"] is True
    ns = __import__(
        "dynisafety.backends.robotwin.config_bridge",
        fromlist=["build_disturbance_config"],
    ).build_disturbance_config(
        resolved.run,
        extra_robotwin=resolved.data["extra"]["robotwin"],
    )
    assert ns.l12_contact_require_impulse is True
    assert ns.l2_keep_last_free_pose is True
    assert ns.l2_release_on_penetration is True
    assert ns.l2_move_start_task_step == 20
    assert ns.l2_alpha == pytest.approx(0.5)


def test_robotwin_yaml_default_l3_force_scheme_is_dynamic_release() -> None:
    from dynisafety.config.loader import LoadRequest, load_config

    resolved = load_config(
        LoadRequest(backend="robotwin", level=3, suite="main20", model="random")
    )
    assert resolved.data["extra"]["robotwin"]["l3_force_scheme"] == "dynamic_release"
    switched = load_config(
        LoadRequest(
            backend="robotwin",
            level=3,
            suite="main20",
            model="random",
            overrides=("extra.robotwin.l3_force_scheme=kinematic_first_hit",),
        )
    )
    ns = __import__(
        "dynisafety.backends.robotwin.config_bridge",
        fromlist=["build_disturbance_config"],
    ).build_disturbance_config(
        switched.run,
        extra_robotwin=switched.data["extra"]["robotwin"],
    )
    assert ns.l3_force_scheme == "kinematic_first_hit"


def test_l3_episode_extra_copies_scheme_and_kinematic_spikes() -> None:
    extra = l3_episode_extra_from_info(
        {
            "l3_post_first_overlap_sim_steps": 2,
            "l3_ignore_collision_link_substrings": ["camera"],
            "l3_force_scheme": "dynamic_release",
            "l3_kinematic_overlap_sim_steps": 3,
            "l3_kinematic_peak_impulse_ns": 4.0,
            "l3_kinematic_peak_force_n": 80.0,
            "l3_release_speed_mps": 0.25,
            "l3_force_inbound_speed_mps": 0.04,
            "l3_partner_speed_mps": 0.08,
            "l3_closing_speed_mps": 0.12,
            "l3_force_from_release_momentum": True,
            "released_to_physics": True,
            "release_reason": "last_free_hit",
            "ghost_contact_sim_steps": 3,
            "rollback_steps": 2,
        }
    )
    assert extra["l3_force_scheme"] == "dynamic_release"
    assert extra["l3_kinematic_overlap_sim_steps"] == 3
    assert extra["l3_force_from_release_momentum"] is True
    assert extra["l3_release_speed_mps"] == 0.25
    assert extra["l3_closing_speed_mps"] == 0.12
    assert extra["l3_force_inbound_speed_mps"] == 0.04
    assert extra["released_to_physics"] is True
    assert extra["release_reason"] == "last_free_hit"


def test_l12_episode_extra_copies_scheme_and_kinematic_spikes() -> None:
    extra = l12_episode_extra_from_info(
        {
            "l12_force_scheme": "newton_relative",
            "l12_kinematic_overlap_sim_steps": 4,
            "l12_kinematic_peak_impulse_ns": 9.0,
            "l12_kinematic_peak_force_n": 180.0,
            "l12_force_inbound_speed_mps": 0.25,
            "l12_partner_speed_mps": 0.08,
            "l12_closing_speed_mps": 0.18,
            "l12_force_from_newton": True,
            "released_to_physics": True,
            "release_reason": "impulse_hit",
            "ghost_contact_sim_steps": 4,
            "rollback_steps": 3,
        }
    )
    assert extra["l12_force_scheme"] == "newton_relative"
    assert extra["l12_kinematic_overlap_sim_steps"] == 4
    assert extra["l12_force_from_newton"] is True
    assert extra["l12_closing_speed_mps"] == 0.18
    assert extra["released_to_physics"] is True
    assert extra["release_reason"] == "impulse_hit"


def test_robotwin_yaml_default_l12_force_scheme_is_kinematic_first_hit() -> None:
    from dynisafety.config.loader import LoadRequest, load_config

    resolved = load_config(
        LoadRequest(backend="robotwin", level=2, suite="main20", model="random")
    )
    assert resolved.data["extra"]["robotwin"]["l12_force_scheme"] == "kinematic_first_hit"
    switched = load_config(
        LoadRequest(
            backend="robotwin",
            level=1,
            suite="main20",
            model="random",
            overrides=("extra.robotwin.l12_force_scheme=newton_relative",),
        )
    )
    ns = __import__(
        "dynisafety.backends.robotwin.config_bridge",
        fromlist=["build_disturbance_config"],
    ).build_disturbance_config(
        switched.run,
        extra_robotwin=switched.data["extra"]["robotwin"],
    )
    assert ns.l12_force_scheme == "newton_relative"


def test_after_step_l12_newton_kinematic_spike_is_ghost_and_requests_release() -> None:
    mgr = _stub_after_step_mgr(level=2)
    mgr.config.l12_force_scheme = "newton_relative"
    mgr._detect_obstacle_collision = (  # type: ignore[method-assign]
        lambda: (True, 0.02, "fl_link7", {"fl_link7"})
    )
    official = mgr.after_step(11)
    assert official is False
    assert mgr._collision_steps == 0
    assert mgr._max_collision_force == 0.0
    assert mgr._ghost_contact_sim_steps == 1
    assert mgr._l12_kinematic_overlap_sim_steps == 1
    assert mgr._l12_kinematic_peak_impulse_ns == 0.02
    assert mgr._l12_kinematic_peak_force_n == pytest.approx(0.02 / 0.05)
    assert mgr.controller.needs_dynamic_release() is True
    assert mgr.controller.release_reason == "impulse_hit"
    np.testing.assert_allclose(mgr._l12_force_linear_vel, np.array([0.25, 0.0, 0.0]))


def test_after_step_l12_newton_overwrites_sapien_spike_after_release() -> None:
    mgr = _stub_after_step_mgr(level=2)
    mgr.config.l12_force_scheme = "newton_relative"
    mgr._actor_is_kinematic = False
    v_obj = np.array([0.25, 0.0, 0.0])
    v_partner = np.zeros(3)
    normal = np.array([1.0, 0.0, 0.0])
    mgr._l12_relative_contact_frame = lambda: (v_obj, v_partner, normal)  # type: ignore[method-assign]
    mgr._detect_obstacle_collision = (  # type: ignore[method-assign]
        lambda: (True, 10.0, "fl_link7", {"fl_link7"})
    )
    official = mgr.after_step(20)
    expected_j = l3_newton_normal_impulse_ns(0.08, v_obj, v_partner, normal)
    assert official is True
    assert mgr._max_collision_impulse == pytest.approx(expected_j)
    assert mgr._max_collision_force == pytest.approx(expected_j / 0.05)
    assert mgr._max_collision_impulse < 1.0
    assert mgr._l12_force_from_newton is True
    assert mgr._l12_closing_speed_mps == pytest.approx(0.25)


def test_after_step_l1_newton_uses_live_closing_speed() -> None:
    mgr = _stub_after_step_mgr(level=1)
    mgr.config.l12_force_scheme = "newton_relative"
    mgr.config.obstacle_mass = 0.15
    mgr._actor_is_kinematic = False
    v_obj = np.array([0.0, 0.0, -0.8])
    v_partner = np.zeros(3)
    normal = np.array([0.0, 0.0, -1.0])
    mgr._l12_relative_contact_frame = lambda: (v_obj, v_partner, normal)  # type: ignore[method-assign]
    mgr._detect_obstacle_collision = (  # type: ignore[method-assign]
        lambda: (True, 40.0, "fl_link7", {"fl_link7"})
    )
    official = mgr.after_step(8)
    expected_j = l3_newton_normal_impulse_ns(0.15, v_obj, v_partner, normal)
    assert official is True
    assert mgr._max_collision_impulse == pytest.approx(expected_j)
    assert mgr._max_collision_impulse == pytest.approx(0.12)
    assert mgr._l12_force_from_newton is True


def test_l12_release_velocity_uses_snapshot_not_zeros() -> None:
    mgr = object.__new__(DisturbanceManager)
    mgr.config = SimpleNamespace(dynamic_level=2, l12_force_scheme="newton_relative")
    mgr._l12_force_linear_vel = np.array([0.3, 0.0, 0.0])
    np.testing.assert_allclose(mgr._l12_release_velocity(), np.array([0.3, 0.0, 0.0]))
    mgr._l12_force_linear_vel = None
    mgr.controller = None
    np.testing.assert_allclose(mgr._l12_release_velocity(), np.zeros(3))
