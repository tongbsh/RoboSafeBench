"""P2: L1/L2 impulse gate, last-free-pose rollback, L3 Scheme B unchanged."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dynisafety.backends.robotwin.contact import (
    SAPIEN_CONTACT_IMPULSE_EPS,
    is_ghost_overlap,
    official_l12_contact,
    should_skip_dynamic_release,
)
from dynisafety.backends.robotwin.disturbance.l2 import SlidingObstacleController
from dynisafety.backends.robotwin.disturbance.manager import DisturbanceManager


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
    mgr.config = SimpleNamespace(dynamic_level=3)
    assert mgr._official_contact_this_step(True, 0.0) is True
    assert mgr._official_contact_this_step(False, 4.0) is False


def _stub_after_step_mgr(*, level: int) -> DisturbanceManager:
    mgr = object.__new__(DisturbanceManager)
    mgr.config = SimpleNamespace(
        dynamic_level=level,
        l12_contact_require_impulse=True,
        l2_keep_last_free_pose=True,
        topple_threshold_deg=30.0,
    )
    mgr.controller = _sliding_ctrl()
    mgr.controller._active = True
    mgr.actor = SimpleNamespace(get_name=lambda: "obstacle")
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
    mgr._l3_post_first_overlap_sim_steps = 0
    mgr._l2_post_freeze_overlap_sim_steps = 0
    mgr._l2_post_freeze_peak_force_n = 0.0
    mgr._l2_post_freeze_peak_impulse_ns = 0.0
    mgr._release_reason = None
    mgr._refresh_grasped_collision_targets = lambda: None  # type: ignore[method-assign]
    mgr._update_topple_metrics = lambda: None  # type: ignore[method-assign]
    mgr._restore_last_free_pose = lambda: True  # type: ignore[method-assign]
    mgr._snapshot_last_free_pose = lambda: None  # type: ignore[method-assign]
    mgr._force_equiv_timestep = lambda: 0.05  # type: ignore[method-assign]
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

    @property
    def active(self) -> bool:
        return self._active

    def handle_collision(self, sim_step: int) -> None:
        del sim_step
        self.hits += 1


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


def test_manager_l12_official_uses_impulse_gate() -> None:
    mgr = object.__new__(DisturbanceManager)
    mgr.config = SimpleNamespace(dynamic_level=2, l12_contact_require_impulse=True)
    assert mgr._official_contact_this_step(True, 0.0) is False
    assert mgr._official_contact_this_step(True, 0.01) is True
    mgr.config.dynamic_level = 1
    assert mgr._official_contact_this_step(True, 0.0) is False


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
    mgr.config = SimpleNamespace(dynamic_level=2, topple_threshold_deg=30.0)
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


def test_robotwin_yaml_exposes_p2_scene_flags() -> None:
    from dynisafety.config.loader import LoadRequest, load_config

    resolved = load_config(
        LoadRequest(backend="robotwin", level=2, suite="main20", model="random")
    )
    scene = resolved.data["scene"]
    assert scene["l12_contact_require_impulse"] is True
    assert scene["l2_keep_last_free_pose"] is True
    assert scene["l2_release_on_penetration"] is False
    ns = __import__(
        "dynisafety.backends.robotwin.config_bridge",
        fromlist=["build_disturbance_config"],
    ).build_disturbance_config(
        resolved.run,
        extra_robotwin=resolved.data["extra"]["robotwin"],
    )
    assert ns.l12_contact_require_impulse is True
    assert ns.l2_keep_last_free_pose is True
    assert ns.l2_release_on_penetration is False
