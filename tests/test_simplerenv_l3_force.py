"""SimplerEnv L3 dynamic_release force pipeline (no live SAPIEN)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dynisafety.backends.simplerenv.disturbance.l3 import SharedReachingController
from dynisafety.core.contact import CONTACT_IMPULSE_EPS
from dynisafety.core.disturbance import Phase
from dynisafety.core.newton_force import l3_newton_normal_impulse_ns


def _l3_ctrl() -> SharedReachingController:
    ctrl = object.__new__(SharedReachingController)
    ctrl.actor = SimpleNamespace(
        name="hand",
        get_pose=lambda: SimpleNamespace(p=np.array([0.1, 0.0, 0.2]), q=np.array([1.0, 0, 0, 0])),
        set_pose=lambda *_a, **_k: None,
    )
    ctrl._force_scheme = "dynamic_release"
    ctrl._pending_release = False
    ctrl._released_to_physics = False
    ctrl.last_free_pos = np.array([0.0, 0.0, 0.2])
    ctrl._last_free_quat = np.array([1.0, 0.0, 0.0, 0.0])
    ctrl._arm_quat = np.array([1.0, 0.0, 0.0, 0.0])
    ctrl._phase = Phase.APPROACH
    ctrl.blocked = False
    ctrl._control_freq = 5
    ctrl._start_pos = np.array([0.0, 0.0, 0.2])
    ctrl._dwell_pos = np.array([0.3, 0.0, 0.15])
    ctrl._entry_far_pos = np.array([-0.1, 0.0, 0.2])
    ctrl._approach_steps = 10
    ctrl._entry_steps = 3
    return ctrl


def test_l3_controller_requests_release_not_flinch() -> None:
    ctrl = _l3_ctrl()
    ctrl.request_dynamic_release()
    assert ctrl.needs_physics_release is True
    assert ctrl._phase == Phase.APPROACH
    ctrl.on_released_to_physics()
    assert ctrl.released_to_physics is True
    assert ctrl.needs_physics_release is False
    ctrl.vanish()
    assert ctrl._phase == Phase.DONE
    assert ctrl.blocked is True


def test_l3_path_velocity_uses_existing_approach_clock() -> None:
    ctrl = _l3_ctrl()
    vel = ctrl.path_velocity()
    expected = (ctrl._dwell_pos - ctrl._start_pos) / (10 * 0.2)
    assert vel == pytest.approx(expected)


def test_wrapper_newton_inject_once_when_sapien_zero() -> None:
    from dynisafety.backends.simplerenv.wrapper import DyniSafetyWrapper

    wrap = object.__new__(DyniSafetyWrapper)
    wrap._level = 3
    wrap._resolved_extra = {"simplerenv": {"l3_force_scheme": "dynamic_release"}}
    wrap._lifecycle = SimpleNamespace(is_kinematic=False, actor=SimpleNamespace(name="hand"))
    wrap._hand_mass_kg = 4.5
    wrap._l3_force_linear_vel = np.array([0.3, 0.0, 0.0])
    wrap._l3_extra = {}
    wrap._l3_relative_contact_frame = lambda: (  # type: ignore[method-assign]
        np.array([1.0, 0.0, 0.0]),
        np.zeros(3),
    )
    injected = wrap._maybe_inject_l3_newton(True, 0.0)
    assert injected == pytest.approx(1.35)
    assert wrap._l3_extra["l3_force_from_release_momentum"] is True
    assert wrap._l3_force_linear_vel is None
    again = wrap._maybe_inject_l3_newton(True, 0.0)
    assert again == pytest.approx(0.0)


def test_wrapper_keeps_nonzero_sapien_impulse() -> None:
    from dynisafety.backends.simplerenv.wrapper import DyniSafetyWrapper

    wrap = object.__new__(DyniSafetyWrapper)
    wrap._level = 3
    wrap._resolved_extra = {"simplerenv": {"l3_force_scheme": "dynamic_release"}}
    wrap._lifecycle = SimpleNamespace(is_kinematic=False, actor=None)
    wrap._hand_mass_kg = 4.5
    wrap._l3_force_linear_vel = np.array([0.3, 0.0, 0.0])
    wrap._l3_extra = {}
    kept = wrap._maybe_inject_l3_newton(True, 0.5)
    assert kept == pytest.approx(0.5)
    assert wrap._l3_force_linear_vel is not None


def test_hand_rgba_matches_robotwin_skin() -> None:
    from dynisafety.backends.simplerenv.actor import HAND_SKIN_COLOR, hand_rgba

    assert hand_rgba([0.78, 0.60, 0.48]) == [0.78, 0.60, 0.48, 1.0]
    assert hand_rgba(HAND_SKIN_COLOR) == [0.78, 0.60, 0.48, 1.0]
    with pytest.raises(ValueError, match="RGB or RGBA"):
        hand_rgba([0.78, 0.60])


def test_seal_released_l3_hand_vanishes_on_miss() -> None:
    from dynisafety.backends.simplerenv.wrapper import DyniSafetyWrapper

    wrap = object.__new__(DyniSafetyWrapper)
    wrap._level = 3
    wrap._resolved_extra = {"simplerenv": {"l3_force_scheme": "dynamic_release"}}
    destroyed: list[bool] = []
    wrap._lifecycle = SimpleNamespace(
        is_kinematic=False,
        actor=SimpleNamespace(name="hand"),
        destroy=lambda: destroyed.append(True),
    )
    wrap._controller = SimpleNamespace(vanish=lambda: None)
    wrap._hand_mass_kg = 4.5
    wrap._l3_force_linear_vel = np.array([0.3, 0.0, 0.0])
    wrap._l3_extra = {}
    wrap._l3_official_force_done = False
    wrap._force_dt = 0.01
    wrap._l3_relative_contact_frame = lambda: (  # type: ignore[method-assign]
        np.array([1.0, 0.0, 0.0]),
        np.zeros(3),
    )
    metrics = wrap._seal_released_l3_hand(
        {"in_contact": False, "force_n": 0.0, "impulse_ns": 0.0}
    )
    assert metrics["in_contact"] is True
    assert metrics["force_n"] == pytest.approx(135.0)
    assert wrap._l3_official_force_done is True
    assert destroyed == [True]


def test_seal_released_l3_hand_keeps_kinematic() -> None:
    from dynisafety.backends.simplerenv.wrapper import DyniSafetyWrapper

    wrap = object.__new__(DyniSafetyWrapper)
    wrap._level = 3
    wrap._resolved_extra = {"simplerenv": {"l3_force_scheme": "dynamic_release"}}
    wrap._lifecycle = SimpleNamespace(is_kinematic=True, destroy=lambda: None)
    wrap._controller = None
    wrap._l3_official_force_done = False
    metrics = wrap._seal_released_l3_hand({"in_contact": False, "force_n": 0.0})
    assert metrics["in_contact"] is False
    assert wrap._l3_official_force_done is False


def test_ghost_kinematic_overlap_is_not_official() -> None:
    from dynisafety.core.newton_force import official_l3_contact

    assert (
        official_l3_contact(
            geometric=True,
            impulse=0.01,
            force_scheme="dynamic_release",
            released_to_physics=False,
            require_impulse=True,
            impulse_eps=CONTACT_IMPULSE_EPS,
        )
        is False
    )
    j = l3_newton_normal_impulse_ns(4.5, np.array([0.3, 0, 0]), np.zeros(3), np.array([1, 0, 0]))
    assert j / 0.01 == pytest.approx(135.0)
