"""Shared L3 Newton helpers used by RoboTwin and SimplerEnv."""

from __future__ import annotations

import numpy as np
import pytest

from dynisafety.core.newton_force import (
    L3_RELEASE_MIN_SPEED_MPS,
    is_ghost_overlap,
    l3_closing_speed_mps,
    l3_newton_normal_impulse_ns,
    l3_release_linear_velocity,
    official_l3_contact,
)


def test_l3_newton_matches_mass_times_closing() -> None:
    v_hand = np.array([0.3, 0.0, 0.0])
    n = np.array([1.0, 0.0, 0.0])
    assert l3_newton_normal_impulse_ns(4.5, v_hand, np.zeros(3), n) == pytest.approx(1.35)
    assert l3_closing_speed_mps(v_hand, np.array([-0.1, 0.0, 0.0]), n) == pytest.approx(0.4)
    assert l3_newton_normal_impulse_ns(4.5, v_hand, v_hand, n) == pytest.approx(0.0)


def test_official_dynamic_release_gates_until_impulse() -> None:
    assert (
        official_l3_contact(
            geometric=True,
            impulse=0.0,
            force_scheme="dynamic_release",
            released_to_physics=False,
            require_impulse=True,
        )
        is False
    )
    assert (
        official_l3_contact(
            geometric=True,
            impulse=0.0,
            force_scheme="dynamic_release",
            released_to_physics=True,
            require_impulse=True,
        )
        is False
    )
    injected = l3_newton_normal_impulse_ns(
        4.5, np.array([0.3, 0.0, 0.0]), np.zeros(3), np.array([1.0, 0.0, 0.0])
    )
    assert (
        official_l3_contact(
            geometric=True,
            impulse=injected,
            force_scheme="dynamic_release",
            released_to_physics=True,
            require_impulse=True,
        )
        is True
    )
    assert is_ghost_overlap(geometric=True, official=False) is True


def test_release_velocity_prefers_last_free_then_path() -> None:
    vel = l3_release_linear_velocity(
        overlap_pos=np.array([0.02, 0.0, 0.0]),
        last_free_pos=np.zeros(3),
        path_velocity=np.array([0.3, 0.0, 0.0]),
        sim_dt=0.2,
        min_speed_mps=0.0,
    )
    assert vel == pytest.approx(np.array([0.1, 0.0, 0.0]))
    floored = l3_release_linear_velocity(
        overlap_pos=np.array([0.02, 0.0, 0.0]),
        last_free_pos=np.zeros(3),
        path_velocity=np.array([0.3, 0.0, 0.0]),
        sim_dt=0.2,
        min_speed_mps=L3_RELEASE_MIN_SPEED_MPS,
    )
    assert float(np.linalg.norm(floored)) == pytest.approx(L3_RELEASE_MIN_SPEED_MPS)
    path_only = l3_release_linear_velocity(
        overlap_pos=np.zeros(3),
        last_free_pos=None,
        path_velocity=np.array([0.3, 0.0, 0.0]),
        sim_dt=0.2,
        min_speed_mps=0.0,
    )
    assert path_only == pytest.approx(np.array([0.3, 0.0, 0.0]))
