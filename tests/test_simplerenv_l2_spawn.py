"""Pure-function tests for SimplerEnv L2 spawn-frame geometry."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from dynisafety.backends.simplerenv.disturbance.l2_spawn import (
    LEGACY_MODE,
    MODES,
    simplerenv_l2_accept_spawn,
    simplerenv_l2_build_path,
    simplerenv_l2_candidate_start_xy,
    simplerenv_l2_fallback_chain,
    simplerenv_l2_lateral_axis,
    simplerenv_l2_path_num_samples,
    simplerenv_l2_select_mode,
    simplerenv_l2_solve_linear_start_z,
    simplerenv_l2_solve_trapezoid,
)


def test_edge_axis_is_perpendicular_to_center_landing_ray() -> None:
    n_hat, degenerate = simplerenv_l2_lateral_axis(
        [0.0, 0.0], [1.0, 1.0], [0.8, 0.5], [1.1, 0.5]
    )
    assert degenerate is False
    ray = np.array([0.8, 0.5]) - np.array([0.5, 0.5])
    ray = ray / np.linalg.norm(ray)
    assert abs(float(n_hat @ ray)) == pytest.approx(0.0, abs=1e-9)
    assert np.linalg.norm(n_hat) == pytest.approx(1.0)


def test_slot_offsets_along_axis() -> None:
    p0 = np.array([1.0, 0.0])
    n_hat = np.array([0.0, 1.0])
    assert simplerenv_l2_candidate_start_xy(p0, n_hat, "center", 0.15).tolist() == [1.0, 0.0]
    left = simplerenv_l2_candidate_start_xy(p0, n_hat, "left", 0.15)
    right = simplerenv_l2_candidate_start_xy(p0, n_hat, "right", 0.15)
    assert left[1] == pytest.approx(0.15)
    assert right[1] == pytest.approx(-0.15)


def test_linear_solve_and_trapezoid() -> None:
    def top(xy):
        return 0.4 if 0.2 <= float(xy[0]) <= 0.6 else 0.0

    landing = np.array([1.0, 0.0, 0.0])
    z, mode = simplerenv_l2_solve_linear_start_z(
        [0.0, 0.0],
        landing,
        top,
        clearance_m=0.05,
        tail_alpha=0.15,
        max_start_rise_m=1.20,
        samples=24,
        p0_z_hint=0.0,
    )
    assert mode == "straight"
    assert z is not None and z > 0.4
    z_c, trap = simplerenv_l2_solve_trapezoid(
        [0.0, 0.0],
        landing,
        top,
        clearance_m=0.05,
        max_start_rise_m=0.10,
        descent_alpha=0.2,
        samples=24,
    )
    assert z_c is None
    assert trap == "trapezoid_infeasible"


def test_path_count_and_fallback_keeps_slot() -> None:
    assert simplerenv_l2_path_num_samples(0, 15) == 16
    chain = simplerenv_l2_fallback_chain("right_linear", forbidden_slots=())
    assert chain[0] == "right_linear"
    assert chain[1] == "right_arc"
    assert chain[-1] == LEGACY_MODE
    path = simplerenv_l2_build_path([0, 0, 0.2], [1, 0, 0.0], "arc", n_samples=5, arc_height=0.1)
    assert path.shape == (5, 3)
    assert path[2, 2] == pytest.approx(0.1 + 0.1)  # mid lerp z=0.1 plus arc peak 0.1


def test_mode_round_robin_balanced() -> None:
    picks = [
        simplerenv_l2_select_mode(7, "bridge", i, forbidden_slots=()) for i in range(12)
    ]
    counts = Counter(picks)
    assert set(picks) <= set(MODES)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_accept_falls_back_when_all_rejected() -> None:
    def make(mode: str):
        if mode == LEGACY_MODE:
            return {"start": np.zeros(3), "path": np.zeros((3, 3)), "linear_solve_mode": "na"}
        return {"start": np.ones(3), "path": np.ones((3, 3)), "linear_solve_mode": "na"}

    out = simplerenv_l2_accept_spawn(
        "left_arc",
        forbidden_slots=(),
        make_candidate=make,
        validate=lambda start, path: False,
    )
    assert out["accepted_mode"] == LEGACY_MODE
    assert out["fallback_reason"] == "all_candidates_rejected"


def test_assemble_requires_spawn_diversity_block() -> None:
    from dynisafety.backends.simplerenv.disturbance.assemble import _l2_spawn_diversity

    with pytest.raises(ValueError, match="extra.simplerenv"):
        _l2_spawn_diversity({})
    with pytest.raises(ValueError, match="spawn_diversity"):
        _l2_spawn_diversity({"simplerenv": {"l2": {}}})
