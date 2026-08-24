"""Pure-function tests for LIBERO L3 reach-frame geometry (no simulator)."""

from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np
import pytest

from dynisafety.backends.libero.disturbance.l3_reach import (
    LEGACY_MODE,
    MODES,
    libero_l3_accept_reach,
    libero_l3_build_path,
    libero_l3_clip_xy,
    libero_l3_fallback_chain,
    libero_l3_far_edge,
    libero_l3_lateral_axis,
    libero_l3_opposite_xy,
    libero_l3_path_num_samples,
    libero_l3_pose_at,
    libero_l3_resolve_depth,
    libero_l3_select_mode,
    libero_l3_slot_xy,
)
from dynisafety.core.disturbance import minimum_jerk


def _dummy_candidate(mode: str, style: str) -> dict:
    if mode == LEGACY_MODE:
        start = np.array([0.45, 0.0, 1.0], dtype=np.float64)
        dwell = np.array([0.0, 0.0, 0.9], dtype=np.float64)
        return {
            "start": start,
            "dwell": dwell,
            "path": np.zeros((4, 3), dtype=np.float64),
            "path_style": "straight",
            "min_clearance_m": 0.0,
            "depth_collapsed": False,
            "slot_collapsed": False,
        }
    slot, _, depth = mode.partition("_")
    x = {"left": -1.0, "opposite": 0.0, "right": 1.0}[slot]
    z = 0.30 if depth == "near" else 0.50
    return {
        "start": np.array([x, 0.0, 1.0], dtype=np.float64),
        "dwell": np.array([0.0, 0.0, z], dtype=np.float64),
        "path": np.zeros((4, 3), dtype=np.float64),
        "path_style": style,
        "min_clearance_m": 0.05,
        "depth_collapsed": False,
        "slot_collapsed": False,
    }


def test_opposite_xy_is_clipped_reflection() -> None:
    robot = np.array([-0.8, 0.1], dtype=np.float64)
    table = np.array([0.0, 0.0], dtype=np.float64)
    half = np.array([0.45, 0.30], dtype=np.float64)
    got = libero_l3_opposite_xy(robot, table, half)
    reflected = 2.0 * table - robot
    expected = libero_l3_clip_xy(reflected, table, half)
    np.testing.assert_allclose(got, expected)
    assert got[0] == pytest.approx(0.45)


def test_far_edge_faces_away_from_robot() -> None:
    table = np.array([0.0, 0.0], dtype=np.float64)
    half = np.array([0.45, 0.30], dtype=np.float64)
    fwd = np.array([1.0, 0.0], dtype=np.float64)
    origin, edge_dir, half_len = libero_l3_far_edge(table, half, fwd)
    np.testing.assert_allclose(origin, [0.45, 0.0])
    assert abs(float(edge_dir[0])) == pytest.approx(0.0)
    assert half_len == pytest.approx(0.30)


def test_left_right_slide_on_far_edge() -> None:
    robot = np.array([-0.7, 0.0], dtype=np.float64)
    table = np.array([0.0, 0.0], dtype=np.float64)
    half = np.array([0.45, 0.30], dtype=np.float64)
    n_hat, degenerate = libero_l3_lateral_axis(robot, table)
    assert degenerate is False
    p_opp = libero_l3_opposite_xy(robot, table, half)
    left, _ = libero_l3_slot_xy(
        "left",
        opposite_xy=p_opp,
        table_center_xy=table,
        start_xy_half=half,
        n_hat=n_hat,
        lateral_width_m=0.15,
    )
    right, _ = libero_l3_slot_xy(
        "right",
        opposite_xy=p_opp,
        table_center_xy=table,
        start_xy_half=half,
        n_hat=n_hat,
        lateral_width_m=0.15,
    )
    opp, collapsed = libero_l3_slot_xy(
        "opposite",
        opposite_xy=p_opp,
        table_center_xy=table,
        start_xy_half=half,
        n_hat=n_hat,
        lateral_width_m=0.15,
    )
    assert collapsed is False
    np.testing.assert_allclose(opp, p_opp)
    assert left[0] == pytest.approx(p_opp[0])
    assert right[0] == pytest.approx(p_opp[0])
    assert left[1] == pytest.approx(p_opp[1] + 0.15)
    assert right[1] == pytest.approx(p_opp[1] - 0.15)


def test_depth_collapses_on_short_corridor() -> None:
    eef = np.array([0.0, 0.0], dtype=np.float64)
    task = np.array([0.15, 0.0], dtype=np.float64)
    alpha, collapsed = libero_l3_resolve_depth(
        "near",
        eef_xy=eef,
        task_xy=task,
        near_alpha=0.30,
        mid_alpha=0.50,
        min_eef_clearance=0.12,
        min_task_clearance=0.0,
        collapse_m=0.08,
    )
    assert collapsed is True
    assert alpha == pytest.approx(0.50)
    alpha2, collapsed2 = libero_l3_resolve_depth(
        "near",
        eef_xy=eef,
        task_xy=np.array([0.40, 0.0], dtype=np.float64),
        near_alpha=0.30,
        mid_alpha=0.50,
        min_eef_clearance=0.12,
        min_task_clearance=0.0,
        collapse_m=0.08,
    )
    assert collapsed2 is False
    assert alpha2 == pytest.approx(0.30)


def test_straight_path_matches_min_jerk_lerp() -> None:
    start = np.array([0.45, 0.0, 1.00], dtype=np.float64)
    dwell = np.array([0.10, 0.0, 0.90], dtype=np.float64)
    n = libero_l3_path_num_samples(10)
    path = libero_l3_build_path(start, dwell, "straight", n_samples=n)
    assert path.shape == (11, 3)
    alphas = np.linspace(0.0, 1.0, n)
    for a, pos in zip(alphas, path, strict=True):
        expected = start + minimum_jerk(float(a)) * (dwell - start)
        np.testing.assert_allclose(pos, expected)
    np.testing.assert_allclose(libero_l3_pose_at(start, dwell, 0.0, "straight"), start)
    np.testing.assert_allclose(libero_l3_pose_at(start, dwell, 1.0, "straight"), dwell)


def test_raised_lifts_only_start() -> None:
    start = np.array([0.45, 0.0, 1.00], dtype=np.float64)
    dwell = np.array([0.10, 0.0, 0.90], dtype=np.float64)
    path = libero_l3_build_path(
        start, dwell, "raised", n_samples=5, hover_rise_m=0.10
    )
    assert path[0, 2] == pytest.approx(1.10)
    np.testing.assert_allclose(path[-1], dwell)


def test_via_cruises_then_drops() -> None:
    start = np.array([0.45, 0.0, 1.00], dtype=np.float64)
    dwell = np.array([0.10, 0.0, 0.90], dtype=np.float64)
    descent = 0.20
    path = libero_l3_build_path(
        start,
        dwell,
        "via",
        n_samples=21,
        hover_rise_m=0.10,
        via_descent_alpha=descent,
    )
    cruise_z = 1.10
    alphas = np.linspace(0.0, 1.0, len(path))
    cruise_end = 1.0 - descent
    for a, pos in zip(alphas, path, strict=True):
        if a <= cruise_end + 1e-12:
            assert pos[2] == pytest.approx(cruise_z)
        else:
            assert pos[2] <= cruise_z + 1e-12
    np.testing.assert_allclose(path[-1], dwell)


def test_select_mode_round_robin_balanced() -> None:
    counts = Counter(
        libero_l3_select_mode(7, 2, i, forbidden_slots=(), forbidden_depths=())
        for i in range(12)
    )
    assert set(counts) == set(MODES)
    assert set(counts.values()) == {2}


def test_select_mode_stable_across_hash_seed() -> None:
    a = libero_l3_select_mode(3, "goal", 4, forbidden_slots=(), forbidden_depths=())
    b = libero_l3_select_mode(3, "goal", 4, forbidden_slots=(), forbidden_depths=())
    assert a == b
    key = b"3|goal"
    rotation = int.from_bytes(hashlib.sha1(key).digest()[:4], "big") % len(MODES)
    assert a == MODES[(4 + rotation) % len(MODES)]


def test_forbidden_depths_drop_near_from_pool() -> None:
    modes = [
        libero_l3_select_mode(1, 0, i, forbidden_slots=(), forbidden_depths=("near",))
        for i in range(6)
    ]
    assert all(m.endswith("_mid") or m == LEGACY_MODE for m in modes)
    assert "left_near" not in modes


def test_fallback_keeps_cell_before_crossing_slot() -> None:
    chain = libero_l3_fallback_chain(
        "left_near", forbidden_slots=(), forbidden_depths=()
    )
    assert chain[0] == ("left_near", "straight")
    assert chain[1] == ("left_near", "raised")
    assert chain[2] == ("left_near", "via")
    assert chain[3] == ("right_near", "straight")
    assert chain[4] == ("opposite_mid", "straight")
    assert chain[-1] == (LEGACY_MODE, None)


def test_fallback_opposite_tries_left_first() -> None:
    chain = libero_l3_fallback_chain(
        "opposite_mid", forbidden_slots=(), forbidden_depths=()
    )
    modes = [item[0] for item in chain]
    assert modes[3] == "left_mid"
    assert modes[-1] == LEGACY_MODE


def test_accept_same_cell_raised_after_straight_reject() -> None:
    seen: list[str] = []

    def validate(_start, _path) -> bool:
        seen.append("x")
        return len(seen) > 1

    result = libero_l3_accept_reach(
        "left_mid",
        forbidden_slots=(),
        forbidden_depths=(),
        make_candidate=_dummy_candidate,
        validate=validate,
    )
    assert result["requested_mode"] == "left_mid"
    assert result["accepted_mode"] == "left_mid"
    assert result["path_style"] == "raised"
    assert result["fallback_reason"] == "candidate_rejected"
    assert result["fallback_chain"][0] == "left_mid:straight"
    assert result["fallback_chain"][-1] == "left_mid:raised"


def test_accept_all_rejected_uses_legacy() -> None:
    result = libero_l3_accept_reach(
        "right_near",
        forbidden_slots=(),
        forbidden_depths=(),
        make_candidate=_dummy_candidate,
        validate=lambda _s, _p: False,
    )
    assert result["accepted_mode"] == LEGACY_MODE
    assert result["fallback_reason"] == "all_candidates_rejected"
    assert result["fallback_chain"][-1] == LEGACY_MODE
