"""Pure-function tests for LIBERO L2 spawn-frame geometry (no simulator)."""
from __future__ import annotations

import os
import subprocess
import sys
from collections import Counter

import numpy as np
import pytest

from dynisafety.backends.libero.disturbance.l2_spawn import (
    LEGACY_MODE,
    MODES,
    PATH_STYLES,
    SLOTS,
    libero_l2_accept_spawn,
    libero_l2_build_path,
    libero_l2_candidate_start_xy,
    libero_l2_fallback_chain,
    libero_l2_lateral_axis,
    libero_l2_path_num_samples,
    libero_l2_select_mode,
    libero_l2_solve_linear_start_z,
    libero_l2_solve_trapezoid,
)

# Mirrors l2.yaml move window (10 → 30) so path-count tests match production.
_MOVE_START = 10
_MOVE_END = 30


def _slot_of(mode: str) -> str:
    return mode.split("_", 1)[0]


def _reference_arc_path(start: np.ndarray, landing: np.ndarray, n: int, arc_height: float) -> np.ndarray:
    """Local copy of ``l2.py`` ``_build_arc_path`` (do not import l2.py)."""
    n = max(2, int(n))
    alphas = np.linspace(0.0, 1.0, n)
    path = np.empty((n, 3), dtype=np.float64)
    for i, a in enumerate(alphas):
        pos = start + a * (landing - start)
        pos[2] += 4.0 * arc_height * a * (1.0 - a)
        path[i] = pos
    return path


def _cabinet_top(
    xy: np.ndarray,
    *,
    height: float,
    x0: float = 0.35,
    x1: float = 0.55,
    floor: float = 0.0,
) -> float:
    x = float(np.asarray(xy, dtype=np.float64).reshape(-1)[0])
    if x0 <= x <= x1:
        return height
    return floor


def _dummy_candidate(mode: str) -> dict:
    if mode == LEGACY_MODE:
        start = np.array([0.0, 0.0, 0.05], dtype=np.float64)
        return {
            "start": start,
            "path": np.zeros((4, 3), dtype=np.float64),
            "linear_solve_mode": "na",
            "min_clearance_m": 0.0,
        }
    slot = _slot_of(mode)
    x = {"left": -1.0, "center": 0.0, "right": 1.0}[slot]
    return {
        "start": np.array([x, 0.0, 0.20], dtype=np.float64),
        "path": np.zeros((4, 3), dtype=np.float64),
        "linear_solve_mode": "straight" if mode.endswith("linear") else "na",
        "min_clearance_m": 0.05,
    }


# --- lateral axis -----------------------------------------------------------


def test_lateral_axis_orthogonal_to_forward() -> None:
    base = np.array([0.0, 0.0], dtype=np.float64)
    table = np.array([1.0, 0.0], dtype=np.float64)
    n_hat, degenerate = libero_l2_lateral_axis(base, table)
    fwd = table - base
    assert degenerate is False
    assert n_hat.shape == (2,)
    assert not np.isnan(n_hat).any()
    assert n_hat @ fwd == pytest.approx(0.0, abs=1e-12)
    assert float(np.linalg.norm(n_hat)) == pytest.approx(1.0)
    np.testing.assert_allclose(n_hat, [0.0, 1.0])


def test_left_right_symmetric_and_collinear() -> None:
    p0 = np.array([0.4, -0.1], dtype=np.float64)
    n_hat, _ = libero_l2_lateral_axis([0.0, 0.0], [0.0, 1.0])
    width = 0.15
    left = libero_l2_candidate_start_xy(p0, n_hat, "left", width)
    center = libero_l2_candidate_start_xy(p0, n_hat, "center", width)
    right = libero_l2_candidate_start_xy(p0, n_hat, "right", width)
    np.testing.assert_allclose(center, p0)
    np.testing.assert_allclose((left + right) / 2.0, p0)
    np.testing.assert_allclose(left - p0, -(right - p0))
    # 2-D cross product of (left-p0) and (right-p0) is 0 ⇒ collinear with P0.
    offset_l = left - p0
    offset_r = right - p0
    assert offset_l[0] * offset_r[1] - offset_l[1] * offset_r[0] == pytest.approx(0.0)
    # Three points share one line: (right-left) ∥ n_hat.
    span = right - left
    assert abs(float(span @ np.array([-n_hat[1], n_hat[0]]))) == pytest.approx(0.0)


def test_degenerate_axis_when_base_equals_table_center() -> None:
    n_hat, degenerate = libero_l2_lateral_axis([0.25, 0.5], [0.25, 0.5])
    assert degenerate is True
    assert not np.isnan(n_hat).any()
    np.testing.assert_allclose(n_hat, [0.0, 1.0])
    left = libero_l2_candidate_start_xy([0.0, 0.0], n_hat, "left", 0.1)
    right = libero_l2_candidate_start_xy([0.0, 0.0], n_hat, "right", 0.1)
    np.testing.assert_allclose(left, [0.0, 0.1])
    np.testing.assert_allclose(right, [0.0, -0.1])


# --- linear invert / trapezoid ----------------------------------------------


def test_linear_solve_clears_step_cabinet() -> None:
    start_xy = np.array([0.0, 0.0], dtype=np.float64)
    landing = np.array([1.0, 0.0, 0.05], dtype=np.float64)
    clearance = 0.03
    tail = 0.15
    samples = 24
    cabinet_h = 0.30
    # Path is x=α. Pin cabinet edges to the solver grid so the worst-case
    # α (right edge of a high plateau) is actually constrained.
    grid = np.linspace(0.0, 1.0 - tail, samples)
    x0 = float(grid[8])
    x1 = float(grid[14])

    def top(xy: np.ndarray) -> float:
        return _cabinet_top(xy, height=cabinet_h, x0=x0, x1=x1)

    z_start, mode = libero_l2_solve_linear_start_z(
        start_xy,
        landing,
        top,
        clearance_m=clearance,
        tail_alpha=tail,
        max_start_rise_m=0.80,
        samples=samples,
        p0_z_hint=float(landing[2]),
    )
    assert mode == "straight"
    assert z_start is not None
    z_land = float(landing[2])
    for alpha in np.linspace(0.0, 1.0 - tail, 64):
        a = float(alpha)
        xy = (1.0 - a) * start_xy + a * landing[:2]
        z = (1.0 - a) * z_start + a * z_land
        assert z + 1e-9 >= top(xy) + clearance


def test_linear_solve_infeasible_when_rise_exceeds_cap() -> None:
    start_xy = np.array([0.0, 0.0], dtype=np.float64)
    landing = np.array([1.0, 0.0, 0.05], dtype=np.float64)

    def top(xy: np.ndarray) -> float:
        return _cabinet_top(xy, height=1.00)

    z_start, mode = libero_l2_solve_linear_start_z(
        start_xy,
        landing,
        top,
        clearance_m=0.03,
        tail_alpha=0.15,
        max_start_rise_m=0.35,
        samples=24,
        p0_z_hint=float(landing[2]),
    )
    assert z_start is None
    assert mode == "linear_solve_infeasible"


def test_trapezoid_cruise_then_terminal_descent() -> None:
    start_xy = np.array([0.0, 0.0], dtype=np.float64)
    landing = np.array([1.0, 0.0, 0.05], dtype=np.float64)
    clearance = 0.03
    descent = 0.20
    cabinet_h = 0.30

    def top(xy: np.ndarray) -> float:
        return _cabinet_top(xy, height=cabinet_h)

    z_cruise, mode = libero_l2_solve_trapezoid(
        start_xy,
        landing,
        top,
        clearance_m=clearance,
        max_start_rise_m=0.35,
        descent_alpha=descent,
        samples=24,
    )
    assert mode == "trapezoid"
    assert z_cruise is not None
    max_top = max(
        top((1.0 - a) * start_xy + a * landing[:2])
        for a in np.linspace(0.0, 1.0, 24)
    )
    assert z_cruise + 1e-12 >= max_top + clearance

    start = np.array([start_xy[0], start_xy[1], z_cruise], dtype=np.float64)
    path = libero_l2_build_path(
        start,
        landing,
        "trapezoid",
        n_samples=21,
        arc_height=0.0,
        cruise_z=z_cruise,
        descent_alpha=descent,
    )
    alphas = np.linspace(0.0, 1.0, len(path))
    cruise_end = 1.0 - descent
    for a, pos in zip(alphas, path, strict=True):
        if a <= cruise_end + 1e-12:
            assert pos[2] == pytest.approx(z_cruise)
        else:
            assert pos[2] < z_cruise + 1e-12
    np.testing.assert_allclose(path[-1], landing)
    # Descent is confined to the last descent_alpha fraction of α.
    assert all(pos[2] == pytest.approx(z_cruise) for a, pos in zip(alphas, path, strict=True) if a <= cruise_end)


def test_trapezoid_infeasible_when_rise_exceeds_cap() -> None:
    start_xy = np.array([0.0, 0.0], dtype=np.float64)
    landing = np.array([1.0, 0.0, 0.05], dtype=np.float64)

    def top(xy: np.ndarray) -> float:
        return _cabinet_top(xy, height=1.00)

    z_cruise, mode = libero_l2_solve_trapezoid(
        start_xy,
        landing,
        top,
        clearance_m=0.03,
        max_start_rise_m=0.35,
        descent_alpha=0.20,
        samples=24,
    )
    assert z_cruise is None
    assert mode == "trapezoid_infeasible"


# --- path construction ------------------------------------------------------


def test_arc_matches_l2_build_arc_path_formula() -> None:
    start = np.array([0.50, -0.10, 0.05], dtype=np.float64)
    landing = np.array([0.00, 0.00, 0.05], dtype=np.float64)
    n = libero_l2_path_num_samples(_MOVE_START, _MOVE_END)
    height = 0.2
    got = libero_l2_build_path(start, landing, "arc", n_samples=n, arc_height=height)
    expected = _reference_arc_path(start, landing, n, height)
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=0.0)
    linear = libero_l2_build_path(start, landing, "linear", n_samples=n, arc_height=height)
    np.testing.assert_allclose(linear, _reference_arc_path(start, landing, n, 0.0))


def test_path_sample_count_matches_move_window() -> None:
    n = libero_l2_path_num_samples(_MOVE_START, _MOVE_END)
    assert n == _MOVE_END - _MOVE_START + 1
    start = np.array([0.5, -0.1, 0.1], dtype=np.float64)
    landing = np.array([0.0, 0.0, 0.05], dtype=np.float64)
    for style, kwargs in (
        ("arc", {"arc_height": 0.2}),
        ("linear", {"arc_height": 0.2}),
        ("trapezoid", {"arc_height": 0.0, "cruise_z": 0.25, "descent_alpha": 0.2}),
    ):
        path = libero_l2_build_path(start, landing, style, n_samples=n, **kwargs)
        assert path.shape == (n, 3)
    assert libero_l2_path_num_samples(5, 4) == 2


# --- fallback / accept ------------------------------------------------------


def test_fallback_left_arc_rejected_lands_on_left_linear() -> None:
    seen: list[str] = []

    def validate(_start, _path) -> bool:
        # Reject the requested style once; the next offer is same-slot linear.
        seen.append("x")
        return len(seen) > 1

    result = libero_l2_accept_spawn(
        "left_arc",
        forbidden_slots=(),
        make_candidate=_dummy_candidate,
        validate=validate,
    )
    assert result["requested_mode"] == "left_arc"
    assert result["accepted_mode"] == "left_linear"
    assert result["fallback_reason"] == "candidate_rejected"
    assert result["fallback_chain"][0] == "left_arc"
    assert result["fallback_chain"][-1] == "left_linear"
    assert "legacy_jitter" not in result["fallback_chain"]


def test_fallback_chain_keeps_slot_before_crossing() -> None:
    assert libero_l2_fallback_chain("right_linear", forbidden_slots=()) == (
        "right_linear",
        "right_arc",
        "left_linear",
        "center_arc",
        LEGACY_MODE,
    )
    assert libero_l2_fallback_chain("left_arc", forbidden_slots=()) == (
        "left_arc",
        "left_linear",
        "right_arc",
        "center_arc",
        LEGACY_MODE,
    )
    assert libero_l2_fallback_chain("center_linear", forbidden_slots=()) == (
        "center_linear",
        "center_arc",
        "left_linear",
        LEGACY_MODE,
    )


def test_accept_spawn_distinguishes_infeasible_reasons() -> None:
    def _none(_mode: str) -> None:
        return None

    none_hit = libero_l2_accept_spawn(
        "left_linear",
        forbidden_slots=(),
        make_candidate=_none,
        validate=lambda _s, _p: True,
    )
    assert none_hit["accepted_mode"] == LEGACY_MODE
    assert none_hit["fallback_reason"] == "all_candidates_rejected"

    def _trap_then_ok(mode: str):
        if mode == "left_linear":
            return {"infeasible_reason": "trapezoid_infeasible"}
        return _dummy_candidate(mode)

    trap = libero_l2_accept_spawn(
        "left_linear",
        forbidden_slots=(),
        make_candidate=_trap_then_ok,
        validate=lambda _s, _p: True,
    )
    assert trap["accepted_mode"] == "left_arc"
    assert trap["fallback_reason"] == "trapezoid_infeasible"

    def _linear_none_then_ok(mode: str):
        if mode == "left_linear":
            return None
        return _dummy_candidate(mode)

    linear = libero_l2_accept_spawn(
        "left_linear",
        forbidden_slots=(),
        make_candidate=_linear_none_then_ok,
        validate=lambda _s, _p: True,
    )
    assert linear["accepted_mode"] == "left_arc"
    assert linear["fallback_reason"] == "linear_solve_infeasible"


def test_fallback_all_rejected_uses_legacy() -> None:
    result = libero_l2_accept_spawn(
        "left_linear",
        forbidden_slots=(),
        make_candidate=_dummy_candidate,
        validate=lambda _start, _path: False,
    )
    assert result["accepted_mode"] == LEGACY_MODE
    assert result["fallback_reason"] == "all_candidates_rejected"
    assert result["fallback_chain"][-1] == LEGACY_MODE
    assert LEGACY_MODE in result["fallback_chain"]


def test_fallback_chain_finite_unique_ends_legacy() -> None:
    for mode in MODES:
        chain = libero_l2_fallback_chain(mode, forbidden_slots=())
        assert chain[-1] == LEGACY_MODE
        assert len(chain) == len(set(chain))
        assert 2 <= len(chain) <= 5
        assert chain[0] == mode
    assert libero_l2_fallback_chain(LEGACY_MODE, forbidden_slots=()) == (LEGACY_MODE,)


def test_forbidden_left_absent_from_select_and_chain() -> None:
    forbidden = ["left"]
    for episode in range(24):
        mode = libero_l2_select_mode(7, "spatial_0", episode, forbidden_slots=forbidden)
        assert _slot_of(mode) != "left"
        assert mode != "left_arc" and mode != "left_linear"
    for requested in MODES:
        chain = libero_l2_fallback_chain(requested, forbidden_slots=forbidden)
        assert all(_slot_of(m) != "left" or m == LEGACY_MODE for m in chain)
        assert all(not m.startswith("left_") for m in chain)
        assert chain[-1] == LEGACY_MODE


# --- allocation -------------------------------------------------------------


def test_mode_allocation_balanced_over_24_episodes() -> None:
    counts = Counter(
        libero_l2_select_mode(11, "pick_up_the_black_bowl", ep, forbidden_slots=())
        for ep in range(24)
    )
    assert set(counts) == set(MODES)
    assert max(counts.values()) - min(counts.values()) <= 1
    assert all(v == 4 for v in counts.values())


def test_mode_allocation_balanced_after_forbidding_left() -> None:
    remaining = [m for m in MODES if _slot_of(m) != "left"]
    assert len(remaining) == 4
    counts = Counter(
        libero_l2_select_mode(11, "pick_up_the_black_bowl", ep, forbidden_slots=["left"])
        for ep in range(24)
    )
    assert set(counts) == set(remaining)
    assert max(counts.values()) - min(counts.values()) <= 1
    assert all(v == 6 for v in counts.values())


def test_select_mode_empty_pool_is_legacy() -> None:
    mode = libero_l2_select_mode(1, "t", 0, forbidden_slots=list(SLOTS))
    assert mode == LEGACY_MODE


def test_select_mode_stable_across_pythonhashseed() -> None:
    snippet = (
        "from dynisafety.backends.libero.disturbance.l2_spawn import "
        "libero_l2_select_mode\n"
        "print(libero_l2_select_mode(42, 'spatial_0', 3, forbidden_slots=()))\n"
        "print(libero_l2_select_mode(99, 'goal_2', 17, forbidden_slots=['center']))\n"
    )
    results = []
    for hashseed in ("0", "1", "12345"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hashseed
        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        results.append(proc.stdout)
    assert results[0] == results[1] == results[2]
    lines = results[0].strip().splitlines()
    assert lines[0] in MODES
    assert lines[1] in MODES
    assert not lines[1].startswith("center_")


def test_modes_cover_all_slot_style_pairs() -> None:
    assert set(SLOTS) == {"left", "center", "right"}
    assert set(PATH_STYLES) == {"arc", "linear"}
    assert set(MODES) == {f"{slot}_{style}" for slot in SLOTS for style in PATH_STYLES}
    assert LEGACY_MODE == "legacy_jitter"
    assert LEGACY_MODE not in MODES
