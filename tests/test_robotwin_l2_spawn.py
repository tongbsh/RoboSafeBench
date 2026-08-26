"""Pure-function tests for RoboTwin L2 spawn-frame geometry."""
from __future__ import annotations

import numpy as np
import pytest

from dynisafety.backends.robotwin.disturbance.l2_spawn import (
    LEGACY_MODE,
    robotwin_l2_accept_spawn,
    robotwin_l2_candidate_start_xy,
    robotwin_l2_fallback_chain,
    robotwin_l2_lateral_axis,
    robotwin_l2_origin_is_valid,
    robotwin_l2_select_mode,
)


def test_axis_relative_to_primary_eef() -> None:
    n_hat, degenerate = robotwin_l2_lateral_axis([0.0, 0.0], [1.0, 0.0])
    assert degenerate is False
    assert n_hat[1] == pytest.approx(1.0) or n_hat[1] == pytest.approx(-1.0)
    left = robotwin_l2_candidate_start_xy([1.0, 0.0], n_hat, "left", 0.2)
    right = robotwin_l2_candidate_start_xy([1.0, 0.0], [0.0, 1.0], "right", 0.2)
    assert right[1] == pytest.approx(-0.2)
    assert float(np.linalg.norm(left - np.array([1.0, 0.0]))) == pytest.approx(0.2)


def test_fallback_and_forbidden() -> None:
    chain = robotwin_l2_fallback_chain("left_arc", forbidden_slots=("right",))
    assert "right_arc" not in chain
    assert chain[-1] == LEGACY_MODE
    mode = robotwin_l2_select_mode(3, "lift_pot", 0, forbidden_slots=("left", "center", "right"))
    assert mode == LEGACY_MODE


def test_origin_is_valid_ignores_path_overlap() -> None:
    start = np.array([0.85, 0.0, 0.90])
    path = np.array(
        [
            [0.85, 0.0, 0.90],
            [0.40, 0.0, 0.90],
            [0.00, 0.0, 0.80],
        ],
        dtype=np.float64,
    )
    table_object = np.array([0.0, 0.0])

    def origin_blocked(xy: np.ndarray) -> bool:
        return float(np.linalg.norm(np.asarray(xy, dtype=np.float64)[:2] - table_object)) < 0.20

    assert robotwin_l2_origin_is_valid(start, path, origin_blocked=origin_blocked) is True
    assert robotwin_l2_origin_is_valid(path[-1], path, origin_blocked=origin_blocked) is False
    assert robotwin_l2_origin_is_valid(None, path, origin_blocked=origin_blocked) is False


def test_accept_spawn_does_not_reject_table_crossing_path() -> None:
    start = np.array([0.85, 0.15, 0.90])
    path = np.vstack(
        [
            start,
            np.array([0.40, 0.08, 0.90]),
            np.array([0.00, 0.00, 0.80]),
        ]
    )

    def make_candidate(mode: str):
        if mode == LEGACY_MODE:
            return {"start": start, "path": path, "linear_solve_mode": "na"}
        return {"start": start, "path": path, "linear_solve_mode": "na"}

    def validate(cand_start, cand_path) -> bool:
        return robotwin_l2_origin_is_valid(
            cand_start,
            cand_path,
            origin_blocked=lambda xy: float(np.linalg.norm(xy[:2])) < 0.20,
        )

    out = robotwin_l2_accept_spawn(
        "left_arc",
        forbidden_slots=(),
        make_candidate=make_candidate,
        validate=validate,
    )
    assert out["accepted_mode"] == "left_arc"
    assert out["fallback_reason"] is None


def test_config_bridge_requires_spawn_diversity() -> None:
    from dynisafety.backends.robotwin.config_bridge import _l2_spawn_diversity

    with pytest.raises(ValueError, match="extra.robotwin.l2"):
        _l2_spawn_diversity({})
    with pytest.raises(ValueError, match="spawn_diversity"):
        _l2_spawn_diversity({"l2": {}})
