"""Pure-function tests for RoboTwin L2 spawn-frame geometry."""
from __future__ import annotations

import numpy as np
import pytest

from dynisafety.backends.robotwin.disturbance.l2_spawn import (
    LEGACY_MODE,
    robotwin_l2_candidate_start_xy,
    robotwin_l2_fallback_chain,
    robotwin_l2_lateral_axis,
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


def test_config_bridge_requires_spawn_diversity() -> None:
    from dynisafety.backends.robotwin.config_bridge import _l2_spawn_diversity

    with pytest.raises(ValueError, match="extra.robotwin.l2"):
        _l2_spawn_diversity({})
    with pytest.raises(ValueError, match="spawn_diversity"):
        _l2_spawn_diversity({"l2": {}})
