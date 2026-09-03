"""Pure-function tests for SimplerEnv L3 reach-frame geometry."""
from __future__ import annotations

import numpy as np
import pytest

from dynisafety.backends.simplerenv.disturbance.diversity_clearance import LEGACY_UNVALIDATED
from dynisafety.backends.simplerenv.disturbance.l3 import SharedReachingController
from dynisafety.backends.simplerenv.disturbance.l3_reach import (
    LEGACY_MODE,
    MODES,
    simplerenv_l3_accept_reach,
    simplerenv_l3_build_path,
    simplerenv_l3_fallback_chain,
    simplerenv_l3_path_intercept_target,
    simplerenv_l3_pose_at,
    simplerenv_l3_resolve_depth,
    simplerenv_l3_select_mode,
    simplerenv_l3_slot_xy,
)


def test_path_intercept_clamps_clearance() -> None:
    eef = np.array([0.0, 0.0, 0.2])
    task = np.array([0.4, 0.0, 0.1])
    mid = simplerenv_l3_path_intercept_target(
        eef,
        task,
        alpha=0.5,
        min_eef_clearance=0.12,
        tip_overshoot=0.0,
        min_task_clearance=0.0,
        target_z=None,
    )
    assert mid[0] == pytest.approx(0.2)
    near = simplerenv_l3_path_intercept_target(
        eef,
        task,
        alpha=0.0,
        min_eef_clearance=0.12,
        tip_overshoot=0.0,
        min_task_clearance=0.0,
        target_z=0.1,
    )
    assert near[0] == pytest.approx(0.12)
    assert near[2] == pytest.approx(0.1)


def test_slot_slides_on_far_edge() -> None:
    n_hat = np.array([0.0, 1.0])
    opp = np.array([1.0, 0.0])
    left, collapsed = simplerenv_l3_slot_xy(
        "left",
        opposite_xy=opp,
        table_mins=[0.0, -0.5],
        table_maxs=[1.0, 0.5],
        n_hat=n_hat,
        lateral_width_m=0.15,
    )
    assert collapsed is False
    assert left[0] == pytest.approx(1.0)
    assert left[1] != pytest.approx(0.0)


def test_depth_collapses_on_short_corridor() -> None:
    alpha, collapsed = simplerenv_l3_resolve_depth(
        "near",
        eef_xy=[0.0, 0.0],
        task_xy=[0.10, 0.0],
        near_alpha=0.3,
        mid_alpha=0.5,
        min_eef_clearance=0.12,
        min_task_clearance=0.0,
        collapse_m=0.08,
    )
    assert collapsed is True
    assert alpha == pytest.approx(0.5)


def test_fallback_and_legacy() -> None:
    chain = simplerenv_l3_fallback_chain(
        "left_near", forbidden_slots=(), forbidden_depths=()
    )
    assert chain[0] == ("left_near", "straight")
    assert chain[-1] == (LEGACY_MODE, None)
    offered_modes = {mode for mode, _style in chain if mode != LEGACY_MODE}
    assert offered_modes == set(MODES)
    assert simplerenv_l3_select_mode(1, 0, 0, forbidden_slots=("left",), forbidden_depths=()) != "left_mid"

    def make(mode: str, style: str):
        if mode == LEGACY_MODE:
            return {"start": np.zeros(3), "dwell": np.ones(3), "path": np.zeros((2, 3))}
        return None

    out = simplerenv_l3_accept_reach(
        "right_mid",
        forbidden_slots=(),
        forbidden_depths=(),
        make_candidate=make,
        validate=lambda s, p: True,
    )
    assert out["accepted_mode"] == LEGACY_MODE
    assert out["legacy_validated"] is True


def test_l3_accept_marks_unvalidated_legacy() -> None:
    def make(mode: str, style: str):
        return {"start": np.zeros(3), "dwell": np.ones(3), "path": np.zeros((2, 3))}

    out = simplerenv_l3_accept_reach(
        "left_near",
        forbidden_slots=(),
        forbidden_depths=(),
        make_candidate=make,
        validate=lambda s, p: False,
    )
    assert out["accepted_mode"] == LEGACY_MODE
    assert out["fallback_reason"] == LEGACY_UNVALIDATED
    assert out["legacy_validated"] is False


def test_assemble_requires_l3_block() -> None:
    from dynisafety.backends.simplerenv.disturbance.assemble import _l3_reach_block

    with pytest.raises(ValueError, match="extra.simplerenv"):
        _l3_reach_block({})
    incomplete = {
        "simplerenv": {
            "l3": {
                "target_mode": "current",
                "intercept_alpha": 0.5,
                "min_eef_clearance": 0.12,
                "min_task_clearance": 0.0,
                "tip_overshoot": 0.0,
            }
        }
    }
    with pytest.raises(TypeError, match="reach_diversity"):
        _l3_reach_block(incomplete)


def _se_l3_shell(
    *,
    start: np.ndarray,
    dwell: np.ndarray,
    path: np.ndarray | None,
    style: str,
    target_z: float = 0.10,
    enter_height: float = 0.08,
) -> SharedReachingController:
    ctrl = object.__new__(SharedReachingController)
    ctrl._start_pos = np.asarray(start, dtype=np.float64)
    ctrl._dwell_pos = np.asarray(dwell, dtype=np.float64)
    ctrl._planned_path = None if path is None else np.ascontiguousarray(path, dtype=np.float64)
    ctrl._path_style = style
    ctrl._target_z = float(target_z)
    ctrl._enter_height = float(enter_height)
    ctrl._reach_diversity = {"hover_rise_m": 0.10, "via_descent_alpha": 0.20}
    return ctrl


def test_default_off_approach_keeps_hold_z_profile() -> None:
    start = np.array([0.0, 0.0, 0.18])
    dwell = np.array([0.4, 0.0, 0.15])
    ctrl = _se_l3_shell(start=start, dwell=dwell, path=None, style="straight", target_z=0.10)
    mid = ctrl._approach_pos_at(0.4)
    assert mid[2] == pytest.approx(0.10 + 0.08)
    end = ctrl._approach_pos_at(1.0)
    assert end[2] == pytest.approx(0.10 + SharedReachingController.DWELL_HEIGHT)


def test_raised_approach_does_not_double_hover() -> None:
    geom = np.array([0.0, 0.0, 0.20])
    dwell = np.array([0.4, 0.0, 0.12])
    path = simplerenv_l3_build_path(geom, dwell, "raised", n_samples=9, hover_rise_m=0.10)
    ctrl = _se_l3_shell(start=path[0], dwell=dwell, path=path, style="raised")
    assert ctrl._approach_pos_at(0.0) == pytest.approx(path[0])
    assert ctrl._approach_pos_at(1.0) == pytest.approx(dwell)
    doubled = simplerenv_l3_pose_at(path[0], dwell, 0.0, "raised", hover_rise_m=0.10)
    assert doubled[2] == pytest.approx(path[0][2] + 0.10)
    assert ctrl._approach_pos_at(0.0)[2] == pytest.approx(path[0][2])


def test_via_approach_follows_validated_path() -> None:
    geom = np.array([0.0, 0.0, 0.20])
    dwell = np.array([0.4, 0.0, 0.12])
    path = simplerenv_l3_build_path(geom, dwell, "via", n_samples=9, hover_rise_m=0.10)
    ctrl = _se_l3_shell(start=path[0], dwell=dwell, path=path, style="via")
    assert ctrl._approach_pos_at(0.0) == pytest.approx(path[0])
    assert ctrl._approach_pos_at(1.0) == pytest.approx(path[-1])
    mid = ctrl._approach_pos_at(0.5)
    assert mid[2] == pytest.approx(path[0][2])


def test_diversity_withdraw_returns_to_planned_start() -> None:
    start = np.array([0.0, 0.0, 0.55])
    dwell = np.array([0.3, 0.0, 0.22])
    path = simplerenv_l3_build_path(start, dwell, "straight", n_samples=5)
    ctrl = _se_l3_shell(
        start=start, dwell=dwell, path=path, style="straight", target_z=0.10, enter_height=0.08
    )
    assert ctrl._withdraw_pos_at(0.0) == pytest.approx(dwell)
    assert ctrl._withdraw_pos_at(1.0) == pytest.approx(start)
    off = _se_l3_shell(
        start=np.array([0.0, 0.0, 0.18]),
        dwell=np.array([0.3, 0.0, 0.15]),
        path=None,
        style="straight",
        target_z=0.10,
        enter_height=0.08,
    )
    assert off._withdraw_pos_at(1.0)[2] == pytest.approx(0.18)


def test_flinch_retreat_uses_planned_start_z() -> None:
    start = np.array([0.0, 0.0, 0.55])
    dwell = np.array([0.3, 0.0, 0.22])
    path = simplerenv_l3_build_path(start, dwell, "straight", n_samples=5)
    ctrl = _se_l3_shell(
        start=start, dwell=dwell, path=path, style="straight", target_z=0.10, enter_height=0.08
    )
    ctrl._control_freq = 5
    ctrl._begin_retreat_to_start(0, from_pos=dwell)
    assert ctrl._flinch_exit_to is not None
    assert ctrl._flinch_exit_to[2] == pytest.approx(0.55)
    assert ctrl._retreat_target()[2] == pytest.approx(0.55)
    off = _se_l3_shell(
        start=np.array([0.0, 0.0, 0.18]),
        dwell=np.array([0.3, 0.0, 0.15]),
        path=None,
        style="straight",
        target_z=0.10,
        enter_height=0.08,
    )
    off._control_freq = 5
    off._begin_retreat_to_start(0, from_pos=np.array([0.3, 0.0, 0.15]))
    assert off._flinch_exit_to is not None
    assert off._flinch_exit_to[2] == pytest.approx(0.18)
