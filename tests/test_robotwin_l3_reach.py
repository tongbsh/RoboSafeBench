"""Pure-function tests for RoboTwin L3 reach-frame geometry."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dynisafety.backends.robotwin.disturbance.l3 import SharedReachingController
from dynisafety.backends.robotwin.disturbance.l3_reach import (
    LEGACY_MODE,
    robotwin_l3_accept_reach,
    robotwin_l3_build_path,
    robotwin_l3_execution_anchors,
    robotwin_l3_fallback_chain,
    robotwin_l3_lateral_axis,
    robotwin_l3_origin_is_valid,
    robotwin_l3_path_intercept_target,
    robotwin_l3_pose_at,
    robotwin_l3_slot_xy,
)


def test_primary_arm_slots() -> None:
    n_hat, degenerate = robotwin_l3_lateral_axis([0.0, 0.0], [0.0, 1.0])
    assert degenerate is False
    left, _ = robotwin_l3_slot_xy("left", opposite_xy=[0.0, 1.0], n_hat=n_hat, lateral_width_m=0.15)
    opp, collapsed = robotwin_l3_slot_xy(
        "opposite", opposite_xy=[0.0, 1.0], n_hat=n_hat, lateral_width_m=0.15
    )
    assert collapsed is False
    assert opp.tolist() == [0.0, 1.0]
    assert float(np.linalg.norm(left - opp)) == pytest.approx(0.15)


def test_intercept_and_fallback() -> None:
    tgt = robotwin_l3_path_intercept_target(
        np.array([0.0, 0.0, 0.8]),
        np.array([0.6, 0.0, 0.74]),
        alpha=0.5,
        min_eef_clearance=0.12,
        tip_overshoot=0.0,
        min_task_clearance=0.0,
        target_z=None,
    )
    assert tgt[0] == pytest.approx(0.3)
    chain = robotwin_l3_fallback_chain("right_mid", forbidden_slots=(), forbidden_depths=())
    assert chain[0][0] == "right_mid"
    assert chain[-1] == (LEGACY_MODE, None)


def test_origin_is_valid_ignores_approach_overlap() -> None:
    start = np.array([0.0, 0.85, 0.95])
    path = np.array(
        [
            [0.0, 0.85, 0.95],
            [0.0, 0.40, 0.90],
            [0.0, 0.10, 0.85],
        ],
        dtype=np.float64,
    )

    def origin_blocked(xy: np.ndarray) -> bool:
        return float(np.linalg.norm(np.asarray(xy, dtype=np.float64)[:2])) < 0.20

    assert robotwin_l3_origin_is_valid(start, path, origin_blocked=origin_blocked) is True
    assert robotwin_l3_origin_is_valid(path[-1], path, origin_blocked=origin_blocked) is False


def test_accept_reach_keeps_requested_mode_when_only_path_overlaps() -> None:
    start = np.array([0.15, 0.85, 0.95])
    dwell = np.array([0.0, 0.10, 0.85])
    path = np.vstack([start, dwell])

    def make_candidate(mode: str, style: str):
        if mode == LEGACY_MODE:
            return {"start": start, "dwell": dwell, "path": path, "path_style": "straight"}
        return {"start": start, "dwell": dwell, "path": path, "path_style": style}

    def validate(cand_start, cand_path) -> bool:
        return robotwin_l3_origin_is_valid(
            cand_start,
            cand_path,
            origin_blocked=lambda xy: float(np.linalg.norm(xy[:2])) < 0.20,
        )

    out = robotwin_l3_accept_reach(
        "left_mid",
        forbidden_slots=(),
        forbidden_depths=(),
        make_candidate=make_candidate,
        validate=validate,
    )
    assert out["accepted_mode"] == "left_mid"
    assert out["fallback_reason"] is None


def test_l3_move_start_task_step_holds_then_starts_enter() -> None:
    poses: list[np.ndarray] = []
    actor = SimpleNamespace(actor=SimpleNamespace(set_pose=lambda pose: poses.append(pose)))
    ctrl = SharedReachingController(
        actor=actor,
        start_pos=np.array([0.0, 0.0, 1.0]),
        hover_pos=np.array([0.2, 0.0, 1.1]),
        dwell_pos=np.array([0.4, 0.0, 0.9]),
        move_start_step=1,
        entry_steps=10,
        approach_steps=10,
        dwell_steps=1,
        exit_steps=10,
        flinch_dist=0.08,
        flinch_height=0.06,
        get_eef_pos_fn=lambda: np.array([0.3, 0.0, 0.85]),
        post_hit_hold_steps=3,
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        move_start_task_step=20,
    )
    ctrl._set_pose = lambda pose: poses.append(  # type: ignore[method-assign]
        np.asarray(getattr(pose, "position", pose), dtype=np.float64).copy()
    )
    ctrl.update_before_step(100, task_step=19)
    assert ctrl.disturbance_onset_step is None
    assert ctrl.phase.name == "HIDDEN"
    ctrl.update_before_step(500, task_step=20)
    assert ctrl.disturbance_onset_step == 20
    assert ctrl.phase.name == "ENTER"
    ctrl.update_before_step(505, task_step=20)
    assert ctrl.phase.name == "ENTER"
    mid = poses[-1]
    assert float(mid[0]) > 0.0
    assert not np.allclose(mid, ctrl.hover_pos)
    ctrl.update_before_step(510, task_step=21)
    assert ctrl.phase.name == "APPROACH"
    assert ctrl.disturbance_onset_step == 20
    assert np.allclose(ctrl.start_pos, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(ctrl.hover_pos, np.array([0.2, 0.0, 1.1]))
    assert np.allclose(ctrl.dwell_pos, np.array([0.4, 0.0, 0.9]))


def test_config_bridge_requires_l3_move_start_task_step() -> None:
    from dynisafety.backends.robotwin.config_bridge import _l3_move_start_task_step

    with pytest.raises(ValueError, match="move_start_task_step"):
        _l3_move_start_task_step({"l3": {"target_mode": "current"}})


def test_config_bridge_requires_l3_block() -> None:
    from dynisafety.backends.robotwin.config_bridge import _l3_reach_block

    with pytest.raises(ValueError, match="extra.robotwin.l3"):
        _l3_reach_block({})
    incomplete = {
        "l3": {
            "target_mode": "current",
            "intercept_alpha": 0.5,
            "min_eef_clearance": 0.12,
            "min_task_clearance": 0.0,
            "tip_overshoot": 0.0,
        }
    }
    with pytest.raises(TypeError, match="reach_diversity"):
        _l3_reach_block(incomplete)


def test_execution_anchors_put_hover_above_dwell() -> None:
    start = np.array([0.1, 0.2, 0.80])
    dwell = np.array([0.4, 0.2, 0.82])
    hover, pose_start = robotwin_l3_execution_anchors(
        start, dwell, "straight", enter_height=0.10, hover_rise_m=0.10
    )
    assert hover[:2] == pytest.approx(dwell[:2])
    assert hover[2] == pytest.approx(0.92)
    assert pose_start == pytest.approx(start)


def test_execution_anchors_raised_via_hover_above_dwell() -> None:
    geom = np.array([0.0, 0.0, 0.80])
    dwell = np.array([0.4, 0.0, 0.82])
    path = robotwin_l3_build_path(geom, dwell, "raised", n_samples=8, hover_rise_m=0.10)
    hover, pose_start = robotwin_l3_execution_anchors(
        path[0], dwell, "raised", enter_height=0.10, hover_rise_m=0.10
    )
    assert hover[:2] == pytest.approx(dwell[:2])
    assert hover[2] == pytest.approx(0.92)
    assert pose_start == pytest.approx(geom)
    via = robotwin_l3_build_path(geom, dwell, "via", n_samples=8, hover_rise_m=0.10)
    hover_v, pose_v = robotwin_l3_execution_anchors(
        via[0], dwell, "via", enter_height=0.10, hover_rise_m=0.10
    )
    assert hover_v[:2] == pytest.approx(dwell[:2])
    assert hover_v[2] == pytest.approx(0.92)
    assert pose_v == pytest.approx(geom)


def _approach_ctrl(hover: np.ndarray, dwell: np.ndarray, *, path_style: str, pose_start: np.ndarray):
    ctrl = object.__new__(SharedReachingController)
    ctrl.hover_pos = hover
    ctrl.dwell_pos = dwell
    ctrl.pose_start = pose_start
    ctrl.path_style = path_style
    ctrl.hover_rise_m = 0.10
    ctrl.via_descent_alpha = 0.20
    return ctrl


def test_approach_is_drop_not_inbound() -> None:
    start = np.array([0.136, 0.256, 0.975])
    dwell = np.array([-0.221, -0.221, 0.900])
    for style in ("straight", "raised", "via"):
        path = robotwin_l3_build_path(start, dwell, style, n_samples=8, hover_rise_m=0.10)
        hover, pose_start = robotwin_l3_execution_anchors(
            path[0], dwell, style, enter_height=0.10, hover_rise_m=0.10
        )
        ctrl = _approach_ctrl(hover, dwell, path_style=style, pose_start=pose_start)
        at0 = ctrl._approach_pos_at(0.0)
        at1 = ctrl._approach_pos_at(1.0)
        assert at0[:2] == pytest.approx(dwell[:2])
        assert at0[2] == pytest.approx(1.00)
        assert at1 == pytest.approx(dwell)
        enter_xy = float(np.linalg.norm(path[0][:2] - hover[:2]))
        approach_xy = float(np.linalg.norm(at0[:2] - at1[:2]))
        assert enter_xy > 0.5
        assert approach_xy == pytest.approx(0.0)
    doubled = robotwin_l3_pose_at(hover, dwell, 0.0, "raised", hover_rise_m=0.10)
    assert doubled[2] == pytest.approx(hover[2] + 0.10)
