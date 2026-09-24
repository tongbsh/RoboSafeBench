"""SimplerEnv diversity fixture-top and 3D block (no SAPIEN)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dynisafety.backends.simplerenv.disturbance.diversity_clearance import (
    BLOCKING_HEIGHT_MARGIN_Z,
    fixture_top_at_xy,
    mover_blocked_by_scene,
)


def _aabb(mins, maxs):
    return (np.asarray(mins, dtype=np.float64), np.asarray(maxs, dtype=np.float64))


def _actor(name: str, mins, maxs) -> SimpleNamespace:
    box = _aabb(mins, maxs)
    return SimpleNamespace(name=name, _aabb=box)


def _get_aabb(actor):
    return actor._aabb


def test_arena_world_box_does_not_raise_fixture_top() -> None:
    arena = _actor("arena", [-2.0, -2.0, 0.0], [2.0, 2.0, 3.05])
    top = fixture_top_at_xy(
        [0.0, 0.0],
        table_height=0.87,
        actors=[arena],
        get_aabb=_get_aabb,
    )
    assert top == pytest.approx(0.87)


def test_prop_top_is_used_when_xy_covers_query() -> None:
    plate = _actor("plate", [-0.1, -0.1, 0.86], [0.1, 0.1, 0.90])
    top = fixture_top_at_xy(
        [0.0, 0.0],
        table_height=0.87,
        actors=[plate],
        get_aabb=_get_aabb,
    )
    assert top == pytest.approx(0.90)


def test_robot_and_self_are_skipped() -> None:
    hand = _actor("hand", [0.0, 0.0, 0.9], [0.1, 0.1, 1.0])
    robot = _actor("widowx_robot", [-1.0, -1.0, 0.0], [1.0, 1.0, 2.0])
    top = fixture_top_at_xy(
        [0.05, 0.05],
        table_height=0.87,
        actors=[hand, robot],
        exclude_actor=hand,
        get_aabb=_get_aabb,
    )
    assert top == pytest.approx(0.87)


def test_flyover_plate_is_not_blocked() -> None:
    plate = _actor("plate", [-0.1, -0.1, 0.86], [0.1, 0.1, 0.90])
    mover = _aabb([-0.05, -0.05, 0.96], [0.05, 0.05, 1.05])
    assert (
        mover_blocked_by_scene(
            mover,
            [plate],
            table_height=0.87,
            get_aabb=_get_aabb,
        )
        is False
    )


def test_bottom_in_plate_is_blocked() -> None:
    plate = _actor("plate", [-0.1, -0.1, 0.86], [0.1, 0.1, 0.90])
    mover = _aabb([-0.05, -0.05, 0.88], [0.05, 0.05, 0.97])
    assert (
        mover_blocked_by_scene(
            mover,
            [plate],
            table_height=0.87,
            margin_z=BLOCKING_HEIGHT_MARGIN_Z,
            get_aabb=_get_aabb,
        )
        is True
    )


def test_arena_xy_overlap_is_not_a_layer_b_block() -> None:
    arena = _actor("arena", [-2.0, -2.0, 0.0], [2.0, 2.0, 3.05])
    mover = _aabb([-0.05, -0.05, 0.88], [0.05, 0.05, 0.97])
    assert (
        mover_blocked_by_scene(
            mover,
            [arena],
            table_height=0.87,
            get_aabb=_get_aabb,
        )
        is False
    )
