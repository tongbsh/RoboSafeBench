"""Unit tests for RoboTwin S3e: seed formula + primary_arm registry (no sapien)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dynisafety.backends.robotwin import primary_arm as pa_mod
from dynisafety.backends.robotwin.primary_arm import resolve_primary_arm_tag
from dynisafety.backends.robotwin.seed import (
    episode_search_start,
    next_candidate_seed,
    seed_formula_repr,
)

EXPECTED_TASKS = frozenset(
    {
        "adjust_bottle",
        "beat_block_hammer",
        "blocks_ranking_rgb",
        "click_bell",
        "handover_block",
        "lift_pot",
        "move_can_pot",
        "move_pillbottle_pad",
        "move_playingcard_away",
        "move_stapler_pad",
        "open_laptop",
        "pick_dual_bottles",
        "place_a2b_left",
        "place_a2b_right",
        "place_empty_cup",
        "place_mouse_pad",
        "shake_bottle",
        "stack_blocks_two",
        "stack_bowls_three",
        "stamp_seal",
    }
)


class FakeArmTag:
    def __init__(self, arm: str) -> None:
        self.arm = arm.lower()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FakeArmTag):
            return self.arm == other.arm
        if isinstance(other, str):
            return self.arm == other.lower()
        return False

    def __repr__(self) -> str:
        return f"FakeArmTag({self.arm!r})"


def _fake_arm_tag(value: str) -> FakeArmTag:
    return FakeArmTag(value)


# Expose registry keys for tests (internal dict is module-private).
PRIMARY_ARM_TASKS = frozenset(pa_mod._REGISTRY.keys())  # noqa: SLF001


def test_episode_search_start_unified_formula() -> None:
    assert episode_search_start(run_seed=0, seed_stride=2000) == 2000
    assert episode_search_start(run_seed=1, seed_stride=2000) == 4000
    assert episode_search_start(run_seed=2, seed_stride=2000) == 6000
    assert next_candidate_seed(2000) == 2001
    assert seed_formula_repr(2000) == "seed_stride*(1+seed)"
    # Explicitly not the deprecated InternVLA formula.
    assert episode_search_start(run_seed=0, seed_stride=2000) != 100000


def test_episode_search_start_validates_inputs() -> None:
    with pytest.raises(ValueError, match="seed_stride"):
        episode_search_start(run_seed=0, seed_stride=0)
    with pytest.raises(ValueError, match="run_seed"):
        episode_search_start(run_seed=-1, seed_stride=2000)


def test_primary_arm_registry_has_all_twenty_tasks() -> None:
    assert PRIMARY_ARM_TASKS == EXPECTED_TASKS
    assert len(PRIMARY_ARM_TASKS) == 20


def test_resolve_primary_arm_tag_unknown_raises_key_error() -> None:
    with pytest.raises(KeyError, match="unknown_task"):
        resolve_primary_arm_tag(SimpleNamespace(), task_name="unknown_task")


@patch.object(pa_mod, "_arm_tag", side_effect=_fake_arm_tag)
def test_resolve_lift_pot_returns_left(_mock_arm: object) -> None:
    task = SimpleNamespace()
    tag = resolve_primary_arm_tag(task, task_name="lift_pot")
    assert tag == FakeArmTag("left")


@patch.object(pa_mod, "_arm_tag", side_effect=_fake_arm_tag)
def test_resolve_place_a2b_right_by_object_x(_mock_arm: object) -> None:
    task = SimpleNamespace(
        object=SimpleNamespace(get_pose=lambda: SimpleNamespace(p=[0.5, 0.0, 0.0]))
    )
    tag = resolve_primary_arm_tag(task, task_name="place_a2b_right")
    assert tag == FakeArmTag("right")

    task.object.get_pose = lambda: SimpleNamespace(p=[-0.2, 0.0, 0.0])
    tag = resolve_primary_arm_tag(task, task_name="place_a2b_right")
    assert tag == FakeArmTag("left")
