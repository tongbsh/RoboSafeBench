"""Phase-stratified L2/L3 hazard-onset scheduling (no simulator)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml

from dynisafety.backends.libero.disturbance.onset_schedule import (
    phase_to_step,
    resolve_onset,
)

REPO = Path(__file__).resolve().parents[1]
BACKEND_YAML = REPO / "configs" / "backends" / "libero.yaml"
BASE_YAML = REPO / "configs" / "base.yaml"


def _extra_libero() -> dict:
    raw = yaml.safe_load(BACKEND_YAML.read_text())
    return raw["extra"]["libero"]


def _schedule() -> dict:
    return _extra_libero()["onset_schedule"]


def test_phase_to_step_anchors_grasp_not_midpoint() -> None:
    # 0 = start, 1 = grasp, 2 = done. Grasp at 40 of a 200-step task is a
    # fifth of the way in, so phase 1.5 must land at 120, not at 100.
    assert phase_to_step(0.0, t_grasp=40, t_done=200) == 0
    assert phase_to_step(1.0, t_grasp=40, t_done=200) == 40
    assert phase_to_step(2.0, t_grasp=40, t_done=200) == 200
    assert phase_to_step(1.5, t_grasp=40, t_done=200) == 120
    assert phase_to_step(0.5, t_grasp=40, t_done=200) == 20


def test_min_onset_step_clears_the_diagnostics_baseline_window() -> None:
    """The floor must be >= num_steps_wait + reaction.window_size.

    Below that the pre-onset baseline sits on the settle phase and every
    episode is dropped as baseline_unusable — the defect this schedule fixes.
    """
    extra = _extra_libero()
    diagnostics = yaml.safe_load(BASE_YAML.read_text())["diagnostics"]
    required = int(extra["num_steps_wait"]) + int(diagnostics["reaction"]["window_size"])
    assert int(extra["onset_schedule"]["min_onset_step"]) >= required


def test_every_shipped_anchor_yields_a_usable_window() -> None:
    """No task may end up with a degenerate (or floor-clipped) bin."""
    block = _schedule()
    floor = int(block["min_onset_step"])
    for suite, tasks in block["phase_anchors"].items():
        for task_id, anchor in tasks.items():
            t_grasp, t_done = int(anchor["t_grasp"]), int(anchor["t_done"])
            assert t_done > t_grasp > 0, (suite, task_id)
            for name, (lo_p, hi_p) in block["bins"].items():
                lo = max(floor, phase_to_step(lo_p, t_grasp=t_grasp, t_done=t_done))
                hi = max(lo, phase_to_step(hi_p, t_grasp=t_grasp, t_done=t_done))
                assert hi > lo, f"{suite} t{task_id} bin {name} collapsed to {lo}"
                assert hi < t_done, f"{suite} t{task_id} bin {name} fires after task end"


def test_bins_are_balanced_and_cover_every_phase() -> None:
    extra = _extra_libero()
    names = list(_schedule()["bins"])
    plans = [
        resolve_onset(
            extra,
            suite="safelibero_object",
            task_id=1,
            episode_index=i,
            plan_seed=7 + i,
            default_step=10,
        )
        for i in range(50)
    ]
    counts = Counter(p.bin_name for p in plans)
    assert set(counts) == set(names)
    # 50 episodes over 3 bins: round-robin, so no bin may drift by >1.
    assert max(counts.values()) - min(counts.values()) <= 1


def test_onset_is_model_independent_and_reproducible() -> None:
    """Same (suite, task, seed, episode) → same onset, always.

    Cross-model comparability of the four axes and of the paired L0 control
    both rest on this: two models must meet an identical hazard schedule.
    """
    extra = _extra_libero()
    kwargs = dict(
        suite="safelibero_spatial", task_id=2, episode_index=11, plan_seed=1234, default_step=10
    )
    first = resolve_onset(extra, **kwargs)
    again = resolve_onset(extra, **kwargs)
    assert first.step == again.step and first.bin_name == again.bin_name
    other_episode = resolve_onset(extra, **{**kwargs, "episode_index": 12})
    assert other_episode.bin_name != first.bin_name


def test_scheduled_onset_never_lands_before_the_floor() -> None:
    extra = _extra_libero()
    floor = int(_schedule()["min_onset_step"])
    for suite, tasks in _schedule()["phase_anchors"].items():
        for task_id in tasks:
            for episode_index in range(12):
                plan = resolve_onset(
                    extra,
                    suite=suite,
                    task_id=int(task_id),
                    episode_index=episode_index,
                    plan_seed=7 + episode_index,
                    default_step=10,
                )
                assert plan.scheduled is True
                assert plan.step >= floor
                assert plan.window is not None
                assert plan.window[0] <= plan.step <= plan.window[1]


def test_unknown_task_falls_back_and_says_so() -> None:
    plan = resolve_onset(
        _extra_libero(),
        suite="not_a_suite",
        task_id=0,
        episode_index=0,
        plan_seed=7,
        default_step=10,
    )
    assert plan.scheduled is False
    assert plan.step == 10
    assert plan.fallback_reason == "no_phase_anchor"
    assert plan.bin_name == "fixed"


def test_disabled_schedule_restores_the_fixed_step() -> None:
    extra = _extra_libero()
    extra["onset_schedule"] = {**extra["onset_schedule"], "enabled": False}
    plan = resolve_onset(
        extra,
        suite="safelibero_goal",
        task_id=0,
        episode_index=3,
        plan_seed=7,
        default_step=10,
    )
    assert plan.scheduled is False
    assert plan.step == 10
    assert plan.fallback_reason == "schedule_disabled"


def test_missing_block_is_an_error_not_a_python_default() -> None:
    with pytest.raises(TypeError, match="onset_schedule"):
        resolve_onset({}, suite="s", task_id=0, episode_index=0, plan_seed=7, default_step=10)


def test_plan_dict_is_jsonable_and_carries_the_provenance() -> None:
    plan = resolve_onset(
        _extra_libero(),
        suite="safelibero_long",
        task_id=0,
        episode_index=1,
        plan_seed=7,
        default_step=10,
    )
    payload = plan.as_dict()
    assert set(payload) == {
        "onset_step",
        "onset_bin",
        "onset_window",
        "t_grasp",
        "t_done",
        "scheduled",
        "fallback_reason",
    }
    assert payload["t_grasp"] == 58
    assert payload["onset_step"] == plan.step
