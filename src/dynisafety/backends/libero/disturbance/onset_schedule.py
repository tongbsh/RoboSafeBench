"""Phase-stratified hazard-onset scheduling for LIBERO L2 / L3.

Both levels used to fire at a constant ``move_start_step`` that happened to
equal ``num_steps_wait``. Three things followed: the pre-onset diagnostics
window landed entirely on the settle phase (see
``diagnostics.reaction.min_baseline_*``), hazard timing was never varied, and
every collision therefore happened in the same task phase.

This module schedules the onset instead. Each episode is assigned a task phase
(approach / grasp / transport) round-robin by ``episode_index``, and the onset
step is drawn uniformly inside that phase's window.

Two properties matter and are deliberate:

- **Model-independent.** The bin and the draw depend only on
  ``(suite, task_id, seed, episode_index)``, never on the policy's own
  trajectory. Every model under test therefore sees an identical hazard
  schedule on the same episode, which is what makes the four axes and the
  paired level-0 control comparable across models. A distance-triggered onset
  (the L1 scheme) would not have this property.
- **Anchors are a protocol constant, not a per-model measurement.** The phase
  boundaries come from disturbance-free rollouts once and live in YAML. They
  are close to model-invariant in practice: across the two reference models
  the median first-grasp step differs by at most 4 task steps on all 16
  LIBERO tasks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

# Own RNG stream id, so scheduling never shifts the spawn / reach diversity draws.
_ONSET_STREAM = 0x04E7

_REQUIRED_KEYS = ("enabled", "min_onset_step", "bins", "phase_anchors")


@dataclass(frozen=True)
class OnsetPlan:
    """Resolved onset for one episode. Serialized into ``extra.onset_plan``."""

    step: int
    bin_name: str
    window: tuple[int, int] | None
    t_grasp: int | None
    t_done: int | None
    scheduled: bool
    fallback_reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "onset_step": int(self.step),
            "onset_bin": self.bin_name,
            "onset_window": None if self.window is None else [int(self.window[0]), int(self.window[1])],
            "t_grasp": None if self.t_grasp is None else int(self.t_grasp),
            "t_done": None if self.t_done is None else int(self.t_done),
            "scheduled": bool(self.scheduled),
            "fallback_reason": self.fallback_reason,
        }


def _schedule_block(extra_libero: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = extra_libero.get("onset_schedule")
    if not isinstance(raw, Mapping):
        raise TypeError(
            "extra.libero.onset_schedule must be a mapping "
            "(configs/backends/libero.yaml); hazard onset must not fall back "
            "to a Python default"
        )
    missing = [key for key in _REQUIRED_KEYS if key not in raw]
    if missing:
        raise ValueError(f"extra.libero.onset_schedule missing keys: {missing}")
    return raw


def phase_to_step(phase: float, *, t_grasp: int, t_done: int) -> int:
    """Map a phase coordinate to a task step.

    ``0`` = policy start, ``1`` = first grasp, ``2`` = task complete. The two
    segments are interpolated separately because grasp does not sit halfway
    through a pick-and-place.
    """
    p = float(phase)
    if p <= 1.0:
        return int(round(p * float(t_grasp)))
    return int(round(float(t_grasp) + (p - 1.0) * float(t_done - t_grasp)))


def _anchors(
    block: Mapping[str, Any], *, suite: str, task_id: int
) -> tuple[int, int] | None:
    table = block.get("phase_anchors")
    if not isinstance(table, Mapping):
        return None
    per_suite = table.get(str(suite))
    if not isinstance(per_suite, Mapping):
        return None
    entry = per_suite.get(int(task_id), per_suite.get(str(task_id)))
    if not isinstance(entry, Mapping):
        return None
    try:
        t_grasp = int(entry["t_grasp"])
        t_done = int(entry["t_done"])
    except (KeyError, TypeError, ValueError):
        return None
    if t_done <= t_grasp or t_grasp <= 0:
        return None
    return t_grasp, t_done


def _bin_items(block: Mapping[str, Any]) -> list[tuple[str, float, float]]:
    bins = block.get("bins")
    if not isinstance(bins, Mapping) or not bins:
        raise ValueError("extra.libero.onset_schedule.bins must be a non-empty mapping")
    out: list[tuple[str, float, float]] = []
    for name, span in bins.items():
        if not isinstance(span, Sequence) or isinstance(span, (str, bytes)) or len(span) != 2:
            raise ValueError(f"onset_schedule.bins[{name!r}] must be a [lo, hi] pair")
        lo, hi = float(span[0]), float(span[1])
        if hi <= lo:
            raise ValueError(f"onset_schedule.bins[{name!r}] needs hi > lo, got {span!r}")
        out.append((str(name), lo, hi))
    return out


def resolve_onset(
    extra_libero: Mapping[str, Any],
    *,
    suite: str,
    task_id: int,
    episode_index: int,
    plan_seed: int,
    default_step: int,
) -> OnsetPlan:
    """Pick this episode's onset step and record how it was chosen.

    Falls back to ``default_step`` (never raises) when the schedule is off or
    the task has no measured anchors: an unscheduled onset is a labelled
    fallback, not a silent constant.
    """
    block = _schedule_block(extra_libero)
    if not bool(block["enabled"]):
        return OnsetPlan(
            step=int(default_step),
            bin_name="fixed",
            window=None,
            t_grasp=None,
            t_done=None,
            scheduled=False,
            fallback_reason="schedule_disabled",
        )

    anchors = _anchors(block, suite=suite, task_id=task_id)
    if anchors is None:
        return OnsetPlan(
            step=int(default_step),
            bin_name="fixed",
            window=None,
            t_grasp=None,
            t_done=None,
            scheduled=False,
            fallback_reason="no_phase_anchor",
        )

    t_grasp, t_done = anchors
    bins = _bin_items(block)
    name, lo_phase, hi_phase = bins[int(episode_index) % len(bins)]
    floor = int(block["min_onset_step"])
    lo = max(floor, phase_to_step(lo_phase, t_grasp=t_grasp, t_done=t_done))
    hi = max(lo, phase_to_step(hi_phase, t_grasp=t_grasp, t_done=t_done))

    rng = np.random.default_rng([int(plan_seed), _ONSET_STREAM])
    step = int(rng.integers(lo, hi + 1))
    return OnsetPlan(
        step=step,
        bin_name=name,
        window=(lo, hi),
        t_grasp=t_grasp,
        t_done=t_done,
        scheduled=True,
        fallback_reason=None,
    )


__all__ = ["OnsetPlan", "phase_to_step", "resolve_onset"]
