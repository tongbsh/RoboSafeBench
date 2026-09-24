"""Put ``first_contact_step`` on the trace's task-step clock.

``EpisodeRecord.first_contact_step`` is written by each backend on whatever
clock its contact manager runs on. RoboTwin's is the physics sub-step
(dt = 1/250), roughly 40–90× the task-step clock the traces and ``t_onset``
carry; LIBERO and SimplerEnv already write task steps. Every pre-contact gate
(``t_react < first_contact_step``) compares against a task-step time, so a
sub-step value passes almost any post-contact deviation as "before contact".

The trace records ``in_contact`` per task step, which is on the right clock by
construction. Consumers that have the trace should call
:func:`resolve_first_contact_step` instead of reading the record field.
"""

from __future__ import annotations

from typing import Any

from dynisafety.diagnostics.schema import Trace

SOURCE_NO_CONTACT = "no_contact"
SOURCE_EXTRA_CONTROL_STEP = "extra_control_step"
SOURCE_TRACE = "trace"
SOURCE_RECORD = "record"
SOURCE_RECORD_OFF_CLOCK = "record_off_clock"
SOURCE_UNKNOWN = "unknown"
SOURCE_L3_HAND_REMOVED = "l3_hand_removed"

UNRESOLVED_SOURCES: frozenset[str] = frozenset({SOURCE_RECORD_OFF_CLOCK, SOURCE_UNKNOWN})

# Backends whose contact manager runs on a finer clock than the task step
# also write the control-step value here (RoboTwin: take_action_cnt). It is
# the number the online diagnostics session was handed, so it is preferred
# even over the trace flag for exact agreement with the stored labels.
EXTRA_CONTROL_STEP_KEY = "first_contact_control_step"


def first_contact_from_trace(trace: Trace) -> int | None:
    """First task step whose trace sample is flagged ``in_contact``."""
    for step in trace.steps:
        if bool(getattr(step, "in_contact", False)):
            return int(step.t)
    return None


def first_contact_from_extra(ep: Any) -> int | None:
    """``extra.first_contact_control_step`` when the record carries one."""
    extra = getattr(ep, "extra", None)
    if not isinstance(extra, dict):
        return None
    value = extra.get(EXTRA_CONTROL_STEP_KEY)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _episode_extra(ep: Any) -> dict[str, Any]:
    extra = getattr(ep, "extra", None)
    return extra if isinstance(extra, dict) else {}


def simplerenv_l3_physical_release(ep: Any) -> bool:
    """True when the L3 hand was released into a dynamic body.

    Official ``in_contact`` stays false during the kinematic overlap, so the
    release is the collision the trace can place in time.
    """
    extra = _episode_extra(ep)
    if int(extra.get("ghost_contact_sim_steps") or 0) > 0:
        return True
    if int(extra.get("l3_kinematic_overlap_sim_steps") or 0) > 0:
        return True
    return bool(extra.get("released_to_physics"))


def simplerenv_l3_hand_removed_step(trace: Trace) -> int | None:
    """First task step on which a visible L3 hand has left the trace.

    The controller deletes the actor on the release step. Samples at and
    after this step include the arm motion caused by that impact.
    """
    if trace.header.backend != "simplerenv" or int(trace.header.level) != 3:
        return None
    seen_hand = False
    for step in trace.steps:
        phase = step.phase.value if step.phase is not None else ""
        if step.obstacle_pos is not None:
            seen_hand = True
            continue
        if seen_hand and phase == "cleared":
            return int(step.t)
    return None


def resolve_first_contact_step(
    ep: Any,
    trace: Trace | None,
    *,
    contact_steps: int | None = None,
    first_contact_step: int | None = None,
) -> tuple[int | None, str]:
    """``(first_contact_step, source)`` on the trace's task-step clock.

    ``ep`` may be an ``EpisodeRecord`` or any object with ``contact_steps`` /
    ``first_contact_step``; the keyword overrides exist for callers that hold
    the values loose (the online diagnostics session).

    Sources, in order of preference: ``no_contact`` (no contact recorded;
    every pre-contact gate passes), ``extra_control_step``
    (``extra.first_contact_control_step``, the backend's own control-step
    value), ``trace`` (first ``in_contact`` sample), ``record`` (trace never
    flagged contact, record is within the trace's span), ``record_off_clock``
    (record lies beyond the trace's last step — the contact time is unknown
    and the gate cannot pass), ``unknown`` (contact happened but no source
    can place it).
    """
    n_contact = (
        int(contact_steps)
        if contact_steps is not None
        else int(getattr(ep, "contact_steps", 0) or 0)
    )
    if n_contact <= 0:
        return None, SOURCE_NO_CONTACT
    from_extra = first_contact_from_extra(ep)
    if from_extra is not None:
        return from_extra, SOURCE_EXTRA_CONTROL_STEP
    if trace is not None:
        from_trace = first_contact_from_trace(trace)
        if from_trace is not None:
            return from_trace, SOURCE_TRACE
    recorded = (
        first_contact_step
        if first_contact_step is not None
        else getattr(ep, "first_contact_step", None)
    )
    if recorded is None:
        return None, SOURCE_UNKNOWN
    if trace is not None and trace.steps:
        last_t = max(int(step.t) for step in trace.steps)
        if int(recorded) > last_t:
            return None, SOURCE_RECORD_OFF_CLOCK
    return int(recorded), SOURCE_RECORD


__all__ = [
    "EXTRA_CONTROL_STEP_KEY",
    "SOURCE_EXTRA_CONTROL_STEP",
    "SOURCE_NO_CONTACT",
    "SOURCE_RECORD",
    "SOURCE_RECORD_OFF_CLOCK",
    "SOURCE_TRACE",
    "SOURCE_UNKNOWN",
    "UNRESOLVED_SOURCES",
    "SOURCE_L3_HAND_REMOVED",
    "first_contact_from_extra",
    "first_contact_from_trace",
    "resolve_first_contact_step",
    "simplerenv_l3_hand_removed_step",
    "simplerenv_l3_physical_release",
]
