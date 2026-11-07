"""Contact-graph semantics: continuous official contact, constraint state, clamp (00 §6, D9 c).

Input is a time-ordered list of :class:`ContactStep` (one per hook step, **including steps with an empty
contact list** - only a recorded empty step proves that contact was lost). The contacts are assumed to be
post-release (``dynamic``) contacts already (the runtime drops kinematic overlaps, R8). Normals point from the
partner toward the hazard.

Definitions used here:

* The *robot side* is ``robot_link`` and ``held_object``; every other non-hazard body is ``scene``, movable
  task objects included (they need not be static); contacts with other ``hazard`` bodies are ignored. A step
  has *robot contact* when it contains a robot-side contact. Contact with the scene alone is not official
  robot contact.
* A step has a *second partner* when a robot-side contact is accompanied by a contact with a different
  non-hazard partner name: a scene body (fixed or movable, e.g. a task object) or another robot part (D10 g).
  The opposing partner of a ``clamp`` is this second partner; it is not required to be static.
* A step has the *clamp configuration* when such a pair has roughly opposing normals:
  ``|pi - angle(n_a, n_b)| < opposing_normals_tol_rad``.
* Runs of steps are merged into one continuous run when the time between the last step of the run and the
  next step with the property is at most ``gap_tolerance_s``, provided that only recorded false steps lie
  between them. Run duration is ``t_last - t_first`` (a lower bound by up to one sample interval).
* ``clamp``: a continuous clamp-configuration run lasting at least ``t_q_s``; else ``constrained`` if any step has
  a second partner; else ``unconstrained``; ``None`` when there is no robot contact at all. The state is the
  maximum over the whole record.
"""
import dataclasses
import enum
import math
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .._numeric import finite_float, non_negative_float, positive_float, unit_vec3
from ..host import ContactObs, PartnerClass

_ROBOT_SIDE = (PartnerClass.ROBOT_LINK, PartnerClass.HELD_OBJECT)


class ConstraintState(str, enum.Enum):
    UNCONSTRAINED = "unconstrained"
    CONSTRAINED = "constrained"
    CLAMP = "clamp"


@dataclasses.dataclass(eq=False)
class ContactStep:
    """Everything the hazard touches at one instant (possibly nothing)."""
    time_s: float
    contacts: List[ContactObs]

    def __post_init__(self) -> None:
        self.time_s = finite_float("time_s", self.time_s)
        self.contacts = list(self.contacts)
        for obs in self.contacts:
            if not isinstance(obs, ContactObs):
                raise ValueError("contacts must hold hazardlib.host.ContactObs, got %r" % (type(obs).__name__,))


@dataclasses.dataclass(eq=False)
class ContactSegment:
    t_start_s: float
    t_end_s: float

    @property
    def duration_s(self) -> float:
        return self.t_end_s - self.t_start_s


@dataclasses.dataclass(eq=False)
class ContactSummary:
    """``contact_duration_s`` is the duration of the first continuous robot-contact segment (the one that
    starts at the first official contact); ``longest_contact_s`` the longest one; ``clamp_duration_s`` the
    longest continuous clamp-configuration run. All ``None`` when there was no such contact."""
    segments: List[ContactSegment]
    first_contact_t_s: Optional[float]
    contact_duration_s: Optional[float]
    longest_contact_s: Optional[float]
    constraint_state: Optional[ConstraintState]
    clamp_duration_s: Optional[float]


def continuous_runs(times_s: Sequence[float], flags: Sequence[bool],
                    gap_tolerance_s: float) -> List[Tuple[float, float]]:
    """``(t_first, t_last)`` of every maximal run of true flags, merging runs separated only by false
    samples when ``t_next_true - t_last_true <= gap_tolerance_s``. Consecutive true samples always belong
    to the same run."""
    gap = non_negative_float("gap_tolerance_s", gap_tolerance_s)
    if len(times_s) != len(flags):
        raise ValueError("times_s and flags must have equal length, got %d and %d" % (len(times_s), len(flags)))
    runs = []  # type: List[Tuple[float, float]]
    start = last = None  # type: Any
    pending_gap = False
    for t, flag in zip(times_s, flags):
        if flag:
            if start is None:
                start = last = t
            elif pending_gap and (t - last) > gap:
                runs.append((start, last))
                start = last = t
            else:
                last = t
            pending_gap = False
        elif start is not None:
            pending_gap = True
    if start is not None:
        runs.append((start, last))
    return runs


def _step_properties(contacts: Sequence[ContactObs], tolerance_rad: float) -> Tuple[bool, bool, bool]:
    relevant = [c for c in contacts if c.partner_class is not PartnerClass.HAZARD]
    robot = [c for c in relevant if c.partner_class in _ROBOT_SIDE]
    if not robot:
        return False, False, False
    second = clamp = False
    for a in robot:
        n_a = unit_vec3("contact normal of %s" % a.partner_name, a.normal)
        for b in relevant:
            if b.partner_name == a.partner_name:
                continue
            second = True
            n_b = unit_vec3("contact normal of %s" % b.partner_name, b.normal)
            angle = math.atan2(float(np.linalg.norm(np.cross(n_a, n_b))), float(np.dot(n_a, n_b)))
            if abs(math.pi - angle) < tolerance_rad:
                clamp = True
    return True, second, clamp


def summarize_contacts(steps: Sequence[ContactStep], t_q_s: float, opposing_normals_tol_rad: float,
                       gap_tolerance_s: float) -> ContactSummary:
    """Duration, constraint state and clamp decision of one hazard's official contacts (see the module
    docstring for every definition). ``t_q_s``, the angle tolerance and the gap tolerance are all arguments."""
    t_q = positive_float("t_q_s", t_q_s)
    tolerance = positive_float("opposing_normals_tol_rad", opposing_normals_tol_rad)
    if tolerance > math.pi:
        raise ValueError("opposing_normals_tol_rad must be <= pi, got %r" % (tolerance,))
    gap = non_negative_float("gap_tolerance_s", gap_tolerance_s)
    times = []  # type: List[float]
    robot_flags, second_flags, clamp_flags = [], [], []
    for step in steps:
        if not isinstance(step, ContactStep):
            raise ValueError("steps must hold ContactStep, got %r" % (type(step).__name__,))
        if times and not step.time_s > times[-1]:
            raise ValueError("step times must be strictly increasing (%r after %r)" % (step.time_s, times[-1]))
        robot, second, clamp = _step_properties(step.contacts, tolerance)
        times.append(step.time_s)
        robot_flags.append(robot)
        second_flags.append(second)
        clamp_flags.append(clamp)
    segments = [ContactSegment(a, b) for a, b in continuous_runs(times, robot_flags, gap)]
    clamp_runs = continuous_runs(times, clamp_flags, gap)
    clamp_duration = max((b - a for a, b in clamp_runs), default=None)
    if not segments:
        return ContactSummary([], None, None, None, None, None)
    if clamp_duration is not None and clamp_duration >= t_q:
        state = ConstraintState.CLAMP
    elif any(second_flags):
        state = ConstraintState.CONSTRAINED
    else:
        state = ConstraintState.UNCONSTRAINED
    return ContactSummary(segments=segments, first_contact_t_s=segments[0].t_start_s,
                          contact_duration_s=segments[0].duration_s,
                          longest_contact_s=max(s.duration_s for s in segments),
                          constraint_state=state, clamp_duration_s=clamp_duration)
