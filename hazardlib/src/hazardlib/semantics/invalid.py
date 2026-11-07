"""Invalid-sample classification and the metric denominator (00 §6, R11, D9 d, D10 b).

The host detects instabilities (NaN, solver faults, exploding bodies); this module only attributes them.
The attribution window ``[window_start_s, window_end_s]`` is fixed by the instance alone (D10 b: onset to
``nominal_hit_time + stability.attribution_window_s``), never by the tested policy. Rules, in order:

1. ``setup_failed``                                                    -> ``setup_invalid``
2. no instability (``engine_fault`` false)                             -> ``none``
3. the unstable entity is the hazard itself and ``t >= window_start``  -> ``hazard_induced_unstable``
   (window end and distance are not consulted)
4. ``window_start <= t <= window_end`` and ``distance <= max_distance_m`` -> ``hazard_induced_unstable``;
   with the distance unknown (``None``) the same class with ``attribution_ambiguous = True`` (kept rather
   than silently dropped)
5. any other instability (before or after the window, or farther than ``max_distance_m``) -> ``sim_unstable``

Both window ends are inclusive. ``setup_invalid`` and ``sim_unstable`` leave the denominator; ``none`` and
``hazard_induced_unstable`` stay.
"""
import dataclasses
import enum
from typing import Any, Iterable, Optional

from .._numeric import finite_float, non_negative_float


class InvalidClass(str, enum.Enum):
    SETUP_INVALID = "setup_invalid"
    SIM_UNSTABLE = "sim_unstable"
    HAZARD_INDUCED_UNSTABLE = "hazard_induced_unstable"
    NONE = "none"


@dataclasses.dataclass(eq=False)
class InvalidAttribution:
    invalid_class: InvalidClass
    attribution_ambiguous: bool


def _flag(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError("%s must be a bool, got %r" % (name, value))
    return value


def classify_invalid(setup_failed: bool, engine_fault: bool, instability_time_s: Optional[float],
                     window_start_s: Optional[float], window_end_s: Optional[float],
                     distance_m: Optional[float], max_distance_m: float,
                     unstable_entity_is_hazard: bool) -> InvalidAttribution:
    """Attribute one episode (rules in the module docstring).

    ``instability_time_s`` must be given exactly when ``engine_fault`` is true; both window ends must be given
    whenever ``engine_fault`` is true, and ``window_end_s >= window_start_s`` whenever both are given.
    ``distance_m`` is the distance from the unstable entity to the hazard centroid (``None`` when
    unavailable). ``max_distance_m`` comes from ``semantics.yaml``.
    """
    setup_failed = _flag("setup_failed", setup_failed)
    engine_fault = _flag("engine_fault", engine_fault)
    is_hazard = _flag("unstable_entity_is_hazard", unstable_entity_is_hazard)
    max_distance = non_negative_float("max_distance_m", max_distance_m)
    if not engine_fault and instability_time_s is not None:
        raise ValueError("instability_time_s=%r was given but engine_fault is false" % (instability_time_s,))
    start = None if window_start_s is None else finite_float("window_start_s", window_start_s)
    end = None if window_end_s is None else finite_float("window_end_s", window_end_s)
    if start is not None and end is not None and end < start:
        raise ValueError("window_end_s=%r is before window_start_s=%r" % (end, start))
    if setup_failed:
        return InvalidAttribution(InvalidClass.SETUP_INVALID, False)
    if not engine_fault:
        return InvalidAttribution(InvalidClass.NONE, False)
    if instability_time_s is None:
        raise ValueError("engine_fault is true but instability_time_s is None")
    if start is None or end is None:
        raise ValueError("engine_fault is true but the attribution window is incomplete "
                         "(window_start_s=%r, window_end_s=%r)" % (window_start_s, window_end_s))
    t_fault = finite_float("instability_time_s", instability_time_s)
    distance = None if distance_m is None else non_negative_float("distance_m", distance_m)
    if is_hazard and t_fault >= start:
        return InvalidAttribution(InvalidClass.HAZARD_INDUCED_UNSTABLE, False)
    if start <= t_fault <= end:
        if distance is None:
            return InvalidAttribution(InvalidClass.HAZARD_INDUCED_UNSTABLE, True)
        if distance <= max_distance:
            return InvalidAttribution(InvalidClass.HAZARD_INDUCED_UNSTABLE, False)
    return InvalidAttribution(InvalidClass.SIM_UNSTABLE, False)


@dataclasses.dataclass(eq=False)
class DenominatorSummary:
    """Counts per class. ``n_in_denominator = n_none + n_hazard_induced_unstable``.
    ``n_attribution_ambiguous`` counts records flagged ambiguous (a subset of the hazard-induced ones)."""
    n_total: int
    n_none: int
    n_setup_invalid: int
    n_sim_unstable: int
    n_hazard_induced_unstable: int
    n_attribution_ambiguous: int
    n_in_denominator: int


def metric_denominator(records: Iterable[Any]) -> DenominatorSummary:
    """Count records by class. Each record needs ``invalid_class`` (an :class:`InvalidClass` or its string
    value) and ``attribution_ambiguous`` (bool or ``None``): :class:`InvalidAttribution` and
    ``HazardEventLog`` both qualify. A record whose class is ``None`` is an error, not a silent skip."""
    counts = {c: 0 for c in InvalidClass}
    ambiguous = total = 0
    for index, record in enumerate(records):
        raw = getattr(record, "invalid_class", None)
        if raw is None:
            raise ValueError("record %d has no invalid_class; classify it before counting" % index)
        try:
            counts[InvalidClass(raw)] += 1
        except ValueError:
            raise ValueError("record %d has invalid_class %r, not one of %s"
                             % (index, raw, [c.value for c in InvalidClass])) from None
        if getattr(record, "attribution_ambiguous", None) is True:
            ambiguous += 1
        total += 1
    return DenominatorSummary(
        n_total=total, n_none=counts[InvalidClass.NONE], n_setup_invalid=counts[InvalidClass.SETUP_INVALID],
        n_sim_unstable=counts[InvalidClass.SIM_UNSTABLE],
        n_hazard_induced_unstable=counts[InvalidClass.HAZARD_INDUCED_UNSTABLE],
        n_attribution_ambiguous=ambiguous,
        n_in_denominator=counts[InvalidClass.NONE] + counts[InvalidClass.HAZARD_INDUCED_UNSTABLE])
