"""``exposure_low`` of an episode (paradigm §5.2, D9 e): delegated to T03's ``is_exposure_low``.

Window ``[onset_time_s, nominal_hit_time_s]``; the point is the instance's ``p*``
(``provenance.anchor_point``: the nominal EEF reference at ``t_hit`` plus the exposure offset, i.e. where the
non-reacting robot is hit; the hazard centroid ``nominal_hit_point`` is ``standoff_m`` away from it); the
radius is ``semantics.yaml: exposure.low_exposure_radius_m``.
"""
import dataclasses
from typing import Any, Mapping, Optional

from .._numeric import non_negative_float
from .._settings import ValueReader
from ..config import load_config
from ..semantics import exposure as _exposure
from ..semantics.event_log import HazardEventLog
from ..spec.instance import HazardInstance


def low_exposure_radius_m(semantics: Mapping[str, Any]) -> float:
    reader = ValueReader("exposure")
    value = reader.get("semantics", semantics, "exposure", "low_exposure_radius_m")
    reader.done()
    return non_negative_float("exposure.low_exposure_radius_m", value)


def load_low_exposure_radius_m(config_dir: Optional[Any] = None) -> float:
    return low_exposure_radius_m(load_config("semantics", config_dir=config_dir))


def exposure_low(instance: HazardInstance, eef_positions: Any, times_s: Any, radius_m: float) -> bool:
    if "anchor_point" not in instance.provenance:
        raise ValueError("instance %r has no provenance.anchor_point (p*)" % (instance.instance_id,))
    return _exposure.is_exposure_low(eef_positions, times_s, instance.provenance["anchor_point"],
                                     instance.onset_time_s, instance.nominal_hit_time_s, radius_m)


def annotate_exposure(log: HazardEventLog, instance: HazardInstance, eef_positions: Any, times_s: Any,
                      radius_m: float) -> HazardEventLog:
    """A copy of ``log`` with ``exposure_low`` filled in (the runtime leaves it ``None``)."""
    if log.instance_id != instance.instance_id:
        raise ValueError("log %r does not belong to instance %r" % (log.instance_id, instance.instance_id))
    return dataclasses.replace(log, exposure_low=exposure_low(instance, eef_positions, times_s, radius_m))
