"""Near-miss metrics from a per-step trace (00 §6): minimum separation and time to collision.

``separation_kind`` says what the distance is: ``geometric`` when the host supplied a true-geometry
minimum distance, otherwise ``surrogate`` (distance between the two origins, which is not a surface
distance). A surrogate is never presented as geometric.

``ttc_series[i]`` is ``separation[i] / closing_speed[i]`` with the closing speed taken as the backward
difference ``-(separation[i] - separation[i-1]) / (t[i] - t[i-1])``. It is ``None`` when it is undefined:
at the first sample (no previous one) and whenever the pair is not approaching (closing speed <= 0). No
epsilon is applied: a closing speed that is positive only because of noise gives a huge value, which
``ttc_min_s`` ignores in favour of any smaller one.
"""
import dataclasses
import enum
from typing import Any, List, Optional

import numpy as np

from .._numeric import finite_array


class SeparationKind(str, enum.Enum):
    GEOMETRIC = "geometric"
    SURROGATE = "surrogate"


@dataclasses.dataclass(eq=False)
class ProximitySummary:
    min_separation_m: float
    t_min_separation_s: float
    separation_kind: SeparationKind
    separation_m: np.ndarray
    ttc_series_s: List[Optional[float]]
    ttc_min_s: Optional[float]


def compute_proximity(times_s: Any, hazard_pos: Any, eef_pos: Any,
                      geometric_distance_m: Optional[Any]) -> ProximitySummary:
    """Near-miss summary of one episode trace.

    ``times_s`` strictly increasing (N >= 1); ``hazard_pos`` and ``eef_pos`` have shape (N, 3). When
    ``geometric_distance_m`` (shape (N,), finite, >= 0) is given it is used and the kind is ``geometric``;
    pass ``None`` to fall back to the origin distance (``surrogate``). A partly missing geometric series
    is rejected rather than mixed with the surrogate.
    """
    times = finite_array("times_s", times_s)
    if times.ndim != 1 or times.size < 1:
        raise ValueError("times_s must be a non-empty 1-D array, got shape %s" % (times.shape,))
    if times.size > 1 and not np.all(np.diff(times) > 0):
        raise ValueError("times_s must be strictly increasing")
    hazard = finite_array("hazard_pos", hazard_pos)
    eef = finite_array("eef_pos", eef_pos)
    for name, arr in (("hazard_pos", hazard), ("eef_pos", eef)):
        if arr.shape != (times.size, 3):
            raise ValueError("%s must have shape (%d, 3), got %s" % (name, times.size, arr.shape))
    if geometric_distance_m is None:
        separation = np.linalg.norm(hazard - eef, axis=1)
        kind = SeparationKind.SURROGATE
    else:
        separation = finite_array("geometric_distance_m", geometric_distance_m)
        if separation.shape != times.shape:
            raise ValueError("geometric_distance_m must have shape %s, got %s" % (times.shape, separation.shape))
        if np.any(separation < 0):
            raise ValueError("geometric_distance_m must be >= 0")
        separation = separation.copy()
        kind = SeparationKind.GEOMETRIC
    ttc = [None]  # type: List[Optional[float]]
    for i in range(1, times.size):
        closing = -(separation[i] - separation[i - 1]) / (times[i] - times[i - 1])
        ttc.append(float(separation[i] / closing) if closing > 0 else None)
    defined = [v for v in ttc if v is not None]
    index = int(np.argmin(separation))
    return ProximitySummary(min_separation_m=float(separation[index]), t_min_separation_s=float(times[index]),
                            separation_kind=kind, separation_m=separation, ttc_series_s=ttc,
                            ttc_min_s=min(defined) if defined else None)
