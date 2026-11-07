"""Exposure (00 §5.2, D9 e): was the tested robot ever near the nominal hit point while the hazard acted?

T06 calls this; it must not reimplement it. The window and the radius are arguments (the radius comes from
``semantics.yaml: exposure.low_exposure_radius_m``, the window is normally ``[t_onset, t_hit]``). Distances
use the sampled end-effector positions only, so a fast pass between two samples can be missed by up to half a
step of travel; the caller should sample at least at the control rate.
"""
from typing import Any

import numpy as np

from .._numeric import finite_array, finite_float, non_negative_float, vec3


def min_distance_to_hit_point(eef_positions: Any, times_s: Any, hit_point: Any, t_start_s: float,
                              t_end_s: float) -> float:
    """Smallest distance (m) from the end-effector samples with ``t_start_s <= t <= t_end_s`` to
    ``hit_point``. Raises if the window holds no sample."""
    times = finite_array("times_s", times_s)
    positions = finite_array("eef_positions", eef_positions)
    if times.ndim != 1 or positions.shape != (times.size, 3):
        raise ValueError("eef_positions must have shape (%d, 3) for %d times, got %s"
                         % (times.size, times.size, positions.shape))
    if times.size > 1 and not np.all(np.diff(times) > 0):
        raise ValueError("times_s must be strictly increasing")
    point = vec3("hit_point", hit_point)
    start = finite_float("t_start_s", t_start_s)
    end = finite_float("t_end_s", t_end_s)
    if end < start:
        raise ValueError("t_end_s=%r is before t_start_s=%r" % (end, start))
    inside = np.logical_and(times >= start, times <= end)
    if not np.any(inside):
        raise ValueError("no end-effector sample inside the window [%r, %r]" % (start, end))
    return float(np.min(np.linalg.norm(positions[inside] - point, axis=1)))


def is_exposure_low(eef_positions: Any, times_s: Any, hit_point: Any, t_start_s: float, t_end_s: float,
                    radius_m: float) -> bool:
    """True when the tested robot stayed farther than ``radius_m`` from ``hit_point`` throughout the
    window (strictly greater), i.e. the instance did not actually test it."""
    radius = non_negative_float("radius_m", radius_m)
    return min_distance_to_hit_point(eef_positions, times_s, hit_point, t_start_s, t_end_s) > radius
