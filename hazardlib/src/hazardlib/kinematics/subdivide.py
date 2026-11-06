"""Sub-stepping so that a hazard never moves more than ``max_step_m`` between two hook calls (R1).

The hook grid is ``t_start, t_start + hook_dt, t_start + 2 hook_dt, ...`` ending exactly at ``t_end``
(the last interval may be shorter). Displacement is the straight-line distance between the positions at
the two ends of an interval, which is what a hook-to-hook teleport actually does. ``subdivide`` keeps
every hook instant and inserts times only where an interval exceeds ``max_step_m``.
"""
import math
from typing import Any, List

import numpy as np

from .._numeric import positive_float
from .path import Path


def hook_times(t_start_s: float, t_end_s: float, hook_dt_s: float) -> np.ndarray:
    """The hook instants over ``[t_start_s, t_end_s]``; the last one is exactly ``t_end_s``."""
    dt = positive_float("hook_dt_s", hook_dt_s)
    span = t_end_s - t_start_s
    if not span > 0:
        raise ValueError("t_end_s=%r must be after t_start_s=%r" % (t_end_s, t_start_s))
    count = int(math.ceil(span / dt))
    times = t_start_s + dt * np.arange(count + 1, dtype=np.float64)
    times[-1] = t_end_s
    return times


def step_displacements(path: Path, times_s: Any) -> np.ndarray:
    """Distance travelled between consecutive entries of ``times_s`` (length ``len(times_s) - 1``)."""
    times = np.asarray(times_s, dtype=np.float64)
    pos = path.pos(times)
    return np.linalg.norm(np.diff(pos, axis=0), axis=1)


def _displacement(path: Path, t0: float, t1: float) -> float:
    return float(np.linalg.norm(path.pos(t1) - path.pos(t0)))


def needs_substeps(path: Path, max_step_m: float, hook_dt_s: float) -> bool:
    """True if some hook interval of ``path`` moves the hazard by more than ``max_step_m``."""
    limit = positive_float("max_step_m", max_step_m)
    grid = hook_times(path.t_start_s, path.t_end_s, hook_dt_s)
    return bool(np.any(step_displacements(path, grid) > limit))


def _split(path: Path, t0: float, t1: float, limit: float) -> List[float]:
    """Times after ``t0`` up to and including ``t1`` such that every step is <= ``limit``."""
    distance = _displacement(path, t0, t1)
    if distance <= limit:
        return [t1]
    count = int(math.ceil(distance / limit))
    edges = [t0 + (t1 - t0) * i / count for i in range(1, count)] + [t1]
    out = []  # type: List[float]
    previous = t0
    for edge in edges:
        out.extend(_bisect(path, previous, edge, limit))
        previous = edge
    return out


def _bisect(path: Path, t0: float, t1: float, limit: float) -> List[float]:
    out = []  # type: List[float]
    stack = [(t0, t1)]
    while stack:
        lo, hi = stack.pop()
        if _displacement(path, lo, hi) <= limit:
            out.append(hi)
            continue
        mid = 0.5 * (lo + hi)
        if not lo < mid < hi:
            raise ValueError("cannot satisfy max_step_m=%r between t=%r and t=%r (path discontinuous "
                             "or the limit is below float resolution)" % (limit, lo, hi))
        stack.append((mid, hi))
        stack.append((lo, mid))
    return out


def subdivide(path: Path, max_step_m: float, hook_dt_s: float) -> np.ndarray:
    """Strictly increasing times over the span of ``path``: every hook instant plus inserted sub-steps,
    such that the displacement between consecutive times is at most ``max_step_m``."""
    limit = positive_float("max_step_m", max_step_m)
    grid = hook_times(path.t_start_s, path.t_end_s, hook_dt_s)
    times = [float(grid[0])]
    for t0, t1 in zip(grid[:-1], grid[1:]):
        times.extend(_split(path, float(t0), float(t1), limit))
    return np.array(times, dtype=np.float64)
