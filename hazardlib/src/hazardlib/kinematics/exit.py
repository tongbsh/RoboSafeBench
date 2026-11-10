"""Exit paths of a hazard that is not released (decision D19): what a slide / reach does after its nominal hit
time when the robot did not meet it. Pure numpy, no defaults.

* :func:`straight_exit_steps` / :func:`bounding_radius_m`: a hazard that keeps its release velocity ("pass")
  leaves the workspace box widened by its bounding-sphere radius; the exit is sampled on the hook grid.
* :class:`RetractPath`: a hand that retracts. The analytic primitive path is followed on to its own end when it
  continues past the hit (a reach with overshoot), then the whole path is retraced, time-reversed, back to its
  start pose.
"""
import math
from typing import Any, Sequence

import numpy as np

from .._numeric import finite_array, finite_float, positive_float, vec3
from .path import Path


class RetractPath(Path):
    """``base`` over ``[t_from, t_turn]`` (``t_turn = base.t_end_s``: the primitive's own end), then ``base``
    time-reversed back to its start: ``pos(t) = base.pos(2 t_turn - t)`` over
    ``[t_turn, t_turn + (t_turn - base.t_start_s)]``. Without a continuation (``t_from == t_turn``) the retrace
    starts at once from the pose at ``t_from``. The velocity of the retrace is the negated base velocity."""

    def __init__(self, base: Path, t_from: float) -> None:
        t_from = finite_float("t_from", t_from)
        if not base.t_start_s < t_from <= base.t_end_s:
            raise ValueError("t_from=%r must lie in (%r, %r]" % (t_from, base.t_start_s, base.t_end_s))
        self.base = base
        self.t_turn_s = base.t_end_s
        super().__init__(t_from, self.t_turn_s + (self.t_turn_s - base.t_start_s))

    def _mirror(self, t: np.ndarray) -> np.ndarray:
        return np.where(t <= self.t_turn_s, t, 2.0 * self.t_turn_s - t)

    def _pos_clamped(self, t: np.ndarray) -> np.ndarray:
        return self.base.pos(self._mirror(t))

    def _vel_inside(self, t: np.ndarray) -> np.ndarray:
        sign = np.where(t < self.t_turn_s, 1.0, -1.0)[:, None]
        return sign * self.base.vel(self._mirror(t))


def bounding_radius_m(shape: str, half_extents_m: Sequence[float]) -> float:
    """Radius of the sphere around the centre that contains the body: the radius of a sphere, the half
    diagonal of a box."""
    h = finite_array("half_extents_m", half_extents_m)
    if shape == "sphere":
        return float(np.max(h))
    if shape == "box":
        return float(np.linalg.norm(h))
    raise ValueError("no bounding radius for shape %r (sphere, box)" % (shape,))


def _strictly_outside(p: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> bool:
    return bool(np.any(p < lo) or np.any(p > hi))


def straight_exit_steps(p0: Any, velocity: Any, lo: Any, hi: Any, hook_dt_s: float) -> int:
    """Smallest ``n >= 1`` such that ``p0 + velocity * (n hook_dt_s)`` is strictly outside the box
    ``[lo, hi]`` (a point on a line that has left a convex box stays outside). The velocity must be non-zero."""
    p, v = vec3("p0", p0), vec3("velocity", velocity)
    lo, hi = vec3("lo", lo), vec3("hi", hi)
    dt = positive_float("hook_dt_s", hook_dt_s)
    if not np.any(v != 0.0):
        raise ValueError("a hazard with zero release velocity never leaves the workspace")
    last_inside = math.inf  # last instant at which the line is not strictly outside the box
    for axis in range(3):
        if v[axis] == 0.0:
            if p[axis] < lo[axis] or p[axis] > hi[axis]:
                last_inside = -math.inf
            continue
        last_inside = min(last_inside, max((lo[axis] - p[axis]) / v[axis], (hi[axis] - p[axis]) / v[axis]))
    n = 1 if last_inside < 0 else int(math.floor(last_inside / dt)) + 1
    while not _strictly_outside(p + v * (n * dt), lo, hi):  # round-off at the boundary
        n += 1
    return n
