"""Time-parameterised point paths (seconds, metres, world frame, z up).

A ``Path`` is the motion of one point (a hazard centroid or the end effector)
over ``[t_start_s, t_end_s]``:

* ``pos(t)``  position. Outside the span the path is **clamped**: before the
  start it sits at the first point, after the end at the last one (a motion that
  has not started or has finished is at rest). Callers that must stay inside the
  span check ``t_start_s``/``t_end_s`` themselves.
* ``vel(t)``  analytic velocity, **zero outside the closed span** and the
  one-sided analytic derivative at the end points (so ``vel(t_end_s)`` is the
  speed the path arrives with, not zero).
* ``quat(t)`` ``wxyz`` orientation, or ``None`` for paths without orientation
  (only ``SampledPath`` built with quaternions has one).
* ``finite_difference_velocity(t, dt)`` backward difference over the last hook
  step (rule R7). It never applies a speed floor.

``t`` may be a scalar (result shape ``(3,)``) or an array (shape ``t.shape + (3,)``).
Subclasses implement ``_pos_clamped`` / ``_vel_inside`` on flat arrays.
"""
from typing import Any, Optional, Tuple

import numpy as np

from .._numeric import finite_array, finite_float, positive_float


def _flatten(name: str, t: Any) -> Tuple[np.ndarray, Tuple[int, ...]]:
    arr = finite_array(name, t)
    return arr.reshape(-1), arr.shape


class Path:
    def __init__(self, t_start_s: float, t_end_s: float) -> None:
        t0 = finite_float("t_start_s", t_start_s)
        t1 = finite_float("t_end_s", t_end_s)
        if not t1 > t0:
            raise ValueError("t_end_s=%r must be after t_start_s=%r" % (t1, t0))
        self.t_start_s = t0
        self.t_end_s = t1

    @property
    def duration_s(self) -> float:
        return self.t_end_s - self.t_start_s

    @property
    def has_orientation(self) -> bool:
        return False

    # -- implemented by subclasses -----------------------------------------
    def _pos_clamped(self, t: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _vel_inside(self, t: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _quat_clamped(self, t: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # -- public queries ------------------------------------------------------
    def pos(self, t: Any) -> np.ndarray:
        flat, shape = _flatten("t", t)
        clamped = np.clip(flat, self.t_start_s, self.t_end_s)
        return self._pos_clamped(clamped).reshape(shape + (3,))

    def vel(self, t: Any) -> np.ndarray:
        flat, shape = _flatten("t", t)
        inside = np.logical_and(flat >= self.t_start_s, flat <= self.t_end_s)
        out = np.zeros((flat.size, 3), dtype=np.float64)
        if np.any(inside):
            out[inside] = self._vel_inside(flat[inside])
        return out.reshape(shape + (3,))

    def quat(self, t: Any) -> Optional[np.ndarray]:
        if not self.has_orientation:
            return None
        flat, shape = _flatten("t", t)
        clamped = np.clip(flat, self.t_start_s, self.t_end_s)
        return self._quat_clamped(clamped).reshape(shape + (4,))

    def sample(self, times_s: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """``(pos, quat_or_None)`` at ``times_s``."""
        return self.pos(times_s), self.quat(times_s)

    def finite_difference_velocity(self, t: float, dt: float) -> np.ndarray:
        """``(pos(t) - pos(t - dt)) / dt``: the velocity of the last hook step ending at ``t``.

        R7: this is the release velocity handed to dynamics. No lower bound is
        applied; a slow or stationary path yields a small or zero velocity.
        ``[t - dt, t]`` must lie inside the span: a clamped step would silently
        under-report the speed, so it is an error instead.
        """
        t = finite_float("t", t)
        dt = positive_float("dt", dt)
        if t - dt < self.t_start_s or t > self.t_end_s:
            raise ValueError(
                "step [%r, %r] is not inside the path span [%r, %r]"
                % (t - dt, t, self.t_start_s, self.t_end_s))
        return (self.pos(t) - self.pos(t - dt)) / dt


class SampledPath(Path):
    """Piecewise-linear path through ``(times_s[i], pos[i])`` samples.

    Velocity is the slope of the segment containing ``t`` (piecewise constant;
    at an interior sample the segment to its right, at the last sample the last
    segment). Orientation, if given, is interpolated by normalised linear
    interpolation after flipping to the shorter arc.
    """

    def __init__(self, times_s: Any, pos: Any, quat: Optional[Any]) -> None:
        times = finite_array("times_s", times_s)
        if times.ndim != 1 or times.size < 2:
            raise ValueError("times_s must be a 1-D array with at least 2 samples, got shape %s"
                             % (times.shape,))
        if not np.all(np.diff(times) > 0):
            raise ValueError("times_s must be strictly increasing")
        points = finite_array("pos", pos)
        if points.shape != (times.size, 3):
            raise ValueError("pos must have shape (%d, 3), got %s" % (times.size, points.shape))
        quats = None
        if quat is not None:
            quats = finite_array("quat", quat)
            if quats.shape != (times.size, 4):
                raise ValueError("quat must have shape (%d, 4), got %s" % (times.size, quats.shape))
            if not np.all(np.linalg.norm(quats, axis=1) > 0):
                raise ValueError("quat contains a zero quaternion")
        super().__init__(float(times[0]), float(times[-1]))
        self.times_s = times.copy()
        self.points = points.copy()
        self.quats = None if quats is None else quats.copy()
        self._seg_dt = np.diff(self.times_s)
        self._seg_vel = np.diff(self.points, axis=0) / self._seg_dt[:, None]

    @classmethod
    def from_pose_sequence(cls, seq: Any) -> "SampledPath":
        """From anything with ``time_s``, ``pos`` and ``quat`` (e.g. ``spec.PoseSequence``)."""
        return cls(seq.time_s, seq.pos, seq.quat)

    @property
    def has_orientation(self) -> bool:
        return self.quats is not None

    def _segment(self, t: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.times_s, t, side="right") - 1
        return np.clip(idx, 0, self.times_s.size - 2)

    def _pos_clamped(self, t: np.ndarray) -> np.ndarray:
        return np.stack([np.interp(t, self.times_s, self.points[:, axis]) for axis in range(3)],
                        axis=-1)

    def _vel_inside(self, t: np.ndarray) -> np.ndarray:
        return self._seg_vel[self._segment(t)]

    def _quat_clamped(self, t: np.ndarray) -> np.ndarray:
        idx = self._segment(t)
        frac = ((t - self.times_s[idx]) / self._seg_dt[idx])[:, None]
        q0 = self.quats[idx]
        q1 = self.quats[idx + 1]
        sign = np.where(np.sum(q0 * q1, axis=1) < 0, -1.0, 1.0)[:, None]
        mixed = (1.0 - frac) * q0 + frac * sign * q1
        return mixed / np.linalg.norm(mixed, axis=1, keepdims=True)
