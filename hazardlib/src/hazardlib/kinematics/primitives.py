"""Motion primitives in seconds and metres (world frame, z up). Pure numpy.

Time profiles (the ``s(u)`` that maps normalised time ``u in [0, 1]`` to the
fraction of the path covered):

* ``"linear"``    s = u (constant speed).
* ``"min_jerk"``  s = 10u^3 - 15u^4 + 6u^5 (starts and stops at rest).

``min_jerk`` and its derivative are *clamped*: for ``u < 0`` they hold the
start value, for ``u > 1`` the end value (derivative 0 outside ``[0, 1]``).

Every function takes all of its numbers explicitly; there are no defaults.
"""
from typing import Any

import numpy as np

from .._numeric import (finite_array, finite_float, non_negative_float, positive_float, vec3)
from .path import Path

PROFILES = ("linear", "min_jerk")


def _unit_interval(name: str, x: Any) -> np.ndarray:
    return finite_array(name, x)


def _scalar_or_array(out: np.ndarray, scalar: bool) -> Any:
    return float(out) if scalar else out


# --------------------------------------------------------------------------
# min-jerk profile
# --------------------------------------------------------------------------

def min_jerk(tau01: Any) -> Any:
    """Minimum-jerk position profile ``s(u) = 10u^3 - 15u^4 + 6u^5``, clamped to [0, 1].
    ``s(0)=0``, ``s(1)=1``, ``s(0.5)=0.5``. Scalar in -> float out; array in -> array out."""
    arr = _unit_interval("tau01", tau01)
    u = np.clip(arr, 0.0, 1.0)
    out = u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)
    return _scalar_or_array(out, arr.ndim == 0)


def min_jerk_derivative(tau01: Any) -> Any:
    """``ds/du = 30 u^2 (1-u)^2`` for ``u`` in [0, 1], 0 outside. Peak 1.875 at u = 0.5."""
    arr = _unit_interval("tau01", tau01)
    u = np.clip(arr, 0.0, 1.0)
    out = 30.0 * (u * (1.0 - u)) ** 2
    return _scalar_or_array(out, arr.ndim == 0)


def min_jerk_inverse(s01: float) -> float:
    """The ``u`` in [0, 1] with ``min_jerk(u) == s01`` (scalar), by bisection to float resolution."""
    target = finite_float("s01", s01)
    if target < 0.0 or target > 1.0:
        raise ValueError("s01 must lie in [0, 1], got %r" % (target,))
    if target == 0.0 or target == 1.0:
        return target
    lo, hi = 0.0, 1.0
    while True:
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if min_jerk(mid) < target:
            lo = mid
        else:
            hi = mid
    return lo if abs(min_jerk(lo) - target) <= abs(min_jerk(hi) - target) else hi


def profile_inverse(speed_profile: str, fraction: float) -> float:
    """Normalised time ``u`` at which a profile has covered ``fraction`` of its path."""
    _check_profile(speed_profile)
    if speed_profile == "linear":
        value = finite_float("fraction", fraction)
        if value < 0.0 or value > 1.0:
            raise ValueError("fraction must lie in [0, 1], got %r" % (value,))
        return value
    return min_jerk_inverse(fraction)


def _check_profile(speed_profile: str) -> None:
    if speed_profile not in PROFILES:
        raise ValueError("speed_profile %r is not one of %s" % (speed_profile, list(PROFILES)))


# --------------------------------------------------------------------------
# free fall
# --------------------------------------------------------------------------

def fall_height_for_tau(tau_s: Any, g: float) -> Any:
    """Height ``h = g tau^2 / 2`` from which a body released at rest takes ``tau_s`` to fall to the
    reference plane. ``g`` (m/s^2, > 0) is passed by the caller (use the engine's own value)."""
    tau = finite_array("tau_s", tau_s)
    if np.any(tau < 0):
        raise ValueError("tau_s must be >= 0")
    gravity = positive_float("g", g)
    return _scalar_or_array(0.5 * gravity * tau * tau, tau.ndim == 0)


def fall_time_for_height(h: Any, g: float) -> Any:
    """Time ``sqrt(2h/g)`` for a body released at rest to fall a height ``h`` (m, >= 0)."""
    height = finite_array("h", h)
    if np.any(height < 0):
        raise ValueError("h must be >= 0")
    gravity = positive_float("g", g)
    return _scalar_or_array(np.sqrt(2.0 * height / gravity), height.ndim == 0)


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

class LinePath(Path):
    """Straight segment ``p_start -> p_end`` over ``[t_start_s, t_end_s]`` with a time profile and an
    optional parabolic lift ``4 H s (1 - s)`` along +z (``H`` = ``arc_height_m``, the extra height at
    mid-path). The lift depends on the path fraction ``s``, so its shape does not depend on the profile."""

    def __init__(self, p_start: Any, p_end: Any, t_start_s: float, t_end_s: float,
                 speed_profile: str, arc_height_m: float) -> None:
        super().__init__(t_start_s, t_end_s)
        _check_profile(speed_profile)
        self.p_start = vec3("p_start", p_start)
        self.p_end = vec3("p_end", p_end)
        self.speed_profile = speed_profile
        self.arc_height_m = non_negative_float("arc_height_m", arc_height_m)
        self._delta = self.p_end - self.p_start

    def _fraction(self, t: np.ndarray):
        u = (t - self.t_start_s) / self.duration_s
        if self.speed_profile == "linear":
            return u, np.ones_like(u)
        return min_jerk(u), min_jerk_derivative(u)

    def _pos_clamped(self, t: np.ndarray) -> np.ndarray:
        s, _ = self._fraction(t)
        pos = self.p_start + np.outer(s, self._delta)
        pos[:, 2] += 4.0 * self.arc_height_m * s * (1.0 - s)
        return pos

    def _vel_inside(self, t: np.ndarray) -> np.ndarray:
        s, ds = self._fraction(t)
        vel = np.outer(ds, self._delta) / self.duration_s
        vel[:, 2] += 4.0 * self.arc_height_m * (1.0 - 2.0 * s) * ds / self.duration_s
        return vel


class BallisticPath(Path):
    """``p0 + v0 (t - t0) + a (t - t0)^2 / 2`` over ``[t_start_s, t_end_s]`` (``t0 = t_start_s``)."""

    def __init__(self, p0: Any, v0: Any, accel: Any, t_start_s: float, t_end_s: float) -> None:
        super().__init__(t_start_s, t_end_s)
        self.p0 = vec3("p0", p0)
        self.v0 = vec3("v0", v0)
        self.accel = vec3("accel", accel)

    def _pos_clamped(self, t: np.ndarray) -> np.ndarray:
        dt = t - self.t_start_s
        return self.p0 + np.outer(dt, self.v0) + 0.5 * np.outer(dt * dt, self.accel)

    def _vel_inside(self, t: np.ndarray) -> np.ndarray:
        return self.v0 + np.outer(t - self.t_start_s, self.accel)


def straight_path(p_start: Any, p_end: Any, t_start: float, t_end: float) -> LinePath:
    """Straight line at constant speed ``|p_end - p_start| / (t_end - t_start)``."""
    return LinePath(p_start, p_end, t_start, t_end, "linear", 0.0)


def arc_path(p_start: Any, p_end: Any, t_start: float, t_end: float, arc_height: float) -> LinePath:
    """Constant-speed (in path fraction) motion whose z is lifted by a parabola of height
    ``arc_height`` m above the straight line at mid-path (``arc_height >= 0``)."""
    return LinePath(p_start, p_end, t_start, t_end, "linear", arc_height)


def reach_path(p_origin: Any, p_end: Any, t_start: float, t_end: float, speed_profile: str) -> LinePath:
    """A limb reaching from ``p_origin`` to ``p_end`` with ``speed_profile`` ("min_jerk": starts and
    ends at rest; "linear": constant speed)."""
    return LinePath(p_origin, p_end, t_start, t_end, speed_profile, 0.0)


def free_fall_path(p_release: Any, t_release_s: float, t_end_s: float, g: float) -> BallisticPath:
    """Fall from rest at ``p_release`` at time ``t_release_s`` under gravity ``g`` along -z."""
    gravity = positive_float("g", g)
    return BallisticPath(p_release, np.zeros(3), np.array([0.0, 0.0, -gravity]), t_release_s, t_end_s)
