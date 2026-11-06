"""Nominal-collision-by-construction (paradigm 00 §5.1-§5.2, decisions D1-D6 in §13): given a hit point
and a hit time on the nominal end-effector path, solve the hazard **centroid** path so that a robot that
does not react meets the hazard at ``t_hit``, with ``onset = t_hit - tau``.

Conventions shared by all families (T06 relies on these):

* **tau and t_hit follow the first surface contact (D1).** ``t_hit`` is the instant the hazard's surface
  first touches the robot, not the instant the centroids coincide. This module works on **centroids**:
  ``hit_point`` is the centroid target at ``t_hit``, so the caller converts before calling,
  ``hit_point = contact_point + standoff * n_hat`` with ``standoff = r_hazard(n_hat) + r_eef_contact``
  (``r_hazard`` the hazard's support radius along the approach direction, ``r_eef_contact`` the host's
  measured effective contact radius, BD field ``eef_contact_radius_m``), and records ``standoff_m`` in the
  instance provenance. Exposure offsets (``HazardSpec.exposure.offset_m``) are added by the caller too.
  Nothing is added here. Conformance check C13 verifies the conversion on each host.
* ``nominal_ee_path`` is used to check that ``t_hit`` lies in its span and to report diagnostics; it
  does not move the hit point. ``diagnostics["nominal_ee_offset_m"]`` is the distance from ``hit_point`` to
  where the nominal path actually is at ``t_hit``; with the conversion above it is expected to be about
  ``standoff_m``.
* ``onset_time_s = t_hit - tau_s`` is where the returned path starts: the instant the hazard starts to
  move (D5). It need not be visible then; the visible lead time tau_vis is a gate diagnostic.
  ``diagnostics["onset_margin_s"]`` is ``onset - nominal.t_start``; a negative value means the onset
  precedes the nominal path, which the gates must reject.
* Directions are world-frame vectors (normalised here).
* **Release velocity (D4, R7).** For ``slide`` and ``reach`` it is the backward difference of the instance
  path over a **fixed window** ``release_window_s`` ending at the release instant, with no floor. The
  caller passes the value of ``release_velocity_window_s`` from ``runtime.yaml``, the same on every
  host. It is **not** the engine hook step: using the hook step would give the same instance different
  release velocities on different engines. The runtime applies the same window when it rolls back at the
  first overlap (R8); the value returned here is the nominal one. ``diagnostics["release_window_s"]``
  records the window used (present for ``slide`` and ``reach`` only).

Families and their ``params`` (all keys required, no extras; unknown or renamed keys are errors):

``intrusion.fall``   ``g``
    Released **at rest at the onset** from ``hit_point + (0, 0, g tau^2 / 2)`` and left to the engine's
    gravity: it reaches ``hit_point`` at ``t_hit``. ``release_time_s = onset``, release velocity 0 (D2).
    The body is dynamic from the onset, so there is no kinematic-overlap phase and R8's rollback does not
    apply. ``g`` must be the host's **measured** free-fall acceleration (BD ``gravity_mps2``); it has no
    default and must not be assumed to be 9.81. Hazard linear and angular damping must be 0 on the host.
    The returned path is the analytic free fall over ``[onset, t_hit]``.
``intrusion.slide``  ``direction``, ``speed_mps``, ``release_window_s``
    Constant speed along ``direction`` from ``hit_point - direction * speed * tau`` (reached at the
    onset) to ``hit_point``. Kinematic until ``t_hit``, then released with the backward difference over
    ``release_window_s`` (about ``speed * direction``).
``intrusion.reach``  ``direction``, ``reach_distance_m``, ``overshoot_m``, ``speed_profile``,
                     ``release_window_s``
    A limb tip that starts at ``hit_point - direction * reach_distance_m`` at the onset and moves toward
    ``hit_point + direction * overshoot_m`` with ``speed_profile`` in ``{min_jerk, linear}``. The total
    duration is solved so that the tip passes ``hit_point`` exactly at ``t_hit``. Kinematic until
    ``t_hit``, then released as for ``slide``. **``overshoot_m = 0`` with ``min_jerk`` ends at rest
    (analytic end speed 0): that is quasi-static occupancy, not an impact, and must not be used for an
    instance that carries severity (D6); such an instance belongs to ``occupancy.dwell``.** This module
    does not enforce it (T06 rejects it).
"""
import dataclasses
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .._numeric import (finite_float, non_negative_float, positive_float, unit_vec3, vec3)
from .path import Path
from .primitives import (PROFILES, fall_height_for_tau, free_fall_path, profile_inverse,
                         reach_path, straight_path)

FAMILY_FALL = "intrusion.fall"
FAMILY_SLIDE = "intrusion.slide"
FAMILY_REACH = "intrusion.reach"
FAMILIES = (FAMILY_FALL, FAMILY_SLIDE, FAMILY_REACH)


@dataclasses.dataclass(eq=False)
class HitPlan:
    """Result of :func:`build_hit_path`.

    ``path`` is the hazard centroid path starting at ``onset_time_s`` (it may extend past
    ``nominal_hit_time_s``; the runtime releases at ``release_time_s`` anyway). ``release_velocity`` is
    linear only (angular velocity is the caller's, normally zero). ``hit_velocity`` is the analytic
    ``path.vel(t_hit)``. ``diagnostics`` always has ``nominal_ee_offset_m``, ``onset_margin_s``,
    ``travel_distance_m``, ``hit_speed_mps`` and ``relative_speed_at_hit_mps``, plus ``release_window_s``
    for ``slide`` and ``reach`` (``fall`` is released at rest and uses no window).
    """
    family: str
    path: Path
    tau_s: float
    onset_time_s: float
    release_time_s: float
    release_velocity: np.ndarray
    nominal_hit_time_s: float
    nominal_hit_point: np.ndarray
    hit_velocity: np.ndarray
    diagnostics: Dict[str, float]


def _take_params(family: str, params: Any, keys: Sequence[str]) -> Mapping[str, Any]:
    if not isinstance(params, Mapping):
        raise ValueError("%s: params must be a mapping, got %r" % (family, params))
    missing = [k for k in keys if k not in params]
    extra = sorted(str(k) for k in params if k not in keys)
    if missing or extra:
        raise ValueError("%s: params must be exactly %s (missing: %s; unexpected: %s)"
                         % (family, list(keys), missing, extra))
    return params


def _build_fall(t_hit: float, hit_point: np.ndarray, tau: float, params: Mapping[str, Any]):
    gravity = positive_float("params.g", params["g"])
    onset = t_hit - tau
    release_point = hit_point + np.array([0.0, 0.0, float(fall_height_for_tau(tau, gravity))])
    path = free_fall_path(release_point, onset, t_hit, gravity)
    return path, onset, np.zeros(3)


def _build_slide(t_hit: float, hit_point: np.ndarray, tau: float, params: Mapping[str, Any]):
    direction = unit_vec3("params.direction", params["direction"])
    speed = positive_float("params.speed_mps", params["speed_mps"])
    window = positive_float("params.release_window_s", params["release_window_s"])
    onset = t_hit - tau
    start = hit_point - direction * (speed * tau)
    path = straight_path(start, hit_point, onset, t_hit)
    return path, t_hit, _release_velocity(path, t_hit, window)


def _build_reach(t_hit: float, hit_point: np.ndarray, tau: float, params: Mapping[str, Any]):
    direction = unit_vec3("params.direction", params["direction"])
    reach = positive_float("params.reach_distance_m", params["reach_distance_m"])
    overshoot = non_negative_float("params.overshoot_m", params["overshoot_m"])
    profile = params["speed_profile"]
    if profile not in PROFILES:
        raise ValueError("params.speed_profile %r is not one of %s" % (profile, list(PROFILES)))
    window = positive_float("params.release_window_s", params["release_window_s"])
    onset = t_hit - tau
    origin = hit_point - direction * reach
    end = hit_point + direction * overshoot
    u_hit = profile_inverse(profile, reach / (reach + overshoot))
    total_s = tau / u_hit
    path = reach_path(origin, end, onset, onset + total_s, profile)
    return path, t_hit, _release_velocity(path, t_hit, window)


def _release_velocity(path: Path, t_release: float, window: float) -> np.ndarray:
    if t_release - window < path.t_start_s:
        raise ValueError("params.release_window_s=%r exceeds the kinematic phase (%r s before the release)"
                         % (window, t_release - path.t_start_s))
    return path.finite_difference_velocity(t_release, window)


_BUILDERS = {
    FAMILY_FALL: (("g",), _build_fall),
    FAMILY_SLIDE: (("direction", "speed_mps", "release_window_s"), _build_slide),
    FAMILY_REACH: (("direction", "reach_distance_m", "overshoot_m", "speed_profile",
                    "release_window_s"),
                   _build_reach),
}  # type: Dict[str, Any]


def build_hit_path(family: str, nominal_ee_path: Path, t_hit: float, hit_point: Any, tau_s: float,
                   params: Mapping[str, Any]) -> HitPlan:
    """Solve the hazard path of ``family`` for the given nominal hit (see the module docstring)."""
    if family not in _BUILDERS:
        raise ValueError("unsupported family %r; supported: %s" % (family, list(FAMILIES)))
    if not isinstance(nominal_ee_path, Path):
        raise ValueError("nominal_ee_path must be a hazardlib.kinematics.Path, got %r"
                         % (type(nominal_ee_path).__name__,))
    t_hit = finite_float("t_hit", t_hit)
    if t_hit < nominal_ee_path.t_start_s or t_hit > nominal_ee_path.t_end_s:
        raise ValueError("t_hit=%r is outside the nominal EEF path span [%r, %r]"
                         % (t_hit, nominal_ee_path.t_start_s, nominal_ee_path.t_end_s))
    hit_point = vec3("hit_point", hit_point)
    tau = positive_float("tau_s", tau_s)
    keys, builder = _BUILDERS[family]
    params = _take_params(family, params, keys)

    path, release_time, release_velocity = builder(t_hit, hit_point, tau, params)
    onset = t_hit - tau
    hit_velocity = path.vel(t_hit)
    ee_pos = nominal_ee_path.pos(t_hit)
    ee_vel = nominal_ee_path.vel(t_hit)
    diagnostics = {
        "nominal_ee_offset_m": float(np.linalg.norm(hit_point - ee_pos)),
        "onset_margin_s": float(onset - nominal_ee_path.t_start_s),
        "travel_distance_m": float(np.linalg.norm(hit_point - path.pos(onset))),
        "hit_speed_mps": float(np.linalg.norm(hit_velocity)),
        "relative_speed_at_hit_mps": float(np.linalg.norm(hit_velocity - ee_vel)),
    }
    if "release_window_s" in params:
        diagnostics["release_window_s"] = float(params["release_window_s"])
    return HitPlan(family=family, path=path, tau_s=tau, onset_time_s=onset,
                   release_time_s=float(release_time), release_velocity=release_velocity,
                   nominal_hit_time_s=t_hit, nominal_hit_point=hit_point,
                   hit_velocity=hit_velocity, diagnostics=diagnostics)
