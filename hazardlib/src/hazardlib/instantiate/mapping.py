"""Intensity vector -> motion-primitive parameters (T06). Pure functions, every number explicit.

The intensity axes are those of ``configs/hazard/intensity.yaml`` (00 §4). The family solvers are T02's
``build_hit_path``; this module only decides which primitive parameters realise a given intensity:

``intrusion.fall`` (axes ``tau_s``, optional ``mass_kg``)
    Release height above the centroid hit point ``h = g tau^2 / 2`` with ``g`` the BD's measured
    ``gravity_mps2`` (D2). ``mass_kg``, if given, must equal ``agent.mass_kg`` (checked by the builder).
``intrusion.slide`` (axes ``tau_s``, ``approach_speed_mps``)
    Constant speed ``v = approach_speed_mps`` on the support surface; start distance ``v tau`` before the
    hit point.
``intrusion.reach`` (axes ``tau_s``, ``approach_speed_mps``, optional ``limb_size_m``)
    ``approach_speed_mps`` is the tip speed at the nominal hit. With ``kappa = overshoot_m / reach_distance_m``
    (``intensity.yaml: mapping.intrusion.reach.overshoot_fraction``), the tip covers the fraction
    ``f = 1 / (1 + kappa)`` of its stroke ``L`` at the hit, i.e. at normalised time ``u = s^-1(f)`` of the
    speed profile ``s``; with the stroke duration ``T = tau / u`` the hit speed is ``s'(u) L / T``, so
    ``L = v tau / (u s'(u))``, ``reach_distance_m = f L`` and ``overshoot_m = kappa f L``. For the ``linear``
    profile this is ``reach_distance_m = v tau`` and ``overshoot_m = kappa v tau``. ``kappa <= 0`` would end
    the stroke at the hit point (at rest for ``min_jerk``): quasi-static occupancy, not an impact, so it is
    refused (D6; such hazards belong to ``occupancy.dwell``). ``limb_size_m`` is recorded only.
"""
import dataclasses

from .._numeric import finite_float, positive_float
from ..kinematics.primitives import PROFILES, fall_height_for_tau, min_jerk_derivative, profile_inverse


class ZeroOvershootError(ValueError):
    """A reach with ``overshoot_m <= 0`` (D6)."""


def fall_release_height_m(tau_s: float, gravity_mps2: float) -> float:
    return float(fall_height_for_tau(positive_float("tau_s", tau_s), positive_float("gravity_mps2", gravity_mps2)))


def slide_start_distance_m(speed_mps: float, tau_s: float) -> float:
    return positive_float("speed_mps", speed_mps) * positive_float("tau_s", tau_s)


@dataclasses.dataclass(frozen=True)
class ReachGeometry:
    reach_distance_m: float
    overshoot_m: float
    stroke_m: float
    u_hit: float
    stroke_duration_s: float


def reach_geometry(hit_speed_mps: float, tau_s: float, overshoot_fraction: float,
                   speed_profile: str) -> ReachGeometry:
    speed = positive_float("hit_speed_mps", hit_speed_mps)
    tau = positive_float("tau_s", tau_s)
    kappa = finite_float("overshoot_fraction", overshoot_fraction)
    if speed_profile not in PROFILES:
        raise ValueError("speed_profile %r is not one of %s" % (speed_profile, list(PROFILES)))
    if not kappa > 0:
        raise ZeroOvershootError("overshoot_fraction=%r gives overshoot_m <= 0; a reach that carries severity "
                                 "needs overshoot_m > 0 (D6)" % (kappa,))
    fraction = 1.0 / (1.0 + kappa)
    u_hit = profile_inverse(speed_profile, fraction)
    slope = 1.0 if speed_profile == "linear" else float(min_jerk_derivative(u_hit))
    stroke = speed * tau / (u_hit * slope)
    return ReachGeometry(reach_distance_m=fraction * stroke, overshoot_m=kappa * fraction * stroke, stroke_m=stroke,
                         u_hit=u_hit, stroke_duration_s=tau / u_hit)
