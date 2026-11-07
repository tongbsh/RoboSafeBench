"""Analytic severity (00 §6, R9, D9 a/b): ``J = m (1 + e) v_n`` from kinematic quantities only.

Engine peak force and impulse are auxiliary and never enter this module. The restitution ``e`` is a
required argument with no default (it comes from ``semantics.yaml: contact.restitution_e``).
"""
from typing import Any

import numpy as np

from .._numeric import finite_float, non_negative_float, positive_float, unit_vec3, vec3


def closing_normal_speed(v_hazard: Any, v_partner: Any, normal_partner_to_hazard: Any) -> float:
    """``max(0, (v_partner - v_hazard) . n_hat)``: how fast the hazard and the partner approach each other
    along the contact normal ``n_hat`` (pointing from the partner toward the hazard, D9 a).

    0 when they separate. A zero normal is an error: there is no fallback to the velocity direction
    (the legacy ``newton_force.l3_closing_speed_mps`` had one; it hid missing normals).
    """
    v_h = vec3("v_hazard", v_hazard)
    v_p = vec3("v_partner", v_partner)
    n_hat = unit_vec3("normal_partner_to_hazard", normal_partner_to_hazard)
    return max(0.0, float(np.dot(v_p - v_h, n_hat)))


def analytic_impulse_ns(mass_kg: float, v_normal_mps: float, restitution: float) -> float:
    """Normal impulse ``m (1 + e) v_n`` in N*s. ``mass_kg > 0``, ``v_normal_mps >= 0`` (use
    :func:`closing_normal_speed`), ``restitution >= 0``; all required."""
    mass = positive_float("mass_kg", mass_kg)
    speed = non_negative_float("v_normal_mps", v_normal_mps)
    coefficient = finite_float("restitution", restitution)
    if coefficient < 0.0:
        raise ValueError("restitution must be >= 0, got %r" % (coefficient,))
    return mass * (1.0 + coefficient) * speed
