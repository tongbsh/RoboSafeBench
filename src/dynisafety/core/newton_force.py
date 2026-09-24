"""Shared L3 Newton impact helpers (engine-agnostic).

Official L3 ``dynamic_release`` force is J=m(1+e)v_n when the solver reports
~0 impulse. RoboTwin and SimplerEnv both import from here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dynisafety.core.contact import CONTACT_IMPULSE_EPS

L3_FORCE_SCHEME_KINEMATIC_FIRST_HIT = "kinematic_first_hit"
L3_FORCE_SCHEME_DYNAMIC_RELEASE = "dynamic_release"
L3_FORCE_SCHEMES = frozenset(
    {L3_FORCE_SCHEME_KINEMATIC_FIRST_HIT, L3_FORCE_SCHEME_DYNAMIC_RELEASE}
)
# Last-millimetre *motion* floor after release. Official force does **not**
# use this floor — it uses the unfloored inbound and the contact-normal
# relative speed.
L3_RELEASE_MIN_SPEED_MPS = 0.25
# Newton restitution for the L3 analytic fallback (e=0: palm stops).
L3_NEWTON_RESTITUTION = 0.0


def _unit_or_zero(vec: np.ndarray) -> tuple[np.ndarray, float]:
    arr = np.asarray(vec, dtype=np.float64).reshape(3)
    speed = float(np.linalg.norm(arr))
    if speed <= 1e-12:
        return np.zeros(3, dtype=np.float64), 0.0
    return arr / speed, speed


def orient_normal_from_actor(
    normal: np.ndarray,
    *,
    actor_pos: np.ndarray,
    contact_pos: np.ndarray,
) -> np.ndarray:
    """Flip ``normal`` so it points from the actor toward the contact."""
    n, nlen = _unit_or_zero(normal)
    if nlen <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    delta = np.asarray(contact_pos, dtype=np.float64).reshape(3) - np.asarray(
        actor_pos, dtype=np.float64
    ).reshape(3)
    if float(np.linalg.norm(delta)) > 1e-8 and float(np.dot(n, delta)) < 0.0:
        return -n
    return n


def l3_closing_speed_mps(
    v_hand: np.ndarray,
    v_partner: np.ndarray,
    normal_from_hand: np.ndarray,
) -> float:
    """Approaching speed along ``normal_from_hand``; 0 if separating or degenerate."""
    n, nlen = _unit_or_zero(normal_from_hand)
    if nlen <= 1e-12:
        n, nlen = _unit_or_zero(v_hand)
        if nlen <= 1e-12:
            return 0.0
    v_rel = np.asarray(v_hand, dtype=np.float64).reshape(3) - np.asarray(
        v_partner, dtype=np.float64
    ).reshape(3)
    return max(0.0, float(np.dot(v_rel, n)))


def l3_newton_normal_impulse_ns(
    mass_kg: float,
    v_hand: np.ndarray,
    v_partner: np.ndarray,
    normal_from_hand: np.ndarray,
    *,
    restitution: float = L3_NEWTON_RESTITUTION,
) -> float:
    """Newton impact impulse with μ≈m_hand (arm ≫ palm): J = m(1+e)v_n.

    ``v_n`` is the closing speed along the contact normal pointing from the
    palm toward the partner. Separating contacts yield 0.
    """
    e = float(restitution)
    if e < 0.0:
        raise ValueError(f"restitution must be >= 0, got {restitution!r}")
    closing = l3_closing_speed_mps(v_hand, v_partner, normal_from_hand)
    return float(mass_kg) * (1.0 + e) * closing


def l3_release_linear_velocity(
    *,
    overlap_pos: np.ndarray,
    last_free_pos: np.ndarray | None,
    path_velocity: np.ndarray,
    sim_dt: float,
    min_speed_mps: float = L3_RELEASE_MIN_SPEED_MPS,
) -> np.ndarray:
    """Inbound velocity for the dynamic palm after last-free rollback.

    Prefer the finite difference overlap−last_free over one step (the
    motion that caused the ghost hit). If that is missing or tiny, use the
    scripted path velocity. Scale up to ``min_speed_mps`` along the same
    direction so the palm continues into the arm instead of settling.
    Official force must pass ``min_speed_mps=0`` (unfloored inbound).
    """
    dt = max(float(sim_dt), 1e-9)
    path = np.asarray(path_velocity, dtype=np.float64).reshape(3)
    measured = np.zeros(3, dtype=np.float64)
    if last_free_pos is not None:
        delta = np.asarray(overlap_pos, dtype=np.float64).reshape(3) - np.asarray(
            last_free_pos, dtype=np.float64
        ).reshape(3)
        if float(np.linalg.norm(delta)) > 1e-8:
            measured = delta / dt
    candidate = measured if float(np.linalg.norm(measured)) > 1e-8 else path
    speed = float(np.linalg.norm(candidate))
    if speed <= 1e-8:
        return np.zeros(3, dtype=np.float64)
    return candidate * (max(speed, float(min_speed_mps)) / speed)


def l3_release_momentum_impulse_ns(mass_kg: float, velocity: np.ndarray) -> float:
    """Impulse if a palm of ``mass_kg`` stops against a stationary wall (e=0)."""
    vel = np.asarray(velocity, dtype=np.float64).reshape(3)
    return l3_newton_normal_impulse_ns(mass_kg, vel, np.zeros(3), vel)


def normalize_l3_force_scheme(raw: Any, *, field_name: str) -> str:
    """Require one of the two mutually exclusive L3 force schemes."""
    scheme = str(raw).strip().lower()
    if scheme not in L3_FORCE_SCHEMES:
        allowed = ", ".join(sorted(L3_FORCE_SCHEMES))
        raise ValueError(f"{field_name} must be one of {{{allowed}}}, got {raw!r}")
    return scheme


def official_l3_contact(
    *,
    geometric: bool,
    impulse: float,
    force_scheme: str,
    released_to_physics: bool,
    require_impulse: bool,
    impulse_eps: float = CONTACT_IMPULSE_EPS,
) -> bool:
    """Official L3 contact under the mutually exclusive force scheme.

    ``kinematic_first_hit``: any matching manifold (Scheme B).
    ``dynamic_release``: no official contact while kinematic; after release,
    the same impulse gate as L1/L2.
    """
    scheme = normalize_l3_force_scheme(force_scheme, field_name="l3_force_scheme")
    if scheme == L3_FORCE_SCHEME_KINEMATIC_FIRST_HIT:
        return bool(geometric)
    if not released_to_physics:
        return False
    if not geometric:
        return False
    if not require_impulse:
        return True
    return float(impulse) > float(impulse_eps)


def l3_freeze_after_first_event(force_scheme: str) -> bool:
    """True when official CR/force snapshot the first event only (Scheme B)."""
    return (
        normalize_l3_force_scheme(force_scheme, field_name="l3_force_scheme")
        == L3_FORCE_SCHEME_KINEMATIC_FIRST_HIT
    )


def is_ghost_overlap(*, geometric: bool, official: bool) -> bool:
    """Kinematic / zero-impulse overlap that must not count as official contact."""
    return bool(geometric) and not bool(official)
