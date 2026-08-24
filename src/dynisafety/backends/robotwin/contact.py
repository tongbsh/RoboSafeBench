"""SAPIEN impulse → equivalent force helpers (RoboTwin backend adaptation).

Control-step rising-edge counting still lives on ``EpisodeMetricsBuilder``.
Sim-step rising-edge (RoboTwin manager) uses :func:`rising_edge_contact`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from dynisafety.backends._engine.sapien_contact import (
    measure_actor_contact,
    read_pair_body_entity,
)
from dynisafety.core.contact import (
    CONTACT_IMPULSE_EPS,
    ContactStepResult,
    reduce_impulse_norms,
    resolve_force_equiv_dt,
    tilt_deg_from_reference,
)
from dynisafety.core.contact import (
    is_toppled as is_toppled,
)

SAPIEN_CONTACT_IMPULSE_EPS = CONTACT_IMPULSE_EPS

measure_tilt_deg = tilt_deg_from_reference


def partner_name_excluded(name: str, substrings: Sequence[str]) -> bool:
    """True when ``name`` contains any exclude substring (case-insensitive)."""
    lowered = str(name).lower()
    return any(str(token).lower() in lowered for token in substrings if str(token).strip())


def filter_collision_names(
    names: Iterable[str],
    substrings: Sequence[str],
) -> set[str]:
    """Drop names matching any exclude substring; empty ``substrings`` is a no-op."""
    tokens = tuple(str(token) for token in substrings if str(token).strip())
    if not tokens:
        return set(names)
    return {str(name) for name in names if not partner_name_excluded(str(name), tokens)}


def rising_edge_contact(prev_in_contact: bool, in_contact: bool) -> tuple[bool, bool]:
    """Return ``(is_new_event, next_prev)`` for official contact rising-edge."""
    now = bool(in_contact)
    return bool(now and not prev_in_contact), now


def official_l12_contact(
    *,
    geometric: bool,
    impulse: float,
    require_impulse: bool,
    impulse_eps: float,
) -> bool:
    """L1/L2 official contact: matching manifold, optionally gated on impulse.

    ``impulse_eps`` is the engine noise floor (``CONTACT_IMPULSE_EPS``), not a
    YAML hyperparameter. L3 does not call this helper (Scheme B: geometry only).
    """
    if not geometric:
        return False
    if not require_impulse:
        return True
    return float(impulse) > float(impulse_eps)


def is_ghost_overlap(*, geometric: bool, official: bool) -> bool:
    """Kinematic / zero-impulse overlap that must not count as official contact."""
    return bool(geometric) and not bool(official)


def should_skip_dynamic_release(*, penetrating: bool, release_on_penetration: bool) -> bool:
    """True when a kinematic→dynamic rebuild would spawn inside a blocker."""
    return bool(penetrating) and not bool(release_on_penetration)


def classify_topple(*, tilt_exceeds: bool, released_to_physics: bool) -> tuple[bool, bool]:
    """Split official topple (post-release) from kinematic/landing tip-over.

    Returns ``(official_toppled, landing_toppled)`` for this sample. Callers
    OR the flags across the episode. Official ``toppled`` stays false until
    the actor is dynamic.
    """
    if not tilt_exceeds:
        return False, False
    if released_to_physics:
        return True, False
    return False, True


def xyz_list(vec: Any) -> list[float]:
    """JSON-safe length-3 XYZ list."""
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        raise ValueError(f"xyz vector must have at least 3 values, got shape {arr.shape}")
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def force_equiv_timestep(
    dt_sim: float,
    *,
    force_equiv_dt: str,
    force_report_window_s: float,
    control_freq_hz: float,
) -> float:
    """Denominator for reported force = impulse / dt. No silent defaults."""
    return resolve_force_equiv_dt(
        force_equiv_dt,
        sim_timestep_s=float(dt_sim),
        control_freq_hz=float(control_freq_hz),
        force_report_window_s=float(force_report_window_s),
    )


def contact_point_impulse_scalar(
    points: Sequence[Any],
    *,
    impulse_reduce: str,
) -> float:
    """Aggregate impulse norms over contact points for one manifold."""
    norms: list[float] = []
    for p in points:
        imp = getattr(p, "impulse", None)
        if imp is None:
            continue
        norms.append(float(np.linalg.norm(np.asarray(imp, dtype=np.float64).ravel())))
    return reduce_impulse_norms(norms, impulse_reduce=impulse_reduce)


def measure_contact_step(
    scene: Any,
    *,
    actor_name: str,
    target_names: set[str],
    dt_sim: float,
    force_equiv_dt: str,
    force_report_window_s: float,
    control_freq_hz: float,
    impulse_reduce: str,
) -> ContactStepResult:
    """Scan SAPIEN contacts between the disturbance actor and target entities.

    RoboTwin contacts expose ``contact.bodies[*].entity.name`` (not actor0/1).

    Scheme B: a matching contact manifold counts as ``in_contact`` even when
    impulse is ~0 (kinematic hand / unsettled obstacle). Force still derives
    from impulse / force_equiv_dt only.
    """
    dt = force_equiv_timestep(
        dt_sim,
        force_equiv_dt=force_equiv_dt,
        force_report_window_s=force_report_window_s,
        control_freq_hz=control_freq_hz,
    )
    return measure_actor_contact(
        scene,
        actor_name=actor_name,
        target_names=target_names,
        pair_reader=read_pair_body_entity,
        force_equiv_dt_s=dt,
        impulse_reduce=impulse_reduce,
    )
