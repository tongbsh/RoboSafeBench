"""SAPIEN impulse → equivalent force helpers (RoboTwin backend adaptation).

Control-step rising-edge counting still lives on ``EpisodeMetricsBuilder``.
Sim-step rising-edge (RoboTwin manager) uses :func:`rising_edge_contact`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
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
from dynisafety.core.newton_force import (  # noqa: F401 — re-export
    L3_FORCE_SCHEME_DYNAMIC_RELEASE,
    L3_FORCE_SCHEME_KINEMATIC_FIRST_HIT,
    L3_FORCE_SCHEMES,
    L3_NEWTON_RESTITUTION,
    L3_RELEASE_MIN_SPEED_MPS,
    is_ghost_overlap,
    l3_closing_speed_mps,
    l3_freeze_after_first_event,
    l3_newton_normal_impulse_ns,
    l3_release_linear_velocity,
    l3_release_momentum_impulse_ns,
    normalize_l3_force_scheme,
    official_l3_contact,
    orient_normal_from_actor,
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


L12_FORCE_SCHEME_KINEMATIC_FIRST_HIT = "kinematic_first_hit"
L12_FORCE_SCHEME_NEWTON_RELATIVE = "newton_relative"
L12_FORCE_SCHEMES = frozenset(
    {L12_FORCE_SCHEME_KINEMATIC_FIRST_HIT, L12_FORCE_SCHEME_NEWTON_RELATIVE}
)


def normalize_l12_force_scheme(raw: Any, *, field_name: str) -> str:
    """Require one of the two mutually exclusive RoboTwin L1/L2 force schemes."""
    scheme = str(raw).strip().lower()
    if scheme not in L12_FORCE_SCHEMES:
        allowed = ", ".join(sorted(L12_FORCE_SCHEMES))
        raise ValueError(f"{field_name} must be one of {{{allowed}}}, got {raw!r}")
    return scheme


def official_l12_contact(
    *,
    geometric: bool,
    impulse: float,
    require_impulse: bool,
    impulse_eps: float,
) -> bool:
    """L1/L2 official contact: matching manifold, optionally gated on impulse.

    ``impulse_eps`` is the engine noise floor (``CONTACT_IMPULSE_EPS``), not a
    YAML hyperparameter. L3 ``kinematic_first_hit`` does not call this helper
    (Scheme B: geometry only). L3 ``dynamic_release`` reuses it after the hand
    is released to physics.
    """
    if not geometric:
        return False
    if not require_impulse:
        return True
    return float(impulse) > float(impulse_eps)


def official_l12_scheme_contact(
    *,
    geometric: bool,
    impulse: float,
    force_scheme: str,
    released_to_physics: bool,
    require_impulse: bool,
    impulse_eps: float,
) -> bool:
    """Official L1/L2 contact under the mutually exclusive force scheme.

    ``kinematic_first_hit``: current impulse-gated first event (solver spike
    may become ``peak_force_n``).
    ``newton_relative``: no official contact while kinematic; after release,
    same impulse gate on the analytic J=m(1+e)v_n (SAPIEN spikes must
    already have been overwritten). Not the L3 name ``dynamic_release``.
    """
    scheme = normalize_l12_force_scheme(force_scheme, field_name="l12_force_scheme")
    if scheme == L12_FORCE_SCHEME_KINEMATIC_FIRST_HIT:
        return official_l12_contact(
            geometric=geometric,
            impulse=impulse,
            require_impulse=require_impulse,
            impulse_eps=impulse_eps,
        )
    if not released_to_physics:
        return False
    return official_l12_contact(
        geometric=geometric,
        impulse=impulse,
        require_impulse=require_impulse,
        impulse_eps=impulse_eps,
    )


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


def l3_episode_extra_from_info(dyn_final: Mapping[str, Any]) -> dict[str, Any]:
    """Copy L3 contact extras from ``DisturbanceManager.build_info`` into EpisodeRecord."""
    extra: dict[str, Any] = {
        "l3_post_first_overlap_sim_steps": int(
            dyn_final.get("l3_post_first_overlap_sim_steps", 0) or 0
        ),
    }
    raw_ignore = dyn_final.get("l3_ignore_collision_link_substrings")
    extra["l3_ignore_collision_link_substrings"] = (
        [str(item) for item in raw_ignore] if raw_ignore else []
    )
    if "l3_force_scheme" in dyn_final:
        extra["l3_force_scheme"] = str(dyn_final["l3_force_scheme"])
    for key in (
        "l3_kinematic_overlap_sim_steps",
        "l3_kinematic_peak_impulse_ns",
        "l3_kinematic_peak_force_n",
        "l3_release_speed_mps",
        "l3_force_inbound_speed_mps",
        "l3_partner_speed_mps",
        "l3_closing_speed_mps",
        "l3_force_from_release_momentum",
        "released_to_physics",
        "ghost_contact_sim_steps",
        "rollback_steps",
    ):
        if key in dyn_final:
            extra[key] = dyn_final[key]
    if "release_reason" in dyn_final:
        extra["release_reason"] = dyn_final["release_reason"]
    return extra


def l12_episode_extra_from_info(dyn_final: Mapping[str, Any]) -> dict[str, Any]:
    """Copy L1/L2 Newton extras from ``DisturbanceManager.build_info`` into EpisodeRecord."""
    extra: dict[str, Any] = {}
    if "l12_force_scheme" in dyn_final:
        extra["l12_force_scheme"] = str(dyn_final["l12_force_scheme"])
    for key in (
        "l12_kinematic_overlap_sim_steps",
        "l12_kinematic_peak_impulse_ns",
        "l12_kinematic_peak_force_n",
        "l12_force_inbound_speed_mps",
        "l12_partner_speed_mps",
        "l12_closing_speed_mps",
        "l12_force_from_newton",
        "released_to_physics",
        "ghost_contact_sim_steps",
        "rollback_steps",
    ):
        if key in dyn_final:
            extra[key] = dyn_final[key]
    if "release_reason" in dyn_final:
        extra["release_reason"] = dyn_final["release_reason"]
    return extra
