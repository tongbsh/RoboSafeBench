"""SAPIEN impulse → equivalent force / tilt helpers (backend adaptation only).

``contact_events`` rising-edge counting belongs to ``EpisodeMetricsBuilder`` (core),
not this module.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from dynisafety.backends._engine.sapien_contact import (
    actor_has_contact,
    measure_actor_contact,
    read_pair_actor01,
)
from dynisafety.backends._engine.sapien_contact import (
    scene_timestep as sapien_scene_timestep,
)
from dynisafety.config.schema import MISSING, RunConfig
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
from dynisafety.core.contact import (
    quat_to_matrix as quat_to_matrix,
)

SAPIEN_CONTACT_IMPULSE_EPS = CONTACT_IMPULSE_EPS

measure_tilt_deg = tilt_deg_from_reference


def filter_collision_link_names(
    link_names: set[str],
    *,
    ignore_substrings: Sequence[str],
) -> set[str]:
    """Drop robot links whose names contain any ignored substring (L3)."""
    if not ignore_substrings:
        return set(link_names)
    tokens = [str(t).lower() for t in ignore_substrings if str(t)]
    out: set[str] = set()
    for name in link_names:
        lowered = name.lower()
        if any(tok in lowered for tok in tokens):
            continue
        out.add(name)
    return out


def scene_timestep(scene: Any, *, control_freq: int) -> float:
    return sapien_scene_timestep(scene, control_freq=control_freq)


def force_equiv_dt_s(
    cfg: RunConfig,
    *,
    scene_timestep_s: float,
    control_freq: int,
) -> float:
    """Denominator for reported force = impulse / dt."""
    contact = cfg.contact
    if contact is MISSING:
        raise ValueError("cfg.contact is MISSING")
    mode = contact.force_equiv_dt
    window = contact.force_report_window_s
    if mode is MISSING:
        raise ValueError("contact.force_equiv_dt is required (YAML)")
    return resolve_force_equiv_dt(
        mode,
        sim_timestep_s=float(scene_timestep_s),
        control_freq_hz=float(control_freq),
        force_report_window_s=None if window is MISSING else float(window),
    )


def contact_impulse_scalar(contact: Any, *, reduce: str) -> float:
    """Aggregate impulse norms over contact points for one manifold."""
    pts = list(getattr(contact, "points", None) or [])
    norms: list[float] = []
    for p in pts:
        imp = getattr(p, "impulse", None)
        if imp is None:
            continue
        norms.append(float(np.linalg.norm(np.asarray(imp, dtype=np.float64).ravel())))
    return reduce_impulse_norms(norms, impulse_reduce=reduce)


def measure_contact_step(
    scene: Any,
    *,
    actor_name: str,
    robot_link_names: set[str],
    ignore_link_substrings: Sequence[str],
    force_equiv_dt: float,
    impulse_reduce: str,
    force_anomaly_threshold_n: float,
    impulse_anomaly_threshold_ns: float,
) -> ContactStepResult:
    """Scan SAPIEN contacts between the disturbance actor and filtered robot links.

    Scheme B: a matching contact manifold counts as ``in_contact`` even when
    impulse is ~0 (common for kinematic pose-set actors). Force/impulse fields
    still come only from measured impulse (may be 0 while in contact).
    Anomaly thresholds stay on the signature for callers; policy lives in
    ``is_force_anomaly``.
    """
    del force_anomaly_threshold_n, impulse_anomaly_threshold_ns
    names = filter_collision_link_names(robot_link_names, ignore_substrings=ignore_link_substrings)
    return measure_actor_contact(
        scene,
        actor_name=actor_name,
        target_names=names,
        pair_reader=read_pair_actor01,
        force_equiv_dt_s=float(force_equiv_dt),
        impulse_reduce=impulse_reduce,
    )


def require_contact_params(cfg: RunConfig) -> tuple[str, float, float, float]:
    """Return (impulse_reduce, anomaly_force_n, anomaly_impulse_ns, topple_deg)."""
    contact = cfg.contact
    metrics = cfg.metrics
    if contact is MISSING:
        raise ValueError("cfg.contact is MISSING")
    if metrics is MISSING:
        raise ValueError("cfg.metrics is MISSING")
    reduce = contact.impulse_reduce
    fanom = metrics.force_anomaly_threshold_n
    ianom = metrics.impulse_anomaly_threshold_ns
    topple = contact.topple_threshold_deg
    if reduce is MISSING:
        raise ValueError("contact.impulse_reduce required (YAML)")
    if fanom is MISSING or ianom is MISSING:
        raise ValueError(
            "metrics.force_anomaly_threshold_n and impulse_anomaly_threshold_ns required (YAML)"
        )
    if topple is MISSING:
        raise ValueError("contact.topple_threshold_deg required (YAML)")
    return str(reduce), float(fanom), float(ianom), float(topple)


def actor_has_impulse_contact(
    scene: Any,
    *,
    actor_name: str,
    robot_link_names: set[str],
) -> bool:
    """True if actor↔robot contact has impulse above numerical noise floor."""
    return actor_has_contact(
        scene,
        actor_name=actor_name,
        target_names=robot_link_names,
        pair_reader=read_pair_actor01,
        require_impulse=True,
    )


def actor_has_geometric_contact(
    scene: Any,
    *,
    actor_name: str,
    robot_link_names: set[str],
) -> bool:
    """True if SAPIEN reports a contact manifold, ignoring impulse magnitude.

    Kinematic pose-set actors may report ~0 impulse in the same substep.
    """
    return actor_has_contact(
        scene,
        actor_name=actor_name,
        target_names=robot_link_names,
        pair_reader=read_pair_actor01,
        require_impulse=False,
    )


def obb_overlaps_links(
    actor_pos: np.ndarray,
    *,
    yaw: float,
    robot_link_positions: np.ndarray,
    obb_half: Sequence[float],
    half_scale: float,
) -> bool:
    """Axis-aligned OBB (yaw about +Z) vs robot link positions."""
    half = np.asarray(obb_half, dtype=np.float64).reshape(3) * float(half_scale)
    cos_neg = math.cos(-float(yaw))
    sin_neg = math.sin(-float(yaw))
    origin = np.asarray(actor_pos, dtype=np.float64)
    for rpos in robot_link_positions:
        diff = np.asarray(rpos, dtype=np.float64) - origin
        lx = cos_neg * diff[0] - sin_neg * diff[1]
        ly = sin_neg * diff[0] + cos_neg * diff[1]
        lz = diff[2]
        if abs(lx) < half[0] and abs(ly) < half[1] and abs(lz) < half[2]:
            return True
    return False
