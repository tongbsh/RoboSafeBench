"""SAPIEN contact helpers (duck-typed; no ``sapien`` import)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from dynisafety.core.contact import (
    CONTACT_IMPULSE_EPS,
    ContactStepResult,
    reduce_impulse_norms,
)

ContactPairReader = Callable[[Any], tuple[str, str] | None]


@dataclass(frozen=True)
class ContactScan:
    """Raw SAPIEN scan before impulse→force conversion."""

    in_contact: bool
    max_impulse_ns: float
    peak_partner: str | None
    partners: dict[str, int]


def read_pair_actor01(contact: Any) -> tuple[str, str] | None:
    """ManiSkill2 / SimplerEnv contacts expose ``actor0`` / ``actor1``."""
    a0 = getattr(contact, "actor0", None)
    a1 = getattr(contact, "actor1", None)
    if a0 is None or a1 is None:
        return None
    n0 = getattr(a0, "name", None)
    n1 = getattr(a1, "name", None)
    if n0 is None or n1 is None:
        return None
    return (str(n0), str(n1))


def read_pair_body_entity(contact: Any) -> tuple[str, str] | None:
    """RoboTwin contacts expose ``bodies[*].entity``, not ``actor0``/``actor1``."""
    bodies = getattr(contact, "bodies", None)
    if bodies is None or len(bodies) < 2:
        return None
    body0, body1 = bodies[0], bodies[1]
    if body0 is None or body1 is None:
        return None
    entity0 = getattr(body0, "entity", None)
    entity1 = getattr(body1, "entity", None)
    if entity0 is None or entity1 is None:
        return None
    n0 = getattr(entity0, "name", None)
    n1 = getattr(entity1, "name", None)
    if n0 is None or n1 is None:
        return None
    return (str(n0), str(n1))


def _point_impulse_norms(points: Any) -> list[float]:
    norms: list[float] = []
    for p in points:
        imp = getattr(p, "impulse", None)
        if imp is None:
            continue
        norms.append(float(np.linalg.norm(np.asarray(imp, dtype=np.float64).ravel())))
    return norms


def scan_actor_contacts(
    scene: Any,
    *,
    actor_name: str,
    target_names: set[str],
    pair_reader: ContactPairReader,
    impulse_reduce: str,
) -> ContactScan:
    """Scan matching manifolds (Scheme B: match ⇒ contact, impulse optional)."""
    if not actor_name or not target_names:
        return ContactScan(
            in_contact=False,
            max_impulse_ns=0.0,
            peak_partner=None,
            partners={},
        )
    mode = str(impulse_reduce).strip().lower()
    if mode not in ("sum", "max"):
        raise ValueError(f"impulse_reduce must be 'sum' or 'max', got {impulse_reduce!r}")

    max_impulse = 0.0
    peak_partner: str | None = None
    partners: dict[str, int] = {}
    in_contact = False
    for contact in scene.get_contacts():
        pair = pair_reader(contact)
        if pair is None:
            continue
        n0, n1 = pair
        if n0 == actor_name and n1 in target_names:
            partner = n1
        elif n1 == actor_name and n0 in target_names:
            partner = n0
        else:
            continue
        in_contact = True
        partners[partner] = partners.get(partner, 0) + 1
        points = getattr(contact, "points", None) or []
        scalar = (
            reduce_impulse_norms(_point_impulse_norms(points), impulse_reduce=mode)
            if points
            else 0.0
        )
        if scalar > max_impulse:
            max_impulse = float(scalar)
            peak_partner = partner
    return ContactScan(
        in_contact=in_contact,
        max_impulse_ns=float(max_impulse),
        peak_partner=peak_partner,
        partners=dict(partners),
    )


def measure_actor_contact(
    scene: Any,
    *,
    actor_name: str,
    target_names: set[str],
    pair_reader: ContactPairReader,
    force_equiv_dt_s: float,
    impulse_reduce: str,
) -> ContactStepResult:
    """Convert a SAPIEN actor scan into ``ContactStepResult``."""
    scan = scan_actor_contacts(
        scene,
        actor_name=actor_name,
        target_names=target_names,
        pair_reader=pair_reader,
        impulse_reduce=impulse_reduce,
    )
    dt = float(force_equiv_dt_s)
    force = float(scan.max_impulse_ns / dt) if dt > 0.0 else 0.0
    impulse_ns = float(scan.max_impulse_ns)
    return ContactStepResult(
        in_contact=scan.in_contact,
        force_n=force,
        impulse_ns=impulse_ns,
        partners=dict(scan.partners),
        force_equiv_dt_s=dt,
        contact_without_force=bool(scan.in_contact and impulse_ns <= CONTACT_IMPULSE_EPS),
        peak_partner=scan.peak_partner,
    )


def actor_has_contact(
    scene: Any,
    *,
    actor_name: str,
    target_names: set[str],
    pair_reader: ContactPairReader,
    require_impulse: bool,
) -> bool:
    """Short-circuit predicate for actor↔target contact.

    ``require_impulse=True``: a matching manifold whose **sum** of point
    impulse norms exceeds ``CONTACT_IMPULSE_EPS``.
    ``require_impulse=False``: any matching manifold (Scheme B).
    """
    if not actor_name or not target_names:
        return False
    for contact in scene.get_contacts():
        pair = pair_reader(contact)
        if pair is None:
            continue
        n0, n1 = pair
        hit = (n0 == actor_name and n1 in target_names) or (n1 == actor_name and n0 in target_names)
        if not hit:
            continue
        if not require_impulse:
            return True
        impulse = sum(_point_impulse_norms(getattr(contact, "points", None) or []))
        if impulse > CONTACT_IMPULSE_EPS:
            return True
    return False


def scene_timestep(scene: Any, *, control_freq: int | None = None) -> float:
    """First positive ``get_timestep`` / ``get_time_step`` / ``timestep`` / ``dt``.

    If every probe fails, ``control_freq`` (when not ``None``) falls back to
    ``1/control_freq``. Omitting it raises — the fallback is the caller's
    explicit choice, not a hidden default.
    """
    for attr in ("get_timestep", "get_time_step"):
        fn = getattr(scene, attr, None)
        if callable(fn):
            try:
                dt = float(fn())
                if dt > 0.0:
                    return dt
            except Exception:
                pass
    for attr in ("timestep", "time_step", "dt"):
        try:
            dt = float(getattr(scene, attr))
            if dt > 0.0:
                return dt
        except Exception:
            pass
    if control_freq is not None:
        hz = int(control_freq)
        if hz <= 0:
            raise ValueError(
                f"control_freq must be > 0 when used as timestep fallback, got {control_freq!r}"
            )
        return 1.0 / float(hz)
    raise ValueError("unable to read a positive scene timestep; pass control_freq to fall back")
