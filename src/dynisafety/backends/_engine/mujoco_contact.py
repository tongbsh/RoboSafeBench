"""MuJoCo geom-pair contact helpers (lazy ``mujoco`` import)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from dynisafety.core.contact import CONTACT_IMPULSE_EPS, ContactStepResult


def _raw_model(model: Any) -> Any:
    return getattr(model, "_model", model)


def _raw_data(data: Any) -> Any:
    return getattr(data, "_data", data)


def _mj_contact_force(raw_model: Any, raw_data: Any, index: int, force_buf: np.ndarray) -> None:
    import mujoco

    mujoco.mj_contactForce(raw_model, raw_data, index, force_buf)


def expand_body_subtree(model: Any, root_body_ids: Iterable[int]) -> set[int]:
    """Return *root_body_ids* plus their **direct** children only.

    LIBERO scans ``body_parentid == root`` per root. RoboCasa grows the parent
    set in the same pass, so increasing body ids can pull in grandchildren.
    Direct-child expansion is deterministic and order-independent.
    """
    roots = {int(i) for i in root_body_ids}
    out = set(roots)
    parentid = model.body_parentid
    for body_idx in range(int(model.nbody)):
        if int(parentid[body_idx]) in roots:
            out.add(int(body_idx))
    return out


def expand_body_descendants(model: Any, root_body_ids: Iterable[int]) -> set[int]:
    """Return *root_body_ids* plus every descendant (children, grandchildren, …).

    GSO pack wrappers put collision meshes on ``gso_align`` under the object
    body. One-level expansion misses those geoms for contact / footprint / AEGIS.
    """
    out = {int(i) for i in root_body_ids}
    parentid = model.body_parentid
    grew = True
    while grew:
        grew = False
        for body_idx in range(int(model.nbody)):
            idx = int(body_idx)
            if idx in out:
                continue
            if int(parentid[idx]) in out:
                out.add(idx)
                grew = True
    return out


def collect_geom_ids_for_bodies(model: Any, body_ids: Iterable[int]) -> set[int]:
    """Geom ids whose ``geom_bodyid`` is in *body_ids*."""
    bodies = {int(i) for i in body_ids}
    out: set[int] = set()
    for geom_id in range(int(model.ngeom)):
        if int(model.geom_bodyid[geom_id]) in bodies:
            out.add(int(geom_id))
    return out


def collect_geom_ids_by_body_prefix(
    model: Any,
    *,
    prefixes: tuple[str, ...],
    require_collision: bool = True,
) -> set[int]:
    """Geom ids whose body name starts with any of *prefixes*.

    When ``require_collision`` is true, keep geoms with ``contype > 0`` or
    ``conaffinity > 0``. Flags are read from the raw mjModel
    (``getattr(model, "_model", model)``). Missing flag arrays default to 1.
    """
    raw = _raw_model(model)
    ids: set[int] = set()
    for geom_id in range(int(model.ngeom)):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = model.body_id2name(body_id)
        if body_name is None:
            continue
        if not str(body_name).startswith(prefixes):
            continue
        if require_collision:
            contype = int(raw.geom_contype[geom_id]) if hasattr(raw, "geom_contype") else 1
            conaff = int(raw.geom_conaffinity[geom_id]) if hasattr(raw, "geom_conaffinity") else 1
            if not (contype > 0 or conaff > 0):
                continue
        ids.add(int(geom_id))
    return ids


def body_quat(sim: Any, body_name: str) -> np.ndarray:
    """Return ``sim.data.body_xquat[body_id]`` (wxyz)."""
    try:
        body_id = int(sim.model.body_name2id(body_name))
    except Exception as exc:
        raise KeyError(f"unknown body {body_name!r}") from exc
    if body_id < 0:
        raise KeyError(f"unknown body {body_name!r}")
    return np.asarray(sim.data.body_xquat[body_id], dtype=np.float64)


def measure_geom_pair_contact(
    sim: Any,
    *,
    side_a_geom_ids: set[int],
    side_b_geom_ids: set[int],
    force_equiv_dt_s: float,
    impulse_reduce: str,
) -> ContactStepResult:
    """Peak |Fn| between two geom sets via ``mj_contactForce``.

    Hits are ``(g1∈A ∧ g2∈B) ∨ (g2∈A ∧ g1∈B)``. The B-side geom's body name
    is the partner key (``geom:{id}`` if unnamed).
    """
    if force_equiv_dt_s <= 0:
        raise ValueError(f"force_equiv_dt_s must be positive, got {force_equiv_dt_s!r}")
    if impulse_reduce not in ("sum", "max"):
        raise ValueError(f"impulse_reduce must be 'sum' or 'max', got {impulse_reduce!r}")
    if not side_b_geom_ids:
        return ContactStepResult(
            in_contact=False,
            force_n=0.0,
            impulse_ns=0.0,
            partners={},
            force_equiv_dt_s=float(force_equiv_dt_s),
            contact_without_force=False,
            peak_partner=None,
        )

    model = sim.model
    data = sim.data
    raw_model = _raw_model(model)
    raw_data = _raw_data(data)
    partner_counts: dict[str, int] = {}
    peak_fn = 0.0
    peak_partner: str | None = None
    force_buf = np.zeros(6)
    side_a = side_a_geom_ids
    side_b = side_b_geom_ids
    for i in range(int(data.ncon)):
        contact = data.contact[i]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        hit = (g1 in side_a and g2 in side_b) or (g2 in side_a and g1 in side_b)
        if not hit:
            continue
        obstacle_gid = g2 if g1 in side_a else g1
        body_id = int(model.geom_bodyid[obstacle_gid])
        body_name = model.body_id2name(body_id)
        key = str(body_name) if body_name is not None else f"geom:{obstacle_gid}"
        partner_counts[key] = partner_counts.get(key, 0) + 1
        _mj_contact_force(raw_model, raw_data, i, force_buf)
        fn = abs(float(force_buf[0]))
        if fn > peak_fn:
            peak_fn = fn
            peak_partner = key

    in_contact = bool(partner_counts)
    force_n = float(peak_fn) if in_contact else 0.0
    # Impulse ≈ force * dt; reduce mode reserved for multi-contact aggregation.
    # Current peak-normal path yields a single scalar — sum/max coincide.
    impulse_ns = force_n * float(force_equiv_dt_s)
    if impulse_reduce == "max":
        impulse_ns = float(impulse_ns)
    return ContactStepResult(
        in_contact=in_contact,
        force_n=force_n,
        impulse_ns=float(impulse_ns),
        partners=dict(partner_counts),
        force_equiv_dt_s=float(force_equiv_dt_s),
        contact_without_force=bool(in_contact and float(impulse_ns) <= CONTACT_IMPULSE_EPS),
        peak_partner=peak_partner,
    )
