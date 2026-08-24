"""RoboCasa contact measurement via ``mj_contactForce``.

Robot geoms (arm / gripper / mobile base / wheels / torso) vs injected
obstacle geoms only. Cabinet/door contacts are not CR.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from dynisafety.backends._engine.mujoco_contact import (
    body_quat,
    collect_geom_ids_by_body_prefix,
    collect_geom_ids_for_bodies,
    expand_body_subtree,
    measure_geom_pair_contact,
)
from dynisafety.backends.robocasa.env import HIDDEN_OBSTACLE_BODY
from dynisafety.core.contact import (
    ContactStepResult,
    resolve_force_equiv_dt,
    tilt_deg_from_body_quat,
)

_ROBOT_BODY_PREFIXES = ("robot0_", "gripper0_", "mobilebase0_", "panda0_")


def collect_robot_geom_ids(env: Any) -> set[int]:
    """Collision geoms for PandaOmron arm, gripper, torso, and mobile base/wheels."""
    return collect_geom_ids_by_body_prefix(env.sim.model, prefixes=_ROBOT_BODY_PREFIXES)


def _obstacle_root_body_name(env: Any) -> str | None:
    meta = getattr(env, "_dynisafety_l12", None)
    if isinstance(meta, Mapping) and meta.get("body_name"):
        return str(meta["body_name"])
    model = env.sim.model
    for candidate in (HIDDEN_OBSTACLE_BODY, f"{HIDDEN_OBSTACLE_BODY}_main"):
        try:
            model.body_name2id(candidate)
        except Exception:
            continue
        return candidate
    return None


def collect_obstacle_geom_ids(env: Any) -> set[int]:
    """Geoms of the embedded L1/L2 floor actor (including visual copies)."""
    model = env.sim.model
    body_name = _obstacle_root_body_name(env)
    if body_name is None:
        return set()
    try:
        body_id = int(model.body_name2id(body_name))
    except Exception:
        return set()
    return collect_geom_ids_for_bodies(model, expand_body_subtree(model, [body_id]))


def measure_contact_step(
    env: Any,
    *,
    robot_geom_ids: set[int],
    obstacle_geom_ids: set[int],
    force_equiv_dt_s: float,
    impulse_reduce: str,
) -> ContactStepResult:
    return measure_geom_pair_contact(
        env.sim,
        side_a_geom_ids=robot_geom_ids,
        side_b_geom_ids=obstacle_geom_ids,
        force_equiv_dt_s=force_equiv_dt_s,
        impulse_reduce=impulse_reduce,
    )


def force_equiv_dt_s(cfg: Any, *, missing: Any) -> float:
    contact = cfg.contact
    if contact is missing:
        raise ValueError("cfg.contact is MISSING")
    mode = contact.force_equiv_dt
    if mode is missing:
        raise ValueError("contact.force_equiv_dt is MISSING")
    window = contact.force_report_window_s
    window_s = None if window is missing else float(window)
    control_freq_hz = None
    runtime = getattr(cfg, "runtime", missing)
    if runtime is not missing:
        freq = getattr(runtime, "control_freq", missing)
        if freq is not missing:
            control_freq_hz = float(freq)
    return resolve_force_equiv_dt(
        mode,
        sim_timestep_s=None,
        control_freq_hz=control_freq_hz,
        force_report_window_s=window_s,
    )


def measure_tilt_deg(
    env: Any,
    body_name: str,
    *,
    ref_quat: np.ndarray | None = None,
) -> float:
    """Tilt of *body_name* from *ref_quat* (wxyz) or world-Z upright."""
    return tilt_deg_from_body_quat(body_quat(env.sim, body_name), ref_quat=ref_quat)
