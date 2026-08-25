"""LIBERO contact / tilt measurement (migrated from ``collision.py``).

Backend adaptation only: geom sets, step contact, force via ``mj_contactForce``,
tilt. Injury/SASR classification stays in ``core`` / config thresholds.

L1/L2 use ``measure_contact_step`` (obstacle vs arm ∪ task objects).
L3 personal safety uses ``measure_l3_person_contact_step`` (hand vs arm/gripper
only — not hand↔props or arm↔scene).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dynisafety.backends._engine.mujoco_contact import (
    body_quat,
    collect_geom_ids_by_body_prefix,
    collect_geom_ids_for_bodies,
    expand_body_descendants,
    expand_body_subtree,
    measure_geom_pair_contact,
)
from dynisafety.core.contact import ContactStepResult, tilt_deg_from_body_quat


def _unwrap_render_env(env: Any) -> Any:
    """Return the OffScreenRenderEnv-like object (has ``.sim`` and ``.env``)."""
    if hasattr(env, "sim") and hasattr(env, "env"):
        return env
    if hasattr(env, "env") and hasattr(env.env, "sim"):
        return env.env
    raise TypeError(f"cannot unwrap render env from {type(env)!r}")


def collect_robot_geom_ids(env: Any) -> set[int]:
    """MuJoCo geom IDs for robot arm + gripper (collision geoms only)."""
    render = _unwrap_render_env(env)
    return collect_geom_ids_by_body_prefix(render.sim.model, prefixes=("robot0_", "gripper0_"))


def collect_obstacle_geom_ids(env: Any) -> set[int]:
    """Geom IDs for bodies whose object name contains ``obstacle``."""
    render = _unwrap_render_env(env)
    model = render.sim.model
    inner = render.env
    obstacle_body_ids: set[int] = set()
    for obj_name, obj_body in {**inner.objects_dict, **inner.fixtures_dict}.items():
        if "obstacle" not in obj_name.lower():
            continue
        try:
            body_id = int(model.body_name2id(obj_body.root_body))
            obstacle_body_ids |= expand_body_descendants(model, [body_id])
        except Exception:
            pass
    return collect_geom_ids_for_bodies(model, obstacle_body_ids)


def collect_task_object_geom_ids(env: Any) -> set[int]:
    """Geom IDs for ``obj_of_interest`` (grasped / manipulated props)."""
    render = _unwrap_render_env(env)
    model = render.sim.model
    inner = render.env
    geom_ids: set[int] = set()

    raw_ooi = getattr(render, "obj_of_interest", None)
    if raw_ooi is None:
        raw_ooi = getattr(inner, "obj_of_interest", None)
    if not raw_ooi:
        return geom_ids

    for name in raw_ooi:
        obj = inner.objects_dict.get(name)
        if obj is None:
            continue
        try:
            body_id = int(model.body_name2id(obj.root_body))
        except Exception:
            continue
        geom_ids |= collect_geom_ids_for_bodies(model, expand_body_subtree(model, [body_id]))
    return geom_ids


def measure_contact_step(
    env: Any,
    *,
    robot_geom_ids: set[int],
    obstacle_geom_ids: set[int],
    force_equiv_dt_s: float,
    impulse_reduce: str,
) -> ContactStepResult:
    """L1/L2 (and L0) contact: disturbance obstacle vs arm ∪ task objects.

    Pairing is the caller's ``robot_geom_ids`` (typically arm+gripper plus
    ``obj_of_interest``) against the active obstacle. Hand-vs-arm-only
    personal-safety metrics belong in ``measure_l3_person_contact_step``.
    """
    return measure_geom_pair_contact(
        _unwrap_render_env(env).sim,
        side_a_geom_ids=robot_geom_ids,
        side_b_geom_ids=obstacle_geom_ids,
        force_equiv_dt_s=force_equiv_dt_s,
        impulse_reduce=impulse_reduce,
    )


def measure_l3_person_contact_step(
    env: Any,
    *,
    arm_geom_ids: set[int],
    hand_geom_ids: set[int],
    force_equiv_dt_s: float,
    impulse_reduce: str,
) -> ContactStepResult:
    """L3 personal-safety contact: human-hand proxy vs robot arm/gripper only.

    Counts neither hand↔scene props nor arm↔table/fixtures/task objects.
    Those contacts may still drive the L3 controller (vanish / ignore) but
    must not enter ``contact_steps`` / ``peak_force_n``.
    """
    return measure_geom_pair_contact(
        _unwrap_render_env(env).sim,
        side_a_geom_ids=set(arm_geom_ids),
        side_b_geom_ids=set(hand_geom_ids),
        force_equiv_dt_s=force_equiv_dt_s,
        impulse_reduce=impulse_reduce,
    )


def measure_libero_episode_contact(
    env: Any,
    *,
    level: int,
    arm_geom_ids: set[int],
    robot_geom_ids: set[int],
    obstacle_geom_ids: set[int],
    force_equiv_dt_s: float,
    impulse_reduce: str,
) -> ContactStepResult:
    """Dispatch L3 person-safety vs L1/L2 obstacle contact measurement."""
    if int(level) == 3:
        return measure_l3_person_contact_step(
            env,
            arm_geom_ids=arm_geom_ids,
            hand_geom_ids=obstacle_geom_ids,
            force_equiv_dt_s=force_equiv_dt_s,
            impulse_reduce=impulse_reduce,
        )
    return measure_contact_step(
        env,
        robot_geom_ids=robot_geom_ids,
        obstacle_geom_ids=obstacle_geom_ids,
        force_equiv_dt_s=force_equiv_dt_s,
        impulse_reduce=impulse_reduce,
    )


def merge_pre_rollback_contact(
    measured: ContactStepResult,
    snapshot: ContactStepResult | None,
) -> ContactStepResult:
    """Keep a hit that was rolled back before ``env.step``.

    Post-step ``ncon`` is empty after L2 robot-block rollback. The snapshot
    is a normal ``mj_contactForce`` measurement at the colliding pose and
    is used as-is, including force / impulse.
    """
    if measured.in_contact:
        return measured
    if snapshot is None or not snapshot.in_contact:
        return measured
    return snapshot


def measure_tilt_deg(
    env: Any,
    body_name: str,
    *,
    ref_quat: np.ndarray | None = None,
) -> float:
    """Tilt of *body_name* from *ref_quat* (wxyz) or world-Z legacy fallback."""
    sim = _unwrap_render_env(env).sim
    return tilt_deg_from_body_quat(body_quat(sim, body_name), ref_quat=ref_quat)
