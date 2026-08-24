"""Read-only disturbance phase probes.

``make_probe`` returns a backend adapter that normalizes a controller snapshot
onto the frozen ``DisturbanceState`` contract
(``docs/behavior_diagnostics.md`` §4.4).

Call ``make_probe`` **once per episode**. The probe latches the first
non-``None`` onset and the first ``CLEARED`` hazard-end on that instance;
those locks are not cleared. Reusing a probe on a later episode leaks the
previous episode's anchors. There is no ``reset()`` — construct a new probe.

Constraints (do not relax):

- Never import ``dynisafety.backends.*`` (simulators are mutually exclusive).
- Never mutate controller state, step the simulator, or influence control flow.
- Access controller fields with ``getattr`` / duck typing, including private
  names such as ``_settle_end_step``. Missing attributes degrade to conservative
  values (``INACTIVE`` / ``None``); a snapshot must not crash a rollout.
- L0 and ``controller is None`` yield ``None`` (no disturbance). Unknown
  backend names raise.
"""

from __future__ import annotations

import logging
from typing import Any

from dynisafety.diagnostics.schema import (
    DisturbancePhase,
    DisturbanceProbe,
    DisturbanceState,
    Vec3,
)

_LOG = logging.getLogger(__name__)

KNOWN_BACKENDS: tuple[str, ...] = ("libero", "simplerenv", "robotwin", "robocasa")

# Unified L3 vocabulary (``dynisafety.core.disturbance.Phase`` plus backend
# subsets). Compared as lowercase strings so probes never import core/backends.
# EXIT / WITHDRAW (and flinch-exit / post-hit-hold) stay on-field, so they
# map to SETTLING. Only DONE is CLEARED — the hand has left and the scene
# is empty. Spec §1.1 said "first EXIT / DONE"; this five-phase mapping is
# the source of truth.
L3_PHASE_MAP: dict[str, DisturbancePhase] = {
    "hidden": DisturbancePhase.INACTIVE,
    "wait": DisturbancePhase.INACTIVE,
    "enter": DisturbancePhase.APPROACHING,
    "approach": DisturbancePhase.APPROACHING,
    "dwell": DisturbancePhase.ACTIVE,
    "flinch": DisturbancePhase.ACTIVE,
    "post_hit_hold": DisturbancePhase.SETTLING,
    "withdraw": DisturbancePhase.SETTLING,
    "flinch_exit": DisturbancePhase.SETTLING,
    "exit": DisturbancePhase.SETTLING,
    "done": DisturbancePhase.CLEARED,
}


def make_probe(backend: str, controller: object, *, level: int) -> DisturbanceProbe | None:
    """Build a read-only probe, or ``None`` when there is no disturbance.

    Always construct a **new** probe for each episode (this factory does
    that). First-seen ``onset_step`` and ``hazard_end_step`` are latched on
    the instance and are not reset.

    ``controller is None`` or ``level == 0`` → ``None``. Unknown ``backend``
    raises ``ValueError``.
    """
    if controller is None:
        return None
    try:
        lvl = int(level)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"level must be an int, got {level!r}") from exc
    if lvl == 0:
        return None
    if lvl not in (1, 2, 3):
        raise ValueError(f"unsupported disturbance level {lvl!r}; expected 0, 1, 2, or 3")

    key = str(backend).strip().lower()
    if key == "libero":
        from dynisafety.diagnostics.probes.libero import LiberoDisturbanceProbe

        return LiberoDisturbanceProbe(controller, level=lvl)
    if key == "simplerenv":
        from dynisafety.diagnostics.probes.simplerenv import SimplerEnvDisturbanceProbe

        return SimplerEnvDisturbanceProbe(controller, level=lvl)
    if key == "robotwin":
        from dynisafety.diagnostics.probes.robotwin import RoboTwinDisturbanceProbe

        return RoboTwinDisturbanceProbe(controller, level=lvl)
    if key == "robocasa":
        from dynisafety.diagnostics.probes.robocasa import RoboCasaDisturbanceProbe

        return RoboCasaDisturbanceProbe(controller, level=lvl)
    raise ValueError(
        f"unknown diagnostics backend {backend!r}; expected one of {KNOWN_BACKENDS}"
    )


def hazard_persists_for_level(level: int) -> bool | None:
    """L1/L2 persist a static obstacle; L3 clears the scene; L0 is undefined."""
    if level in (1, 2):
        return True
    if level == 3:
        return False
    return None


def phase_token(raw: object) -> str:
    """Lowercase phase name from an enum, string, or ``None``."""
    if raw is None:
        return ""
    value = getattr(raw, "value", raw)
    return str(value).strip().lower()


def map_l3_phase(raw: object) -> DisturbancePhase:
    """Map a backend L3 phase token onto ``DisturbancePhase``.

    ``exit`` / ``withdraw`` remain on-field (hand still retracting) and map
    to ``SETTLING``. Only ``done`` is ``CLEARED``.

    Empty / missing → ``INACTIVE``. Unknown tokens → ``ACTIVE`` (fail-safe:
    treat a live but unrecognized phase as still hazardous).
    """
    token = phase_token(raw)
    if not token:
        return DisturbancePhase.INACTIVE
    mapped = L3_PHASE_MAP.get(token)
    if mapped is not None:
        return mapped
    return DisturbancePhase.ACTIVE


def first_int(ctrl: object, *names: str) -> int | None:
    """Return the first attribute that coerces to ``int``, else ``None``."""
    for name in names:
        value = getattr(ctrl, name, None)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def flag(ctrl: object, *names: str) -> bool:
    """True if any named attribute is truthy."""
    return any(bool(getattr(ctrl, name, False)) for name in names)


def as_vec3(value: Any) -> Vec3 | None:
    """Coerce an xyz-like object to ``Vec3``. Never raises to the caller."""
    if value is None:
        return None
    try:
        xyz = getattr(value, "p", value)
        xyz = getattr(xyz, "position", xyz)
        items = list(xyz)[:3]
        if len(items) < 3:
            return None
        return (float(items[0]), float(items[1]), float(items[2]))
    except Exception:
        return None


def read_obstacle_pos(ctrl: object) -> Vec3 | None:
    """Best-effort body origin. Duck-typed; returns ``None`` if unreadable.

    Order: ``_get_pos()``, ``actor`` / ``entity`` ``get_pose().p``, a cached
    3-vector ``obstacle_pos``, then a MuJoCo-shaped ``env.sim`` / ``objects_dict``
    path (LIBERO ``read_obstacle_table_pose`` / RoboCasa ``read_obstacle_pos``
    without importing those modules).
    """
    getter = getattr(ctrl, "_get_pos", None)
    if callable(getter):
        pos = as_vec3(getter())
        if pos is not None:
            return pos
    for attr in ("actor", "entity"):
        obj = getattr(ctrl, attr, None)
        if obj is None:
            continue
        get_pose = getattr(obj, "get_pose", None)
        if not callable(get_pose):
            continue
        try:
            pos = as_vec3(get_pose())
        except Exception:
            pos = None
        if pos is not None:
            return pos
    cached = getattr(ctrl, "obstacle_pos", None)
    if cached is not None and not callable(cached):
        pos = as_vec3(cached)
        if pos is not None:
            return pos
    return _read_mujoco_body_pos(ctrl)


def _read_mujoco_body_pos(ctrl: object) -> Vec3 | None:
    name: object | None = None
    for attr in ("obstacle_name", "hand_name", "body_name"):
        name = getattr(ctrl, attr, None)
        if name:
            break
    if not name:
        name = getattr(ctrl, "active_obstacle_name", None)
    if not name:
        return None
    env = getattr(ctrl, "env", None)
    if env is None:
        handle = getattr(ctrl, "handle", None)
        env = getattr(handle, "env", None) if handle is not None else None
    if env is None:
        return None
    try:
        sim = getattr(env, "sim", None)
        if sim is not None:
            body_id = sim.model.body_name2id(str(name))
            return as_vec3(sim.data.body_xpos[body_id])
    except Exception:
        pass
    try:
        inner = getattr(env, "env", env)
        objects = getattr(inner, "objects_dict", None)
        if objects is None:
            return None
        obj = objects.get(str(name)) if hasattr(objects, "get") else None
        if obj is None:
            return None
        sim = getattr(inner, "sim", None)
        if sim is None:
            return None
        body_id = sim.model.body_name2id(obj.root_body)
        return as_vec3(sim.data.body_xpos[body_id])
    except Exception:
        return None


def l1_settle_end(ctrl: object, drop_step: int) -> int:
    """First CLEARED step for a drop-style L1, matching the LIBERO oracle window.

    Prefer ``_settle_end_step``. Else ``drop_step + max(settle_steps, 1)``.
    Controllers that do not store a settle window get a 1-step ACTIVE tick
    (YAML ``settle_steps: 0`` is what LIBERO L1 actually ships).
    """
    explicit = first_int(ctrl, "_settle_end_step")
    if explicit is not None:
        return explicit
    steps = first_int(ctrl, "settle_steps")
    window = max(int(steps), 1) if steps is not None else 1
    return int(drop_step) + window


def l2_settle_end(ctrl: object) -> int | None:
    """``move_end + settle`` if the controller exposes it."""
    explicit = first_int(ctrl, "_settle_end_step", "hold_end_step")
    if explicit is not None:
        return explicit
    move_end = first_int(ctrl, "move_end_step")
    steps = first_int(ctrl, "settle_steps")
    if move_end is None or steps is None:
        return None
    return int(move_end) + int(steps)


def map_drop_l1(
    ctrl: object, task_step: int, *, appear_at_rest: bool = False
) -> tuple[DisturbancePhase, int | None]:
    """Desktop / spawn L1: hidden → drop (ACTIVE) → settle → CLEARED.

    ``appear_at_rest=True`` (RoboCasa floor box) skips falling/settle: the
    object is already parked when it becomes visible, so the first dropped
    step is ``CLEARED``.
    """
    dropped = flag(ctrl, "_dropped")
    drop_step = first_int(ctrl, "drop_step", "_placed_sim_step")
    if not dropped:
        return DisturbancePhase.INACTIVE, None
    if drop_step is None:
        return DisturbancePhase.CLEARED, None
    if appear_at_rest:
        if task_step < drop_step:
            return DisturbancePhase.INACTIVE, None
        return DisturbancePhase.CLEARED, drop_step
    settle_end = l1_settle_end(ctrl, drop_step)
    if task_step < drop_step:
        return DisturbancePhase.INACTIVE, None
    if task_step < settle_end:
        if task_step == drop_step:
            return DisturbancePhase.ACTIVE, None
        return DisturbancePhase.SETTLING, None
    return DisturbancePhase.CLEARED, settle_end


def map_desktop_l2(
    ctrl: object,
    task_step: int,
    *,
    end_inclusive: bool = False,
) -> tuple[DisturbancePhase, int | None]:
    """Scheduled slide: pre-move → first-half APPROACHING → ACTIVE → settle → CLEARED.

    ``end_inclusive=True`` matches RoboTwin (``sim_step <= move_end_step`` is
    still moving). Early ``released`` / ``blocked`` / ``_frozen`` skip to
    ``CLEARED``. Predicted ``hazard_end_step`` is the settle end on a clean
    finish, or ``block_task_step`` when the controller latched a contact step.
    """
    start = first_int(ctrl, "move_start_step")
    end = first_int(ctrl, "move_end_step")
    settle_end = l2_settle_end(ctrl)
    released = flag(ctrl, "released", "_released", "_frozen")
    blocked = flag(ctrl, "blocked")
    reason = phase_token(getattr(ctrl, "release_reason", None) or getattr(ctrl, "_release_reason", None))

    def _cleared_predicted() -> int | None:
        if reason == "settle" and settle_end is not None:
            return settle_end
        if blocked:
            hit = first_int(ctrl, "block_task_step", "_block_task_step")
            if hit is not None:
                return hit
        if settle_end is not None and (released or task_step >= settle_end):
            if reason in ("", "settle") and not blocked:
                return settle_end
        return None

    if start is None or end is None:
        if released or blocked:
            return DisturbancePhase.CLEARED, _cleared_predicted()
        return DisturbancePhase.INACTIVE, None

    if released or blocked:
        return DisturbancePhase.CLEARED, _cleared_predicted()
    if task_step < start:
        return DisturbancePhase.INACTIVE, None
    still_moving = task_step <= end if end_inclusive else task_step < end
    if still_moving:
        mid = float(start) + (float(end) - float(start)) / 2.0
        if task_step < mid:
            return DisturbancePhase.APPROACHING, None
        return DisturbancePhase.ACTIVE, None
    if settle_end is not None and task_step < settle_end:
        return DisturbancePhase.SETTLING, None
    return DisturbancePhase.CLEARED, settle_end


def unwrap_manager(ctrl: object) -> object:
    """If ``ctrl`` looks like RoboTwin ``DisturbanceManager``, return its controller."""
    inner = getattr(ctrl, "controller", None)
    if inner is not None and getattr(ctrl, "dynamic_level", None) is not None:
        return inner
    return ctrl


class SnapshotProbe:
    """Shared snapshot lock + fail-soft wrapper. Subclasses implement ``_phase_at``.

    First non-``None`` onset and first ``CLEARED`` hazard-end are latched
    here so a controller that later nulls those fields (RoboTwin L1
    ``_placed_sim_step`` after ``on_released_to_physics``) cannot wipe the
    episode anchors. There is no ``reset()``: call ``make_probe`` again for
    the next episode.
    """

    def __init__(self, controller: object, *, level: int) -> None:
        self._ctrl = controller
        self._level = int(level)
        self._onset_locked: int | None = None
        self._hazard_end_locked: int | None = None
        self._warned = False

    def snapshot(self, task_step: int) -> DisturbanceState:
        try:
            return self._snapshot_impl(int(task_step))
        except Exception:
            self._warn_once()
            return self._conservative()

    def _snapshot_impl(self, task_step: int) -> DisturbanceState:
        phase, predicted_end = self._phase_at(task_step)
        onset = self._lock_onset()
        hazard_end = self._lock_hazard_end(phase, task_step, predicted_end)
        pos = None
        if self._should_read_pos(phase):
            pos = read_obstacle_pos(self._ctrl)
        return DisturbanceState(
            phase=phase,
            obstacle_pos=pos,
            onset_step=onset,
            hazard_end_step=hazard_end,
            hazard_persists=hazard_persists_for_level(self._level),
        )

    def _phase_at(self, task_step: int) -> tuple[DisturbancePhase, int | None]:
        raise NotImplementedError

    def _read_onset(self) -> int | None:
        return first_int(self._ctrl, "disturbance_onset_step")

    def _lock_onset(self) -> int | None:
        """Return the first non-``None`` onset seen on this instance.

        Subsequent controller clears (``None``) do not unlock it. Shared by
        every backend probe — not a RoboTwin-only workaround.
        """
        if self._onset_locked is not None:
            return self._onset_locked
        value = self._read_onset()
        if value is not None:
            self._onset_locked = int(value)
        return self._onset_locked

    def _lock_hazard_end(
        self,
        phase: DisturbancePhase,
        task_step: int,
        predicted: int | None,
    ) -> int | None:
        if self._hazard_end_locked is not None:
            return self._hazard_end_locked
        if phase is not DisturbancePhase.CLEARED:
            return None
        self._hazard_end_locked = int(predicted) if predicted is not None else int(task_step)
        return self._hazard_end_locked

    def _should_read_pos(self, phase: DisturbancePhase) -> bool:
        if phase is DisturbancePhase.INACTIVE:
            return self._level == 2
        if phase is DisturbancePhase.CLEARED and self._level == 3:
            return False
        return True

    def _conservative(self) -> DisturbanceState:
        return DisturbanceState(
            phase=DisturbancePhase.INACTIVE,
            obstacle_pos=None,
            onset_step=self._onset_locked,
            hazard_end_step=self._hazard_end_locked,
            hazard_persists=hazard_persists_for_level(self._level),
        )

    def _warn_once(self) -> None:
        if self._warned:
            return
        self._warned = True
        _LOG.warning(
            "diagnostics probe snapshot failed; returning conservative DisturbanceState",
            exc_info=True,
        )


__all__ = [
    "KNOWN_BACKENDS",
    "L3_PHASE_MAP",
    "SnapshotProbe",
    "as_vec3",
    "first_int",
    "flag",
    "hazard_persists_for_level",
    "make_probe",
    "map_desktop_l2",
    "map_drop_l1",
    "map_l3_phase",
    "phase_token",
    "read_obstacle_pos",
    "unwrap_manager",
]
