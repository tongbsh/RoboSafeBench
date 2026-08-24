"""RoboCasa read-only phase probe.

Kitchen **floor navigation**, not tabletop drop / slide / shared-reaching.
Do not reuse desktop L1-fall / L2-arc semantics.

RoboCasa L2 has no hazard-end: the obstacle ping-pongs across the destination
apron for the rest of the episode. ``hazard_end_step`` stays None on clean
episodes, so the Recovery axis is **n/a** for this backend/level pair. Tables
must render n/a, never 0.0.

Mapping (controller internals → ``DisturbancePhase``). Attributes in
parentheses are private.

L1 Frontal Sudden Appear (``_dropped``, ``drop_step``, ``blocked``,
``spawn_failed``):

- not dropped (including ``spawn_failed``) → ``inactive``
- dropped → ``cleared`` immediately (the box is already at rest on the
  floor; there is no falling / settle window). ``hazard_end_step = drop_step``.
  ``hazard_persists=True``.

L2 dest-apron ping-pong (``_spawned``, ``onset_step``, ``released``,
``blocked``, ``move_start_step``, ``move_end_step``, ``_one_way_steps``):

``move_start_step`` / ``move_end_step`` are **not** a global timeline —
they only define the one-way crossing duration. Onset is distance-triggered
(``onset_step``). The box is visible parked at an aisle end *before* onset.

- not ``_spawned`` or ``onset_step is None`` → ``inactive``
- first half of the first crossing → ``approaching``
- remainder of the ping-pong → ``active``
- ``released`` / ``blocked`` (handed to physics) → ``cleared``
- there is **no** settle phase on a clean episode: the box keeps crossing
  until contact or timeout. ``hazard_end_step`` stays ``None`` (intentional;
  Recovery is n/a — do not invent an end step).

Do **not** fall back to ``move_start_step`` as onset.

L3 Approaching Human (``_phase``, ``onset_step``, ``_spawned``). Floor walk
onto the dest apron, dwell, withdraw, hide:

- ``hidden`` → ``inactive``
- ``enter`` / ``approach`` → ``approaching``
- ``dwell`` / ``flinch`` → ``active`` (flinch is a short floor recoil)
- ``withdraw`` → ``settling``
  (``WITHDRAW`` is still on-field, so it is ``SETTLING``; only ``DONE``
  is ``CLEARED``)
- ``done`` → ``cleared`` (``hazard_persists=False``)

Onset is ``onset_step`` / ``disturbance_onset_step`` only (distance-triggered;
may stay ``None`` for the whole episode).

Read-only. Never imports ``dynisafety.backends``.
"""

from __future__ import annotations

from dynisafety.diagnostics.probes import (
    SnapshotProbe,
    first_int,
    flag,
    map_drop_l1,
    map_l3_phase,
)
from dynisafety.diagnostics.schema import DisturbancePhase


def _map_l2_pingpong(ctrl: object, task_step: int) -> tuple[DisturbancePhase, int | None]:
    spawned = flag(ctrl, "_spawned")
    onset = first_int(ctrl, "onset_step", "disturbance_onset_step")
    released = flag(ctrl, "released", "_released")
    blocked = flag(ctrl, "blocked")
    if released or (blocked and spawned and onset is not None):
        return DisturbancePhase.CLEARED, None
    if not spawned or onset is None:
        return DisturbancePhase.INACTIVE, None
    elapsed = int(task_step) - int(onset)
    one_way = first_int(ctrl, "_one_way_steps")
    if one_way is None:
        start = first_int(ctrl, "move_start_step")
        end = first_int(ctrl, "move_end_step")
        if start is not None and end is not None and end > start:
            one_way = int(end) - int(start)
    if one_way is not None and elapsed < (float(one_way) / 2.0):
        return DisturbancePhase.APPROACHING, None
    return DisturbancePhase.ACTIVE, None


class RoboCasaDisturbanceProbe(SnapshotProbe):
    """Duck-typed adapter over a RoboCasa L1/L2/L3 floor-nav controller."""

    def _read_onset(self) -> int | None:
        # Never use move_start_step — on L2 that is a duration bound, not onset.
        if self._level == 1:
            return first_int(self._ctrl, "disturbance_onset_step", "drop_step")
        return first_int(self._ctrl, "disturbance_onset_step", "onset_step")

    def _phase_at(self, task_step: int) -> tuple[DisturbancePhase, int | None]:
        if self._level == 1:
            return map_drop_l1(self._ctrl, task_step, appear_at_rest=True)
        if self._level == 2:
            return _map_l2_pingpong(self._ctrl, task_step)
        raw = getattr(self._ctrl, "phase", None)
        if raw is None:
            raw = getattr(self._ctrl, "_phase", None)
        return map_l3_phase(raw), None
