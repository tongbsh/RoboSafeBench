"""LIBERO read-only phase probe.

Mapping (controller internals → ``DisturbancePhase``). Attributes in
parentheses are private and read with ``getattr(ctrl, "_name", ...)``.

L1 SuddenDrop (``_dropped``, ``drop_step``, optional ``settle_steps`` /
``_settle_end_step``):

- not dropped / no ``drop_step`` → ``inactive``
- ``t == drop_step`` → ``active`` (teleport + fall)
- ``drop_step < t < settle_end`` → ``settling``
- ``t >= settle_end`` → ``cleared`` (static obstacle remains;
  ``hazard_persists=True``)
- ``settle_end = _settle_end_step`` or ``drop_step + max(settle_steps, 1)``
  or ``drop_step + 1`` when the controller does not store a window
  (LIBERO L1 YAML ships ``settle_steps: 0``; the oracle uses the same
  ``max(..., 1)`` tick so the drop edge is observable)

L2 DynamicObstacle (``move_start_step``, ``move_end_step``,
``_settle_end_step``, ``released``, ``blocked``, ``_block_task_step``):

- ``t < move_start_step`` → ``inactive`` (already posed at start, not hidden)
- first half of ``[move_start, move_end)`` → ``approaching``
- second half → ``active``
- ``[move_end, _settle_end_step)`` → ``settling``
- ``t >= _settle_end_step`` or ``released`` or ``blocked`` → ``cleared``
  (obstacle stays; ``hazard_persists=True``)

L3 SharedReaching (``_phase`` / ``core.disturbance.Phase``,
``_entry_start_step``):

- ``hidden`` / ``wait`` → ``inactive``
- ``enter`` / ``approach`` → ``approaching``
- ``dwell`` / ``flinch`` → ``active``
- ``withdraw`` / ``post_hit_hold`` / ``flinch_exit`` / ``exit`` → ``settling``
  (``EXIT`` / ``WITHDRAW`` are still on-field, so they are ``SETTLING``;
  only ``DONE`` is ``CLEARED``)
- ``done`` → ``cleared`` (hand gone; ``hazard_persists=False``)
- unknown token → ``active`` (fail-safe)

Read-only. Never imports ``dynisafety.backends``.
"""

from __future__ import annotations

from dynisafety.diagnostics.probes import (
    SnapshotProbe,
    first_int,
    map_desktop_l2,
    map_drop_l1,
    map_l3_phase,
)
from dynisafety.diagnostics.schema import DisturbancePhase


class LiberoDisturbanceProbe(SnapshotProbe):
    """Duck-typed adapter over a LIBERO L1/L2/L3 disturbance controller."""

    def _read_onset(self) -> int | None:
        if self._level == 1:
            return first_int(self._ctrl, "disturbance_onset_step", "drop_step")
        if self._level == 2:
            return first_int(self._ctrl, "disturbance_onset_step", "move_start_step")
        return first_int(self._ctrl, "disturbance_onset_step", "_entry_start_step")

    def _phase_at(self, task_step: int) -> tuple[DisturbancePhase, int | None]:
        if self._level == 1:
            return map_drop_l1(self._ctrl, task_step)
        if self._level == 2:
            return map_desktop_l2(self._ctrl, task_step)
        raw = getattr(self._ctrl, "phase", None)
        if raw is None:
            raw = getattr(self._ctrl, "_phase", None)
        return map_l3_phase(raw), None
