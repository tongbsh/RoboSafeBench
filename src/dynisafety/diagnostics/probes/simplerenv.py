"""SimplerEnv read-only phase probe.

Mapping (controller internals → ``DisturbancePhase``). Attributes in
parentheses are private.

L1 SuddenDrop (``_dropped``, ``drop_step``, ``blocked``):

- not dropped → ``inactive``
- ``t == drop_step`` → ``active`` (kinematic teleport, then physics)
- ``t > drop_step`` → ``cleared`` (1-step window; no settle field on the
  controller; ``hazard_persists=True``)

L2 SlidingObstacle (``move_start_step``, ``move_end_step``,
``_settle_end_step``, ``_released``, ``blocked``):

- ``t < move_start_step`` → ``inactive``
- first half of ``[move_start, move_end)`` → ``approaching``
- second half → ``active``
- ``[move_end, _settle_end_step)`` → ``settling``
- ``t >= _settle_end_step`` or ``_released`` or ``blocked`` → ``cleared``
  (``blocked`` also sets ``_released`` on the real controller)

``released`` is **private** here (``_released``), unlike LIBERO L2.

L3 SharedReaching (public ``phase`` property, ``_phase``,
``_entry_start_step``):

- ``hidden`` / ``wait`` → ``inactive``
- ``enter`` / ``approach`` → ``approaching``
- ``dwell`` / ``flinch`` → ``active``
- ``withdraw`` / ``post_hit_hold`` / ``flinch_exit`` / ``exit`` → ``settling``
  (``EXIT`` / ``WITHDRAW`` are still on-field, so they are ``SETTLING``;
  only ``DONE`` is ``CLEARED``)
- ``done`` → ``cleared`` (``hazard_persists=False``)

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


class SimplerEnvDisturbanceProbe(SnapshotProbe):
    """Duck-typed adapter over a SimplerEnv L1/L2/L3 disturbance controller."""

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
