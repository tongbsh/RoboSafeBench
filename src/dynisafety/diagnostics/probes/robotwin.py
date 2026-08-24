"""RoboTwin read-only phase probe.

L1/L2 controllers are thin; episode bookkeeping lives on
``DisturbanceManager``. This probe accepts either the inner controller or a
manager-shaped object (``controller`` + ``dynamic_level``) and unwraps.

Mapping (controller internals → ``DisturbancePhase``). Attributes in
parentheses are private.

L1 SuddenDrop (``_dropped``, ``_placed_sim_step``, ``_xy_latched``).
There is **no** ``drop_step`` and **no** ``disturbance_onset_step``:

- not ``_dropped`` → ``inactive`` (includes XY-latched but spawn-blocked)
- ``t == _placed_sim_step`` → ``active``
- later / ``_dropped`` after physics release → ``cleared``
  (``_placed_sim_step`` is cleared back to ``None`` on
  ``on_released_to_physics``; onset must be locked on first sighting)

**OnsetReporting gap (probe-side compensation).** RoboTwin L1 does not
implement a durable ``disturbance_onset_step`` / ``drop_step``. The only
onset signal is ``_placed_sim_step``, which ``on_released_to_physics``
nulls. ``SnapshotProbe`` latches the first non-``None`` onset so the four
diagnostic axes survive the release. The correct long-term fix is a
read-only property on the controller; that would touch red-line
``backends/*/disturbance/**`` files, so the probe carries the
compensation for now.

L2 SlidingObstacle (``move_start_step``, ``move_end_step``,
``hold_end_step``, ``_released``, ``_frozen``, ``release_reason``):

- ``t < move_start_step`` → ``inactive``
- first half of ``[move_start, move_end]`` (end **inclusive**, matching
  ``sim_step <= move_end_step``) → ``approaching``
- second half → ``active``
- ``move_end < t <= hold_end_step`` → ``settling``
- past ``hold_end_step`` or ``_released`` / ``_frozen`` → ``cleared``

``move_end_step`` defaults to 4000 in YAML (almost a full episode), so the
APPROACHING/ACTIVE split is a coarse half-path heuristic, not a geometric
closest-point test. There is no ``_settle_end_step``; settle is
``hold_end_step = move_end_step + hold_steps`` (``hold_steps`` comes from
YAML ``settle_steps``).

L3 SharedReaching (**public** ``phase`` and ``phase_start``,
``move_start_step``). No ``WITHDRAW``: dwell goes straight to ``EXIT``:

- ``hidden`` → ``inactive``
- ``enter`` / ``approach`` → ``approaching``
- ``dwell`` / ``flinch`` → ``active``
- ``post_hit_hold`` / ``exit`` → ``settling``
  (``EXIT`` is still on-field — hand withdrawing — so it is ``SETTLING``;
  only ``DONE`` is ``CLEARED``)
- ``done`` → ``cleared`` (``hazard_persists=False``)

Onset is ``move_start_step`` (L2/L3) or the locked ``_placed_sim_step`` (L1).
Read-only. Never imports ``dynisafety.backends``.
"""

from __future__ import annotations

from dynisafety.diagnostics.probes import (
    SnapshotProbe,
    first_int,
    map_desktop_l2,
    map_drop_l1,
    map_l3_phase,
    unwrap_manager,
)
from dynisafety.diagnostics.schema import DisturbancePhase


class RoboTwinDisturbanceProbe(SnapshotProbe):
    """Duck-typed adapter over a RoboTwin controller or DisturbanceManager."""

    def __init__(self, controller: object, *, level: int) -> None:
        super().__init__(unwrap_manager(controller), level=level)

    def _read_onset(self) -> int | None:
        if self._level == 1:
            return first_int(
                self._ctrl,
                "disturbance_onset_step",
                "drop_step",
                "_placed_sim_step",
            )
        if self._level == 2:
            return first_int(self._ctrl, "disturbance_onset_step", "move_start_step")
        return first_int(self._ctrl, "disturbance_onset_step", "move_start_step")

    def _phase_at(self, task_step: int) -> tuple[DisturbancePhase, int | None]:
        if self._level == 1:
            return map_drop_l1(self._ctrl, task_step)
        if self._level == 2:
            return map_desktop_l2(self._ctrl, task_step, end_inclusive=True)
        raw = getattr(self._ctrl, "phase", None)
        if raw is None:
            raw = getattr(self._ctrl, "_phase", None)
        return map_l3_phase(raw), None
