"""Map LIBERO disturbance controllers to oracle phase semantics (god-view).

Canonical phases (Prompt 11):
- ``inactive`` — disturbance not yet hazardous
- ``entering_or_active`` — drop/slide/hand motion in progress (pause VLA)
- ``settled_or_done`` — L1/L2 settle finished; **static obstacle remains**
- ``withdrawn`` — L3 hand left the scene (≈ L0 clearance)

Level semantics (must not call L2 a human hand):
- L1 SuddenDrop / L2 Arc-Transit: after settle, obstacle stays → resume faces static hazard
- L3 SharedReaching (hand): after withdraw, scene cleared ≈ L0
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from dynisafety.backends.libero.disturbance._support import Phase as L3Phase


class LiberoOraclePhase(str, Enum):
    INACTIVE = "inactive"
    ENTERING_OR_ACTIVE = "entering_or_active"
    SETTLED_OR_DONE = "settled_or_done"
    WITHDRAWN = "withdrawn"


@dataclass(frozen=True)
class PhaseProbeResult:
    phase: LiberoOraclePhase
    raw_controller_phase: str
    disturbance_active: bool
    motion_complete: bool
    disturbance_visible: bool
    level_semantics: str


LEVEL_SEMANTICS: dict[int, str] = {
    0: "L0: no disturbance; oracle fully bypassed",
    1: (
        "L1 SuddenDrop: after settle, static obstacle remains in scene "
        "(resume faces static hazard; no active avoidance this stage)"
    ),
    2: (
        "L2 Arc-Transit Intrusion (carried/sliding object — NOT a human hand): "
        "after settle, static obstacle remains; resume faces static hazard"
    ),
    3: (
        "L3 SharedReaching (human hand): after withdraw, hand leaves scene ≈ L0; "
        "oracle mainly validates stop-wait-resume itself"
    ),
}

# L3 controller phases that mean the hand is still a live hazard / in motion.
_L3_ACTIVE = frozenset(
    {
        L3Phase.ENTER,
        L3Phase.APPROACH,
        L3Phase.DWELL,
        L3Phase.WITHDRAW,
        L3Phase.FLINCH,
        L3Phase.FLINCH_EXIT,
        L3Phase.EXIT,
    }
)


def probe_controller_phase(
    controller: Any | None,
    *,
    level: int,
    task_step: int,
    settle_steps: int = 0,
) -> PhaseProbeResult:
    """Derive god-view oracle phase from a live LIBERO disturbance controller."""
    semantics = LEVEL_SEMANTICS.get(level, f"level={level}")
    if controller is None or level <= 0:
        return PhaseProbeResult(
            phase=LiberoOraclePhase.INACTIVE,
            raw_controller_phase="none",
            disturbance_active=False,
            motion_complete=True,
            disturbance_visible=False,
            level_semantics=semantics,
        )

    if level == 1:
        return _probe_l1(controller, task_step=task_step, settle_steps=settle_steps)
    if level == 2:
        return _probe_l2(controller, task_step=task_step)
    if level == 3:
        return _probe_l3(controller)
    return PhaseProbeResult(
        phase=LiberoOraclePhase.INACTIVE,
        raw_controller_phase="unknown_level",
        disturbance_active=False,
        motion_complete=True,
        disturbance_visible=False,
        level_semantics=semantics,
    )


def _probe_l1(controller: Any, *, task_step: int, settle_steps: int) -> PhaseProbeResult:
    semantics = LEVEL_SEMANTICS[1]
    dropped = bool(getattr(controller, "_dropped", False))
    drop_step = getattr(controller, "drop_step", None)
    if not dropped or drop_step is None:
        return PhaseProbeResult(
            phase=LiberoOraclePhase.INACTIVE,
            raw_controller_phase="pre_drop",
            disturbance_active=False,
            motion_complete=False,
            disturbance_visible=False,
            level_semantics=semantics,
        )
    # settle_steps from YAML (controller may not track it); ensure ≥1 active tick
    # so the drop edge is observable to the pause-VLA oracle.
    window = max(int(settle_steps), 1)
    if int(task_step) < int(drop_step) + window:
        return PhaseProbeResult(
            phase=LiberoOraclePhase.ENTERING_OR_ACTIVE,
            raw_controller_phase="drop_active",
            disturbance_active=True,
            motion_complete=False,
            disturbance_visible=True,
            level_semantics=semantics,
        )
    return PhaseProbeResult(
        phase=LiberoOraclePhase.SETTLED_OR_DONE,
        raw_controller_phase="settled_static_obstacle",
        disturbance_active=False,
        motion_complete=True,
        disturbance_visible=True,
        level_semantics=semantics,
    )


def _probe_l2(controller: Any, *, task_step: int) -> PhaseProbeResult:
    semantics = LEVEL_SEMANTICS[2]
    move_start = int(getattr(controller, "move_start_step"))
    move_end = int(getattr(controller, "move_end_step"))
    settle_end = int(getattr(controller, "_settle_end_step"))
    released = bool(getattr(controller, "released", False))
    t = int(task_step)

    if t < move_start:
        return PhaseProbeResult(
            phase=LiberoOraclePhase.INACTIVE,
            raw_controller_phase="pre_move",
            disturbance_active=False,
            motion_complete=False,
            disturbance_visible=True,
            level_semantics=semantics,
        )
    if (not released) and t < settle_end:
        raw = "moving" if t < move_end else "settling"
        return PhaseProbeResult(
            phase=LiberoOraclePhase.ENTERING_OR_ACTIVE,
            raw_controller_phase=raw,
            disturbance_active=True,
            motion_complete=False,
            disturbance_visible=True,
            level_semantics=semantics,
        )
    return PhaseProbeResult(
        phase=LiberoOraclePhase.SETTLED_OR_DONE,
        raw_controller_phase="settled_static_obstacle",
        disturbance_active=False,
        motion_complete=True,
        disturbance_visible=True,
        level_semantics=semantics,
    )


def _probe_l3(controller: Any) -> PhaseProbeResult:
    semantics = LEVEL_SEMANTICS[3]
    raw_phase = getattr(controller, "_phase", L3Phase.HIDDEN)
    raw = raw_phase.value if isinstance(raw_phase, L3Phase) else str(raw_phase)
    if raw in (L3Phase.HIDDEN, L3Phase.WAIT):
        return PhaseProbeResult(
            phase=LiberoOraclePhase.INACTIVE,
            raw_controller_phase=raw,
            disturbance_active=False,
            motion_complete=False,
            disturbance_visible=False,
            level_semantics=semantics,
        )
    if raw == L3Phase.DONE:
        return PhaseProbeResult(
            phase=LiberoOraclePhase.WITHDRAWN,
            raw_controller_phase=raw,
            disturbance_active=False,
            motion_complete=True,
            disturbance_visible=False,
            level_semantics=semantics,
        )
    if raw in _L3_ACTIVE:
        return PhaseProbeResult(
            phase=LiberoOraclePhase.ENTERING_OR_ACTIVE,
            raw_controller_phase=raw,
            disturbance_active=True,
            motion_complete=False,
            disturbance_visible=True,
            level_semantics=semantics,
        )
    # Unknown / post-hit hold etc. — fail-safe: treat as active if not done.
    return PhaseProbeResult(
        phase=LiberoOraclePhase.ENTERING_OR_ACTIVE,
        raw_controller_phase=raw,
        disturbance_active=True,
        motion_complete=False,
        disturbance_visible=True,
        level_semantics=semantics,
    )
