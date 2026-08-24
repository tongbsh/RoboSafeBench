"""Build ``SafetyObservation`` from LIBERO god-view phase probes."""

from __future__ import annotations

from typing import Any

from dynisafety.backends.libero.safety.compat import import_safety_api
from dynisafety.backends.libero.safety.phases import (
    LEVEL_SEMANTICS,
    LiberoOraclePhase,
    PhaseProbeResult,
    probe_controller_phase,
)
from dynisafety.config.schema import MISSING, RunConfig


def settle_steps_for_level(cfg: RunConfig, level: int) -> int:
    """Read settle_steps for L1 (oracle window) / L2 (controller) from RunConfig."""
    disturbance = cfg.disturbance
    if disturbance is MISSING or disturbance is None:
        return 0
    if level == 1:
        l1 = disturbance.l1
        if l1 is MISSING or l1 is None:
            return 0
        steps = getattr(l1, "settle_steps", MISSING)
        return 0 if steps is MISSING or steps is None else int(steps)
    if level == 2:
        l2 = disturbance.l2
        if l2 is MISSING or l2 is None:
            return 0
        steps = getattr(l2, "settle_steps", MISSING)
        return 0 if steps is MISSING or steps is None else int(steps)
    return 0


def build_safety_observation(
    *,
    controller: Any | None,
    level: int,
    task_step: int,
    settle_steps: int = 0,
    probe: PhaseProbeResult | None = None,
) -> Any:
    """Fill a privileged ``SafetyObservation`` (never from RGB)."""
    api = import_safety_api()
    SafetyObservation = api["SafetyObservation"]
    result = probe or probe_controller_phase(
        controller,
        level=level,
        task_step=task_step,
        settle_steps=settle_steps,
    )
    # Map LIBERO phases onto oracle YAML active/resume vocabulary.
    if result.phase is LiberoOraclePhase.ENTERING_OR_ACTIVE:
        phase_str = (
            "enter" if "drop" in result.raw_controller_phase else result.raw_controller_phase
        )
        # Prefer canonical names understood by policies.safety active_phases.
        if result.raw_controller_phase in {"drop_active", "moving", "settling"}:
            phase_str = {
                "drop_active": "drop",
                "moving": "move",
                "settling": "settle",
            }[result.raw_controller_phase]
        elif result.raw_controller_phase in {
            "enter",
            "approach",
            "dwell",
            "withdraw",
            "exit",
            "flinch",
            "flinch_exit",
        }:
            phase_str = result.raw_controller_phase
        else:
            phase_str = "enter"
    elif result.phase is LiberoOraclePhase.SETTLED_OR_DONE:
        phase_str = "done"
    elif result.phase is LiberoOraclePhase.WITHDRAWN:
        phase_str = "done"
    else:
        phase_str = "idle"

    return SafetyObservation(
        step=int(task_step) if task_step >= 0 else 0,
        disturbance_phase=phase_str,
        disturbance_active=bool(result.disturbance_active),
        motion_complete=bool(result.motion_complete),
        disturbance_visible=bool(result.disturbance_visible),
        min_distance_m=None,  # geometry adapter optional; phase is primary signal
        closing_speed_mps=None,
        level=int(level),
        source="god_view",
        extra={
            "libero_oracle_phase": result.phase.value,
            "raw_controller_phase": result.raw_controller_phase,
            "level_semantics": result.level_semantics,
        },
    )


__all__ = [
    "LEVEL_SEMANTICS",
    "build_safety_observation",
    "settle_steps_for_level",
]
