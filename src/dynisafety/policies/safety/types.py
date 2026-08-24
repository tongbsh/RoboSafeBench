"""Backend-agnostic safety observation / intervention contracts (Prompt 10).

Privileged god-view inputs only. Missing observations must fail-closed when the
oracle is enabled — never silent pass-through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from dynisafety.policies.base import ActionChunk


class OracleState(str, Enum):
    """Simplified pause-VLA state machine (no RETREAT / active avoidance)."""

    PASS_THROUGH = "pass_through"
    STOP = "stop"
    WAIT = "wait"
    RESUME = "resume"


class InterventionType(str, Enum):
    NONE = "none"
    STOP = "stop"
    WAIT = "wait"
    RESUME = "resume"
    FAIL_CLOSED = "fail_closed"


class OracleMode(str, Enum):
    STOP_WAIT_RESUME = "stop_wait_resume"
    STOP_ONLY = "stop_only"


@dataclass(frozen=True)
class SafetyObservation:
    """God-view disturbance snapshot injected by a backend adapter.

    Not derived from RGB. ``disturbance_active`` / ``motion_complete`` / phase
    drive the pause-VLA oracle; geometry fields are optional for adapters and
    future shields.
    """

    step: int
    disturbance_phase: str
    disturbance_active: bool
    motion_complete: bool
    disturbance_visible: bool = False
    min_distance_m: float | None = None
    closing_speed_mps: float | None = None
    timestamp_s: float | None = None
    level: int | None = None
    source: str = "god_view"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ValueError(f"SafetyObservation.step must be int >= 0, got {self.step!r}")
        if not isinstance(self.disturbance_phase, str) or not self.disturbance_phase.strip():
            raise ValueError(
                f"SafetyObservation.disturbance_phase must be non-empty str, "
                f"got {self.disturbance_phase!r}"
            )
        if self.source != "god_view" and "rgb" in self.source.lower():
            raise ValueError(
                "SafetyObservation.source must not claim RGB vision; "
                f"got {self.source!r}"
            )


@dataclass(frozen=True)
class SafetyInterventionResult:
    """One control-step decision from the oracle / shield layer."""

    executed_action: ActionChunk
    nominal_action: ActionChunk | None
    intervention_type: InterventionType
    reason: str
    state_before: OracleState
    state_after: OracleState
    thresholds: dict[str, Any]
    policy_inferred: bool
    oracle_paused: bool
    policy_infer_steps: int
    oracle_pause_steps: int


@dataclass
class OracleStepStats:
    """Episode-level counters: pause steps are excluded from VLA inference count."""

    policy_infer_steps: int = 0
    oracle_pause_steps: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "policy_infer_steps": int(self.policy_infer_steps),
            "oracle_pause_steps": int(self.oracle_pause_steps),
        }
