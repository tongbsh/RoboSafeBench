"""Minimal local stand-ins when ``dynisafety.policies.safety`` is not installed.

Prompt 10 owns the canonical API under ``policies/safety/``. Until that lands on
the integration branch, LIBERO adapters import through :func:`import_safety_api`
so unit tests and wiring stay unblocked. Mapping notes live in the Prompt 11
handoff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class _OracleState(str, Enum):
    PASS_THROUGH = "pass_through"
    STOP = "stop"
    WAIT = "wait"
    RESUME = "resume"


class _InterventionType(str, Enum):
    NONE = "none"
    STOP = "stop"
    WAIT = "wait"
    RESUME = "resume"
    FAIL_CLOSED = "fail_closed"


@dataclass(frozen=True)
class _SafetyObservation:
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


def import_safety_api() -> dict[str, Any]:
    """Return SafetyObservation / wrap_policy / OracleState symbols."""
    try:
        from dynisafety.policies.safety import (  # type: ignore[attr-defined]
            InterventionType,
            OracleState,
            SafetyObservation,
            wrap_policy,
        )
        from dynisafety.policies.safety.config import load_oracle_config
        from dynisafety.policies.safety.types import SafetyInterventionResult

        return {
            "SafetyObservation": SafetyObservation,
            "OracleState": OracleState,
            "InterventionType": InterventionType,
            "SafetyInterventionResult": SafetyInterventionResult,
            "wrap_policy": wrap_policy,
            "load_oracle_config": load_oracle_config,
            "source": "policies.safety",
        }
    except ImportError:
        return {
            "SafetyObservation": _SafetyObservation,
            "OracleState": _OracleState,
            "InterventionType": _InterventionType,
            "SafetyInterventionResult": None,
            "wrap_policy": None,
            "load_oracle_config": None,
            "source": "libero.safety.compat",
        }
