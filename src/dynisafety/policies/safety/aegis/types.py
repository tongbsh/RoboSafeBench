"""AEGIS ellipsoid / filter debug contracts (Prompt A0).

Pure geometry + QP I/O types. No VLA weights, no backend rollout, no Oracle FSM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class SolverFailurePolicy(str, Enum):
    """What to do when the CBF-QP solver fails or returns no solution."""

    FALLBACK_NOMINAL = "fallback_nominal"
    RAISE = "raise"


@dataclass(frozen=True)
class EllipsoidState:
    """Oriented ellipsoid in world frame (EEF or obstacle).

    Attributes
    ----------
    position :
        Center ``p`` in world coordinates, shape ``(3,)``.
    axes_diag :
        Semi-axis lengths ``Q_diag`` (local frame), shape ``(3,)``.
    rotation :
        World-from-local rotation ``R`` (``3×3`` orthonormal).
    """

    position: np.ndarray
    axes_diag: np.ndarray
    rotation: np.ndarray

    def __post_init__(self) -> None:
        pos = np.asarray(self.position, dtype=np.float64).reshape(3)
        axes = np.asarray(self.axes_diag, dtype=np.float64).reshape(3)
        rot = np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)
        object.__setattr__(self, "position", pos)
        object.__setattr__(self, "axes_diag", axes)
        object.__setattr__(self, "rotation", rot)
        if not np.all(np.isfinite(pos)):
            raise ValueError("EllipsoidState.position must be finite")
        if not np.all(np.isfinite(axes)) or np.any(axes <= 0):
            raise ValueError("EllipsoidState.axes_diag must be finite and > 0")
        if not np.all(np.isfinite(rot)):
            raise ValueError("EllipsoidState.rotation must be finite")


@dataclass(frozen=True)
class AegisDebugInfo:
    """Per-call diagnostics from :meth:`AegisActionFilter.filter`."""

    enabled: bool
    intervened: bool
    constraint_active: bool
    qp_failed: bool
    fallback_to_nominal: bool
    solver_status: str
    h: float | None = None
    nominal_slack: float | None = None
    filtered_slack: float | None = None
    z_direction: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        z = None if self.z_direction is None else self.z_direction.tolist()
        return {
            "enabled": self.enabled,
            "intervened": self.intervened,
            "constraint_active": self.constraint_active,
            "qp_failed": self.qp_failed,
            "fallback_to_nominal": self.fallback_to_nominal,
            "solver_status": self.solver_status,
            "h": self.h,
            "nominal_slack": self.nominal_slack,
            "filtered_slack": self.filtered_slack,
            "z_direction": z,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class CbfQpResult:
    """Raw QP solve output (internal to the aegis package)."""

    u: np.ndarray | None
    status: str
    h: float
    nominal_slack: float
    constraint_active: bool
    failed: bool
    a_u_v: np.ndarray
    a_u_omega: np.ndarray
    a_uz: np.ndarray
    mu_row: np.ndarray
    u_ref: np.ndarray
