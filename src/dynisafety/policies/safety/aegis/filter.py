"""Stateful AEGIS action filter: nominal 7D → CBF-QP filtered 7D."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dynisafety.policies.safety.aegis.cbf_qp import (
    default_z_direction,
    map_u_to_action7,
    solve_cbf_qp,
    update_z_direction,
)
from dynisafety.policies.safety.aegis.config import (
    AegisConfig,
    aegis_config_with_enabled,
    load_aegis_config,
)
from dynisafety.policies.safety.aegis.types import (
    AegisDebugInfo,
    EllipsoidState,
    SolverFailurePolicy,
)


class AegisActionFilter:
    """CBF-QP filter over delta/abs EE 7D actions.

    Parameters
    ----------
    config :
        Fully-specified :class:`AegisConfig` (thresholds from YAML only).

    Notes
    -----
    * ``enabled=false`` → return the same ndarray object when possible
      (bit-identical values always).
    * Gripper dim (``config.gripper_index``) is never rewritten by the QP.
    * Solver failure (default): fall back to nominal and set
      ``debug.qp_failed=True`` / ``fallback_to_nominal=True``.
    """

    def __init__(self, config: AegisConfig) -> None:
        self._config = config
        self._z: np.ndarray | None = None

    @property
    def config(self) -> AegisConfig:
        return self._config

    @classmethod
    def from_yaml(
        cls,
        path: str | Path | None = None,
        *,
        enabled: bool | None = None,
    ) -> AegisActionFilter:
        cfg = load_aegis_config(path)
        if enabled is not None:
            cfg = aegis_config_with_enabled(cfg, enabled)
        return cls(cfg)

    def reset(self, z_direction: np.ndarray | None = None) -> None:
        """Clear or set the separating-axis direction state."""
        if z_direction is None:
            self._z = None
            return
        z = np.asarray(z_direction, dtype=np.float64).reshape(3)
        n = np.linalg.norm(z)
        if n < 1e-12:
            raise ValueError("z_direction must be non-zero")
        self._z = z / n

    @property
    def z_direction(self) -> np.ndarray | None:
        return None if self._z is None else self._z.copy()

    def filter(
        self,
        nominal_7d: np.ndarray,
        eef_state: EllipsoidState,
        obstacle_state: EllipsoidState,
    ) -> tuple[np.ndarray, AegisDebugInfo]:
        """Filter a nominal 7D action through the CBF-QP (or pass through).

        Returns
        -------
        filtered_7d, debug
        """
        nominal_in = np.asarray(nominal_7d)
        if nominal_in.shape != (7,):
            raise ValueError(f"nominal_7d must have shape (7,), got {nominal_in.shape}")

        cfg = self._config
        if not cfg.enabled:
            # Bit-identical path: no dtype cast, no copy, no QP.
            return nominal_in, AegisDebugInfo(
                enabled=False,
                intervened=False,
                constraint_active=False,
                qp_failed=False,
                fallback_to_nominal=False,
                solver_status="disabled",
                h=None,
                nominal_slack=None,
                filtered_slack=None,
                z_direction=None,
            )

        nominal = np.asarray(nominal_in, dtype=np.float64)

        if self._z is None:
            self._z = default_z_direction(eef_state, obstacle_state)

        result = solve_cbf_qp(nominal, eef_state, obstacle_state, self._z, cfg)

        if result.failed or result.u is None:
            if cfg.on_solver_failure is SolverFailurePolicy.RAISE:
                raise RuntimeError(
                    f"AEGIS CBF-QP solver failed with status={result.status!r}"
                )
            return nominal.copy(), AegisDebugInfo(
                enabled=True,
                intervened=False,
                constraint_active=result.constraint_active,
                qp_failed=True,
                fallback_to_nominal=True,
                solver_status=result.status,
                h=result.h,
                nominal_slack=result.nominal_slack,
                filtered_slack=None,
                z_direction=self._z.copy(),
                extra={"failure_policy": cfg.on_solver_failure.value},
            )

        filtered = map_u_to_action7(
            result.u,
            rotation=eef_state.rotation,
            gripper=float(nominal[cfg.gripper_index]),
            cfg=cfg,
        )
        # Integrate z with the solved u_z (upstream main_aegis.py).
        self._z = update_z_direction(self._z, result.u[6:], dt=cfg.dt)

        filtered_slack = float(
            result.a_u_v @ result.u[:3]
            + result.a_u_omega @ result.u[3:6]
            + result.a_uz @ result.u[6:]
            + float(cfg.cbf_h_gain) * result.h
        )
        intervened = not np.allclose(
            filtered[:6], nominal[:6], atol=float(cfg.intervene_atol), rtol=0.0
        )
        return filtered, AegisDebugInfo(
            enabled=True,
            intervened=intervened,
            constraint_active=result.constraint_active,
            qp_failed=False,
            fallback_to_nominal=False,
            solver_status=result.status,
            h=result.h,
            nominal_slack=result.nominal_slack,
            filtered_slack=filtered_slack,
            z_direction=self._z.copy(),
            extra={
                "alpha_declared": cfg.alpha,
                "max_vel": cfg.max_vel,
                "motion_mode": cfg.motion_mode.value,
            },
        )
