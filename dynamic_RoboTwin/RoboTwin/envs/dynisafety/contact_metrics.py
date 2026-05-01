"""Pure helpers for DyniSafety collision impulse/force reporting (no SAPIEN).

Mirrors ``dynamic_SimplerEnv/dynisafety/wrapper.py`` logic for dt modes and
per-manifold point reduction. ``RoboTwinDyniSafetyManager`` delegates here.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def force_equiv_timestep(
    dt_sim: float,
    *,
    collision_force_equiv_dt: str,
    collision_force_report_window_s: float,
    collision_force_control_freq_hz: float,
) -> float:
    """Denominator for reported force = impulse / dt."""
    mode = str(collision_force_equiv_dt or "impact_window").lower()
    dt_s = max(float(dt_sim), 1e-18)
    if mode == "sim":
        return dt_s
    if mode == "control":
        hz = max(float(collision_force_control_freq_hz or 20.0), 1.0)
        return 1.0 / hz
    if mode == "impact_window":
        w = float(collision_force_report_window_s or 0.05)
        return max(w, 1e-6)
    if mode == "blend":
        hz = max(float(collision_force_control_freq_hz or 20.0), 1.0)
        t_c = 1.0 / hz
        return float(np.sqrt(max(dt_s * t_c, 1e-18)))
    hz = max(float(collision_force_control_freq_hz or 20.0), 1.0)
    return 1.0 / hz


def contact_point_impulse_scalar(
    points: Sequence[Any],
    collision_impulse_point_reduce: str,
) -> float:
    """Aggregate impulse norms over contact points for one manifold."""
    norms: list[float] = []
    for p in points:
        imp = getattr(p, "impulse", None)
        if imp is None:
            continue
        norms.append(float(np.linalg.norm(np.asarray(imp, dtype=np.float64).ravel())))
    if not norms:
        return 0.0
    mode = str(collision_impulse_point_reduce or "max").lower()
    if mode == "sum":
        return float(sum(norms))
    return float(max(norms))
