"""Engine-agnostic contact / tilt primitives (stdlib + numpy only).

Downstream: ``EpisodeMetricsBuilder.update_contact`` / ``update_topple``.
Threshold policy (anomaly, topple) lives here as pure functions; measurement
facts live on ``ContactStepResult``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

# Numerical noise floor for impulse-gated checks — not a tunable hyperparameter.
CONTACT_IMPULSE_EPS = 1e-6

_SUPPORTED_FORCE_EQUIV_MODES = ("sim", "control", "impact_window", "blend")


@dataclass(frozen=True)
class ContactStepResult:
    """One control-step contact measurement (engine-agnostic).

    ``in_contact``
        Whether a matching contact manifold exists this step. SAPIEN Scheme B:
        a geometric pair match counts as contact even when measured impulse is
        ~0 (common for kinematic pose-set actors).

    ``force_n``
        MuJoCo: peak |normal force|. SAPIEN: ``max_impulse / force_equiv_dt_s``.

    ``impulse_ns``
        MuJoCo: ``force_n * force_equiv_dt_s``. SAPIEN: peak manifold impulse.

    ``partners``
        Counterpart entity name → match count this step.

    ``force_equiv_dt_s``
        Denominator used to convert impulse ↔ force this step.

    ``contact_without_force``
        ``in_contact and impulse_ns <= CONTACT_IMPULSE_EPS``.

    ``peak_partner``
        Counterpart that produced the peak impulse (or peak |Fn| on MuJoCo).
        ``None`` when no impulse/force sample exceeded the running max (all
        zeros or no matches).
    """

    in_contact: bool
    force_n: float
    impulse_ns: float
    partners: dict[str, int]
    force_equiv_dt_s: float
    contact_without_force: bool = False
    peak_partner: str | None = None


def resolve_force_equiv_dt(
    mode: Any,
    *,
    sim_timestep_s: float | None,
    control_freq_hz: float | None,
    force_report_window_s: float | None,
) -> float:
    """Resolve ``contact.force_equiv_dt`` to a positive interval in seconds.

    Named modes: ``sim``, ``control``, ``impact_window``, ``blend``. A value
    parseable as a positive float is treated as that many seconds. Missing
    inputs required by a named mode raise ``ValueError`` — no silent fallback.
    """
    mode_s = str(mode).strip().lower()
    if mode_s == "sim":
        if sim_timestep_s is None:
            raise ValueError("resolve_force_equiv_dt(mode='sim') requires sim_timestep_s")
        value = float(sim_timestep_s)
    elif mode_s == "control":
        if control_freq_hz is None:
            raise ValueError("resolve_force_equiv_dt(mode='control') requires control_freq_hz")
        hz = float(control_freq_hz)
        if hz <= 0.0:
            raise ValueError(f"control_freq_hz must be > 0, got {control_freq_hz!r}")
        value = 1.0 / hz
    elif mode_s == "impact_window":
        if force_report_window_s is None:
            raise ValueError(
                "resolve_force_equiv_dt(mode='impact_window') requires force_report_window_s"
            )
        value = float(force_report_window_s)
    elif mode_s == "blend":
        if sim_timestep_s is None:
            raise ValueError("resolve_force_equiv_dt(mode='blend') requires sim_timestep_s")
        if control_freq_hz is None:
            raise ValueError("resolve_force_equiv_dt(mode='blend') requires control_freq_hz")
        hz = float(control_freq_hz)
        if hz <= 0.0:
            raise ValueError(f"control_freq_hz must be > 0, got {control_freq_hz!r}")
        t_c = 1.0 / hz
        dt_s = float(sim_timestep_s)
        if dt_s <= 0.0:
            raise ValueError(f"sim_timestep_s must be > 0 for blend, got {dt_s}")
        value = float(np.sqrt(dt_s * t_c))
    else:
        try:
            value = float(mode_s)
        except (TypeError, ValueError) as exc:
            supported = "|".join(_SUPPORTED_FORCE_EQUIV_MODES)
            raise ValueError(
                f"unsupported force_equiv_dt={mode!r}; use {supported} or a positive seconds value"
            ) from exc
    if value <= 0.0:
        raise ValueError(f"force_equiv_dt_s must be positive, got {value}")
    return value


def reduce_impulse_norms(norms: Sequence[float], *, impulse_reduce: str) -> float:
    """Aggregate per-point impulse norms for one contact manifold."""
    if not norms:
        return 0.0
    mode = str(impulse_reduce).strip().lower()
    if mode == "sum":
        return float(sum(norms))
    if mode == "max":
        return float(max(norms))
    raise ValueError(f"impulse_reduce must be 'sum' or 'max', got {impulse_reduce!r}")


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert a wxyz quaternion to a 3x3 rotation matrix.

    A near-zero quaternion (norm ``< 1e-12``) returns the identity.
    """
    q = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = (float(v) for v in (q / norm))
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def tilt_deg_from_reference(*, ref_quat: np.ndarray, current_quat: np.ndarray) -> float:
    """Tilt of the reference upright axis, ignoring pure yaw about world z."""
    ref_rot = quat_to_matrix(ref_quat)
    cur_rot = quat_to_matrix(current_quat)
    local_up = ref_rot.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    current_up = cur_rot @ local_up
    up_norm = float(np.linalg.norm(current_up))
    if up_norm < 1e-12:
        return 0.0
    cos_angle = float(np.clip(current_up[2] / up_norm, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def tilt_deg_from_body_quat(quat: np.ndarray, *, ref_quat: np.ndarray | None = None) -> float:
    """MuJoCo-style tilt from an already-extracted wxyz body quaternion.

    With ``ref_quat``, the relative-quaternion angle ``2*arccos(|rel_w|)``.
    Without, world-Z upright ``arccos(1 - 2*(x²+y²))``. Fetching the quaternion
    from a simulator is the engine adapter's job.
    """
    q = np.asarray(quat, dtype=np.float64)
    if ref_quat is not None:
        rw = float(ref_quat[0])
        rx, ry, rz = -float(ref_quat[1]), -float(ref_quat[2]), -float(ref_quat[3])
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        rel_w = rw * w - rx * x - ry * y - rz * z
        rel_w = float(np.clip(abs(rel_w), 0.0, 1.0))
        return float(np.degrees(2.0 * np.arccos(rel_w)))
    x, y = float(q[1]), float(q[2])
    up_z = float(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))
    return float(np.degrees(np.arccos(up_z)))


def is_toppled(tilt_deg: float, *, threshold_deg: float) -> bool:
    """Whether tilt exceeds the configured topple threshold.

    Uses strict ``>`` so ``tilt_deg == threshold_deg`` is not toppled. In
    particular ``threshold_deg=0`` and ``tilt_deg=0`` (upright) must be False;
    ``>=`` would mark a still object as toppled. ``threshold_deg < 0`` is
    invalid (not a silent clamp).
    """
    if threshold_deg < 0:
        raise ValueError(f"threshold_deg must be >= 0, got {threshold_deg!r}")
    return float(tilt_deg) > float(threshold_deg)


def is_force_anomaly(
    *,
    force_n: float,
    impulse_ns: float,
    force_threshold_n: float,
    impulse_threshold_ns: float,
) -> bool:
    """Threshold policy: either gate exceeding its configured bound."""
    return bool(
        float(impulse_ns) > float(impulse_threshold_ns) or float(force_n) > float(force_threshold_n)
    )
