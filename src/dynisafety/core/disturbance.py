"""Engine-agnostic disturbance primitives (stdlib + numpy only).

Motion / interpolation helpers, wxyz quaternion products, the union L3
``Phase`` enum, YAML-required config guards, and layered controller protocols.

Call-stack split (not unified here): LIBERO / RoboCasa runners invoke
``step(t, obs)``; SimplerEnv / RoboTwin wrappers invoke
``update_before_step``. Protocols pin the shared surface only — they do
not merge those two call shapes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np

from dynisafety.config.schema import MISSING

# 15/8 = peak of d/dτ (10τ³ − 15τ⁴ + 6τ⁵). Shared by LIBERO and SimplerEnv.
MIN_JERK_PEAK_COEFF = 15.0 / 8.0


def minimum_jerk(tau: float) -> float:
    """Fifth-order minimum-jerk scalar ``10τ³ − 15τ⁴ + 6τ⁵`` (τ clipped to [0, 1])."""
    tau = float(np.clip(tau, 0.0, 1.0))
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def compute_approach_steps(distance: float, peak_velocity: float, *, control_freq: int) -> int:
    """Control steps for a min-jerk segment of length ``distance``.

    ``duration = MIN_JERK_PEAK_COEFF * distance / max(peak_velocity, 0.01)``,
    then ``max(ceil(duration * control_freq), 2)``.

    The ``0.01`` floor is the existing four-backend behaviour (not a YAML
    hyperparameter). ``control_freq <= 0`` raises.
    """
    if control_freq <= 0:
        raise ValueError(f"control_freq must be positive, got {control_freq!r}")
    duration_sec = MIN_JERK_PEAK_COEFF * distance / max(peak_velocity, 0.01)
    return max(int(math.ceil(duration_sec * control_freq)), 2)


def arc_lift(alpha: float, height: float) -> float:
    """Parabolic lift ``4 * height * α * (1 − α)`` (α clipped to [0, 1]).

    Midpoint ``α = 0.5`` returns ``height``. Not used by RoboCasa (ground
    navigation is a straight floor lerp; no parabola).
    """
    a = float(np.clip(alpha, 0.0, 1.0))
    return 4.0 * float(height) * a * (1.0 - a)


def lerp(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    """Linear interpolation of same-shaped arrays (α clipped to [0, 1])."""
    a = float(np.clip(alpha, 0.0, 1.0))
    p0 = np.asarray(start, dtype=np.float64)
    p1 = np.asarray(end, dtype=np.float64)
    return p0 + a * (p1 - p0)


def perpendicular_2d(direction: np.ndarray, *, flip: bool = False) -> np.ndarray:
    """Unit left normal about +Z: ``(-uy, ux)``. ``flip=True`` negates.

    Accepts a 2- or 3-vector and returns the same dimension, so RoboCasa's
    planar heading and the 3-vector forms in SimplerEnv / RoboTwin can share
    one implementation. Only the XY projection is used; a 3-vector's Z is
    ignored and the result's Z is 0. A zero XY projection raises
    ``ValueError``.
    """
    vec = np.asarray(direction, dtype=np.float64).reshape(-1)
    if vec.size not in (2, 3):
        raise ValueError(f"direction must have 2 or 3 elements, got shape {np.shape(direction)}")
    norm = float(np.linalg.norm(vec[:2]))
    if norm <= 1e-12:
        raise ValueError("direction must have a non-zero XY projection")
    ux, uy = float(vec[0]) / norm, float(vec[1]) / norm
    normal = np.array([-uy, ux], dtype=np.float64)
    if flip:
        normal = -normal
    if vec.size == 3:
        return np.array([normal[0], normal[1], 0.0], dtype=np.float64)
    return normal


def interpolate_toward(origin: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    """Full-dimension α interpolation from ``origin`` toward ``target``.

    Z-axis policy is **not** applied here. LIBERO uses midpoint-Z or an
    explicit ``target_z``; SimplerEnv pins Z to the object; RoboTwin
    interpolates Z with the rest. Backends that need a special Z must
    overwrite that component after calling this function.
    """
    return lerp(origin, target, alpha)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions: ``q1 * q2``.

    First rotate by ``q2`` in the local frame, then by ``q1`` in the world
    frame.
    """
    w1, x1, y1, z1 = (float(v) for v in q1)
    w2, x2, y2, z2 = (float(v) for v in q2)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_about_axis(axis: str, angle_deg: float) -> np.ndarray:
    """wxyz quaternion rotating ``angle_deg`` about world axis ``x`` / ``y`` / ``z``."""
    rad = math.radians(angle_deg)
    c, s = math.cos(rad / 2), math.sin(rad / 2)
    if axis == "x":
        return np.array([c, s, 0.0, 0.0], dtype=np.float64)
    if axis == "y":
        return np.array([c, 0.0, s, 0.0], dtype=np.float64)
    if axis == "z":
        return np.array([c, 0.0, 0.0, s], dtype=np.float64)
    raise ValueError(f"Unknown axis: {axis}")


def arm_quat_from_yaw(yaw: float) -> np.ndarray:
    """Hand / palm orientation used by LIBERO and SimplerEnv L3 (wxyz).

    Applies x90 → x180 → z180 → z180, then yaw about z. The two z180
    multiplies cancel algebraically; they are an intentional historical
    convention shared verbatim by both backends — do not simplify.
    ``yaw`` is in radians.
    """
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q = quat_mul(quat_about_axis("x", 90), q)
    q = quat_mul(quat_about_axis("x", 180), q)
    q = quat_mul(quat_about_axis("z", 180), q)
    q = quat_mul(quat_about_axis("z", 180), q)
    q = quat_mul(quat_about_axis("z", math.degrees(yaw)), q)
    return q / np.linalg.norm(q)


class Phase(str, Enum):
    """Union of L3 phase names across the four backends.

    Each backend uses a subset. A member being listed here does not mean
    that backend implements the phase.
    """

    HIDDEN = "hidden"
    ENTER = "enter"
    APPROACH = "approach"
    DWELL = "dwell"
    WITHDRAW = "withdraw"
    POST_HIT_HOLD = "post_hit_hold"
    FLINCH = "flinch"
    FLINCH_EXIT = "flinch_exit"
    EXIT = "exit"
    WAIT = "wait"
    DONE = "done"


def require_configured(value: Any, name: str) -> Any:
    """Return ``value`` unless it is the schema ``MISSING`` sentinel or ``None``.

    Identity checks only (``is MISSING`` / ``is None``). Do not use
    truthiness — ``MISSING.__bool__`` is False, so ``if not value`` would
    also reject ``0``, ``""``, and ``[]``.
    """
    if value is MISSING or value is None:
        raise ValueError(f"{name} is MISSING; set it in YAML (iron law 1)")
    return value


def require_map_key(block: Mapping[str, Any], key: str, *, where: str) -> Any:
    """Return ``block[key]`` or raise if the key is absent (no Python default)."""
    if key not in block:
        raise ValueError(f"{where}.{key} missing from YAML (no Python default)")
    return block[key]


@runtime_checkable
class OnsetReporting(Protocol):
    @property
    def disturbance_onset_step(self) -> int | None: ...


@runtime_checkable
class SteppableDisturbance(OnsetReporting, Protocol):
    def update_before_step(self, step: int) -> None: ...


@runtime_checkable
class PhysicsReleasable(Protocol):
    """Kinematic actor that must be handed back to the physics engine."""

    def needs_physics_release(self) -> bool: ...

    def on_released_to_physics(self) -> None: ...
