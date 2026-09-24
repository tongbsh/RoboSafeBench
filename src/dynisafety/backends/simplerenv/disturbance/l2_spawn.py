"""SimplerEnv L2 spawn-frame geometry: edge slots, path solve, mode rotation.

Pure functions only. Callers inject scene queries. This module does not
import LIBERO, RoboTwin, MuJoCo, or SAPIEN.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from dynisafety.backends.simplerenv.disturbance.diversity_clearance import LEGACY_UNVALIDATED
from dynisafety.core.disturbance import arc_lift, lerp

SLOTS: tuple[str, ...] = ("left", "center", "right")
PATH_STYLES: tuple[str, ...] = ("arc", "linear")
MODES: tuple[str, ...] = (
    "left_arc",
    "right_arc",
    "center_arc",
    "left_linear",
    "right_linear",
    "center_linear",
)
LEGACY_MODE: str = "legacy_jitter"

_CENTER_ARC = "center_arc"
_DEGENERATE_N_HAT = np.array([0.0, 1.0], dtype=np.float64)
_NORM_EPS = 1e-12
_ALPHA_DENOM_EPS = 1e-12
_OPPOSITE_SLOT = {"left": "right", "right": "left", "center": "left"}

FixtureTopFn = Callable[[np.ndarray], float]
MakeCandidateFn = Callable[[str], dict[str, Any] | None]
ValidateFn = Callable[[Any, Any], bool]

__all__ = [
    "SLOTS",
    "PATH_STYLES",
    "MODES",
    "LEGACY_MODE",
    "simplerenv_l2_lateral_axis",
    "simplerenv_l2_candidate_start_xy",
    "simplerenv_l2_path_num_samples",
    "simplerenv_l2_solve_linear_start_z",
    "simplerenv_l2_solve_trapezoid",
    "simplerenv_l2_build_path",
    "simplerenv_l2_select_mode",
    "simplerenv_l2_fallback_chain",
    "simplerenv_l2_accept_spawn",
    "simplerenv_l2_path_sha1",
]


def _as_xy(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"{name} must have at least 2 components, got shape {np.shape(value)}")
    return np.array([float(arr[0]), float(arr[1])], dtype=np.float64)


def _as_xyz(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        raise ValueError(f"{name} must be a 3-vector, got shape {np.shape(value)}")
    return np.array([float(arr[0]), float(arr[1]), float(arr[2])], dtype=np.float64)


def _split_mode(mode: str) -> tuple[str, str]:
    slot, sep, style = str(mode).partition("_")
    if not sep:
        return slot, ""
    return slot, style


def _other_style(style: str) -> str:
    if style == "arc":
        return "linear"
    if style == "linear":
        return "arc"
    raise ValueError(f"unknown L2 path style {style!r}; expected one of {PATH_STYLES}")


def simplerenv_l2_path_sha1(path: np.ndarray) -> str:
    arr = np.ascontiguousarray(path, dtype=np.float64)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def simplerenv_l2_lateral_axis(
    table_mins: Any | None,
    table_maxs: Any | None,
    landing_xy: Any,
    p0_xy: Any,
) -> tuple[np.ndarray, bool]:
    """Unit axis along the hit table edge (perpendicular to the outside ray).

    Default start already sits on the table-center → landing ray, past the
    AABB edge. Sliding along ``rot90(that ray)`` keeps the outside distance.
    Missing bounds or a degenerate ray fall back to ``rot90(P0 − landing)``,
    then world ``+Y``.
    """
    land = _as_xy(landing_xy, name="landing_xy")
    p0 = _as_xy(p0_xy, name="p0_xy")
    if table_mins is not None and table_maxs is not None:
        mins = _as_xy(table_mins, name="table_mins")
        maxs = _as_xy(table_maxs, name="table_maxs")
        center = 0.5 * (mins + maxs)
        ray = land - center
        norm = float(np.linalg.norm(ray))
        if np.isfinite(norm) and norm > _NORM_EPS:
            ray = ray / norm
            n_hat = np.array([-ray[1], ray[0]], dtype=np.float64)
            n_norm = float(np.linalg.norm(n_hat))
            if np.isfinite(n_norm) and n_norm > _NORM_EPS:
                return n_hat / n_norm, False
    along = p0 - land
    norm = float(np.linalg.norm(along))
    if (not np.isfinite(norm)) or norm <= _NORM_EPS:
        return _DEGENERATE_N_HAT.copy(), True
    along = along / norm
    n_hat = np.array([-along[1], along[0]], dtype=np.float64)
    n_norm = float(np.linalg.norm(n_hat))
    if (not np.isfinite(n_norm)) or n_norm <= _NORM_EPS:
        return _DEGENERATE_N_HAT.copy(), True
    return n_hat / n_norm, False


def simplerenv_l2_candidate_start_xy(
    p0: Any,
    n_hat: Any,
    slot: str,
    lateral_width_m: float,
) -> np.ndarray:
    if slot not in SLOTS:
        raise ValueError(f"unknown L2 slot {slot!r}; expected one of {SLOTS}")
    origin = _as_xy(p0, name="p0")
    axis = _as_xy(n_hat, name="n_hat")
    width = float(lateral_width_m)
    if slot == "center":
        return origin
    if slot == "left":
        return origin + width * axis
    return origin - width * axis


def simplerenv_l2_path_num_samples(move_start_step: int, move_end_step: int) -> int:
    return max(2, int(move_end_step) - int(move_start_step) + 1)


def simplerenv_l2_solve_linear_start_z(
    start_xy: Any,
    landing: Any,
    fixture_top_fn: FixtureTopFn,
    *,
    clearance_m: float,
    tail_alpha: float,
    max_start_rise_m: float,
    samples: int,
    p0_z_hint: float | None = None,
) -> tuple[float | None, str]:
    start = _as_xy(start_xy, name="start_xy")
    land = _as_xyz(landing, name="landing")
    land_xy = land[:2]
    z_land = float(land[2])
    hint = z_land if p0_z_hint is None else float(p0_z_hint)
    clearance = float(clearance_m)
    n = max(1, int(samples))
    tail = float(tail_alpha)
    end_alpha = 1.0 - tail
    if end_alpha <= 0.0:
        alphas = np.array([0.0], dtype=np.float64)
    else:
        alphas = np.linspace(0.0, end_alpha, n, dtype=np.float64)

    z_start = hint
    for alpha in alphas:
        a = float(alpha)
        denom = 1.0 - a
        if denom <= _ALPHA_DENOM_EPS:
            continue
        xy = (1.0 - a) * start + a * land_xy
        required = (float(fixture_top_fn(xy)) + clearance - a * z_land) / denom
        if np.isfinite(required):
            z_start = max(z_start, required)

    if (not np.isfinite(z_start)) or (z_start - z_land > float(max_start_rise_m)):
        return None, "linear_solve_infeasible"
    return float(z_start), "straight"


def simplerenv_l2_solve_trapezoid(
    start_xy: Any,
    landing: Any,
    fixture_top_fn: FixtureTopFn,
    *,
    clearance_m: float,
    max_start_rise_m: float,
    descent_alpha: float,
    samples: int,
) -> tuple[float | None, str]:
    del descent_alpha
    start = _as_xy(start_xy, name="start_xy")
    land = _as_xyz(landing, name="landing")
    land_xy = land[:2]
    z_land = float(land[2])
    clearance = float(clearance_m)
    n = max(1, int(samples))
    max_top_plus_c = -np.inf
    for alpha in np.linspace(0.0, 1.0, n, dtype=np.float64):
        a = float(alpha)
        xy = (1.0 - a) * start + a * land_xy
        max_top_plus_c = max(max_top_plus_c, float(fixture_top_fn(xy)) + clearance)
    if not np.isfinite(max_top_plus_c):
        max_top_plus_c = z_land
    z_cruise = float(max_top_plus_c)
    if z_cruise > z_land + float(max_start_rise_m):
        return None, "trapezoid_infeasible"
    return z_cruise, "trapezoid"


def simplerenv_l2_build_path(
    start: Any,
    landing: Any,
    path_style: str,
    *,
    n_samples: int,
    arc_height: float,
    cruise_z: float | None = None,
    descent_alpha: float | None = None,
) -> np.ndarray:
    start_xyz = _as_xyz(start, name="start")
    land_xyz = _as_xyz(landing, name="landing")
    n = max(2, int(n_samples))
    alphas = np.linspace(0.0, 1.0, n)
    path = np.empty((n, 3), dtype=np.float64)

    if path_style == "trapezoid":
        if cruise_z is None:
            raise ValueError("cruise_z is required for path_style='trapezoid'")
        if descent_alpha is None:
            raise ValueError("descent_alpha is required for path_style='trapezoid'")
        z_cruise = float(cruise_z)
        descent = float(descent_alpha)
        if descent <= 0.0 or descent > 1.0:
            raise ValueError(f"descent_alpha must be in (0, 1], got {descent}")
        cruise_end = 1.0 - descent
        start_xy = start_xyz[:2]
        land_xy = land_xyz[:2]
        z_land = float(land_xyz[2])
        for i, raw_alpha in enumerate(alphas):
            a = float(raw_alpha)
            if a <= cruise_end:
                t = 0.0 if cruise_end <= _ALPHA_DENOM_EPS else a / cruise_end
                xy = (1.0 - t) * start_xy + t * land_xy
                path[i] = (float(xy[0]), float(xy[1]), z_cruise)
            else:
                t = (a - cruise_end) / descent
                path[i] = (float(land_xy[0]), float(land_xy[1]), z_cruise + t * (z_land - z_cruise))
        return path

    if path_style == "linear":
        height = 0.0
    elif path_style == "arc":
        height = float(arc_height)
    else:
        raise ValueError(
            f"unknown L2 path style {path_style!r}; expected 'arc', 'linear', or 'trapezoid'"
        )
    for i, raw_alpha in enumerate(alphas):
        a = float(raw_alpha)
        pos = lerp(start_xyz, land_xyz, a)
        pos[2] += arc_lift(a, height)
        path[i] = pos
    return path


def simplerenv_l2_select_mode(
    seed: Any,
    task_id: Any,
    episode_index: int,
    *,
    forbidden_slots: Sequence[str],
) -> str:
    banned = {str(slot) for slot in forbidden_slots}
    pool = [mode for mode in MODES if _split_mode(mode)[0] not in banned]
    if not pool:
        return LEGACY_MODE
    key = f"{seed}|{task_id}".encode()
    rotation = int.from_bytes(hashlib.sha1(key).digest()[:4], "big") % len(pool)
    return pool[(int(episode_index) + rotation) % len(pool)]


def simplerenv_l2_fallback_chain(
    requested_mode: str,
    *,
    forbidden_slots: Sequence[str],
) -> tuple[str, ...]:
    if requested_mode == LEGACY_MODE:
        return (LEGACY_MODE,)
    banned = {str(slot) for slot in forbidden_slots}
    ordered: list[str] = []

    def _offer(mode: str) -> None:
        if mode == LEGACY_MODE or mode in ordered:
            return
        slot, _style = _split_mode(mode)
        if slot in banned:
            return
        ordered.append(mode)

    _offer(requested_mode)
    slot, style = _split_mode(requested_mode)
    if slot in SLOTS and style in PATH_STYLES:
        _offer(f"{slot}_{_other_style(style)}")
        _offer(f"{_OPPOSITE_SLOT[slot]}_{style}")
    _offer(_CENTER_ARC)
    for mode in MODES:
        _offer(mode)
    ordered.append(LEGACY_MODE)
    return tuple(ordered)


def _candidate_fields(cand: dict[str, Any] | None) -> dict[str, Any]:
    if cand is None:
        return {
            "start": None,
            "path": None,
            "linear_solve_mode": "na",
            "min_clearance_m": None,
        }
    return {
        "start": cand.get("start"),
        "path": cand.get("path"),
        "linear_solve_mode": cand.get("linear_solve_mode", "na"),
        "min_clearance_m": cand.get("min_clearance_m"),
    }


def _reject_reason_from_candidate(cand: dict[str, Any] | None) -> str | None:
    if cand is None:
        return "linear_solve_infeasible"
    if cand.get("path") is not None:
        return None
    reason = cand.get("infeasible_reason")
    return "linear_solve_infeasible" if reason is None else str(reason)


def simplerenv_l2_accept_spawn(
    requested_mode: str,
    *,
    forbidden_slots: Sequence[str],
    make_candidate: MakeCandidateFn,
    validate: ValidateFn,
) -> dict[str, Any]:
    chain = simplerenv_l2_fallback_chain(requested_mode, forbidden_slots=forbidden_slots)
    tried: list[str] = []
    first_reason: str | None = None

    for mode in chain:
        tried.append(mode)
        if mode == LEGACY_MODE:
            cand = make_candidate(mode)
            fields = _candidate_fields(cand)
            validated = False
            if fields.get("path") is not None:
                validated = bool(validate(fields.get("start"), fields.get("path")))
            if requested_mode == LEGACY_MODE and len(tried) == 1:
                reason = None if validated else LEGACY_UNVALIDATED
            elif validated:
                reason = first_reason or "all_candidates_rejected"
            else:
                reason = LEGACY_UNVALIDATED
            return {
                "requested_mode": requested_mode,
                "accepted_mode": LEGACY_MODE,
                "fallback_reason": reason,
                "fallback_chain": tuple(tried),
                "legacy_validated": validated,
                **fields,
            }
        cand = make_candidate(mode)
        reject = _reject_reason_from_candidate(cand)
        if reject is not None:
            if first_reason is None:
                first_reason = reject
            continue
        if not validate(cand.get("start"), cand.get("path")):
            if first_reason is None:
                first_reason = "candidate_rejected"
            continue
        return {
            "requested_mode": requested_mode,
            "accepted_mode": mode,
            "fallback_reason": None if mode == requested_mode else first_reason,
            "fallback_chain": tuple(tried),
            **_candidate_fields(cand),
        }

    return {
        "requested_mode": requested_mode,
        "accepted_mode": LEGACY_MODE,
        "fallback_reason": "all_candidates_rejected",
        "fallback_chain": tuple(tried),
        "start": None,
        "path": None,
        "linear_solve_mode": "na",
        "min_clearance_m": None,
    }
