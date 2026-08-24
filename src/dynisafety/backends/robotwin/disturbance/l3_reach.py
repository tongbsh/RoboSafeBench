"""RoboTwin L3 reach-frame geometry: primary-arm slots, intercept depth.

Pure functions only. Does not import LIBERO, SimplerEnv, MuJoCo, or SAPIEN.
left/right are ± lateral_width along ``rot90(primary_eef → opposite)``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from dynisafety.core.disturbance import lerp, minimum_jerk

SLOTS: tuple[str, ...] = ("left", "opposite", "right")
DEPTHS: tuple[str, ...] = ("near", "mid")
PATH_STYLES: tuple[str, ...] = ("straight", "raised", "via")
MODES: tuple[str, ...] = (
    "left_mid",
    "right_mid",
    "opposite_mid",
    "left_near",
    "right_near",
    "opposite_near",
)
LEGACY_MODE: str = "legacy_opposite"

_OPPOSITE_SLOT = {"left": "right", "right": "left", "opposite": "left"}
_NORM_EPS = 1e-12
_ALPHA_DENOM_EPS = 1e-12

MakeCandidateFn = Callable[[str, str], dict[str, Any] | None]
ValidateFn = Callable[[Any, Any], bool]

__all__ = [
    "SLOTS",
    "DEPTHS",
    "PATH_STYLES",
    "MODES",
    "LEGACY_MODE",
    "robotwin_l3_lateral_axis",
    "robotwin_l3_slot_xy",
    "robotwin_l3_resolve_depth",
    "robotwin_l3_path_intercept_target",
    "robotwin_l3_path_num_samples",
    "robotwin_l3_pose_at",
    "robotwin_l3_build_path",
    "robotwin_l3_select_mode",
    "robotwin_l3_fallback_chain",
    "robotwin_l3_accept_reach",
    "robotwin_l3_path_sha1",
    "robotwin_l3_execution_anchors",
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
    slot, sep, depth = str(mode).partition("_")
    if not sep:
        return slot, ""
    return slot, depth


def robotwin_l3_lateral_axis(primary_eef_xy: Any, opposite_xy: Any) -> tuple[np.ndarray, bool]:
    eef = _as_xy(primary_eef_xy, name="primary_eef_xy")
    opp = _as_xy(opposite_xy, name="opposite_xy")
    fwd = opp - eef
    norm = float(np.linalg.norm(fwd))
    if (not np.isfinite(norm)) or norm <= _NORM_EPS:
        return np.array([0.0, 1.0], dtype=np.float64), True
    fwd = fwd / norm
    n_hat = np.array([-fwd[1], fwd[0]], dtype=np.float64)
    n_norm = float(np.linalg.norm(n_hat))
    if (not np.isfinite(n_norm)) or n_norm <= _NORM_EPS:
        return np.array([0.0, 1.0], dtype=np.float64), True
    return n_hat / n_norm, False


def robotwin_l3_slot_xy(
    slot: str,
    *,
    opposite_xy: Any,
    n_hat: Any,
    lateral_width_m: float,
) -> tuple[np.ndarray, bool]:
    if slot not in SLOTS:
        raise ValueError(f"unknown L3 slot {slot!r}; expected one of {SLOTS}")
    p_opp = _as_xy(opposite_xy, name="opposite_xy")
    if slot == "opposite":
        return p_opp.copy(), False
    axis = _as_xy(n_hat, name="n_hat")
    an = float(np.linalg.norm(axis))
    if (not np.isfinite(an)) or an <= _NORM_EPS:
        return p_opp.copy(), True
    axis = axis / an
    width = float(lateral_width_m)
    point = p_opp + width * axis if slot == "left" else p_opp - width * axis
    collapsed = bool(np.linalg.norm(point - p_opp) <= _NORM_EPS)
    return point, collapsed


def robotwin_l3_resolve_depth(
    depth: str,
    *,
    eef_xy: Any,
    task_xy: Any,
    near_alpha: float,
    mid_alpha: float,
    min_eef_clearance: float,
    min_task_clearance: float,
    collapse_m: float,
) -> tuple[float, bool]:
    if depth not in DEPTHS:
        raise ValueError(f"unknown L3 depth {depth!r}; expected one of {DEPTHS}")
    eef = _as_xy(eef_xy, name="eef_xy")
    task = _as_xy(task_xy, name="task_xy")
    dist = float(np.linalg.norm(task - eef))
    usable = dist - float(min_eef_clearance) - float(min_task_clearance)
    if usable < float(collapse_m):
        return float(mid_alpha), True
    if depth == "near":
        return float(near_alpha), False
    return float(mid_alpha), False


def robotwin_l3_path_intercept_target(
    eef_pos: np.ndarray,
    task_obj_pos: np.ndarray,
    *,
    alpha: float,
    min_eef_clearance: float,
    tip_overshoot: float,
    min_task_clearance: float,
    target_z: float | None,
) -> np.ndarray:
    eef = np.asarray(eef_pos, dtype=np.float64).reshape(3)
    task = np.asarray(task_obj_pos, dtype=np.float64).reshape(3)
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f"intercept alpha must be in [0, 1], got {alpha!r}")
    if float(min_eef_clearance) < 0.0:
        raise ValueError(f"min_eef_clearance must be >= 0, got {min_eef_clearance!r}")
    if float(tip_overshoot) < 0.0:
        raise ValueError(f"tip_overshoot must be >= 0, got {tip_overshoot!r}")
    if float(min_task_clearance) < 0.0:
        raise ValueError(f"min_task_clearance must be >= 0, got {min_task_clearance!r}")

    delta = task[:2] - eef[:2]
    dist = float(np.linalg.norm(delta))
    floor = float(min_eef_clearance) + float(tip_overshoot)
    task_cap = float(min_task_clearance)
    if dist < 1e-8:
        xy = eef[:2] + np.array([floor, 0.0], dtype=np.float64)
    else:
        unit = delta / dist
        along = float(alpha) * dist
        hi = dist - task_cap if task_cap > 0.0 else dist
        if hi < floor:
            along = floor if floor > 0.0 else max(hi, 0.0)
        else:
            along = min(max(along, floor), hi)
        xy = eef[:2] + along * unit
    if target_z is None:
        z = (1.0 - float(alpha)) * float(eef[2]) + float(alpha) * float(task[2])
    else:
        z = float(target_z)
    return np.array([xy[0], xy[1], z], dtype=np.float64)


def robotwin_l3_path_num_samples(n_samples: int) -> int:
    return max(2, int(n_samples))


def robotwin_l3_execution_anchors(
    start: Any,
    dwell: Any,
    path_style: str,
    *,
    enter_height: float,
    hover_rise_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map accepted ``path[0]`` to ``(hover, pose_start)`` for the controller.

    ``start`` is the executed waypoint 0. For ``raised`` / ``via`` that
    already includes ``hover_rise_m``; ``pose_start`` strips the rise so
    ``pose_at(..., "via")`` matches the validated path. ``straight`` keeps
    today's hover at ``dwell_z + enter_height``.
    """
    start_xyz = _as_xyz(start, name="start")
    dwell_xyz = _as_xyz(dwell, name="dwell")
    style = str(path_style)
    if style in ("raised", "via"):
        hover = start_xyz.copy()
        pose_start = start_xyz.copy()
        pose_start[2] = float(start_xyz[2]) - float(hover_rise_m)
        return hover, pose_start
    hover = start_xyz.copy()
    hover[2] = float(dwell_xyz[2]) + float(enter_height)
    return hover, start_xyz.copy()


def robotwin_l3_pose_at(
    start: Any,
    dwell: Any,
    alpha: float,
    path_style: str,
    *,
    hover_rise_m: float = 0.0,
    via_descent_alpha: float = 0.20,
) -> np.ndarray:
    start_xyz = _as_xyz(start, name="start")
    dwell_xyz = _as_xyz(dwell, name="dwell")
    a = float(np.clip(alpha, 0.0, 1.0))
    if path_style == "straight":
        return lerp(start_xyz, dwell_xyz, minimum_jerk(a))
    if path_style == "raised":
        lifted = start_xyz.copy()
        lifted[2] += float(hover_rise_m)
        return lerp(lifted, dwell_xyz, minimum_jerk(a))
    if path_style != "via":
        raise ValueError(f"unknown L3 path style {path_style!r}; expected one of {PATH_STYLES}")
    descent = float(via_descent_alpha)
    if descent <= 0.0 or descent > 1.0:
        raise ValueError(f"via_descent_alpha must be in (0, 1], got {descent}")
    cruise_end = 1.0 - descent
    cruise_z = float(start_xyz[2]) + float(hover_rise_m)
    start_xy = start_xyz[:2]
    dwell_xy = dwell_xyz[:2]
    z_dwell = float(dwell_xyz[2])
    if a <= cruise_end:
        t = 0.0 if cruise_end <= _ALPHA_DENOM_EPS else a / cruise_end
        xy = lerp(start_xy, dwell_xy, minimum_jerk(t))
        return np.array([float(xy[0]), float(xy[1]), cruise_z], dtype=np.float64)
    t = (a - cruise_end) / descent
    z = cruise_z + minimum_jerk(t) * (z_dwell - cruise_z)
    return np.array([float(dwell_xy[0]), float(dwell_xy[1]), float(z)], dtype=np.float64)


def robotwin_l3_build_path(
    start: Any,
    dwell: Any,
    path_style: str,
    *,
    n_samples: int,
    hover_rise_m: float = 0.0,
    via_descent_alpha: float = 0.20,
) -> np.ndarray:
    n = max(2, int(n_samples))
    alphas = np.linspace(0.0, 1.0, n)
    path = np.empty((n, 3), dtype=np.float64)
    for i, raw_alpha in enumerate(alphas):
        path[i] = robotwin_l3_pose_at(
            start,
            dwell,
            float(raw_alpha),
            path_style,
            hover_rise_m=hover_rise_m,
            via_descent_alpha=via_descent_alpha,
        )
    return path


def robotwin_l3_path_sha1(path: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(path, dtype=np.float64).tobytes()).hexdigest()


def _mode_allowed(
    mode: str,
    *,
    forbidden_slots: Sequence[str],
    forbidden_depths: Sequence[str],
) -> bool:
    if mode == LEGACY_MODE:
        return True
    slot, depth = _split_mode(mode)
    if slot in {str(s) for s in forbidden_slots}:
        return False
    if depth in {str(d) for d in forbidden_depths}:
        return False
    return True


def robotwin_l3_select_mode(
    seed: Any,
    task_id: Any,
    episode_index: int,
    *,
    forbidden_slots: Sequence[str],
    forbidden_depths: Sequence[str],
) -> str:
    banned_slots = {str(slot) for slot in forbidden_slots}
    banned_depths = {str(depth) for depth in forbidden_depths}
    pool = [
        mode
        for mode in MODES
        if _split_mode(mode)[0] not in banned_slots and _split_mode(mode)[1] not in banned_depths
    ]
    if not pool:
        return LEGACY_MODE
    key = f"{seed}|{task_id}".encode()
    rotation = int.from_bytes(hashlib.sha1(key).digest()[:4], "big") % len(pool)
    return pool[(int(episode_index) + rotation) % len(pool)]


def robotwin_l3_fallback_chain(
    requested_mode: str,
    *,
    forbidden_slots: Sequence[str],
    forbidden_depths: Sequence[str],
) -> tuple[tuple[str, str | None], ...]:
    if requested_mode == LEGACY_MODE:
        return ((LEGACY_MODE, None),)
    ordered: list[tuple[str, str | None]] = []

    def _offer(mode: str, style: str | None) -> None:
        if (mode, style) in ordered:
            return
        if not _mode_allowed(
            mode, forbidden_slots=forbidden_slots, forbidden_depths=forbidden_depths
        ):
            return
        ordered.append((mode, style))

    _offer(requested_mode, "straight")
    slot, depth = _split_mode(requested_mode)
    if slot in SLOTS and depth in DEPTHS:
        _offer(requested_mode, "raised")
        _offer(requested_mode, "via")
        _offer(f"{_OPPOSITE_SLOT[slot]}_{depth}", "straight")
    _offer("opposite_mid", "straight")
    ordered.append((LEGACY_MODE, None))
    return tuple(ordered)


def _candidate_fields(cand: dict[str, Any] | None) -> dict[str, Any]:
    if cand is None:
        return {
            "start": None,
            "dwell": None,
            "path": None,
            "path_style": None,
            "min_clearance_m": None,
            "depth_collapsed": False,
            "slot_collapsed": False,
        }
    return {
        "start": cand.get("start"),
        "dwell": cand.get("dwell"),
        "path": cand.get("path"),
        "path_style": cand.get("path_style"),
        "min_clearance_m": cand.get("min_clearance_m"),
        "depth_collapsed": bool(cand.get("depth_collapsed", False)),
        "slot_collapsed": bool(cand.get("slot_collapsed", False)),
    }


def _reject_reason_from_candidate(cand: dict[str, Any] | None) -> str | None:
    if cand is None:
        return "candidate_rejected"
    if cand.get("path") is not None:
        return None
    reason = cand.get("infeasible_reason")
    return "candidate_rejected" if reason is None else str(reason)


def robotwin_l3_accept_reach(
    requested_mode: str,
    *,
    forbidden_slots: Sequence[str],
    forbidden_depths: Sequence[str],
    make_candidate: MakeCandidateFn,
    validate: ValidateFn,
) -> dict[str, Any]:
    chain = robotwin_l3_fallback_chain(
        requested_mode,
        forbidden_slots=forbidden_slots,
        forbidden_depths=forbidden_depths,
    )
    tried: list[str] = []
    first_reason: str | None = None

    for mode, style in chain:
        label = mode if style is None else f"{mode}:{style}"
        tried.append(label)
        if mode == LEGACY_MODE:
            reason = None if requested_mode == LEGACY_MODE and len(tried) == 1 else "all_candidates_rejected"
            return {
                "requested_mode": requested_mode,
                "accepted_mode": LEGACY_MODE,
                "path_style": None,
                "fallback_reason": reason,
                "fallback_chain": tuple(tried),
                **_candidate_fields(make_candidate(mode, "straight")),
            }
        use_style = "straight" if style is None else style
        cand = make_candidate(mode, use_style)
        reject = _reject_reason_from_candidate(cand)
        if reject is not None:
            if first_reason is None:
                first_reason = reject
            continue
        if not validate(cand.get("start"), cand.get("path")):
            if first_reason is None:
                first_reason = "candidate_rejected"
            continue
        accepted_style = str(cand.get("path_style") or use_style)
        return {
            "requested_mode": requested_mode,
            "accepted_mode": mode,
            "path_style": accepted_style,
            "fallback_reason": None
            if (mode == requested_mode and use_style == "straight")
            else first_reason,
            "fallback_chain": tuple(tried),
            **_candidate_fields(cand),
        }

    return {
        "requested_mode": requested_mode,
        "accepted_mode": LEGACY_MODE,
        "path_style": None,
        "fallback_reason": "all_candidates_rejected",
        "fallback_chain": tuple(tried),
        "start": None,
        "dwell": None,
        "path": None,
        "min_clearance_m": None,
        "depth_collapsed": False,
        "slot_collapsed": False,
    }
