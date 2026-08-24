"""LIBERO L3 reach-frame geometry: far-rim slots, intercept depth, mode rotation.

Pure functions only. Callers inject scene queries (``make_candidate``,
``validate``); this module never imports MuJoCo, LIBERO, or robosuite.
Slot / depth labels are the cross-backend contract; the embeddings here
are LIBERO-only (far-rim of ``start_xy_half``, ``mid`` = today's α).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from dynisafety.backends.libero.disturbance.l2_spawn import libero_l2_lateral_axis
from dynisafety.core.disturbance import lerp, minimum_jerk

SLOTS: tuple[str, ...] = ("left", "opposite", "right")
DEPTHS: tuple[str, ...] = ("near", "mid")
PATH_STYLES: tuple[str, ...] = ("straight", "raised", "via")
# Adjacent episodes alternate left/right before visiting opposite, then
# the near depths. Round-robin over this tuple is balanced when N % 6 == 0.
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
    "libero_l3_lateral_axis",
    "libero_l3_clip_xy",
    "libero_l3_opposite_xy",
    "libero_l3_far_edge",
    "libero_l3_slot_xy",
    "libero_l3_resolve_depth",
    "libero_l3_path_num_samples",
    "libero_l3_pose_at",
    "libero_l3_build_path",
    "libero_l3_select_mode",
    "libero_l3_fallback_chain",
    "libero_l3_accept_reach",
    "libero_l3_path_sha1",
]


def libero_l3_lateral_axis(
    robot_base_xy: Any,
    table_center_xy: Any,
) -> tuple[np.ndarray, bool]:
    """Robot-anchored left/right axis — same construction as L2."""
    return libero_l2_lateral_axis(robot_base_xy, table_center_xy)


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


def libero_l3_clip_xy(xy: Any, table_center_xy: Any, start_xy_half: Any) -> np.ndarray:
    """Axis-aligned clip of ``xy`` into ``table_center ± start_xy_half``."""
    point = _as_xy(xy, name="xy")
    center = _as_xy(table_center_xy, name="table_center_xy")
    half = _as_xy(start_xy_half, name="start_xy_half")
    if half[0] <= 0.0 or half[1] <= 0.0:
        raise ValueError(f"start_xy_half must be > 0, got {half.tolist()!r}")
    return np.array(
        [
            float(np.clip(point[0], center[0] - half[0], center[0] + half[0])),
            float(np.clip(point[1], center[1] - half[1], center[1] + half[1])),
        ],
        dtype=np.float64,
    )


def libero_l3_opposite_xy(
    robot_base_xy: Any,
    table_center_xy: Any,
    start_xy_half: Any,
) -> np.ndarray:
    """Today's ``start_mode=2`` XY: reflect the base through the table, then clip."""
    base = _as_xy(robot_base_xy, name="robot_base_xy")
    table = _as_xy(table_center_xy, name="table_center_xy")
    reflected = 2.0 * table - base
    return libero_l3_clip_xy(reflected, table, start_xy_half)


def libero_l3_far_edge(
    table_center_xy: Any,
    start_xy_half: Any,
    fwd: Any,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Far edge of the clip AABB (outward normal most aligned with ``+fwd``).

    Returns ``(origin, edge_dir, half_length)`` where ``origin`` is the
    edge midpoint and ``edge_dir`` is a unit vector along the edge
    (right-handed: ``rot90(outward)`` so it matches ``n_hat`` when
    ``fwd`` aligns with a box axis).
    """
    center = _as_xy(table_center_xy, name="table_center_xy")
    half = _as_xy(start_xy_half, name="start_xy_half")
    forward = _as_xy(fwd, name="fwd")
    fn = float(np.linalg.norm(forward))
    if (not np.isfinite(fn)) or fn <= _NORM_EPS:
        forward = np.array([1.0, 0.0], dtype=np.float64)
    else:
        forward = forward / fn
    hx, hy = float(half[0]), float(half[1])
    # (outward, midpoint, half_length, edge_dir = rot90(outward))
    edges = (
        (
            np.array([1.0, 0.0], dtype=np.float64),
            np.array([center[0] + hx, center[1]], dtype=np.float64),
            hy,
            np.array([0.0, 1.0], dtype=np.float64),
        ),
        (
            np.array([-1.0, 0.0], dtype=np.float64),
            np.array([center[0] - hx, center[1]], dtype=np.float64),
            hy,
            np.array([0.0, -1.0], dtype=np.float64),
        ),
        (
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([center[0], center[1] + hy], dtype=np.float64),
            hx,
            np.array([-1.0, 0.0], dtype=np.float64),
        ),
        (
            np.array([0.0, -1.0], dtype=np.float64),
            np.array([center[0], center[1] - hy], dtype=np.float64),
            hx,
            np.array([1.0, 0.0], dtype=np.float64),
        ),
    )
    outward, origin, half_len, edge_dir = max(edges, key=lambda item: float(item[0] @ forward))
    del outward
    return origin.copy(), edge_dir.copy(), float(half_len)


def _project_onto_edge(
    xy: np.ndarray,
    origin: np.ndarray,
    edge_dir: np.ndarray,
    half_len: float,
) -> np.ndarray:
    t = float((xy - origin) @ edge_dir)
    t = float(np.clip(t, -half_len, half_len))
    return origin + t * edge_dir


def libero_l3_slot_xy(
    slot: str,
    *,
    opposite_xy: Any,
    table_center_xy: Any,
    start_xy_half: Any,
    n_hat: Any,
    lateral_width_m: float,
) -> tuple[np.ndarray, bool]:
    """LIBERO far-rim slot around today's opposite entry.

    ``opposite`` is today's clipped reflection (not re-projected).
    ``left`` / ``right`` slide along the far edge by ``lateral_width_m``
    from the projection of ``opposite`` onto that edge, then clamp to
    the edge segment. ``collapsed`` is true when the requested offset
    was clipped back onto ``opposite``.
    """
    if slot not in SLOTS:
        raise ValueError(f"unknown L3 slot {slot!r}; expected one of {SLOTS}")
    p_opp = _as_xy(opposite_xy, name="opposite_xy")
    if slot == "opposite":
        return p_opp.copy(), False

    table = _as_xy(table_center_xy, name="table_center_xy")
    # Reconstruct fwd as rot-90^{-1}(n_hat) = (n_y, -n_x) so the far
    # edge is the one opposite the robot (n_hat = rot90(fwd)).
    axis = _as_xy(n_hat, name="n_hat")
    an = float(np.linalg.norm(axis))
    if (not np.isfinite(an)) or an <= _NORM_EPS:
        axis = np.array([0.0, 1.0], dtype=np.float64)
    else:
        axis = axis / an
    fwd = np.array([axis[1], -axis[0]], dtype=np.float64)
    origin, edge_dir, half_len = libero_l3_far_edge(table, start_xy_half, fwd)
    # Prefer sliding along +n_hat when the edge is parallel to it.
    if float(edge_dir @ axis) < 0.0:
        edge_dir = -edge_dir
    t0 = float((p_opp - origin) @ edge_dir)
    width = float(lateral_width_m)
    delta = width if slot == "left" else -width
    t = float(np.clip(t0 + delta, -half_len, half_len))
    point = origin + t * edge_dir
    collapsed = bool(np.linalg.norm(point - p_opp) <= _NORM_EPS)
    return point, collapsed


def libero_l3_resolve_depth(
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
    """Return ``(alpha, collapsed)``. Short corridors force ``mid``."""
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


def libero_l3_path_num_samples(approach_steps: int) -> int:
    """Waypoint count shared by validation and min-jerk execution."""
    return max(2, int(approach_steps) + 1)


def libero_l3_pose_at(
    start: Any,
    dwell: Any,
    alpha: float,
    path_style: str,
    *,
    hover_rise_m: float = 0.0,
    via_descent_alpha: float = 0.20,
) -> np.ndarray:
    """Single-α pose matching ``SharedReachingController._approach_pos_at``.

    * ``straight``: min-jerk lerp ``start → dwell``.
    * ``raised``: same lerp after lifting ``start.z`` by ``hover_rise_m``.
    * ``via``: XY min-jerk to ``dwell.xy`` at cruise height
      ``start.z + hover_rise_m``; last ``via_descent_alpha`` drops Z
      onto ``dwell.z`` while XY stays at the dwell.
    """
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
        raise ValueError(
            f"unknown L3 path style {path_style!r}; expected one of {PATH_STYLES}"
        )
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


def libero_l3_build_path(
    start: Any,
    dwell: Any,
    path_style: str,
    *,
    n_samples: int,
    hover_rise_m: float = 0.0,
    via_descent_alpha: float = 0.20,
) -> np.ndarray:
    """Build a ``(n_samples, 3)`` waypoint array from :func:`libero_l3_pose_at`."""
    n = max(2, int(n_samples))
    alphas = np.linspace(0.0, 1.0, n)
    path = np.empty((n, 3), dtype=np.float64)
    for i, raw_alpha in enumerate(alphas):
        path[i] = libero_l3_pose_at(
            start,
            dwell,
            float(raw_alpha),
            path_style,
            hover_rise_m=hover_rise_m,
            via_descent_alpha=via_descent_alpha,
        )
    return path


def libero_l3_path_sha1(path: np.ndarray) -> str:
    arr = np.ascontiguousarray(path, dtype=np.float64)
    return hashlib.sha1(arr.tobytes()).hexdigest()


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


def libero_l3_select_mode(
    seed: Any,
    task_id: Any,
    episode_index: int,
    *,
    forbidden_slots: Sequence[str],
    forbidden_depths: Sequence[str],
) -> str:
    """Round-robin a reach mode, stable across processes.

    Rotation is ``sha1(f"{seed}|{task_id}")[:4]`` as a big-endian uint32
    modulo ``len(pool)`` — not builtin ``hash()``. Then
    ``pool[(episode_index + rotation) % len(pool)]``.
    """
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


def libero_l3_fallback_chain(
    requested_mode: str,
    *,
    forbidden_slots: Sequence[str],
    forbidden_depths: Sequence[str],
) -> tuple[tuple[str, str | None], ...]:
    """Ordered ``(mode, path_style)`` pairs ending in ``(legacy, None)``.

    Slot / depth are the scientific cell; path style is only a means:

    1. requested + straight
    2. same cell + raised
    3. same cell + via
    4. opposite slot, same depth + straight (``opposite`` tries ``left``)
    5. ``opposite_mid`` + straight
    6. ``legacy_opposite``
    """
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
    if reason is None:
        return "candidate_rejected"
    return str(reason)


def libero_l3_accept_reach(
    requested_mode: str,
    *,
    forbidden_slots: Sequence[str],
    forbidden_depths: Sequence[str],
    make_candidate: MakeCandidateFn,
    validate: ValidateFn,
) -> dict[str, Any]:
    """Walk the fallback chain until a candidate is accepted.

    ``make_candidate(mode, path_style)`` — ``path_style`` is ``\"straight\"``
    for ``legacy_opposite``. ``validate(start, path)`` is the injected sweep.
    ``LEGACY_MODE`` is accepted without calling ``validate``.
    """
    chain = libero_l3_fallback_chain(
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
            if requested_mode == LEGACY_MODE and len(tried) == 1:
                reason: str | None = None
            else:
                reason = "all_candidates_rejected"
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
            "fallback_reason": None if (mode == requested_mode and use_style == "straight") else first_reason,
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
