"""LIBERO L2 spawn-frame geometry: lateral slots, path solve, mode rotation.

Pure functions only. Callers inject scene queries (``fixture_top_fn``,
``make_candidate``, ``validate``); this module never imports MuJoCo,
LIBERO, or robosuite and never touches a simulator.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from dynisafety.core.disturbance import arc_lift, lerp

SLOTS: tuple[str, ...] = ("left", "center", "right")
PATH_STYLES: tuple[str, ...] = ("arc", "linear")
# Adjacent episodes alternate left/right before visiting center, then the
# other path style. Round-robin over this tuple is balanced when N % 6 == 0.
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
    "libero_l2_lateral_axis",
    "libero_l2_candidate_start_xy",
    "libero_l2_path_num_samples",
    "libero_l2_solve_linear_start_z",
    "libero_l2_solve_trapezoid",
    "libero_l2_build_path",
    "libero_l2_select_mode",
    "libero_l2_fallback_chain",
    "libero_l2_accept_spawn",
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
    """Map a non-buildable ``make_candidate`` result to a fallback reason.

    * ``None`` → ``"linear_solve_infeasible"`` (legacy / default).
    * ``{"infeasible_reason": "..."}`` (no ``path``) → that reason, so
      Track B can distinguish linear vs trapezoid failure.
    * A dict with ``path`` is a real candidate (returns ``None`` here).
    """
    if cand is None:
        return "linear_solve_infeasible"
    if cand.get("path") is not None:
        return None
    reason = cand.get("infeasible_reason")
    if reason is None:
        return "linear_solve_infeasible"
    return str(reason)


def libero_l2_lateral_axis(
    robot_base_xy: Any,
    table_center_xy: Any,
) -> tuple[np.ndarray, bool]:
    """Robot-anchored left/right axis in the world XY plane.

    ``fwd = normalize(table_center_xy - robot_base_xy)`` and
    ``n_hat = rot90(fwd)`` (counter-clockwise: ``(x, y) -> (-y, x)``).

    When the two points coincide (or the difference is non-finite) this
    returns world ``+Y`` ``[0.0, 1.0]`` and ``degenerate=True``. Never
    returns NaN.

    Returns:
        ``(n_hat, degenerate)`` with ``n_hat`` a unit ``(2,)`` float64 vector.
    """
    base = _as_xy(robot_base_xy, name="robot_base_xy")
    table = _as_xy(table_center_xy, name="table_center_xy")
    fwd = table - base
    norm = float(np.linalg.norm(fwd))
    if (not np.isfinite(norm)) or norm <= _NORM_EPS:
        return _DEGENERATE_N_HAT.copy(), True
    fwd = fwd / norm
    n_hat = np.array([-fwd[1], fwd[0]], dtype=np.float64)
    n_norm = float(np.linalg.norm(n_hat))
    if (not np.isfinite(n_norm)) or n_norm <= _NORM_EPS:
        return _DEGENERATE_N_HAT.copy(), True
    return n_hat / n_norm, False


def libero_l2_candidate_start_xy(
    p0: Any,
    n_hat: Any,
    slot: str,
    lateral_width_m: float,
) -> np.ndarray:
    """Lateral slot around ``P0`` along ``n_hat``.

    Only XY is moved. ``center`` returns ``P0.xy`` unchanged;
    ``left = P0 + w * n_hat``; ``right = P0 - w * n_hat``. Z is left
    to the caller or the Z solvers.
    """
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


def libero_l2_path_num_samples(move_start_step: int, move_end_step: int) -> int:
    """Waypoint count shared by validation and execution.

    Matches ``l2.py`` ``_build_arc_path``:
    ``max(2, move_end_step - move_start_step + 1)``.
    """
    return max(2, int(move_end_step) - int(move_start_step) + 1)


def libero_l2_solve_linear_start_z(
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
    """Invert a straight-line Z profile so the path clears fixtures.

    Path model: ``xy(α)`` lerps ``start_xy → landing.xy`` and
    ``z(α) = (1-α) * z_start + α * z_land``. Constraints are enforced on
    ``α ∈ [0, 1 - tail_alpha]`` (uniform ``samples`` points). The landing
    tail is excluded so the solver never divides by ``1-α → 0`` and so
    the shallow descent onto the table is not forced to clear furniture.

    Closed form, per sample (``α < 1``)::

        z_start >= (fixture_top_fn(xy(α)) + clearance_m - α * z_land) / (1 - α)
        z_start  = max(p0_z_hint, max over α of the right-hand side)

    ``p0_z_hint`` is the authored ``P0.z`` (typically ``landing.z +
    start_offset.z``). When omitted, ``landing[2]`` is used — correct
    iff ``start_offset.z == 0``.

    Returns:
        ``(z_start, "straight")`` or ``(None, "linear_solve_infeasible")``
        when ``z_start - z_land > max_start_rise_m``.
    """
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
        top = float(fixture_top_fn(xy))
        required = (top + clearance - a * z_land) / denom
        if np.isfinite(required):
            z_start = max(z_start, required)

    if (not np.isfinite(z_start)) or (z_start - z_land > float(max_start_rise_m)):
        return None, "linear_solve_infeasible"
    return float(z_start), "straight"


def libero_l2_solve_trapezoid(
    start_xy: Any,
    landing: Any,
    fixture_top_fn: FixtureTopFn,
    *,
    clearance_m: float,
    max_start_rise_m: float,
    descent_alpha: float,
    samples: int,
) -> tuple[float | None, str]:
    """Trapezoid downgrade: horizontal cruise then a vertical drop.

    Sample ``top(xy(α)) + clearance_m`` uniformly on the full XY segment
    ``start_xy → landing.xy``. The cruise height that makes clearance
    hold by construction is ``z_cruise = max_α (top + clearance)``.

    The rise cap is a hard feasibility cut, not a silent clamp: if
    ``z_cruise > z_land + max_start_rise_m`` this returns
    ``(None, "trapezoid_infeasible")`` so the caller can walk the
    fallback chain. Otherwise returns ``(z_cruise, "trapezoid")``.

    ``descent_alpha`` is accepted for a stable caller signature; cruise
    height does not depend on it. The descent itself is applied by
    ``libero_l2_build_path(..., path_style="trapezoid")``.
    """
    del descent_alpha  # reserved: consumed by libero_l2_build_path
    start = _as_xy(start_xy, name="start_xy")
    land = _as_xyz(landing, name="landing")
    land_xy = land[:2]
    z_land = float(land[2])
    clearance = float(clearance_m)
    n = max(1, int(samples))
    alphas = np.linspace(0.0, 1.0, n, dtype=np.float64)
    max_top_plus_c = -np.inf
    for alpha in alphas:
        a = float(alpha)
        xy = (1.0 - a) * start + a * land_xy
        max_top_plus_c = max(max_top_plus_c, float(fixture_top_fn(xy)) + clearance)
    if not np.isfinite(max_top_plus_c):
        max_top_plus_c = z_land
    z_cruise = float(max_top_plus_c)
    if z_cruise > z_land + float(max_start_rise_m):
        return None, "trapezoid_infeasible"
    return z_cruise, "trapezoid"


def libero_l2_build_path(
    start: Any,
    landing: Any,
    path_style: str,
    *,
    n_samples: int,
    arc_height: float,
    cruise_z: float | None = None,
    descent_alpha: float | None = None,
) -> np.ndarray:
    """Build a ``(n_samples, 3)`` waypoint array.

    ``n_samples`` must be the same count used at execution
    (``libero_l2_path_num_samples``); this function does not look up
    move-window steps. Internally ``n = max(2, n_samples)``.

    * ``"arc"``: ``pos = start + α*(landing-start)`` then
      ``pos[2] += 4.0 * arc_height * α * (1-α)``. Bitwise-identical
      loop to ``l2.py`` ``_build_arc_path``.
    * ``"linear"``: the same lerp with no parabolic lift
      (equivalent to ``arc_height=0``).
    * ``"trapezoid"``: for ``α <= 1-descent_alpha``, XY lerps
      ``start.xy → landing.xy`` while Z stays at ``cruise_z``; for
      ``α > 1-descent_alpha``, XY stays at ``landing.xy`` and Z lerps
      ``cruise_z → landing[2]``. ``start[2]`` is unused.

    Returns:
        ``float64`` array of shape ``(n, 3)``.
    """
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


def libero_l2_select_mode(
    seed: Any,
    task_id: Any,
    episode_index: int,
    *,
    forbidden_slots: Sequence[str],
) -> str:
    """Round-robin a diversity mode, stable across processes.

    ``pool = [m for m in MODES if slot(m) not in forbidden_slots]``.
    An empty pool returns ``LEGACY_MODE``.

    Rotation is ``sha1(f"{seed}|{task_id}")[:4]`` as a big-endian uint32
    modulo ``len(pool)`` — **not** builtin ``hash()``, which is salted
    by ``PYTHONHASHSEED``. Then
    ``pool[(episode_index + rotation) % len(pool)]``.
    """
    banned = {str(slot) for slot in forbidden_slots}
    pool = [mode for mode in MODES if _split_mode(mode)[0] not in banned]
    if not pool:
        return LEGACY_MODE
    key = f"{seed}|{task_id}".encode()
    rotation = int.from_bytes(hashlib.sha1(key).digest()[:4], "big") % len(pool)
    return pool[(int(episode_index) + rotation) % len(pool)]


def libero_l2_fallback_chain(
    requested_mode: str,
    *,
    forbidden_slots: Sequence[str],
) -> tuple[str, ...]:
    """Ordered, de-duplicated, finite fallback list ending in ``LEGACY_MODE``.

    Slot diversity is the goal; path style is only a means. The chain
    therefore keeps the requested slot before crossing the table:

    1. ``requested_mode``
    2. same slot, the other style
    3. opposite slot, same style (``center`` tries ``left`` first;
       ``left`` tries ``right``; ``right`` tries ``left``)
    4. ``center_arc``
    5. ``legacy_jitter`` (always last; never stripped)

    Modes whose slot is in ``forbidden_slots`` are omitted, except
    ``LEGACY_MODE`` which is always appended.
    """
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
    ordered.append(LEGACY_MODE)
    return tuple(ordered)


def libero_l2_accept_spawn(
    requested_mode: str,
    *,
    forbidden_slots: Sequence[str],
    make_candidate: MakeCandidateFn,
    validate: ValidateFn,
) -> dict[str, Any]:
    """Walk the fallback chain until a candidate is accepted.

    ``make_candidate(mode)`` returns one of:

    * a buildable dict ``{"start", "path", "linear_solve_mode",
      "min_clearance_m"}``
    * ``None`` — recorded as ``linear_solve_infeasible``
    * ``{"infeasible_reason": "..."}`` (no ``path``) — that string is
      recorded instead, so Track B can write
      ``"linear_solve_infeasible"`` vs ``"trapezoid_infeasible"`` into
      episode extra

    ``validate(start, path)`` is the caller-injected sweep. A false
    result is ``candidate_rejected``.

    ``LEGACY_MODE`` is accepted without calling ``validate``.
    ``make_candidate`` is still invoked so the caller can attach the
    status-quo jitter+arc payload; a ``None`` there yields empty
    ``start`` / ``path`` and ``linear_solve_mode="na"``.

    Returns:
        Dict with ``requested_mode``, ``accepted_mode``,
        ``fallback_reason`` (``None`` / ``"candidate_rejected"`` /
        ``"linear_solve_infeasible"`` / ``"trapezoid_infeasible"`` /
        ``"all_candidates_rejected"``),
        ``fallback_chain`` (modes actually tried, in order), plus the
        accepted candidate fields.
    """
    chain = libero_l2_fallback_chain(requested_mode, forbidden_slots=forbidden_slots)
    tried: list[str] = []
    first_reason: str | None = None

    for mode in chain:
        tried.append(mode)
        if mode == LEGACY_MODE:
            if requested_mode == LEGACY_MODE and len(tried) == 1:
                reason: str | None = None
            else:
                reason = "all_candidates_rejected"
            return {
                "requested_mode": requested_mode,
                "accepted_mode": LEGACY_MODE,
                "fallback_reason": reason,
                "fallback_chain": tuple(tried),
                **_candidate_fields(make_candidate(mode)),
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
