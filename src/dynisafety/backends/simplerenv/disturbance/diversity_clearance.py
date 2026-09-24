"""SimplerEnv L2/L3 diversity clearance only.

Used by spawn/reach planners. Do not call from landing, support-surface, or
L1 drop logic. LIBERO / RoboTwin have their own probes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from dynisafety.backends.simplerenv.targets import get_actor_world_aabb, xy_aabb_overlaps

# Same tokens as the former controller skip lists (substring match).
_ACTOR_SKIP_TOKENS: tuple[str, ...] = ("robot", "camera", "light", "dynisafety", "goal_site")
_TABLE_FLOOR_TOKENS: tuple[str, ...] = ("arena", "ground", "table")
LAYER_B_SKIP_TOKENS: tuple[str, ...] = _ACTOR_SKIP_TOKENS + _TABLE_FLOOR_TOKENS

# Z slack for "bottom actually meets a prop top". Matches former L2 layer-B margin.
BLOCKING_HEIGHT_MARGIN_Z = 0.02

LEGACY_UNVALIDATED = "legacy_unvalidated"

Aabb = tuple[np.ndarray, np.ndarray]
AabbFn = Callable[[Any], Aabb | None]


def actor_name_matches(name: str, tokens: Iterable[str]) -> bool:
    lowered = str(name or "").lower()
    return any(token in lowered for token in tokens)


def is_table_or_floor_name(name: str) -> bool:
    return actor_name_matches(name, _TABLE_FLOOR_TOKENS)


def _xy_inside_aabb(xy: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> bool:
    return (
        float(mins[0]) <= float(xy[0]) <= float(maxs[0])
        and float(mins[1]) <= float(xy[1]) <= float(maxs[1])
    )


def fixture_top_at_xy(
    xy: Any,
    *,
    table_height: float,
    actors: Iterable[Any],
    exclude_actor: Any = None,
    get_aabb: AabbFn | None = None,
) -> float:
    """Highest support under ``xy`` for diversity layer A / L3 brick clearance.

    ``arena`` / ``table`` / ``ground`` contribute ``table_height`` only — never
    the world AABB max Z (WidowX arena boxes are ~3 m tall). Task props
    contribute their real AABB top. Robot / camera / lights / self are skipped.
    """
    point = np.asarray(xy, dtype=np.float64).reshape(-1)
    if point.size < 2:
        raise ValueError(f"xy must have at least 2 components, got shape {np.shape(xy)}")
    aabb_fn = get_actor_world_aabb if get_aabb is None else get_aabb
    best = float(table_height)
    for scene_actor in actors:
        if scene_actor is exclude_actor:
            continue
        name = str(getattr(scene_actor, "name", "") or "")
        if actor_name_matches(name, _ACTOR_SKIP_TOKENS):
            continue
        aabb = aabb_fn(scene_actor)
        if aabb is None:
            continue
        mins, maxs = aabb
        if not _xy_inside_aabb(point, mins, maxs):
            continue
        if is_table_or_floor_name(name):
            top = float(table_height)
        else:
            top = float(maxs[2])
        if np.isfinite(top):
            best = max(best, top)
    return float(best)


def mover_blocked_by_scene(
    mover_aabb: Aabb,
    actors: Iterable[Any],
    *,
    exclude_actor: Any = None,
    table_height: float,
    margin_z: float = BLOCKING_HEIGHT_MARGIN_Z,
    get_aabb: AabbFn | None = None,
) -> bool:
    """True when the mover's footprint meets a prop in 3D.

    Table / arena / ground are ignored (layer B skip). A prop whose XY overlaps
    the mover blocks only if the mover bottom is at or below the prop top plus
    ``margin_z``. Flying over a plate with air gap is not a block.
    """
    del table_height
    aabb_fn = get_actor_world_aabb if get_aabb is None else get_aabb
    mover_mins = np.asarray(mover_aabb[0], dtype=np.float64)
    mover_maxs = np.asarray(mover_aabb[1], dtype=np.float64)
    mover_bottom = float(mover_mins[2])
    slack = float(margin_z)
    for scene_actor in actors:
        if scene_actor is exclude_actor:
            continue
        name = str(getattr(scene_actor, "name", "") or "")
        if actor_name_matches(name, LAYER_B_SKIP_TOKENS):
            continue
        scene_aabb = aabb_fn(scene_actor)
        if scene_aabb is None:
            continue
        scene_mins = np.asarray(scene_aabb[0], dtype=np.float64)
        scene_maxs = np.asarray(scene_aabb[1], dtype=np.float64)
        if not xy_aabb_overlaps(mover_mins[:2], mover_maxs[:2], scene_mins[:2], scene_maxs[:2]):
            continue
        if mover_bottom <= float(scene_maxs[2]) + slack:
            return True
    return False
