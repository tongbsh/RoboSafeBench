"""XY landing avoid / push helpers (disk or AABB). No simulator import."""

from __future__ import annotations

import numpy as np


def quat_wxyz_to_rot(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in np.asarray(quat, dtype=np.float64).reshape(-1)[:4])
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def world_xy_aabb_from_obb(
    *,
    center: np.ndarray,
    quat_wxyz: np.ndarray,
    half_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """World-XY AABB of an oriented box. ``half_xyz`` is local half-extents."""
    origin = np.asarray(center, dtype=np.float64).reshape(-1)[:3]
    half = np.asarray(half_xyz, dtype=np.float64).reshape(-1)[:3]
    rot = quat_wxyz_to_rot(quat_wxyz)
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                local = np.array([sx * half[0], sy * half[1], sz * half[2]], dtype=np.float64)
                corners.append(origin + rot @ local)
    pts = np.stack(corners, axis=0)
    return pts[:, :2].min(axis=0), pts[:, :2].max(axis=0)


def nudge_xy_disk(
    obstacle_xy: np.ndarray,
    actor_xy: np.ndarray,
    min_dist: float,
) -> np.ndarray:
    """Push ``obstacle_xy`` onto the circle of radius ``min_dist`` around actor."""
    obs = np.asarray(obstacle_xy, dtype=np.float64).reshape(-1)[:2].copy()
    act = np.asarray(actor_xy, dtype=np.float64).reshape(-1)[:2]
    diff = obs - act
    dist = float(np.linalg.norm(diff))
    radius = float(min_dist)
    if dist >= radius:
        return obs
    if dist <= 1e-8:
        diff = np.array([1.0, 0.0], dtype=np.float64)
        dist = 1.0
    return act + diff / dist * radius


def nudge_xy_aabb(
    obstacle_xy: np.ndarray,
    obstacle_half_xy: tuple[float, float],
    actor_mins: np.ndarray,
    actor_maxs: np.ndarray,
    margin: float,
) -> np.ndarray:
    """If the obstacle AABB overlaps the actor AABB, push onto the nearest face."""
    obs = np.asarray(obstacle_xy, dtype=np.float64).reshape(-1)[:2].copy()
    ohx, ohy = float(obstacle_half_xy[0]), float(obstacle_half_xy[1])
    pad = float(margin)
    expanded_mins = np.asarray(actor_mins, dtype=np.float64).reshape(-1)[:2] - np.array(
        [ohx + pad, ohy + pad], dtype=np.float64
    )
    expanded_maxs = np.asarray(actor_maxs, dtype=np.float64).reshape(-1)[:2] + np.array(
        [ohx + pad, ohy + pad], dtype=np.float64
    )
    x, y = float(obs[0]), float(obs[1])
    inside = (
        expanded_mins[0] < x < expanded_maxs[0] and expanded_mins[1] < y < expanded_maxs[1]
    )
    if not inside:
        return obs
    left = x - float(expanded_mins[0])
    right = float(expanded_maxs[0]) - x
    bottom = y - float(expanded_mins[1])
    top = float(expanded_maxs[1]) - y
    nearest = min(left, right, bottom, top)
    if nearest == left:
        return np.array([float(expanded_mins[0]), y], dtype=np.float64)
    if nearest == right:
        return np.array([float(expanded_maxs[0]), y], dtype=np.float64)
    if nearest == bottom:
        return np.array([x, float(expanded_mins[1])], dtype=np.float64)
    return np.array([x, float(expanded_maxs[1])], dtype=np.float64)


def push_xy_away(
    point_xy: np.ndarray,
    origin_xy: np.ndarray,
    extra_m: float,
) -> np.ndarray:
    """Move ``point_xy`` farther from ``origin_xy`` by ``extra_m`` (no-op if ≤0)."""
    extra = float(extra_m)
    point = np.asarray(point_xy, dtype=np.float64).reshape(-1)[:2].copy()
    if extra <= 0.0:
        return point
    origin = np.asarray(origin_xy, dtype=np.float64).reshape(-1)[:2]
    delta = point - origin
    dist = float(np.linalg.norm(delta))
    if dist <= 1e-8:
        delta = np.array([1.0, 0.0], dtype=np.float64)
        dist = 1.0
    return origin + delta / dist * (dist + extra)
