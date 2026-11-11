"""Oriented-box geometry (numpy only, vectorised) for the generator's geometric checks.

A box is ``(centre (3,), rotation (3, 3), half extents (3,))``; the columns of ``rotation`` are the box axes in
the world frame. Every function broadcasts over leading dimensions.
"""
from typing import Tuple

import numpy as np


def quat_to_mat(quat) -> np.ndarray:
    """Rotation matrices of unit quaternions ``(..., 4)`` in ``(w, x, y, z)`` order (normalised here)."""
    q = np.asarray(quat, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    out = np.empty(q.shape[:-1] + (3, 3))
    out[..., 0, 0] = 1 - 2 * (y * y + z * z)
    out[..., 0, 1] = 2 * (x * y - z * w)
    out[..., 0, 2] = 2 * (x * z + y * w)
    out[..., 1, 0] = 2 * (x * y + z * w)
    out[..., 1, 1] = 1 - 2 * (x * x + z * z)
    out[..., 1, 2] = 2 * (y * z - x * w)
    out[..., 2, 0] = 2 * (x * z - y * w)
    out[..., 2, 1] = 2 * (y * z + x * w)
    out[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return out


def box_overlap_depth(c1, r1, h1, c2, r2, h2) -> np.ndarray:
    """Separating-axis test of two oriented boxes: the smallest overlap of their projections over the 15 axes
    (3 + 3 face normals, 9 edge cross products).

    Positive: the boxes overlap and the value is the penetration depth (the length of the shortest translation
    that separates them). Negative: they are disjoint and ``-value`` is the largest gap found along a
    separating axis (a lower bound of the true distance, exact for face-to-face separation). Degenerate edge
    cross products (parallel edges, zero length) are skipped: they add no axis. Any extra unit axis can only
    lengthen the minimum, so the depth is never below the true one.
    """
    c1, r1, h1, c2, r2, h2 = (np.asarray(a, dtype=np.float64) for a in (c1, r1, h1, c2, r2, h2))
    d = c2 - c1
    axes = [r1[..., :, i] for i in range(3)] + [r2[..., :, j] for j in range(3)]
    for i in range(3):
        for j in range(3):
            cross = np.cross(r1[..., :, i], r2[..., :, j])
            norm = np.linalg.norm(cross, axis=-1, keepdims=True)
            usable = norm > 0
            axes.append(np.where(usable, cross / np.where(usable, norm, 1.0), np.nan))
    best = None
    for axis in axes:
        proj1 = np.sum(h1 * np.abs(np.einsum("...ki,...k->...i", r1, axis)), axis=-1)
        proj2 = np.sum(h2 * np.abs(np.einsum("...ki,...k->...i", r2, axis)), axis=-1)
        overlap = proj1 + proj2 - np.abs(np.sum(d * axis, axis=-1))
        overlap = np.where(np.isnan(overlap), np.inf, overlap)
        best = overlap if best is None else np.minimum(best, overlap)
    return best


def points_box_distance(points, centre, rotation, half) -> np.ndarray:
    """Euclidean distance from ``points (..., 3)`` to the solid box (0 inside)."""
    points = np.asarray(points, dtype=np.float64)
    local = (points - np.asarray(centre, dtype=np.float64)).dot(np.asarray(rotation, dtype=np.float64))
    outside = np.maximum(np.abs(local) - np.asarray(half, dtype=np.float64), 0.0)
    return np.linalg.norm(outside, axis=-1)


def points_boxes_distance(points, centres, rotations, halves) -> np.ndarray:
    """``(P, K)`` distances from ``P`` points to ``K`` boxes (``centres (K, 3)``, ``rotations (K, 3, 3)``,
    ``halves (K, 3)``)."""
    points = np.asarray(points, dtype=np.float64)
    if not len(centres):
        return np.empty((len(points), 0))
    rel = points[:, None, :] - np.asarray(centres, dtype=np.float64)[None, :, :]
    local = np.einsum("kji,pkj->pki", np.asarray(rotations, dtype=np.float64), rel)
    outside = np.maximum(np.abs(local) - np.asarray(halves, dtype=np.float64)[None, :, :], 0.0)
    return np.linalg.norm(outside, axis=-1)


def box_corners(centre, rotation, half) -> np.ndarray:
    """The 8 corners ``(8, 3)`` of one box."""
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
    return np.asarray(centre, dtype=np.float64) + (signs * np.asarray(half, dtype=np.float64)).dot(
        np.asarray(rotation, dtype=np.float64).T)


def world_boxes(local_centre, local_rot, local_half, body_pos, body_rot) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boxes given in a body frame -> world frame: ``(centres (K, 3), rotations (K, 3, 3), halves (K, 3))``."""
    centre = np.asarray(body_pos, dtype=np.float64) + np.asarray(local_centre, dtype=np.float64).dot(
        np.asarray(body_rot, dtype=np.float64).T)
    rot = np.einsum("ij,kjl->kil", np.asarray(body_rot, dtype=np.float64), np.asarray(local_rot, dtype=np.float64))
    return centre, rot, np.asarray(local_half, dtype=np.float64)
