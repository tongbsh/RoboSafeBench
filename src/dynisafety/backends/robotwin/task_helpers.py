"""Pose helpers used by RoboTwin task classes (migrated from envs/dynisafety)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

DEFAULT_QUAT = (1.0, 0.0, 0.0, 0.0)


def pose_like_to_list(pose_like, default_quat: Sequence[float] = DEFAULT_QUAT) -> list[float]:
    """Convert xyz / xyzquat / sapien.Pose into a plain xyzquat list.

    ``sapien.Pose`` is matched structurally on ``.p`` / ``.q`` rather than by
    ``isinstance``, so this module imports without SAPIEN present. This file is
    also vendored into RoboTwin's ``envs/dynisafety/``.
    """
    if hasattr(pose_like, "p") and hasattr(pose_like, "q"):
        return (
            np.asarray(pose_like.p, dtype=np.float64).tolist()
            + np.asarray(pose_like.q, dtype=np.float64).tolist()
        )

    pose_arr = np.asarray(pose_like, dtype=np.float64).reshape(-1)
    if pose_arr.shape[0] >= 7:
        return pose_arr[:7].tolist()
    if pose_arr.shape[0] < 3:
        raise ValueError("Pose-like value must contain at least xyz.")
    return pose_arr[:3].tolist() + list(default_quat)


def actor_pose_to_list(actor, default_quat: Sequence[float] = DEFAULT_QUAT) -> list[float]:
    pose = actor.get_pose()
    return (
        pose.p.tolist() + pose.q.tolist()
        if hasattr(pose, "q")
        else pose_like_to_list(pose, default_quat=default_quat)
    )


def midpoint_pose_list(
    poses: Iterable[Sequence[float]],
    quat: Sequence[float] = DEFAULT_QUAT,
) -> list[float]:
    pose_list = [pose_like_to_list(pose) for pose in poses]
    if not pose_list:
        raise ValueError("midpoint_pose_list requires at least one pose.")
    xyz = np.mean(np.asarray([pose[:3] for pose in pose_list], dtype=np.float64), axis=0)
    return xyz.tolist() + list(quat)
