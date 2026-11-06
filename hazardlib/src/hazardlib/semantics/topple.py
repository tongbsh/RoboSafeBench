"""Toppling (00 §6, R10): the tilt of the object's upright axis, ignoring rotation about world z (yaw).

Quaternions are ``wxyz``. The reference quaternion defines which body axis is "up"; the tilt is the angle
between world +z and that axis after the current rotation. Pure yaw therefore never counts as toppling,
unlike the legacy relative-quaternion total angle used by the LIBERO and RoboCasa backends (P0 F3).
"""
import math
from typing import Any

import numpy as np

from .._numeric import finite_float, non_negative_float


def _rotation(name: str, quat: Any) -> np.ndarray:
    q = np.array(quat, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("%s must be a finite wxyz quaternion of shape (4,), got %r" % (name, quat))
    norm = float(np.linalg.norm(q))
    if not norm > 0:
        raise ValueError("%s must be a non-zero quaternion" % name)
    w, x, y, z = (float(v) for v in q / norm)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def tilt_deg_up_axis(ref_quat: Any, cur_quat: Any) -> float:
    """Tilt in degrees, in [0, 180], of the reference up axis under ``cur_quat``, ignoring yaw about z.
    Computed with ``atan2`` so tiny tilts keep their precision."""
    local_up = _rotation("ref_quat", ref_quat).T @ np.array([0.0, 0.0, 1.0])
    up = _rotation("cur_quat", cur_quat) @ local_up
    return math.degrees(math.atan2(math.hypot(float(up[0]), float(up[1])), float(up[2])))


def is_toppled(tilt_deg: float, threshold_deg: float) -> bool:
    """``tilt_deg > threshold_deg`` (strict, so an upright object with threshold 0 is not toppled).
    ``threshold_deg >= 0`` comes from ``semantics.yaml: toppling.tilt_threshold_deg``."""
    return finite_float("tilt_deg", tilt_deg) > non_negative_float("threshold_deg", threshold_deg)
