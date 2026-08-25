"""Per-env-slot EE override / proprio (Infer is stateless).

Official X-VLA RoboTwin proprio is recursive: the next 20-D vector uses the
last commanded xyz+quat, not the latest env endpose. That state used to live
on ``XVLAAdapter``. Parallel Infer cannot hold it (one process, many slots),
so each Env worker keeps one ``SlotState``.

Formula baseline is ``dynisafety.policies.adapters.xvla`` — the functions
below are re-exported, not rewritten. A second copy would make P0 alignment
meaningless. Reset clears overrides to ``None`` (new-stack adapter), it does
not snapshot the current endpose (that is the leftover ``_robotwin_impl``
path and must not be copied).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dynisafety.policies.adapters.xvla import (
    _bool_gripper_to_official,
    _official_robotwin_row_to_ee16,
    _pred_gripper_to_ee,
    _quat_xyzw_to_rot6d,
)
from dynisafety.policies.base import Observation

# Re-export under public names. Do not replace these with local implementations.
quat_xyzw_to_rot6d = _quat_xyzw_to_rot6d
bool_gripper_to_official = _bool_gripper_to_official
official_robotwin_row_to_ee16 = _official_robotwin_row_to_ee16
pred_gripper_to_ee = _pred_gripper_to_ee


def _require_gripper_threshold(gripper_threshold: float) -> float:
    if isinstance(gripper_threshold, bool) or not isinstance(gripper_threshold, (int, float)):
        raise TypeError(
            f"gripper_threshold must be a number, got {gripper_threshold!r}"
        )
    return float(gripper_threshold)


class SlotState:
    """One Env slot's recurrent EE command, matching ``XVLAAdapter`` official RoboTwin."""

    def __init__(self, *, gripper_threshold: float) -> None:
        self._gripper_threshold = _require_gripper_threshold(gripper_threshold)
        self._left_endpose_override: np.ndarray | None = None
        self._right_endpose_override: np.ndarray | None = None

    @property
    def gripper_threshold(self) -> float:
        return self._gripper_threshold

    @property
    def left_endpose_override(self) -> np.ndarray | None:
        return self._left_endpose_override

    @property
    def right_endpose_override(self) -> np.ndarray | None:
        return self._right_endpose_override

    def reset(self) -> None:
        """Clear commanded EE. First proprio after this reads env endpose."""
        self._left_endpose_override = None
        self._right_endpose_override = None

    def proprio_20d(self, obs: Observation) -> np.ndarray:
        """Assemble official 20-D proprio. Gripper always comes from the env."""
        endpose = _require_endpose(obs)
        left_ee = (
            self._left_endpose_override
            if self._left_endpose_override is not None
            else np.asarray(endpose["left_endpose"], dtype=np.float32)
        )
        right_ee = (
            self._right_endpose_override
            if self._right_endpose_override is not None
            else np.asarray(endpose["right_endpose"], dtype=np.float32)
        )
        left_grip = bool_gripper_to_official(float(endpose["left_gripper"]))
        right_grip = bool_gripper_to_official(float(endpose["right_gripper"]))
        return np.concatenate(
            [
                left_ee[:3],
                quat_xyzw_to_rot6d(left_ee[3:7]),
                np.array([left_grip], dtype=np.float64),
                right_ee[:3],
                quat_xyzw_to_rot6d(right_ee[3:7]),
                np.array([right_grip], dtype=np.float64),
            ],
            axis=0,
        ).astype(np.float32)

    def actions_20_to_ee16(self, actions_20: np.ndarray) -> np.ndarray:
        """Decode ``(T, 20)`` official rows to EE 16-D and latch ``actions[-1]``."""
        raw = np.asarray(actions_20, dtype=np.float32)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if raw.ndim != 2 or raw.shape[1] < 20:
            raise ValueError(f"official X-VLA RoboTwin actions must be (T, 20), got {raw.shape}")
        ee16 = np.stack(
            [
                official_robotwin_row_to_ee16(row, gripper_threshold=self._gripper_threshold)
                for row in raw
            ],
            axis=0,
        )
        last = ee16[-1]
        self._left_endpose_override = np.asarray(last[:7], dtype=np.float32).copy()
        self._right_endpose_override = np.asarray(last[8:15], dtype=np.float32).copy()
        return ee16


def _require_endpose(obs: Observation) -> dict[str, Any]:
    extra = obs.extra if obs.extra is not None else {}
    endpose = extra.get("endpose")
    if not isinstance(endpose, dict):
        raise KeyError(
            "official X-VLA RoboTwin path needs Observation.extra['endpose'] "
            "(left/right_endpose + grippers)"
        )
    return endpose


__all__ = [
    "SlotState",
    "bool_gripper_to_official",
    "official_robotwin_row_to_ee16",
    "pred_gripper_to_ee",
    "quat_xyzw_to_rot6d",
]
