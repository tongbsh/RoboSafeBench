"""Backend-agnostic hold / zero-motion action factory for the pause-VLA oracle."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from dynisafety.policies.action_space import dim_for_space
from dynisafety.policies.base import ActionChunk, Observation
from dynisafety.policies.safety.config import OracleStopActionConfig


class HoldActionFactory(Protocol):
    """Produce a stop/hold chunk without assuming a fixed action dimension."""

    def hold_action(
        self,
        *,
        space: str,
        last_action: ActionChunk | None,
        obs: Observation,
    ) -> ActionChunk: ...


class ZeroMotionHoldFactory:
    """Zero motion dims; keep last gripper (or YAML default) to avoid release."""

    def __init__(self, cfg: OracleStopActionConfig) -> None:
        if cfg.strategy != "hold_gripper_zero_motion":
            raise ValueError(
                f"unsupported stop_action.strategy {cfg.strategy!r}; "
                "expected 'hold_gripper_zero_motion'"
            )
        self._cfg = cfg

    def hold_action(
        self,
        *,
        space: str,
        last_action: ActionChunk | None,
        obs: Observation,
    ) -> ActionChunk:
        del obs  # reserved for abs-pose hold adapters
        dim = dim_for_space(space)
        chunk_size = int(self._cfg.chunk_size)
        row = np.zeros(dim, dtype=np.float64)

        grip_idx = self._cfg.gripper_indices_by_space.get(space)
        if grip_idx is None:
            raise ValueError(
                f"no gripper indices configured for action space {space!r}; "
                f"known: {sorted(self._cfg.gripper_indices_by_space)}"
            )

        if last_action is not None:
            prev = np.asarray(last_action.actions, dtype=np.float64)
            if prev.ndim != 2 or prev.shape[1] != dim:
                raise ValueError(
                    f"last_action shape {prev.shape} incompatible with space={space!r} dim={dim}"
                )
            src = prev[-1]
            for i in grip_idx:
                if i >= dim:
                    raise ValueError(
                        f"gripper index {i} out of range for space={space!r} dim={dim}"
                    )
                row[i] = float(src[i])
        else:
            for i in grip_idx:
                if i >= dim:
                    raise ValueError(
                        f"gripper index {i} out of range for space={space!r} dim={dim}"
                    )
                row[i] = float(self._cfg.default_gripper_hold)

        actions = np.tile(row, (chunk_size, 1))
        return ActionChunk(actions=actions, space=space)
