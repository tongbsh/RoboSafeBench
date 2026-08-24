"""Isaac-GR00T ZMQ adapter (PolicyClient via ZmqTransport)."""

from __future__ import annotations

from typing import Any

import numpy as np

from dynisafety.policies.adapters._common import (
    actions_to_chunk,
    connect_transport,
    require_image,
    require_state,
)
from dynisafety.policies.base import (
    ActionChunk,
    EpisodeContext,
    ModelConfig,
    Observation,
)
from dynisafety.policies.transports import Transport

_LANG_KEY = "annotation.human.action.task_description"


def _normalize_gripper_action(action: np.ndarray, *, binarize: bool) -> np.ndarray:
    a = np.asarray(action, dtype=np.float32).copy()
    a[..., -1] = 2 * (a[..., -1] - 0.0) / (1.0 - 0.0) - 1
    if binarize:
        a[..., -1] = np.sign(a[..., -1])
    return a


def _invert_gripper_action(action: np.ndarray) -> np.ndarray:
    a = np.asarray(action, dtype=np.float32).copy()
    a[..., -1] = a[..., -1] * -1.0
    return a


def _obs_to_gr00t_flat(obs: Observation) -> dict[str, Any]:
    img = require_image(obs, "primary")
    wrist = require_image(obs, "wrist")
    state = require_state(obs).astype(np.float32).reshape(-1)
    if state.shape[0] != 8:
        raise ValueError(f"gr00t expects 8D state, got {state.shape}")
    x, y, z = float(state[0]), float(state[1]), float(state[2])
    roll, pitch, yaw = float(state[3]), float(state[4]), float(state[5])
    gripper = state[6:8].astype(np.float32)
    return {
        "video.image": img[None, None, ...],
        "video.wrist_image": wrist[None, None, ...],
        "state.x": np.array([[[x]]], dtype=np.float32),
        "state.y": np.array([[[y]]], dtype=np.float32),
        "state.z": np.array([[[z]]], dtype=np.float32),
        "state.roll": np.array([[[roll]]], dtype=np.float32),
        "state.pitch": np.array([[[pitch]]], dtype=np.float32),
        "state.yaw": np.array([[[yaw]]], dtype=np.float32),
        "state.gripper": gripper.reshape(1, 1, -1),
        _LANG_KEY: (obs.instruction,),
    }


def _flat_action_to_steps(flat_action: dict[str, Any]) -> np.ndarray:
    keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    parts = [np.asarray(flat_action[f"action.{k}"], dtype=np.float32) for k in keys]
    t0 = parts[0].shape[1]
    rows: list[np.ndarray] = []
    for t in range(t0):
        vec = np.concatenate(
            [np.asarray(flat_action[f"action.{k}"][0, t], dtype=np.float32).reshape(-1) for k in keys]
        )
        vec = _normalize_gripper_action(vec, binarize=True)
        vec = _invert_gripper_action(vec)
        rows.append(vec.astype(np.float32))
    return np.stack(rows, axis=0)


class Gr00tAdapter:
    """ZMQ GR00T policy (LIBERO_PANDA + sim wrapper server)."""

    def __init__(self, cfg: ModelConfig, transport: Transport) -> None:
        self._cfg = cfg
        self._transport = transport

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> Gr00tAdapter:
        if cfg.transport != "zmq":
            raise ValueError("Gr00tAdapter requires transport='zmq'")
        return cls(cfg, connect_transport(cfg))

    def reset(self, ctx: EpisodeContext) -> None:
        del ctx
        self._transport.reset(None)

    def act(self, obs: Observation) -> ActionChunk:
        payload = _obs_to_gr00t_flat(obs)
        result = self._transport.infer(payload)
        flat = result.get("action", result)
        if not isinstance(flat, dict):
            raise TypeError(f"gr00t action must be dict, got {type(flat)!r}")
        actions = _flat_action_to_steps(flat)
        return actions_to_chunk(actions, space=self._cfg.action_space)

    def close(self) -> None:
        self._transport.close()
