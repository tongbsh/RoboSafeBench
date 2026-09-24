"""VLA-Adapter-Pro HTTP adapter — examples-style /reset + /act payload."""

from __future__ import annotations

from typing import Any

import numpy as np

from dynisafety.policies.adapters._common import (
    actions_to_chunk,
    as_jsonable,
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


class VLAAdapter:
    """HTTP client for ``scripts/vla_adapter/deploy_vla_adapter_dynisafety_server.py``.

    Payload matches ``examples/example_agent``: ``images`` / ``state`` /
    ``instruction`` / ``step`` / ``space`` / ``horizon``. Response is
    ``{"ok": true, "actions": (C, 7)}`` already in environment units
    (delta EE 7D, gripper already binarized and inverted on the server).

    LIBERO images arriving here are already 180°-rotated by the backend;
    this adapter does not flip, crop, unnormalize, or invert the gripper.
    """

    def __init__(self, cfg: ModelConfig, transport: Transport) -> None:
        self._cfg = cfg
        self._transport = transport

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> VLAAdapter:
        if cfg.transport != "http":
            raise ValueError("VLAAdapter requires transport='http'")
        return cls(cfg, connect_transport(cfg))

    def reset(self, ctx: EpisodeContext) -> None:
        self._transport.reset(
            {"instruction": ctx.instruction, "episode_index": ctx.episode_index}
        )

    def act(self, obs: Observation) -> ActionChunk:
        images = {key: require_image(obs, key) for key in self._cfg.observation_images}
        payload: dict[str, Any] = {
            "images": as_jsonable(images),
            "state": as_jsonable(require_state(obs)),
            "instruction": obs.instruction,
            "step": obs.step,
            "space": self._cfg.action_space,
            "horizon": self._cfg.chunk_size,
        }
        result = self._transport.infer(payload)
        if not result.get("ok"):
            raise ValueError(f"vla_adapter: act response not ok; body={result!r}")
        if "actions" not in result:
            raise KeyError(
                f"vla_adapter: act response missing 'actions'; keys={sorted(result)}"
            )
        actions = np.asarray(result["actions"], dtype=np.float32)
        return actions_to_chunk(actions, space=self._cfg.action_space)

    def close(self) -> None:
        self._transport.close()
