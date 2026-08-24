"""Dream-VLA HTTP adapter (SimplerEnv Bridge zero-shot path)."""

from __future__ import annotations

from typing import Any

import numpy as np

from dynisafety.policies.adapters._common import (
    actions_to_chunk,
    as_jsonable,
    connect_transport,
    require_image,
)
from dynisafety.policies.base import (
    ActionChunk,
    EpisodeContext,
    ModelConfig,
    Observation,
)
from dynisafety.policies.transports import Transport


class DreamVLAAdapter:
    """HTTP Dream-VLA client. Prefers JSON ``/act`` via HttpTransport."""

    def __init__(self, cfg: ModelConfig, transport: Transport) -> None:
        self._cfg = cfg
        self._transport = transport
        self._queue: list[np.ndarray] = []

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> DreamVLAAdapter:
        if cfg.transport != "http":
            raise ValueError("DreamVLAAdapter requires transport='http'")
        return cls(cfg, connect_transport(cfg))

    def reset(self, ctx: EpisodeContext) -> None:
        del ctx
        self._queue.clear()
        self._transport.reset({})

    def act(self, obs: Observation) -> ActionChunk:
        if not self._queue:
            payload: dict[str, Any] = {
                "image": as_jsonable(require_image(obs, "primary")),
                "instruction": obs.instruction,
            }
            if "unnorm_key" in self._cfg.extra:
                payload["unnorm_key"] = self._cfg.extra["unnorm_key"]
            result = self._transport.infer(payload)
            raw = np.asarray(
                result.get("action", result.get("actions")), dtype=np.float32
            )
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            self._queue = [raw[i] for i in range(raw.shape[0])]
        step = self._queue.pop(0)
        return actions_to_chunk(step.reshape(1, -1), space=self._cfg.action_space)

    def close(self) -> None:
        self._transport.close()
