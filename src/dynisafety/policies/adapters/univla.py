"""UniVLA HTTP adapter — replaces UniVLAHTTPClient registry bypass."""

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


class UniVLAAdapter:
    """HTTP ``/reset`` + ``/act`` client (LIBERO / SimplerEnv).

    Requires ``extra.reset_path``, ``extra.act_path``, ``extra.act_timeout_s``.
    """

    def __init__(self, cfg: ModelConfig, transport: Transport) -> None:
        self._cfg = cfg
        self._transport = transport

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> UniVLAAdapter:
        if cfg.transport != "http":
            raise ValueError("UniVLAAdapter requires transport='http'")
        return cls(cfg, connect_transport(cfg))

    def reset(self, ctx: EpisodeContext) -> None:
        del ctx
        self._transport.reset({})

    def act(self, obs: Observation) -> ActionChunk:
        payload: dict[str, Any] = {
            "image": as_jsonable(require_image(obs, "primary")),
            "task": obs.instruction,
            "proprio": as_jsonable(require_state(obs).astype(np.float64)),
        }
        result = self._transport.infer(payload)
        if "action" not in result:
            raise KeyError(f"univla response missing 'action'; keys={sorted(result)}")
        actions = np.asarray(result["action"], dtype=np.float32)
        return actions_to_chunk(actions, space=self._cfg.action_space)

    def close(self) -> None:
        self._transport.close()
