"""π₀ / π₀.₅ / openpi-fast WebSocket adapter (shared client protocol)."""

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


class Pi0Adapter:
    """OpenPI WebSocket policy (``pi0``, ``pi05``, ``openpifast`` YAML stems).

    Requires ``extra.ws_codec: openpi`` and ``extra.openpi_client_src``.
    """

    def __init__(self, cfg: ModelConfig, transport: Transport) -> None:
        self._cfg = cfg
        self._transport = transport

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> Pi0Adapter:
        if cfg.transport != "websocket":
            raise ValueError("Pi0Adapter requires transport='websocket'")
        return cls(cfg, connect_transport(cfg))

    def reset(self, ctx: EpisodeContext) -> None:
        del ctx
        self._transport.reset(None)

    def act(self, obs: Observation) -> ActionChunk:
        primary = require_image(obs, "primary")
        payload: dict[str, Any] = {
            "observation/image": primary,
            "observation/state": require_state(obs).astype(np.float32),
            "prompt": obs.instruction,
        }
        if "wrist" in obs.images:
            payload["observation/wrist_image"] = require_image(obs, "wrist")
        result = self._transport.infer(payload)
        if "actions" not in result:
            raise KeyError(f"pi0 response missing 'actions'; keys={sorted(result)}")
        actions = np.asarray(result["actions"], dtype=np.float32)
        return actions_to_chunk(actions, space=self._cfg.action_space)

    def close(self) -> None:
        self._transport.close()


# Alias for YAML stems that share the same protocol.
Pi05Adapter = Pi0Adapter
OpenPiFastAdapter = Pi0Adapter
