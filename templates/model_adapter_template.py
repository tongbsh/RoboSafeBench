"""Copy this file to ``src/dynisafety/policies/adapters/<my_model>.py``.

Then:
1. Rename ``TemplateAdapter`` → ``MyModelAdapter``.
2. Fill in the TODO blocks (no ports / timeouts / chunk sizes here — those
   belong in ``configs/models/<my_model>.yaml`` only).
3. Point the YAML ``adapter:`` field at this class.
4. Run ``dynisafety models`` and a smoke eval (see ``docs/user/adding_a_model.md``).

Remote models MUST use ``build_transport`` / ``connect_transport`` — never
``requests``, ``websockets``, or ``zmq`` inside an adapter.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dynisafety.policies.adapters._common import (
    actions_to_chunk,
    as_jsonable,
    connect_transport,
    require_extra,
    require_image,
)
from dynisafety.policies.base import (
    ActionChunk,
    EpisodeContext,
    ModelConfig,
    Observation,
)
from dynisafety.policies.transports import Transport


class TemplateAdapter:
    """Skeleton Policy adapter — replace TODOs before registering."""

    def __init__(self, cfg: ModelConfig, transport: Transport | None) -> None:
        self._cfg = cfg
        self._transport = transport
        # TODO: read model-specific required keys from cfg.extra (no defaults).
        # Example: self._foo = require_extra(cfg, "foo")

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> TemplateAdapter:
        # TODO: enforce cfg.transport in {"local", "http", "websocket", "zmq"}.
        transport: Transport | None = None
        if cfg.transport != "local":
            transport = connect_transport(cfg)
        return cls(cfg, transport)

    def reset(self, ctx: EpisodeContext) -> None:
        # TODO: clear episode buffers; for HTTP call transport.reset({}).
        del ctx
        if self._transport is not None:
            self._transport.reset({})

    def act(self, obs: Observation) -> ActionChunk:
        # TODO: pack Observation → model payload using contract image keys
        # (primary / wrist / left / right). Missing keys must raise.
        _ = require_image(obs, "primary")
        if self._transport is not None:
            payload: dict[str, Any] = {
                "image": as_jsonable(require_image(obs, "primary")),
                "instruction": obs.instruction,
            }
            result = self._transport.infer(payload)
            actions = np.asarray(
                result.get("action", result.get("actions")), dtype=np.float32
            )
        else:
            # TODO: local inference — lazy-import heavy deps here only.
            raise NotImplementedError("TemplateAdapter local act() not implemented")
        return actions_to_chunk(actions, space=self._cfg.action_space)

    def close(self) -> None:
        if self._transport is not None:
            self._transport.close()
