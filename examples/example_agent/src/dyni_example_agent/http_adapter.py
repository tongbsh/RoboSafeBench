"""HTTP example-agent adapter.

Why HTTP rather than WebSocket: ``build_transport``'s websocket path requires
``extra.ws_codec`` ∈ ``{openpi, starvla, lingbot, xiaomi}``, all of which are
model-specific codecs in ``src/dynisafety/policies/transports/codecs/``. A
generic toy websocket server is therefore not possible.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from dynisafety.policies.action_space import dim_for_space
from dynisafety.policies.adapters._common import (
    actions_to_chunk,
    as_jsonable,
    connect_transport,
    require_extra,
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


def _require_int_extra(cfg: ModelConfig, key: str) -> int:
    value = require_extra(cfg, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"extra.{key} must be int, got {value!r}")
    return value


def _require_float_extra(cfg: ModelConfig, key: str) -> float:
    value = require_extra(cfg, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"extra.{key} must be float, got {value!r}")
    return float(value)


class ExampleAgentHttpAdapter:
    """``transport: http`` Policy. Remote I/O goes only through ``connect_transport``.

    Remote planner + local buffer is how real remote WAM-style policies work:
    one ``infer`` requests ``horizon=cfg.chunk_size``, then this adapter emits
    at most ``replan_steps`` rows per ``act()``. That is why ``replan_steps`` is
    a live YAML knob on the HTTP path, not a dead extra.
    """

    def __init__(self, cfg: ModelConfig, transport: Transport) -> None:
        self._cfg = cfg
        self._transport = transport
        self._seed = _require_int_extra(cfg, "seed")
        self._replan_steps = _require_int_extra(cfg, "replan_steps")
        self._action_scale = _require_float_extra(cfg, "action_scale")
        self._feature_grid = _require_int_extra(cfg, "feature_grid")
        if not 1 <= self._replan_steps <= cfg.chunk_size:
            raise ValueError(
                f"extra.replan_steps must satisfy 1 <= replan_steps <= "
                f"chunk_size ({cfg.chunk_size}), got {self._replan_steps}"
            )
        self._dim = dim_for_space(cfg.action_space)
        self._plan: deque[np.ndarray] = deque()
        self._episode_index = 0
        self._instruction = ""

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> ExampleAgentHttpAdapter:
        if cfg.transport == "local":
            raise ValueError(
                "ExampleAgentHttpAdapter does not support transport='local'; "
                "use ExampleAgentAdapter"
            )
        return cls(cfg, connect_transport(cfg))

    def reset(self, ctx: EpisodeContext) -> None:
        self._plan.clear()
        self._episode_index = ctx.episode_index
        self._instruction = ctx.instruction
        self._transport.reset({"instruction": ctx.instruction, "episode_index": ctx.episode_index})

    def act(self, obs: Observation) -> ActionChunk:
        images = {key: require_image(obs, key) for key in self._cfg.observation_images}
        state: np.ndarray | None
        if self._cfg.observation_needs_state:
            state = require_state(obs)
        else:
            state = obs.state
        instruction = obs.instruction if obs.instruction else self._instruction
        if not self._plan:
            payload: dict[str, Any] = {
                "images": as_jsonable(images),
                "state": as_jsonable(state),
                "instruction": instruction,
                "step": obs.step,
                "space": self._cfg.action_space,
                "horizon": self._cfg.chunk_size,
                "episode_index": self._episode_index,
                # Forward YAML extras so the toy server invents no Python defaults.
                "action_dim": self._dim,
                "seed": self._seed,
                "feature_grid": self._feature_grid,
                "action_scale": self._action_scale,
            }
            result = self._transport.infer(payload)
            if not result.get("ok"):
                raise ValueError(f"example_agent_http: act response not ok; body={result!r}")
            if "actions" not in result:
                raise KeyError(
                    f"example_agent_http: act response missing 'actions'; keys={sorted(result)}"
                )
            planned = np.asarray(result["actions"], dtype=np.float32)
            if planned.ndim == 1:
                planned = planned.reshape(1, -1)
            if planned.ndim != 2 or planned.shape[0] < 1:
                raise ValueError(
                    f"example_agent_http: actions must be (horizon, dim), got {planned.shape}"
                )
            self._plan.extend(planned)
        n_take = min(self._replan_steps, len(self._plan))
        rows = [self._plan.popleft() for _ in range(n_take)]
        return actions_to_chunk(np.stack(rows, axis=0), space=self._cfg.action_space)

    def close(self) -> None:
        self._plan.clear()
        self._transport.close()
