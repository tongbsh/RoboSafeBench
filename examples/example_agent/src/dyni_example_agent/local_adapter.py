"""Local (in-process) example-agent adapter.

``dynisafety.policies.adapters._common`` has a leading underscore, but it is
the sanctioned helper module: the official template and every in-tree adapter
import from it.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from dyni_example_agent.mock_backbone import MockBackbone
from dynisafety.policies.action_space import dim_for_space
from dynisafety.policies.adapters._common import (
    actions_to_chunk,
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


class ExampleAgentAdapter:
    """``transport: local`` Policy that demonstrates both chunk modes.

    The backend ``ActionChunkQueue.refill`` takes only
    ``min(YAML chunk_size, returned_rows)`` rows and **discards the rest**.
    This adapter therefore never returns more than ``replan_steps`` rows:

    * ``replan_steps == chunk_size`` — return the full ``(chunk_size, D)``
      block; the harness executes it open-loop (mirrors openvla_oft / starvla).
    * ``replan_steps == 1`` — return ``(1, D)`` every ``act()``. The YAML
      ``chunk_size`` does **not** make the harness run open-loop; it is only
      the internal plan horizon (mirrors internvla_a1 / fastwam).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg
        seed = _require_int_extra(cfg, "seed")
        replan_steps = _require_int_extra(cfg, "replan_steps")
        action_scale = _require_float_extra(cfg, "action_scale")
        feature_grid = _require_int_extra(cfg, "feature_grid")
        if not 1 <= replan_steps <= cfg.chunk_size:
            raise ValueError(
                f"extra.replan_steps must satisfy 1 <= replan_steps <= "
                f"chunk_size ({cfg.chunk_size}), got {replan_steps}"
            )
        if feature_grid < 1:
            raise ValueError(f"extra.feature_grid must be >= 1, got {feature_grid}")
        self._replan_steps = replan_steps
        self._dim = dim_for_space(cfg.action_space)
        self._backbone: MockBackbone | None = MockBackbone.from_checkpoint(
            cfg.checkpoint,
            action_dim=self._dim,
            feature_grid=feature_grid,
            action_scale=action_scale,
            seed=seed,
        )
        self._plan: deque[np.ndarray] = deque()
        self._instruction = ""
        self._episode_index = 0

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> ExampleAgentAdapter:
        if cfg.transport != "local":
            raise ValueError(
                f"ExampleAgentAdapter requires transport='local', got {cfg.transport!r}"
            )
        return cls(cfg)

    def reset(self, ctx: EpisodeContext) -> None:
        self._plan.clear()
        self._instruction = ctx.instruction
        self._episode_index = ctx.episode_index

    def act(self, obs: Observation) -> ActionChunk:
        if self._backbone is None:
            raise RuntimeError("ExampleAgentAdapter.close() has already been called")
        images = {key: require_image(obs, key) for key in self._cfg.observation_images}
        state = require_state(obs) if self._cfg.observation_needs_state else obs.state
        instruction = obs.instruction if obs.instruction else self._instruction
        if not self._plan:
            planned = self._backbone.plan(
                images,
                state,
                instruction,
                obs.step,
                horizon=self._cfg.chunk_size,
                episode_index=self._episode_index,
            )
            if planned.ndim != 2 or planned.shape[0] < 1:
                raise ValueError(
                    f"MockBackbone.plan must return (horizon, dim), got {planned.shape}"
                )
            self._plan.extend(planned)
        n_take = min(self._replan_steps, len(self._plan))
        rows = [self._plan.popleft() for _ in range(n_take)]
        return actions_to_chunk(np.stack(rows, axis=0), space=self._cfg.action_space)

    def close(self) -> None:
        self._backbone = None
        self._plan.clear()
