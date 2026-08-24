"""Random-action adapter for smoke / adding-a-model verification (Tier-0)."""

from __future__ import annotations

import numpy as np

from dynisafety.policies.action_space import dim_for_space
from dynisafety.policies.adapters._common import actions_to_chunk, require_extra
from dynisafety.policies.base import (
    ActionChunk,
    EpisodeContext,
    ModelConfig,
    Observation,
)


class RandomAdapter:
    """Emits Gaussian noise chunks in ``cfg.action_space`` (verification only)."""

    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg
        seed = require_extra(cfg, "seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"extra.seed must be int, got {seed!r}")
        self._rng = np.random.default_rng(seed)
        self._dim = dim_for_space(cfg.action_space)
        self._chunk = cfg.chunk_size

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> RandomAdapter:
        if cfg.transport != "local":
            raise ValueError("RandomAdapter requires transport='local'")
        return cls(cfg)

    def reset(self, ctx: EpisodeContext) -> None:
        del ctx

    def act(self, obs: Observation) -> ActionChunk:
        del obs
        actions = self._rng.standard_normal((self._chunk, self._dim)).astype(np.float32)
        return actions_to_chunk(actions, space=self._cfg.action_space)

    def close(self) -> None:
        return None
