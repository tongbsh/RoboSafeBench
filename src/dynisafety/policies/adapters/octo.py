"""Octo adapter — local (base/small) or HTTP server via ``extra.variant``."""

from __future__ import annotations

from typing import Any

import numpy as np

from dynisafety.policies.adapters._common import (
    actions_to_chunk,
    as_jsonable,
    connect_transport,
    ensure_repo_on_path,
    require_extra,
    require_image,
)
from dynisafety.policies.adapters.spatialvla import _simpler_dict_to_7d
from dynisafety.policies.base import (
    ActionChunk,
    EpisodeContext,
    ModelConfig,
    Observation,
)
from dynisafety.policies.transports import Transport


class OctoAdapter:
    """Octo policy.

    ``extra.variant``: ``base`` | ``small`` | ``server``.
    Local variants need ``extra.repo_root``, ``extra.policy_setup``, ``extra.action_scale``.
    Server variant uses ``transport: http``.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        *,
        impl: Any | None = None,
        transport: Transport | None = None,
    ) -> None:
        self._cfg = cfg
        self._impl = impl
        self._transport = transport
        self._variant = str(require_extra(cfg, "variant"))

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> OctoAdapter:
        variant = str(require_extra(cfg, "variant"))
        if variant == "server":
            if cfg.transport != "http":
                raise ValueError("OctoAdapter variant=server requires transport='http'")
            return cls(cfg, transport=connect_transport(cfg))
        if cfg.transport != "local":
            raise ValueError("OctoAdapter base/small requires transport='local'")
        ensure_repo_on_path(str(require_extra(cfg, "repo_root")))
        from simpler_env.policies.octo.octo_model import OctoInference  # noqa: PLC0415

        policy_setup = require_extra(cfg, "policy_setup")
        action_scale = require_extra(cfg, "action_scale")
        init_rng = require_extra(cfg, "octo_init_rng")
        model_type = cfg.checkpoint if cfg.checkpoint else f"octo-{variant}"
        impl = OctoInference(
            model_type=model_type,
            policy_setup=str(policy_setup),
            init_rng=int(init_rng),
            action_scale=float(action_scale),
        )
        return cls(cfg, impl=impl)

    def reset(self, ctx: EpisodeContext) -> None:
        if self._impl is not None:
            self._impl.reset(ctx.instruction)
        elif self._transport is not None:
            self._transport.reset({})

    def act(self, obs: Observation) -> ActionChunk:
        if self._transport is not None:
            payload = {
                "image": as_jsonable(require_image(obs, "primary")),
                "instruction": obs.instruction,
            }
            result = self._transport.infer(payload)
            actions = np.asarray(
                result.get("action", result.get("actions")), dtype=np.float32
            )
            return actions_to_chunk(actions, space=self._cfg.action_space)
        assert self._impl is not None
        raw, action = self._impl.step(require_image(obs, "primary"), obs.instruction)
        del raw
        vec = _simpler_dict_to_7d(action)
        return actions_to_chunk(vec.reshape(1, -1), space=self._cfg.action_space)

    def close(self) -> None:
        if self._transport is not None:
            self._transport.close()
        if self._impl is not None:
            closer = getattr(self._impl, "close", None)
            if callable(closer):
                closer()
