"""SpatialVLA adapters — in-process local and HTTP server variants."""

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
from dynisafety.policies.base import (
    ActionChunk,
    EpisodeContext,
    ModelConfig,
    Observation,
)
from dynisafety.policies.transports import Transport


class SpatialVLAAdapter:
    """In-process SpatialVLA (SimplerEnv). ``extra.repo_root`` required."""

    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg
        repo = require_extra(cfg, "repo_root")
        ensure_repo_on_path(str(repo))
        # Lazy heavy import
        from simpler_env.policies.spatialvla.spatialvla_model import (  # noqa: PLC0415
            SpatialVLAInference,
        )

        policy_setup = require_extra(cfg, "policy_setup")
        action_scale = require_extra(cfg, "action_scale")
        if not isinstance(action_scale, (int, float)) or isinstance(action_scale, bool):
            raise ValueError(f"extra.action_scale must be float, got {action_scale!r}")
        self._impl = SpatialVLAInference(
            saved_model_path=cfg.checkpoint,
            policy_setup=str(policy_setup),
            action_scale=float(action_scale),
        )

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> SpatialVLAAdapter:
        if cfg.transport != "local":
            raise ValueError("SpatialVLAAdapter requires transport='local'")
        return cls(cfg)

    def reset(self, ctx: EpisodeContext) -> None:
        self._impl.reset(ctx.instruction)

    def act(self, obs: Observation) -> ActionChunk:
        image = require_image(obs, "primary")
        raw, action = self._impl.step(image, obs.instruction)
        del raw
        vec = _simpler_dict_to_7d(action)
        return actions_to_chunk(vec.reshape(1, -1), space=self._cfg.action_space)

    def close(self) -> None:
        closer = getattr(self._impl, "close", None)
        if callable(closer):
            closer()


class SpatialVLAServerAdapter:
    """HTTP SpatialVLA server client (LIBERO / Simpler ``spatialvla_server``)."""

    def __init__(self, cfg: ModelConfig, transport: Transport) -> None:
        self._cfg = cfg
        self._transport = transport

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> SpatialVLAServerAdapter:
        if cfg.transport != "http":
            raise ValueError("SpatialVLAServerAdapter requires transport='http'")
        return cls(cfg, connect_transport(cfg))

    def reset(self, ctx: EpisodeContext) -> None:
        del ctx
        self._transport.reset({})

    def act(self, obs: Observation) -> ActionChunk:
        payload: dict[str, Any] = {
            "image": as_jsonable(require_image(obs, "primary")),
            "instruction": obs.instruction,
        }
        if "unnorm_key" in self._cfg.extra:
            payload["unnorm_key"] = self._cfg.extra["unnorm_key"]
        result = self._transport.infer(payload)
        actions = np.asarray(
            result.get("action", result.get("actions")), dtype=np.float32
        )
        # LIBERO SpatialVLA maps grip {0,1}→{-1,+1}
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.shape[-1] >= 7:
            g = actions[:, -1]
            if np.all((g >= 0.0) & (g <= 1.0)):
                actions = actions.copy()
                actions[:, -1] = np.where(g >= 0.5, 1.0, -1.0)
        return actions_to_chunk(actions, space=self._cfg.action_space)

    def close(self) -> None:
        self._transport.close()


def _simpler_dict_to_7d(action: dict[str, Any]) -> np.ndarray:
    wv = np.asarray(action["world_vector"], dtype=np.float32).reshape(-1)
    rot = np.asarray(action["rot_axangle"], dtype=np.float32).reshape(-1)
    grip = np.asarray(action["gripper"], dtype=np.float32).reshape(-1)
    return np.concatenate([wv[:3], rot[:3], grip[:1]]).astype(np.float32)
