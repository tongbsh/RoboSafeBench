"""UniVLA in-process (latent + action decoder) adapter."""

from __future__ import annotations

from typing import Any

from dynisafety.policies.adapters._common import (
    actions_to_chunk,
    ensure_repo_on_path,
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


class UniVLALocalAdapter:
    """Local UniVLA (LIBERO ``univla_local`` / Simpler twin).

    Required ``extra``: ``repo_root``, ``unnorm_key``, ``center_crop``,
    ``window_size``, ``action_decoder_path``.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg
        ensure_repo_on_path(str(require_extra(cfg, "repo_root")))
        from dynisafety.policies._libero_impl.univla_local_policy import (  # noqa: PLC0415
            UniVLALocalPolicy,
        )

        ns = _Shim(
            pretrained_checkpoint=cfg.checkpoint,
            unnorm_key=str(require_extra(cfg, "unnorm_key")),
            center_crop=bool(require_extra(cfg, "center_crop")),
            window_size=int(require_extra(cfg, "window_size")),
            action_decoder_path=str(require_extra(cfg, "action_decoder_path")),
        )
        for key in ("attn_implementation", "task_suite_name"):
            if key in cfg.extra:
                setattr(ns, key, cfg.extra[key])
        self._impl = UniVLALocalPolicy(ns)

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> UniVLALocalAdapter:
        if cfg.transport != "local":
            raise ValueError("UniVLALocalAdapter requires transport='local'")
        return cls(cfg)

    def reset(self, ctx: EpisodeContext) -> None:
        del ctx
        reset = getattr(self._impl, "reset", None)
        if callable(reset):
            reset()

    def act(self, obs: Observation) -> ActionChunk:
        legacy_obs = {
            "full_image": require_image(obs, "primary"),
            "state": require_state(obs),
        }
        steps = self._impl.predict_actions(legacy_obs, obs.instruction)
        return actions_to_chunk(steps, space=self._cfg.action_space)

    def close(self) -> None:
        closer = getattr(self._impl, "close", None)
        if callable(closer):
            closer()


class _Shim:
    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
