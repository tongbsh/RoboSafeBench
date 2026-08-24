"""OpenVLA-OFT RLinf local adapter — RLinf HF weights via the standard OFT loader.

Paper runs used the same ``openvla-oft`` env and ``robot_utils.get_model`` path as
``openvla_oft``; the checkpoint directory already bundles ``modeling_prismatic.py``.
This adapter keeps a distinct model id while delegating load/act to
``OpenVLAOFTAdapter`` (no ``rlinf.models`` / DTensor dependency).
"""

from __future__ import annotations

from dynisafety.policies.adapters.openvla_oft import OpenVLAOFTAdapter
from dynisafety.policies.base import (
    ActionChunk,
    EpisodeContext,
    ModelConfig,
    Observation,
)


class OpenVLAOFTRLinfAdapter:
    """In-process RLinf GRPO OpenVLA-OFT weights loaded like vanilla OFT.

    Required ``extra`` matches ``OpenVLAOFTAdapter`` (``repo_root``,
    ``model_family``, ``unnorm_key``, ``center_crop``, ``use_proprio``,
    ``use_l1_regression``, ``use_diffusion``, ``use_film``,
    ``num_images_in_input``, ``load_in_8bit``, ``load_in_4bit``). Paper口径:
    single primary image, ``use_proprio=false``, action head baked into the
    checkpoint (``use_l1_regression=false``).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg
        self._inner = OpenVLAOFTAdapter.from_config(cfg)

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> OpenVLAOFTRLinfAdapter:
        if cfg.transport != "local":
            raise ValueError("OpenVLAOFTRLinfAdapter requires transport='local'")
        return cls(cfg)

    def reset(self, ctx: EpisodeContext) -> None:
        self._inner.reset(ctx)

    def act(self, obs: Observation) -> ActionChunk:
        return self._inner.act(obs)

    def close(self) -> None:
        self._inner.close()
