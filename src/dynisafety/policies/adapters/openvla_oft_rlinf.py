"""OpenVLA-OFT RLinf local adapter — RLinf HF weights via the standard OFT loader.

Paper runs used the same ``openvla-oft`` env and ``robot_utils.get_model`` path as
``openvla_oft``; the checkpoint directory already bundles ``modeling_prismatic.py``.
This adapter keeps a distinct model id while delegating load to
``OpenVLAOFTAdapter`` (no ``rlinf.models`` / DTensor dependency).

Action decode: vanilla OFT ``predict_action`` hardcodes ``do_sample=False`` and
never reaches the GRPO sampling path. Official RLinf LIBERO eval uses
``generate_action_verl`` with ``do_sample=True`` and ``temperature=1.6``. The
checkpoint README also lists ``top_p=1.0``; ``_verl_discrete_prediction`` has
no nucleus filter, so ``top_p=1.0`` is a no-op on that softmax. When YAML
sets those extras, this adapter calls ``generate_action_verl`` instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dynisafety.policies.adapters._common import (
    actions_to_chunk,
    require_extra,
    require_image,
)
from dynisafety.policies.adapters.openvla_oft import OpenVLAOFTAdapter
from dynisafety.policies.base import (
    ActionChunk,
    EpisodeContext,
    ModelConfig,
    Observation,
)

# Matches OpenVLA ``predict_action`` empty-token insert (tokenizer id 29871).
_EMPTY_TOKEN_ID = 29871
_TOP_P_OFFICIAL = 1.0


def parse_rlinf_sampling_extras(cfg: ModelConfig) -> tuple[bool, float, float]:
    """Read official RLinf decode extras. ``top_p`` must be 1.0 when sampling."""
    do_sample = require_extra(cfg, "do_sample")
    if not isinstance(do_sample, bool):
        raise ValueError(f"extra.do_sample must be bool, got {do_sample!r}")
    temperature = require_extra(cfg, "temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError(f"extra.temperature must be a number, got {temperature!r}")
    temperature_f = float(temperature)
    if do_sample and temperature_f <= 0.0:
        raise ValueError(
            f"extra.temperature must be > 0 when do_sample=true, got {temperature_f}"
        )
    top_p = require_extra(cfg, "top_p")
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
        raise ValueError(f"extra.top_p must be a number, got {top_p!r}")
    top_p_f = float(top_p)
    if not 0.0 < top_p_f <= 1.0:
        raise ValueError(f"extra.top_p must be in (0, 1], got {top_p_f}")
    if do_sample and abs(top_p_f - _TOP_P_OFFICIAL) > 1e-9:
        raise ValueError(
            "RLinf generate_action_verl has no nucleus filter; official eval "
            f"uses top_p=1.0 (softmax over the 256 action bins). got {top_p_f}"
        )
    return bool(do_sample), temperature_f, top_p_f


def episode_sampling_seed(ctx: EpisodeContext) -> int:
    """Deterministic CUDA/CPU seed for one sampled episode (signed 32-bit)."""
    mix = (
        int(ctx.seed)
        + 1_000_003 * int(ctx.episode_index)
        + 10_007 * int(ctx.task_id)
        + 1009 * int(ctx.level)
    )
    return mix % (2**31 - 1)


def verl_output_to_chunk(action: Any) -> np.ndarray:
    """``generate_action_verl`` returns ``[B, T, 7]`` or ``[T, 7]``; coerce to ``[T, 7]``."""
    arr = np.asarray(action, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(
                f"generate_action_verl batch dim must be 1 for local eval, got {arr.shape}"
            )
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"expected action chunk [T, 7], got {arr.shape}")
    return arr


class OpenVLAOFTRLinfAdapter:
    """In-process RLinf GRPO OpenVLA-OFT weights loaded like vanilla OFT.

    Required ``extra`` matches ``OpenVLAOFTAdapter`` plus ``do_sample``,
    ``temperature``, and ``top_p`` (RLinf official eval: true / 1.6 / 1.0).
    Paper口径: single primary image, ``use_proprio=false``, action head
    baked into the checkpoint (``use_l1_regression=false``).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg
        self._inner = OpenVLAOFTAdapter.from_config(cfg)
        do_sample, temperature_f, top_p_f = parse_rlinf_sampling_extras(cfg)
        self._do_sample = do_sample
        self._temperature = temperature_f
        self._top_p = top_p_f
        if do_sample and not hasattr(self._inner._model, "generate_action_verl"):
            raise TypeError(
                "RLinf sampling requires checkpoint modeling_prismatic.generate_action_verl; "
                f"loaded type is {type(self._inner._model)!r}"
            )
        print(
            "[openvla_oft_rlinf] unnorm_key=%s do_sample=%s temperature=%s top_p=%s"
            % (
                self._inner._ns.unnorm_key,
                self._do_sample,
                self._temperature,
                self._top_p,
            ),
            flush=True,
        )

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> OpenVLAOFTRLinfAdapter:
        if cfg.transport != "local":
            raise ValueError("OpenVLAOFTRLinfAdapter requires transport='local'")
        return cls(cfg)

    def reset(self, ctx: EpisodeContext) -> None:
        self._inner.reset(ctx)
        if not self._do_sample:
            return
        import torch

        seed = episode_sampling_seed(ctx)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def act(self, obs: Observation) -> ActionChunk:
        if not self._do_sample:
            return self._inner.act(obs)
        return self._act_sampled(obs)

    def _act_sampled(self, obs: Observation) -> ActionChunk:
        import torch
        from experiments.robot.openvla_utils import (  # noqa: PLC0415
            DEVICE,
            prepare_images_for_vla,
        )

        inner = self._inner
        ns = inner._ns
        primary = inner._resize_image_for_policy(
            require_image(obs, "primary"), inner._resize
        )
        images = prepare_images_for_vla([primary], ns)
        prompt = (
            f"In: What action should the robot take to {obs.instruction.lower()}?\nOut:"
        )
        inputs = inner._processor(prompt, images[0]).to(DEVICE, dtype=torch.bfloat16)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        if not torch.all(input_ids[:, -1] == _EMPTY_TOKEN_ID):
            empty = torch.tensor(
                [[_EMPTY_TOKEN_ID]], device=input_ids.device, dtype=input_ids.dtype
            )
            input_ids = torch.cat((input_ids, empty), dim=1)
            ones = torch.ones(
                (attention_mask.shape[0], 1),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat((attention_mask, ones), dim=1)

        with torch.inference_mode():
            raw, _ = inner._model.generate_action_verl(
                input_ids=input_ids,
                pixel_values=inputs["pixel_values"],
                attention_mask=attention_mask,
                unnorm_key=ns.unnorm_key,
                do_sample=True,
                temperature=self._temperature,
                # Unpadded B=1: sentinel that cannot appear in token ids, so the
                # RLinf pad-to-end shuffle is a no-op (same layout as OFT).
                padding_idx=-1,
            )
        arr = verl_output_to_chunk(raw)
        out = []
        for row in arr:
            a = inner._normalize(row, binarize=True)
            a = inner._invert(a)
            out.append(a.astype(np.float32))
        return actions_to_chunk(np.stack(out, axis=0), space=self._cfg.action_space)

    def close(self) -> None:
        self._inner.close()
