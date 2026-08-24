"""UniVLA local (in-process) policy for DyniSafety-LIBERO.

Mirrors ``external/model_repos/UniVLA/experiments/robot/libero/run_libero_eval.py`` latent
action + temporal :class:`ActionDecoder` path. Intended for the same layout as
``openvla-oft`` eval: :func:`~dynisafety_libero.policies.create_policy` +
``run_dynisafety_eval.run_task``.

Path resolution
---------------
Default: ``<workspace>/external/model_repos/UniVLA``. Override with ``UNIVLA_ROOT``.

Environment
-----------
UniVLA depends on TensorFlow (image resize), Prismatic/OpenVLA stack, etc. The
project conda docs suggest a dedicated ``univla`` env; if your ``openvla-oft``
env already satisfies these versions, local inference can run there without an
HTTP server. Set ``UNIVLA_ATTN_IMPLEMENTATION=sdpa`` when Flash-Attention is
unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from .base_policy import VLAPolicy

logger = logging.getLogger(__name__)

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
_UNIVLA_ROOT = Path(os.environ.get("UNIVLA_ROOT", str(_WORKSPACE_ROOT / "external" / "model_repos" / "UniVLA")))

_SAFELIBERO_TO_UNNORM_KEY = {
    "safelibero_spatial": "libero_spatial_no_noops",
    "safelibero_object": "libero_object_no_noops",
    "safelibero_goal": "libero_goal_no_noops",
    "safelibero_long": "libero_10_no_noops",
}

LATENT_ACTION_DETOKENIZE = [f"<ACT_{i}>" for i in range(32)]


def _ensure_univla_on_path() -> None:
    if not _UNIVLA_ROOT.exists():
        raise RuntimeError(
            f"UniVLA root not found at '{_UNIVLA_ROOT}'. "
            "Clone the repo or set UNIVLA_ROOT."
        )
    r = str(_UNIVLA_ROOT)
    if r not in sys.path:
        sys.path.insert(0, r)


_ensure_univla_on_path()

from experiments.robot.libero.libero_utils import resize_image  # noqa: E402
from experiments.robot.openvla_utils import DEVICE, get_processor, get_vla_latent_action  # noqa: E402
from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action  # noqa: E402
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig  # noqa: E402
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction  # noqa: E402
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor  # noqa: E402
from prismatic.models.policy.transformer_utils import MAPBlock  # noqa: E402


class ActionDecoderHead(torch.nn.Module):
    def __init__(self, window_size=5):
        super().__init__()
        self.latent_action_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=512, n_heads=8)
        self.visual_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=512, n_heads=8)
        self.proj = nn.Sequential(
            nn.Linear(512, 7 * window_size),
            nn.Tanh(),
        )

    def forward(self, latent_action_tokens, visual_embed):
        latent_action_tokens = latent_action_tokens[:, -4:]
        visual_embed = self.visual_pool(visual_embed)
        action = self.proj(self.latent_action_pool(latent_action_tokens, init_embed=visual_embed))
        return action


class ActionDecoder(nn.Module):
    def __init__(self, window_size=5):
        super().__init__()
        self.net = ActionDecoderHead(window_size=window_size)
        self.temporal_size = window_size
        self.temporal_mask = torch.flip(
            torch.triu(torch.ones(self.temporal_size, self.temporal_size, dtype=torch.bool)), dims=[1]
        ).numpy()
        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], 7))
        self.action_buffer_mask = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_)
        balancing_factor = 0.1
        self.temporal_weights = np.array([np.exp(-1 * balancing_factor * i) for i in range(self.temporal_size)])[
            :, None
        ]

    def reset_buffers(self):
        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], 7))
        self.action_buffer_mask = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_)

    def forward(self, latent_actions, visual_embed, mask, action_low, action_high):
        pred_action = self.net(latent_actions.to(torch.float), visual_embed.to(torch.float)).reshape(
            -1, self.temporal_size, 7
        )
        pred_action = np.array(pred_action.tolist())
        self.action_buffer[1:, :, :] = self.action_buffer[:-1, :, :]
        self.action_buffer_mask[1:, :] = self.action_buffer_mask[:-1, :]
        self.action_buffer[:, :-1, :] = self.action_buffer[:, 1:, :]
        self.action_buffer_mask[:, :-1] = self.action_buffer_mask[:, 1:]
        self.action_buffer_mask = self.action_buffer_mask * self.temporal_mask
        self.action_buffer[0] = pred_action
        self.action_buffer_mask[0] = np.array([True] * self.temporal_mask.shape[0], dtype=np.bool_)
        action_prediction = np.sum(
            self.action_buffer[:, 0, :] * self.action_buffer_mask[:, 0:1] * self.temporal_weights, axis=0
        ) / np.sum(self.action_buffer_mask[:, 0:1] * self.temporal_weights)
        action_prediction = np.where(
            mask,
            0.5 * (action_prediction + 1) * (action_high - action_low) + action_low,
            action_prediction,
        )
        return action_prediction


def _load_univla_vla(pretrained_checkpoint: str, load_in_8bit: bool, load_in_4bit: bool) -> torch.nn.Module:
    attn = os.environ.get("UNIVLA_ATTN_IMPLEMENTATION", "flash_attention_2")
    logger.info("Loading UniVLA VLA with attn_implementation=%s", attn)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    vla = AutoModelForVision2Seq.from_pretrained(
        pretrained_checkpoint,
        attn_implementation=attn,
        torch_dtype=torch.bfloat16,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if not load_in_8bit and not load_in_4bit:
        vla = vla.to(DEVICE)

    stats_path = os.path.join(pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(stats_path):
        with open(stats_path, "r") as f:
            vla.norm_stats = json.load(f)
    else:
        logger.warning("dataset_statistics.json not found next to checkpoint.")

    return vla


class UniVLALocalPolicy(VLAPolicy):
    """UniVLA with latent tokens + learned action decoder (one action per ``predict_actions``)."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        ckpt = os.path.abspath(str(cfg.pretrained_checkpoint))
        assert ckpt, "pretrained_checkpoint must not be empty"
        if "image_aug" in str(cfg.pretrained_checkpoint):
            assert cfg.center_crop, "Expect center_crop=True for checkpoints trained with image augmentations."

        dec_path = getattr(cfg, "action_decoder_path", "") or ""
        if not dec_path:
            dec_path = os.path.join(ckpt, "action_decoder.pt")
        dec_path = os.path.abspath(str(dec_path))
        if not os.path.isfile(dec_path):
            raise FileNotFoundError(f"action_decoder.pt not found: {dec_path}")

        self._window_size = int(getattr(cfg, "univla_window_size", 12))
        self.model = _load_univla_vla(ckpt, bool(cfg.load_in_8bit), bool(cfg.load_in_4bit))
        self.processor = get_processor(SimpleNamespace(pretrained_checkpoint=ckpt))

        self.action_decoder = ActionDecoder(window_size=self._window_size)
        self.action_decoder.net.load_state_dict(torch.load(dec_path, map_location="cpu"))
        self.action_decoder.eval().to(DEVICE)

        self._resolve_unnorm_key()
        self._prev_hist_action: list[str] = [""]

    def reset(self) -> None:
        self.action_decoder.reset_buffers()
        self._prev_hist_action = [""]

    @property
    def action_chunk_size(self) -> int:
        return 1

    def predict_actions(self, obs: dict, task_description: str) -> List[np.ndarray]:
        img = np.asarray(obs["full_image"], dtype=np.uint8)
        img224 = resize_image(img, (224, 224))
        state = np.asarray(obs["state"], dtype=np.float32)
        observation = {"full_image": img224, "state": state}

        hist_action = self._prev_hist_action[-1]
        latent_action, visual_embed, generated_ids = get_vla_latent_action(
            self.model,
            self.processor,
            str(self.cfg.pretrained_checkpoint),
            observation,
            task_description,
            self.cfg.unnorm_key,
            center_crop=bool(self.cfg.center_crop),
            hist_action=hist_action,
        )

        hist_str = ""
        for latent_action_ids in generated_ids[0]:
            tid = int(latent_action_ids.item())
            hist_str += LATENT_ACTION_DETOKENIZE[tid - 32001]
        self._prev_hist_action.append(hist_str)

        action_norm_stats = self.model.get_action_stats(self.cfg.unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high = np.array(action_norm_stats["q99"])
        action_low = np.array(action_norm_stats["q01"])

        action = self.action_decoder(latent_action, visual_embed, mask, action_low, action_high)
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
        return [action.astype(np.float32)]

    def _resolve_unnorm_key(self) -> None:
        key = self.cfg.task_suite_name
        if hasattr(self.model, "norm_stats") and self.model.norm_stats:
            if key in self.model.norm_stats:
                self.cfg.unnorm_key = key
                return
            if f"{key}_no_noops" in self.model.norm_stats:
                self.cfg.unnorm_key = f"{key}_no_noops"
                return
            mapped = _SAFELIBERO_TO_UNNORM_KEY.get(key)
            if mapped and mapped in self.model.norm_stats:
                logger.warning("Using UniVLA unnorm key '%s' for task suite '%s'.", mapped, key)
                self.cfg.unnorm_key = mapped
                return
        raise AssertionError(
            f"Could not resolve unnorm_key for task_suite_name={key}. "
            f"Available stats keys: {list(getattr(self.model, 'norm_stats', {}) or {})}"
        )
