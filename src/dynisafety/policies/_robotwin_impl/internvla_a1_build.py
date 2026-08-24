"""InternVLA-A1 checkpoint load + transforms (extracted from legacy inference.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import load_json
from lerobot.policies.InternVLA_A1_3B.modeling_internvla_a1 import QwenA1Config, QwenA1Policy
from lerobot.policies.InternVLA_A1_3B.transform_internvla_a1 import Qwen3_VLProcessorTransformFn
from lerobot.transforms.core import (
    NormalizeTransformFn,
    RemapImageKeyTransformFn,
    ResizeImagesWithPadFn,
    UnNormalizeTransformFn,
    compose,
)
from lerobot.utils.constants import OBS_IMAGES


def resolve_ckpt_dir(ckpt_path: str | Path) -> Path:
    ckpt_str = str(ckpt_path)
    local_dir = Path(ckpt_str).expanduser()
    if local_dir.exists():
        return local_dir.resolve()
    snapshot_dir = snapshot_download(repo_id=ckpt_str)
    return Path(snapshot_dir)


def build_policy_and_transforms(
    ckpt_path: str | Path,
    stats_key: str,
    resize_size: int,
    dtype: torch.dtype,
) -> tuple[Any, Any, Any]:
    ckpt_dir = resolve_ckpt_dir(ckpt_path)
    config = PreTrainedConfig.from_pretrained(ckpt_dir)
    if not isinstance(config, QwenA1Config):
        raise ValueError(f"Expected QwenA1Config, got {type(config)}")

    policy = QwenA1Policy.from_pretrained(config=config, pretrained_name_or_path=ckpt_dir)
    policy.cuda().to(dtype).eval()

    stats = load_json(ckpt_dir / "stats.json")[stats_key]
    stat_keys = ["min", "max", "mean", "std"]

    state_concat = {k: np.asarray(stats["observation.state"][k]) for k in stat_keys}
    state_stat = {"observation.state": state_concat}

    action_concat = {k: np.asarray(stats["action"][k]) for k in stat_keys}
    action_stat = {"action": action_concat}

    unnormalize_fn = UnNormalizeTransformFn(
        selected_keys=["action"],
        mode="mean_std",
        norm_stats=action_stat,
    )

    image_keys = [f"{OBS_IMAGES}.image{i}" for i in range(3)]

    input_transforms = compose(
        [
            ResizeImagesWithPadFn(height=resize_size, width=resize_size),
            RemapImageKeyTransformFn(mapping={k: k for k in image_keys}),
            Qwen3_VLProcessorTransformFn(),
            NormalizeTransformFn(
                selected_keys=["observation.state"], norm_stats=state_stat
            ),
        ]
    )

    return policy, input_transforms, unnormalize_fn
