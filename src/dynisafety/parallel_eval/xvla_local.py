"""X-VLA load + batch forward with caller-supplied init noise.

Upstream ``generate_actions`` draws ``x1 = torch.randn(B, T, D)`` inside the
model, so the same observation changes when the dynamic batch size changes
(plan 0.13). This module copies that denoising loop and takes ``x1`` from the
caller so each slot is a function of its own ``rng_key`` only.

YAML must add ``extra.parallel_eval.rng_seed`` (required, no Python default).
Env hashes it with ``(episode_uid, step)`` via ``mix_rng_key`` into
``ObsMsg.rng_key``; Infer stays stateless.

``XVLAProcessor`` treats a flat ``List[PIL]`` as one sample's views (B=1).
Batch construction therefore always passes nested ``List[List[PIL]]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from dynisafety.parallel_eval.protocol import ACTION_DIM, PROPRIO_DIM, ObsMsg
from dynisafety.parallel_eval.protocol import mix_rng_key as mix_rng_key
from dynisafety.policies.adapters._common import ensure_repo_on_path

_ALLOWED_TORCH_DTYPE = "float32"
_PROCESSOR_KEYS = ("input_ids", "image_input", "image_mask")


@dataclass(frozen=True)
class XvlaRuntime:
    """Loaded Infer-side objects. Tests inject a fake instead of 879M weights."""

    model: Any
    processor: Any
    device: str
    dtype: torch.dtype


def slot_noise(
    rng_key: int,
    *,
    num_actions: int,
    dim_action: int,
) -> np.ndarray:
    """``(1, T, D)`` float32 noise. Same key ⇒ same tensor, regardless of B."""
    if isinstance(rng_key, bool) or not isinstance(rng_key, int):
        raise TypeError(f"rng_key must be int, got {type(rng_key).__name__}")
    if isinstance(num_actions, bool) or not isinstance(num_actions, int) or num_actions < 1:
        raise ValueError(f"num_actions must be int >= 1, got {num_actions!r}")
    if isinstance(dim_action, bool) or not isinstance(dim_action, int) or dim_action < 1:
        raise ValueError(f"dim_action must be int >= 1, got {dim_action!r}")
    rng = np.random.default_rng(rng_key)
    return rng.standard_normal((1, num_actions, dim_action)).astype(np.float32)


def stack_slot_noise(
    messages: Sequence[ObsMsg],
    *,
    num_actions: int,
    dim_action: int,
) -> np.ndarray:
    """Stack per-slot ``slot_noise`` in message order to ``(B, T, D)``."""
    if not messages:
        raise ValueError("stack_slot_noise requires at least one ObsMsg")
    parts = [
        slot_noise(msg.rng_key, num_actions=num_actions, dim_action=dim_action)
        for msg in messages
    ]
    return np.concatenate(parts, axis=0)


def unique_denoising_steps(messages: Sequence[ObsMsg]) -> int:
    """One ``steps`` value for the batch. Mixing would silently change numerics."""
    if not messages:
        raise ValueError("unique_denoising_steps requires at least one ObsMsg")
    values = {msg.denoising_steps for msg in messages}
    if len(values) != 1:
        raise ValueError(
            f"ObsMsg.denoising_steps must be identical in a batch, got {sorted(values)}"
        )
    return next(iter(values))


def nested_pil_views(messages: Sequence[ObsMsg]) -> list[list[Image.Image]]:
    """One inner list per sample.

    A flat ``List[PIL]`` is rewritten to ``[views]`` inside ``XVLAProcessor``
    (B=1, multi-view). ``[views, views]`` is the only shape that is B=2.
    """
    if not messages:
        raise ValueError("nested_pil_views requires at least one ObsMsg")
    batch: list[list[Image.Image]] = []
    for msg in messages:
        views = [Image.fromarray(np.ascontiguousarray(image)) for image in msg.images]
        batch.append(views)
    return batch


def load_xvla(
    *,
    repo_root: str,
    model_path: str,
    processor_path: str | None,
    device: str,
    torch_dtype: str,
) -> XvlaRuntime:
    """Load like ``deploy.py``. ``torch_dtype`` other than float32 is rejected."""
    if torch_dtype != _ALLOWED_TORCH_DTYPE:
        raise ValueError(
            f"torch_dtype must be {_ALLOWED_TORCH_DTYPE!r} because upstream "
            f"deploy.py hard-codes float32; changing it would change numerics. "
            f"got {torch_dtype!r}"
        )
    _require_path(repo_root, "repo_root")
    _require_path(model_path, "model_path")
    _require_path(device, "device")
    ensure_repo_on_path(repo_root)
    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor

    proc_path = model_path if processor_path is None else processor_path
    _require_path(proc_path, "processor_path")
    processor = XVLAProcessor.from_pretrained(proc_path)
    model = (
        XVLA.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        .to(device)
        .to(torch.float32)
    )
    model.eval()
    return XvlaRuntime(
        model=model,
        processor=processor,
        device=device,
        dtype=torch.float32,
    )


@torch.no_grad()
def generate_actions_with_noise(
    model: Any,
    *,
    input_ids: torch.Tensor,
    image_input: torch.Tensor,
    image_mask: torch.Tensor,
    domain_id: torch.Tensor,
    proprio: torch.Tensor,
    x1: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """Same loop as ``XVLA.generate_actions`` with caller-supplied ``x1``.

    Copied from ``external/model_repos/X-VLA/models/modeling_xvla.py``
    (``generate_actions``). Internal ``torch.randn`` is replaced by ``x1`` so
    dynamic B cannot reshuffle the RNG stream per slot.
    """
    if not isinstance(x1, torch.Tensor):
        raise TypeError(f"x1 must be a torch.Tensor, got {type(x1)!r}")
    batch = int(input_ids.shape[0])
    expected = (batch, int(model.num_actions), int(model.action_space.dim_action))
    if tuple(x1.shape) != expected:
        raise ValueError(
            f"x1.shape must be {expected} (B, num_actions, dim_action), "
            f"got {tuple(x1.shape)}"
        )
    if int(proprio.shape[0]) != batch:
        raise ValueError(
            f"proprio batch {int(proprio.shape[0])} != input_ids batch {batch}"
        )
    if int(domain_id.shape[0]) != batch:
        raise ValueError(
            f"domain_id batch {int(domain_id.shape[0])} != input_ids batch {batch}"
        )

    model.eval()
    enc = model.forward_vlm(input_ids, image_input, image_mask)
    x1 = x1.to(device=proprio.device, dtype=proprio.dtype)
    action = torch.zeros_like(x1)
    steps_n = max(1, int(steps))
    for i in range(steps_n, 0, -1):
        t = torch.full((batch,), i / steps_n, device=proprio.device, dtype=proprio.dtype)
        x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
        proprio_m, x_t_m = model.action_space.preprocess(proprio, x_t)
        action = model.transformer(
            domain_id=domain_id,
            action_with_noise=x_t_m,
            proprio=proprio_m,
            t=t,
            **enc,
        )
    return model.action_space.postprocess(action)


def forward_obs_batch(
    runtime: XvlaRuntime,
    messages: Sequence[ObsMsg],
    *,
    x1: np.ndarray | torch.Tensor,
) -> list[np.ndarray]:
    """Encode N ``ObsMsg`` as one nested batch and return N ``(T, 20)`` chunks.

    Output is numpy so nothing torch-typed can reach the wire.
    """
    if not messages:
        raise ValueError("forward_obs_batch requires at least one ObsMsg")
    steps = unique_denoising_steps(messages)
    nested = nested_pil_views(messages)
    texts = [msg.instruction for msg in messages]
    encoded = runtime.processor(nested, texts)
    missing = [key for key in _PROCESSOR_KEYS if key not in encoded]
    if missing:
        raise KeyError(f"processor output missing keys {missing}")

    device = torch.device(runtime.device)
    dtype = runtime.dtype
    input_ids = _to_model(encoded["input_ids"], device=device, dtype=dtype)
    image_input = _to_model(encoded["image_input"], device=device, dtype=dtype)
    image_mask = _to_model(encoded["image_mask"], device=device, dtype=dtype)
    proprio = _to_model(
        np.stack([msg.proprio for msg in messages], axis=0),
        device=device,
        dtype=dtype,
    )
    if tuple(proprio.shape) != (len(messages), PROPRIO_DIM):
        raise ValueError(
            f"stacked proprio must be {(len(messages), PROPRIO_DIM)}, got {tuple(proprio.shape)}"
        )
    domain_id = torch.as_tensor(
        [msg.domain_id for msg in messages],
        dtype=torch.long,
        device=device,
    )
    x1_tensor = _as_x1_tensor(x1)
    actions = generate_actions_with_noise(
        runtime.model,
        input_ids=input_ids,
        image_input=image_input,
        image_mask=image_mask,
        domain_id=domain_id,
        proprio=proprio,
        x1=x1_tensor,
        steps=steps,
    )
    out = actions.detach().to(dtype=torch.float32).cpu().numpy()
    if out.ndim != 3 or out.shape[0] != len(messages) or out.shape[2] != ACTION_DIM:
        raise ValueError(
            f"expected actions {(len(messages), 'T', ACTION_DIM)}, got {out.shape}"
        )
    return [np.array(out[index], dtype=np.float32, copy=True) for index in range(out.shape[0])]


def _as_x1_tensor(x1: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(x1, torch.Tensor):
        return x1
    if isinstance(x1, np.ndarray):
        if x1.dtype != np.float32:
            raise TypeError(f"x1 numpy dtype must be float32, got {x1.dtype}")
        return torch.as_tensor(x1)
    raise TypeError(f"x1 must be numpy.ndarray or torch.Tensor, got {type(x1)!r}")


def _to_model(value: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.is_floating_point():
        return tensor.to(device=device, dtype=dtype)
    return tensor.to(device=device)


def _require_path(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty str, got {value!r}")
