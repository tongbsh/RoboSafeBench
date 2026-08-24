"""Xiaomi-Robotics-1 RoboCasa365 adapter (client-side processor + TCP serve)."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from PIL import Image

from dynisafety.policies.action_space import dim_for_space
from dynisafety.policies.adapters._common import (
    actions_to_chunk,
    connect_transport,
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
from dynisafety.policies.transports import Transport

_IMAGE_ORDER = ("left", "right", "wrist")


def _require_int_extra(cfg: ModelConfig, key: str, *, min_value: int = 1) -> int:
    value = require_extra(cfg, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"extra.{key} must be int, got {value!r}")
    if value < min_value:
        raise ValueError(f"extra.{key} must be >= {min_value}, got {value}")
    return value


def _require_float_extra(cfg: ModelConfig, key: str) -> float:
    value = require_extra(cfg, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"extra.{key} must be float, got {value!r}")
    return float(value)


def center_crop(image: np.ndarray, crop_ratio: float) -> Image.Image:
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if crop_ratio >= 1.0:
        return pil_image
    width, height = pil_image.size
    crop_width = max(1, int(width * crop_ratio))
    crop_height = max(1, int(height * crop_ratio))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = pil_image.crop((left, top, left + crop_width, top + crop_height))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return cropped.resize((width, height), resampling)


def sample_history(history: deque[np.ndarray], length: int, interval: int) -> np.ndarray:
    items = list(history)
    if not items:
        raise ValueError("Observation history cannot be empty")
    indices = [
        max(0, len(items) - 1 - (length - 1 - index) * interval)
        for index in range(length)
    ]
    return np.ascontiguousarray(np.stack([items[index] for index in indices], axis=0))


class XiaomiAdapter:
    """RoboCasa365 Xiaomi client: local AutoProcessor, remote ``deploy/server.py``."""

    def __init__(self, cfg: ModelConfig, transport: Transport) -> None:
        self._cfg = cfg
        self._transport = transport
        self._robot_type = require_extra(cfg, "robot_type")
        if not isinstance(self._robot_type, str) or not self._robot_type.strip():
            raise ValueError(f"extra.robot_type must be a non-empty str, got {self._robot_type!r}")
        self._robot_type = self._robot_type.strip()
        self._state_pad_dim = _require_int_extra(cfg, "state_pad_dim")
        self._obs_history = _require_int_extra(cfg, "obs_history")
        self._obs_interval = _require_int_extra(cfg, "obs_interval")
        self._replan_steps = _require_int_extra(cfg, "replan_steps")
        self._crop_ratio = _require_float_extra(cfg, "crop_ratio")
        if not 0.0 < self._crop_ratio <= 1.0:
            raise ValueError(f"extra.crop_ratio must be in (0, 1], got {self._crop_ratio}")
        self._processor: Any = None
        self._instruction = ""
        queue_len = (self._obs_history - 1) * self._obs_interval + 1
        self._image_queues: dict[str, deque[np.ndarray]] = {
            key: deque(maxlen=queue_len) for key in _IMAGE_ORDER
        }
        self._state_queue: deque[np.ndarray] = deque(maxlen=queue_len)
        self._action_plan: deque[np.ndarray] = deque()

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> XiaomiAdapter:
        if cfg.transport != "websocket":
            raise ValueError("XiaomiAdapter requires transport='websocket'")
        if cfg.extra.get("ws_codec") != "xiaomi":
            raise ValueError("XiaomiAdapter requires extra.ws_codec='xiaomi'")
        return cls(cfg, connect_transport(cfg))

    def reset(self, ctx: EpisodeContext) -> None:
        self._instruction = ctx.instruction
        for queue in self._image_queues.values():
            queue.clear()
        self._state_queue.clear()
        self._action_plan.clear()
        self._transport.reset(None)

    def act(self, obs: Observation) -> ActionChunk:
        for key in _IMAGE_ORDER:
            self._image_queues[key].append(np.ascontiguousarray(require_image(obs, key)))
        self._state_queue.append(np.asarray(require_state(obs), dtype=np.float32).reshape(-1))
        if not self._action_plan:
            chunk = self._infer(obs.instruction or self._instruction)
            if len(chunk) < self._replan_steps:
                raise RuntimeError(
                    f"Xiaomi returned {len(chunk)} actions, "
                    f"extra.replan_steps is {self._replan_steps}"
                )
            self._action_plan.extend(chunk[: self._replan_steps])
        action = self._action_plan.popleft()
        return actions_to_chunk(action, space=self._cfg.action_space)

    def close(self) -> None:
        self._transport.close()

    def _processor_or_load(self) -> Any:
        if self._processor is not None:
            return self._processor
        from transformers import AutoProcessor  # noqa: PLC0415

        processor = AutoProcessor.from_pretrained(
            self._cfg.checkpoint, trust_remote_code=True, use_fast=False
        )
        robot_types = processor.list_robot_types()
        if self._robot_type not in robot_types:
            raise ValueError(
                f"Robot type {self._robot_type!r} is missing from the checkpoint. "
                f"Available robot types: {robot_types}."
            )
        self._processor = processor
        return processor

    def _build_messages(
        self, image_history: dict[str, np.ndarray], instruction: str
    ) -> list[dict[str, Any]]:
        videos = {
            key: [center_crop(frame, self._crop_ratio) for frame in image_history[key]]
            for key in _IMAGE_ORDER
        }
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Left camera: "},
                    {"type": "video", "video": videos["left"]},
                    {"type": "text", "text": "\nRight camera: "},
                    {"type": "video", "video": videos["right"]},
                    {"type": "text", "text": "\nWrist camera: "},
                    {"type": "video", "video": videos["wrist"]},
                    {
                        "type": "text",
                        "text": f"\n\nGenerate robot actions for the task:\n{instruction} /no_cot",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<cot></cot>"}],
            },
        ]

    def _infer(self, instruction: str) -> np.ndarray:
        processor = self._processor_or_load()
        states = sample_history(self._state_queue, self._obs_history, self._obs_interval)
        images = {
            key: sample_history(queue, self._obs_history, self._obs_interval)
            for key, queue in self._image_queues.items()
        }
        state = np.zeros((1, states.shape[0], self._state_pad_dim), dtype=np.float32)
        state[0, :, : states.shape[-1]] = states
        inputs = processor.apply_chat_template(
            self._build_messages(images, instruction),
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            do_resize=False,
            state=state,
            robot_type=self._robot_type,
        )
        request = dict(inputs)
        request["task_id"] = self._robot_type
        result = self._transport.infer(request)
        if "actions" not in result:
            raise KeyError(f"xiaomi response missing 'actions'; keys={sorted(result)}")
        raw = processor.decode_action(result["actions"], robot_type=self._robot_type)
        action_dim = dim_for_space(self._cfg.action_space)
        actions = raw[0, :, :action_dim].float().cpu().numpy()
        return np.asarray(actions, dtype=np.float32)
