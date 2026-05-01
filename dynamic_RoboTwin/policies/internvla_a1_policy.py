"""InternVLA-A1 in-process policy adapter for DyniSafety-RoboTwin.

This module reproduces the per-step inference logic from the vendored
``dynamic_RoboTwin/inference.py`` (the ``infer_once`` loop)
behind the :class:`~policies.base_policy.RoboTwinPolicy` interface. The
heavy lifting - building the QwenA1 policy + transforms - lives in
``build_policy_and_transforms`` of that file; we simply import and reuse it.

Action stream contract
----------------------
The model emits joint-position deltas. They are unnormalized, summed with
the initial action (delta mode), and dispatched to RoboTwin via
``task_env.take_action(action, action_type='qpos')`` exactly once per
simulator step. Action chunks are buffered locally in a :class:`deque` so
that a single model forward serves ``infer_horizon`` steps.
"""

from __future__ import annotations

import logging
import sys
from collections import deque
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from policies import register_policy
from policies.base_policy import RoboTwinPolicy

logger = logging.getLogger(__name__)


def _ensure_internvla_on_path() -> Path:
    """Make the vendored ``dynamic_RoboTwin/inference.py`` importable."""
    here = Path(__file__).resolve()
    dynamic_robotwin_root = here.parents[1]  # dynamic_RoboTwin/
    if str(dynamic_robotwin_root) not in sys.path:
        sys.path.insert(0, str(dynamic_robotwin_root))
    return dynamic_robotwin_root


@register_policy("internvla_a1")
class InternVLAA1Policy(RoboTwinPolicy):
    """Wraps the QwenA1 policy + transforms used by InternVLA-A1.

    Constructor arguments mirror the relevant fields of
    ``dynamic_RoboTwin/inference.py:InferenceArgs`` so existing scripts can
    pass them through verbatim.
    """

    name = "internvla_a1"
    action_type = "qpos"

    def __init__(
        self,
        policy: Any,
        input_transforms: Any,
        unnormalize_fn: Any,
        *,
        dtype: torch.dtype,
        infer_horizon: int = 30,
        action_horizon_size: int = 50,
        image_history_interval: int = 15,
        action_mode: str = "delta",
        decode_image_flag: bool = False,
        robot_type: tuple[int, ...] = (6, 1, 6, 1),
    ) -> None:
        self._policy = policy
        self._input_transforms = input_transforms
        self._unnormalize_fn = unnormalize_fn
        self._dtype = dtype
        self.infer_horizon = int(infer_horizon)
        self.action_horizon_size = int(action_horizon_size)
        self.image_history_interval = int(image_history_interval)
        self.action_mode = str(action_mode)
        self.decode_image_flag = bool(decode_image_flag)
        self.robot_type = tuple(int(v) for v in robot_type)

        action_dim = sum(self.robot_type)
        self.action_dim = action_dim
        self.left_gripper_idx = sum(self.robot_type[0:2]) - 1
        self.right_gripper_idx = sum(self.robot_type[0:4]) - 1

        self._action_plan: deque = deque([], maxlen=self.action_horizon_size)
        self._head_color_list: list = []
        self._left_wrist_color_list: list = []
        self._right_wrist_color_list: list = []

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, **kwargs: Any) -> "InternVLAA1Policy":
        """Build the QwenA1 policy + transforms.

        Required kwargs:
          * ``ckpt_path`` - local dir or HuggingFace repo id

        Optional kwargs (defaults match ``InferenceArgs``):
          * stats_key (str) = "aloha"
          * resize_size (int) = 224
          * dtype (str) = "float32"
          * infer_horizon (int) = 30
          * action_horizon_size (int) = 50
          * image_history_interval (int) = 15
          * action_mode (str) = "delta"
          * decode_image_flag (bool) = False
          * robot_type (tuple) = (6, 1, 6, 1)
        """
        _ensure_internvla_on_path()

        # Reuse the canonical policy builder shipped with InternVLA-A1.
        from inference import build_policy_and_transforms  # type: ignore

        ckpt_path = kwargs.pop("ckpt_path")
        stats_key = kwargs.pop("stats_key", "aloha")
        resize_size = int(kwargs.pop("resize_size", 224))
        dtype_str = str(kwargs.pop("dtype", "float32"))
        dtype = torch.float32 if dtype_str == "float32" else torch.bfloat16

        policy, input_transforms, unnormalize_fn = build_policy_and_transforms(
            ckpt_path, stats_key, resize_size, dtype
        )

        return cls(
            policy=policy,
            input_transforms=input_transforms,
            unnormalize_fn=unnormalize_fn,
            dtype=dtype,
            infer_horizon=int(kwargs.pop("infer_horizon", 30)),
            action_horizon_size=int(kwargs.pop("action_horizon_size", 50)),
            image_history_interval=int(kwargs.pop("image_history_interval", 15)),
            action_mode=str(kwargs.pop("action_mode", "delta")),
            decode_image_flag=bool(kwargs.pop("decode_image_flag", False)),
            robot_type=tuple(kwargs.pop("robot_type", (6, 1, 6, 1))),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        instruction: str,
        task_env: Any,
        initial_obs: Mapping[str, Any],
    ) -> None:
        del instruction, task_env, initial_obs  # not needed for InternVLA-A1
        self._policy.reset()
        self._action_plan = deque([], maxlen=self.action_horizon_size)
        self._head_color_list = []
        self._left_wrist_color_list = []
        self._right_wrist_color_list = []

    def step(self, task_env: Any, observation: Mapping[str, Any]) -> None:
        from lerobot.utils.constants import OBS_IMAGES  # type: ignore

        img = observation["observation"]["head_camera"]["rgb"]

        # Refresh wrist-camera history when the action buffer is shallow,
        # mirroring the cadence of inference.py to keep behaviour identical.
        if len(self._action_plan) <= self.image_history_interval:
            left_wrist_img = observation["observation"]["left_camera"]["rgb"]
            right_wrist_img = observation["observation"]["right_camera"]["rgb"]

            self._head_color_list.append(
                torch.as_tensor(img).contiguous().cuda().to(self._dtype) / 255.0
            )
            self._left_wrist_color_list.append(
                torch.as_tensor(left_wrist_img).contiguous().cuda().to(self._dtype) / 255.0
            )
            self._right_wrist_color_list.append(
                torch.as_tensor(right_wrist_img).contiguous().cuda().to(self._dtype) / 255.0
            )

            while len(self._head_color_list) > self.image_history_interval + 1:
                self._head_color_list.pop(0)
                self._left_wrist_color_list.pop(0)
                self._right_wrist_color_list.pop(0)

            past_idx = max(len(self._head_color_list) - self.image_history_interval - 1, 0)
            self._image_head_with_history = torch.stack(
                [self._head_color_list[past_idx], self._head_color_list[-1]], dim=0
            )
            self._image_hand_left_with_history = torch.stack(
                [self._left_wrist_color_list[past_idx], self._left_wrist_color_list[-1]], dim=0
            )
            self._image_hand_right_with_history = torch.stack(
                [self._right_wrist_color_list[past_idx], self._right_wrist_color_list[-1]], dim=0
            )

        if not self._action_plan:
            init_action = (
                torch.as_tensor(observation["joint_action"]["vector"][None])
                .contiguous()
                .cuda()
            )
            state = torch.from_numpy(observation["joint_action"]["vector"]).float().cuda()
            task = task_env.get_instruction()

            sample = {
                f"{OBS_IMAGES}.image0": self._image_head_with_history,
                f"{OBS_IMAGES}.image1": self._image_hand_left_with_history,
                f"{OBS_IMAGES}.image2": self._image_hand_right_with_history,
                "observation.state": state,
                "task": task,
            }
            for key in list(sample.keys()):
                if OBS_IMAGES in key and "mask" not in key:
                    image = sample[key].permute(0, 3, 1, 2)
                    sample[key] = image

            sample = self._input_transforms(sample)

            inputs = {}
            for key in sample.keys():
                if key == "task":
                    inputs[key] = [sample[key]]
                elif sample[key].dtype == torch.int64:
                    inputs[key] = sample[key][None].cuda()
                else:
                    inputs[key] = sample[key][None].cuda().to(dtype=self._dtype)

            inputs.update({
                f"{OBS_IMAGES}.image0_mask": torch.tensor([True]).cuda(),
                f"{OBS_IMAGES}.image1_mask": torch.tensor([True]).cuda(),
                f"{OBS_IMAGES}.image2_mask": torch.tensor([True]).cuda(),
            })

            with torch.no_grad():
                action_pred, _ = self._policy.predict_action_chunk(
                    inputs, decode_image=self.decode_image_flag
                )

            action_pred = action_pred[0, : self.infer_horizon, : self.action_dim]
            action_pred = self._unnormalize_fn({"action": action_pred})["action"]

            if self.action_mode == "delta":
                init_action[:, self.left_gripper_idx] = 0.0
                init_action[:, self.right_gripper_idx] = 0.0
                action_pred += init_action

            self._action_plan.extend(action_pred.cpu().numpy())

        action = self._action_plan.popleft()
        action[self.left_gripper_idx] = 0 if action[self.left_gripper_idx] < 0.5 else 1
        action[self.right_gripper_idx] = 0 if action[self.right_gripper_idx] < 0.5 else 1
        task_env.take_action(action, action_type="qpos")


__all__ = ["InternVLAA1Policy"]
