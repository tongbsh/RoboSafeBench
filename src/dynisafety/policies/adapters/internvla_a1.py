"""InternVLA-A1 local adapter (RoboTwin qpos)."""

from __future__ import annotations

from typing import Any

import numpy as np

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


class InternVLAA1Adapter:
    """Wraps RoboTwin InternVLA-A1 inference without calling env ``take_action``.

    Required ``extra``: ``repo_root``, ``robot_type``, plus any kwargs the
    legacy ``InternVLAA1Policy.load`` needs under ``extra.policy_kwargs`` (dict).

    The legacy policy only exposes ``step(task_env, obs)`` which ends in
    ``task_env.take_action``. We pass a capture shim so the backend remains the
    sole executor of actions (iron law / M4 contract).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg
        ensure_repo_on_path(str(require_extra(cfg, "repo_root")))
        from dynisafety.policies._robotwin_impl.internvla_a1_policy import (  # noqa: PLC0415
            InternVLAA1Policy,
        )

        kwargs = require_extra(cfg, "policy_kwargs")
        if not isinstance(kwargs, dict):
            raise ValueError("extra.policy_kwargs must be a mapping")
        robot_type = require_extra(cfg, "robot_type")
        # Legacy InferenceArgs uses tuple[int,...] e.g. (6,1,6,1). A bare
        # string like "aloha" would be iterated char-wise into int() and fail.
        if isinstance(robot_type, str):
            raise ValueError(
                "extra.robot_type must be a list/tuple of ints (e.g. [6, 1, 6, 1]), "
                f"not a name string {robot_type!r}"
            )
        load_kwargs = {
            **kwargs,
            "ckpt_path": cfg.checkpoint,
            "robot_type": tuple(robot_type),
        }
        self._impl = InternVLAA1Policy.load(**load_kwargs)
        self._instruction = ""

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> InternVLAA1Adapter:
        if cfg.transport != "local":
            raise ValueError("InternVLAA1Adapter requires transport='local'")
        return cls(cfg)

    def reset(self, ctx: EpisodeContext) -> None:
        self._instruction = str(ctx.instruction)
        shim = _CaptureEnvShim(instruction=self._instruction)
        initial_obs = {
            "observation": {
                "head_camera": {"rgb": np.zeros((224, 224, 3), dtype=np.uint8)},
                "left_camera": {"rgb": np.zeros((224, 224, 3), dtype=np.uint8)},
                "right_camera": {"rgb": np.zeros((224, 224, 3), dtype=np.uint8)},
            },
            "joint_action": {"vector": np.zeros(14, dtype=np.float32)},
        }
        self._impl.reset(ctx.instruction, shim, initial_obs)

    def act(self, obs: Observation) -> ActionChunk:
        # Build RoboTwin-like observation mapping from contract images.
        rt_obs: dict[str, Any] = {
            "observation": {
                "head_camera": {"rgb": require_image(obs, "primary")},
            },
            "joint_action": {"vector": require_state(obs).astype(np.float32)},
        }
        if "left" in obs.images:
            rt_obs["observation"]["left_camera"] = {"rgb": require_image(obs, "left")}
        if "right" in obs.images:
            rt_obs["observation"]["right_camera"] = {"rgb": require_image(obs, "right")}

        instruction = obs.instruction or self._instruction
        self._instruction = instruction
        shim = _CaptureEnvShim(instruction=instruction)
        self._impl.step(shim, rt_obs)
        if not shim.actions:
            raise RuntimeError(
                "InternVLAA1Adapter: legacy step() did not capture any action "
                "(expected take_action on capture shim)"
            )
        action = np.asarray(shim.actions[0], dtype=np.float32).reshape(1, -1)
        return actions_to_chunk(action, space=self._cfg.action_space)

    def close(self) -> None:
        closer = getattr(self._impl, "close", None)
        if callable(closer):
            closer()


class _CaptureEnvShim:
    """Stand-in for ``task_env``: record actions instead of executing them."""

    def __init__(self, *, instruction: str) -> None:
        self._instruction = instruction
        self.actions: list[np.ndarray] = []

    def get_instruction(self) -> str:
        return self._instruction

    def take_action(self, action: Any, action_type: str = "qpos") -> None:
        del action_type
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
