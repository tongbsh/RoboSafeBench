"""FastWAM local adapter (RoboTwin qpos)."""

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


class FastWAMAdapter:
    """Wraps FastWAM deploy_policy without env stepping.

    Required ``extra``: ``repo_root``, ``dataset_stats_path``, ``policy_kwargs``.

    Legacy ``FastWAMPolicy`` only exposes ``step(task_env, obs)`` which ends in
    ``task_env.take_action``. We use a capture shim so the backend remains the
    sole executor (same pattern as InternVLAA1Adapter).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg
        ensure_repo_on_path(str(require_extra(cfg, "repo_root")))
        from dynisafety.policies._robotwin_impl.fastwam_policy import (  # noqa: PLC0415
            FastWAMPolicy,
        )

        kwargs = require_extra(cfg, "policy_kwargs")
        if not isinstance(kwargs, dict):
            raise ValueError("extra.policy_kwargs must be a mapping")
        load_kwargs = {
            **kwargs,
            "ckpt_path": cfg.checkpoint,
            "dataset_stats_path": require_extra(cfg, "dataset_stats_path"),
        }
        self._impl = FastWAMPolicy.load(**load_kwargs)
        self._instruction = ""

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> FastWAMAdapter:
        if cfg.transport != "local":
            raise ValueError("FastWAMAdapter requires transport='local'")
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
                "FastWAMAdapter: legacy step() did not capture any action "
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
