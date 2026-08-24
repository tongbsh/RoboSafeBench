"""LingBot-VA WebSocket adapter (RoboTwin dual-arm absolute EE).

Mirrors ``_robotwin_impl/lingbot_va_policy.py`` / archive
``eval_polict_client_openpi.py``: ``obs`` wrapper, episode ``reset``,
euler/16-D → abs EE 16-D, and post-chunk ``compute_kv_cache``.
Isolated to this adapter + YAML; does not change the transport codec API.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

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


class LingBotVAAdapter:
    """LingBot-VA remote policy (``extra.ws_codec: lingbot``).

    Required ``extra``: ``connect_timeout_s``, ``api_key`` (may be JSON null),
    ``dump_artifacts`` (bool; must be false in paper reruns).
    """

    def __init__(self, cfg: ModelConfig, transport: Transport) -> None:
        self._cfg = cfg
        self._transport = transport
        dump = require_extra(cfg, "dump_artifacts")
        if not isinstance(dump, bool):
            raise ValueError(f"extra.dump_artifacts must be bool, got {dump!r}")
        self._dump_artifacts = dump
        self._video_guidance = float(cfg.extra.get("video_guidance_scale", 5.0))
        self._action_guidance = float(cfg.extra.get("action_guidance_scale", 1.0))

        self._instruction = ""
        self._init_eef_pose: np.ndarray | None = None
        self._first_obs: dict[str, Any] | None = None
        self._first = True
        self._need_server_reset = False
        self._action_queue: list[tuple[np.ndarray, int]] = []
        self._action_per_frame = 1
        self._pending_key_frames: list[dict[str, Any]] = []
        self._last_raw_action: np.ndarray | None = None

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> LingBotVAAdapter:
        if cfg.transport != "websocket":
            raise ValueError("LingBotVAAdapter requires transport='websocket'")
        return cls(cfg, connect_transport(cfg))

    def reset(self, ctx: EpisodeContext) -> None:
        self._instruction = str(ctx.instruction)
        self._first = True
        self._need_server_reset = True
        self._init_eef_pose = None
        self._first_obs = None
        self._action_queue = []
        self._pending_key_frames = []
        self._last_raw_action = None
        # Codec reset is a no-op; real episode reset is sent on first act().

    def act(self, obs: Observation) -> ActionChunk:
        if self._cfg.action_space != "abs_ee_16d":
            raise ValueError(
                "LingBotVAAdapter requires action_space='abs_ee_16d' "
                f"(got {self._cfg.action_space!r})"
            )
        instruction = obs.instruction or self._instruction
        self._instruction = instruction
        rt_obs = _rt_obs_from_contract(obs)
        formatted = _format_obs(rt_obs, instruction)

        if self._need_server_reset:
            endpose = (obs.extra or {}).get("endpose")
            if not isinstance(endpose, dict):
                raise KeyError(
                    "LingBot-VA needs Observation.extra['endpose'] on first act "
                    "(left/right_endpose + grippers)"
                )
            self._init_eef_pose = _init_eef_from_endpose(endpose)
            self._transport.infer(
                {
                    "reset": True,
                    "prompt": instruction,
                    "save_visualization": self._dump_artifacts,
                }
            )
            self._first_obs = formatted
            self._need_server_reset = False

        if not self._action_queue:
            self._refill_queue(rt_obs=rt_obs, formatted=formatted)

        if not self._action_queue:
            raise RuntimeError("LingBot-VA action queue empty after infer")

        ee_action, frame_idx = self._action_queue.pop(0)

        if (frame_idx + 1) % self._action_per_frame == 0:
            self._pending_key_frames.append(_format_obs(rt_obs, instruction))

        if not self._action_queue and self._pending_key_frames:
            self._transport.infer(
                {
                    "obs": self._pending_key_frames,
                    "compute_kv_cache": True,
                    "imagine": False,
                    "save_visualization": self._dump_artifacts,
                    "state": self._last_raw_action,
                }
            )
            self._pending_key_frames = []

        return actions_to_chunk(
            np.asarray(ee_action, dtype=np.float32).reshape(1, -1),
            space="abs_ee_16d",
        )

    def close(self) -> None:
        self._transport.close()

    def _refill_queue(
        self, *, rt_obs: dict[str, Any], formatted: dict[str, Any]
    ) -> None:
        if self._first:
            obs_to_send = self._first_obs or formatted
            self._first_obs = formatted
        else:
            obs_to_send = formatted

        ret = self._transport.infer(
            {
                "obs": obs_to_send,
                "prompt": self._instruction,
                "save_visualization": self._dump_artifacts,
                "video_guidance_scale": self._video_guidance,
                "action_guidance_scale": self._action_guidance,
            }
        )
        action = np.asarray(ret["action"])
        self._last_raw_action = action
        if action.ndim != 3 or action.shape[2] % 4 != 0:
            raise ValueError(
                f"LingBot action shape {action.shape}; expect [dim,H,F] with F%4==0"
            )
        self._action_per_frame = int(action.shape[2]) // 4
        if self._init_eef_pose is None:
            raise RuntimeError("LingBot init EE pose missing before decode")

        start_idx = 1 if self._first else 0
        for i in range(start_idx, action.shape[1]):
            for j in range(action.shape[2]):
                raw_step = action[:, i, j].flatten()
                ee = _raw_step_to_ee16(raw_step, self._init_eef_pose)
                self._action_queue.append((ee, j))
        self._first = False
        self._pending_key_frames = []


def _rt_obs_from_contract(obs: Observation) -> dict[str, Any]:
    return {
        "observation": {
            "head_camera": {"rgb": require_image(obs, "primary")},
            "left_camera": {"rgb": require_image(obs, "left")},
            "right_camera": {"rgb": require_image(obs, "right")},
        },
        "joint_action": {"vector": require_state(obs).astype(np.float64)},
    }


def _format_obs(observation: dict[str, Any], prompt: str) -> dict[str, Any]:
    return {
        "observation.images.cam_high": observation["observation"]["head_camera"]["rgb"],
        "observation.images.cam_left_wrist": observation["observation"]["left_camera"][
            "rgb"
        ],
        "observation.images.cam_right_wrist": observation["observation"]["right_camera"][
            "rgb"
        ],
        "observation.state": observation["joint_action"]["vector"],
        "task": prompt,
    }


def _init_eef_from_endpose(endpose: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        list(endpose["left_endpose"])
        + [float(endpose["left_gripper"])]
        + list(endpose["right_endpose"])
        + [float(endpose["right_gripper"])],
        dtype=np.float64,
    )


def _euler2quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return R.from_euler("xyz", [roll, pitch, yaw], degrees=False).as_quat()


def _add_eef_pose(new_pose: np.ndarray, init_pose: np.ndarray) -> np.ndarray:
    new_pose_r = R.from_quat(new_pose[3:7][None])
    init_pose_r = R.from_quat(init_pose[3:7][None])
    out_rot = (init_pose_r * new_pose_r).as_quat().reshape(-1)
    out_trans = new_pose[:3] + init_pose[:3]
    return np.concatenate([out_trans, out_rot, new_pose[7:8]])


def _add_init_pose(new_pose: np.ndarray, init_pose: np.ndarray) -> np.ndarray:
    left = _add_eef_pose(new_pose[:8], init_pose[:8])
    right = _add_eef_pose(new_pose[8:], init_pose[8:])
    return np.concatenate([left, right])


def _raw_step_to_ee16(raw_action_step: np.ndarray, init_eef: np.ndarray) -> np.ndarray:
    step = np.asarray(raw_action_step, dtype=np.float64).reshape(-1)
    if step.shape[0] == 14:
        return np.concatenate(
            [
                step[:3],
                _euler2quat(step[3], step[4], step[5]),
                step[6:10],
                _euler2quat(step[10], step[11], step[12]),
                step[13:14],
            ]
        ).astype(np.float32)
    if step.shape[0] == 16:
        ee = _add_init_pose(step, init_eef)
        return np.concatenate(
            [
                ee[:3],
                ee[3:7] / (np.linalg.norm(ee[3:7]) + 1e-8),
                ee[7:11],
                ee[11:15] / (np.linalg.norm(ee[11:15]) + 1e-8),
                ee[15:16],
            ]
        ).astype(np.float32)
    raise NotImplementedError(f"Unsupported lingbot action_dim={step.shape[0]}")
