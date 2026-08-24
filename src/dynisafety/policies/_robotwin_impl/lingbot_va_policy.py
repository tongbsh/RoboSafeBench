"""LingBot-VA websocket-client policy adapter for DyniSafety-RoboTwin.

This adapter mirrors the inference loop in
``lingbot-va/evaluation/robotwin/eval_polict_client_openpi.py``. The model
itself runs in a separate ``lingbot_va`` conda env (the ``wan_va_server.py``
process) and we talk to it over a websocket. The client side only needs
``websockets``, ``msgpack`` and ``scipy`` (all available in the
``internvla_a1`` env where RoboTwin lives), so the two envs stay
decoupled.

Action stream contract
----------------------
The server returns an action chunk shaped ``[action_dim, horizon, frames]``
with ``action_dim`` either 14 (Euler form) or 16 (quaternion form). After
unrolling, every action is converted to an absolute end-effector pose and
dispatched via ``task_env.take_action(action, action_type='ee')``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from . import register_policy
from .base_policy import RoboTwinPolicy
from .lingbot_client.websocket_client_policy import WebsocketClientPolicy

logger = logging.getLogger(__name__)


def _euler2quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """[rx, ry, rz, rw] quaternion from XYZ Euler angles (radians).

    Matches the convention used by ``lingbot-va/evaluation/robotwin/geometry.py``
    so existing model checkpoints stay compatible.
    """
    rot = R.from_euler("xyz", [roll, pitch, yaw], degrees=False)
    return rot.as_quat()  # [x, y, z, w]


def _add_eef_pose(new_pose: np.ndarray, init_pose: np.ndarray) -> np.ndarray:
    """Compose a delta EE pose with the initial pose recorded at episode start.

    Both poses are 8-dim ``[x, y, z, qx, qy, qz, qw, gripper]`` arrays.
    """
    new_pose_R = R.from_quat(new_pose[3:7][None])
    init_pose_R = R.from_quat(init_pose[3:7][None])
    out_rot = (init_pose_R * new_pose_R).as_quat().reshape(-1)
    out_trans = new_pose[:3] + init_pose[:3]
    return np.concatenate([out_trans, out_rot, new_pose[7:8]])


def _add_init_pose(new_pose: np.ndarray, init_pose: np.ndarray) -> np.ndarray:
    """Apply ``_add_eef_pose`` to both arms (16-dim dual-arm pose)."""
    left_pose = _add_eef_pose(new_pose[:8], init_pose[:8])
    right_pose = _add_eef_pose(new_pose[8:], init_pose[8:])
    return np.concatenate([left_pose, right_pose])


def _format_obs(observation: Mapping[str, Any], prompt: str) -> dict:
    """Match the format expected by ``wan_va_server`` exactly."""
    return {
        "observation.images.cam_high": observation["observation"]["head_camera"]["rgb"],
        "observation.images.cam_left_wrist": observation["observation"]["left_camera"]["rgb"],
        "observation.images.cam_right_wrist": observation["observation"]["right_camera"]["rgb"],
        "observation.state": observation["joint_action"]["vector"],
        "task": prompt,
    }


@register_policy("lingbot_va")
class LingBotVAPolicy(RoboTwinPolicy):
    """Talks to a remote ``wan_va_server`` over websocket.

    Constructor kwargs accepted via :py:meth:`load`:
      * host (str) = "0.0.0.0"
      * port (int) = 29056
      * api_key (str | None) = None
      * video_guidance_scale (float) = 5.0
      * action_guidance_scale (float) = 1.0
      * save_visualization (bool) = False
    """

    name = "lingbot_va"
    action_type = "ee"

    def __init__(
        self,
        client: Any,
        *,
        video_guidance_scale: float = 5.0,
        action_guidance_scale: float = 1.0,
        save_visualization: bool = False,
    ) -> None:
        self._client = client
        self._video_guidance_scale = float(video_guidance_scale)
        self._action_guidance_scale = float(action_guidance_scale)
        self._save_visualization = bool(save_visualization)

        # Per-episode mutable state
        self._instruction: str = ""
        self._init_eef_pose: np.ndarray | None = None
        self._first_obs: dict | None = None
        self._first: bool = True
        # Flat queue of single-step actions (already converted to absolute EE
        # pose form) waiting to be dispatched on the env.
        self._action_queue: list[np.ndarray] = []
        # Frames-per-action divisor (action.shape[2] / 4) used to decide when
        # to push key frames back to the server for kv-cache updates.
        self._action_per_frame: int = 1
        # Buffer of key-frame observations that we feed back to the server
        # after each chunk completes (matches the reference loop).
        self._pending_key_frames: list[dict] = []
        # Most recently received raw action chunk -> needed for the kv-cache
        # call right after the chunk is consumed.
        self._last_raw_action: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, **kwargs: Any) -> LingBotVAPolicy:
        host = str(kwargs.pop("host", "0.0.0.0"))
        port = int(kwargs.pop("port", 29056))
        api_key = kwargs.pop("api_key", None)
        client = WebsocketClientPolicy(host=host, port=port, api_key=api_key)

        return cls(
            client=client,
            video_guidance_scale=float(kwargs.pop("video_guidance_scale", 5.0)),
            action_guidance_scale=float(kwargs.pop("action_guidance_scale", 1.0)),
            save_visualization=bool(kwargs.pop("save_visualization", False)),
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
        self._instruction = instruction
        self._first = True
        self._action_queue = []
        self._pending_key_frames = []
        self._last_raw_action = None

        # Snapshot initial EE pose so 16-dim delta predictions can be
        # rebased on it (mirrors eval_polict_client_openpi.py).
        endpose = initial_obs["endpose"]
        init_eef = (
            list(endpose["left_endpose"])
            + [endpose["left_gripper"]]
            + list(endpose["right_endpose"])
            + [endpose["right_gripper"]]
        )
        self._init_eef_pose = np.array(init_eef, dtype=np.float64)

        # Tell the server to drop any prior episode state.
        self._client.infer({
            "reset": True,
            "prompt": instruction,
            "save_visualization": self._save_visualization,
        })

        # Cache the first formatted observation so the very first call to
        # ``step`` already has something to send (the reference loop does
        # the same).
        self._first_obs = _format_obs(initial_obs, instruction)

    def step(self, task_env: Any, observation: Mapping[str, Any]) -> None:
        # When the local action buffer is empty we need to fetch a fresh
        # chunk from the server. The reference loop sends ``first_obs`` for
        # the very first request and the live observation thereafter.
        if not self._action_queue:
            if self._first:
                obs_to_send = self._first_obs
                # Refresh first_obs from the live observation so subsequent
                # chunks see the up-to-date scene.
                self._first_obs = _format_obs(observation, self._instruction)
            else:
                obs_to_send = _format_obs(observation, self._instruction)

            ret = self._client.infer({
                "obs": obs_to_send,
                "prompt": self._instruction,
                "save_visualization": self._save_visualization,
                "video_guidance_scale": self._video_guidance_scale,
                "action_guidance_scale": self._action_guidance_scale,
            })
            action = ret["action"]
            self._last_raw_action = action

            assert action.shape[2] % 4 == 0, (
                f"Server returned action chunk with shape {action.shape}; "
                "the third dim must be divisible by 4 to match the four "
                "intra-step frames produced by lingbot-va."
            )
            self._action_per_frame = action.shape[2] // 4

            start_idx = 1 if self._first else 0
            for i in range(start_idx, action.shape[1]):
                for j in range(action.shape[2]):
                    raw_action_step = action[:, i, j].flatten()

                    if action.shape[0] == 14:
                        ee_action = np.concatenate([
                            raw_action_step[:3],
                            _euler2quat(
                                raw_action_step[3],
                                raw_action_step[4],
                                raw_action_step[5],
                            ),
                            raw_action_step[6:10],
                            _euler2quat(
                                raw_action_step[10],
                                raw_action_step[11],
                                raw_action_step[12],
                            ),
                            raw_action_step[13:14],
                        ])
                    elif action.shape[0] == 16:
                        ee_action = _add_init_pose(raw_action_step, self._init_eef_pose)
                        ee_action = np.concatenate([
                            ee_action[:3],
                            ee_action[3:7] / np.linalg.norm(ee_action[3:7]),
                            ee_action[7:11],
                            ee_action[11:15] / np.linalg.norm(ee_action[11:15]),
                            ee_action[15:16],
                        ])
                    else:
                        raise NotImplementedError(
                            f"Unsupported action_dim={action.shape[0]} from lingbot server"
                        )

                    self._action_queue.append((ee_action, j))

            self._first = False
            self._pending_key_frames = []

        ee_action, frame_idx = self._action_queue.pop(0)
        task_env.take_action(ee_action, action_type="ee")

        # Mirror the reference loop: every ``action_per_frame`` steps we
        # snapshot the current observation as a key frame. After the whole
        # chunk is exhausted we ship the key frames back to the server so
        # it can refresh its kv-cache for the next chunk.
        if (frame_idx + 1) % self._action_per_frame == 0:
            self._pending_key_frames.append(_format_obs(observation, self._instruction))

        if not self._action_queue and self._pending_key_frames:
            self._client.infer({
                "obs": self._pending_key_frames,
                "compute_kv_cache": True,
                "imagine": False,
                "save_visualization": self._save_visualization,
                "state": self._last_raw_action,
            })
            self._pending_key_frames = []


__all__ = ["LingBotVAPolicy"]
