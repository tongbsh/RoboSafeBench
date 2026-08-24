"""X-VLA HTTP-client policy adapter for DyniSafety-RoboTwin.

This adapter mirrors the inference contract in
``external/model_repos/X-VLA/evaluation/robotwin-2.0/client.py`` while fitting the
``RoboTwinPolicy`` interface used by ``dynamic_RoboTwin/main_inference.py``.

The X-VLA model itself runs behind a FastAPI server (``deploy.py``). The
RoboTwin side only sends observations to ``/act`` and converts the returned
action chunk into single-step dual-arm EE actions.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np
import requests
from scipy.spatial.transform import Rotation as R

from . import register_policy
from .base_policy import RoboTwinPolicy

logger = logging.getLogger(__name__)


def _json_numpy_compat_dumps(array: np.ndarray) -> str:
    arr = np.asarray(array)
    return json.dumps(
        {
            "__numpy__": base64.b64encode(arr.tobytes()).decode("ascii"),
            "dtype": arr.dtype.str,
            "shape": list(arr.shape),
        }
    )


def _quat_to_rotate6d(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    return R.from_quat(quat_xyzw).as_matrix()[:, :2].reshape(6)


def _rotate6d_to_quat(v6: np.ndarray) -> np.ndarray:
    v6 = np.asarray(v6, dtype=np.float64).reshape(6)
    a1 = v6[0:5:2]
    a2 = v6[1:6:2]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    proj = np.dot(b1, a2) * b1
    b2_raw = a2 - proj
    b2 = b2_raw / (np.linalg.norm(b2_raw) + 1e-8)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)
    quat_xyzw = R.from_matrix(rot_mats).as_quat()
    return quat_xyzw / (np.linalg.norm(quat_xyzw) + 1e-8)


def _bool_gripper_to_official(gripper_value: float) -> float:
    # Official client maps RoboTwin gripper [0,1] -> {+1 open, -1 close}.
    return 1.0 - float(gripper_value) * 2.0


def _pred_gripper_to_ee(gripper_pred: float, threshold: float) -> float:
    return 1.0 - 2.0 * float(gripper_pred > threshold)


def _front_camera_fallback(observation: Mapping[str, Any]) -> np.ndarray:
    cameras = observation["observation"]
    if "front_camera" in cameras and "rgb" in cameras["front_camera"]:
        return cameras["front_camera"]["rgb"]
    return cameras["head_camera"]["rgb"]


@register_policy("xvla")
class XVLAPolicy(RoboTwinPolicy):
    """HTTP client for the official X-VLA RoboTwin service."""

    name = "xvla"
    action_type = "ee"

    def __init__(
        self,
        *,
        session: requests.Session,
        host: str,
        port: int,
        domain_id: int,
        steps: int,
        request_timeout: float,
        server_timeout: float,
        gripper_threshold: float,
        model_path: str = "",
    ) -> None:
        self._session = session
        self._host = host
        self._port = int(port)
        self._act_url = f"http://{host}:{port}/act"
        self._health_url = f"http://{host}:{port}/openapi.json"
        self._domain_id = int(domain_id)
        self._steps = int(steps)
        self._request_timeout = float(request_timeout)
        self._server_timeout = float(server_timeout)
        self._gripper_threshold = float(gripper_threshold)
        self._model_path = str(model_path)

        self._instruction: str = ""
        self._action_queue: deque[np.ndarray] = deque()
        self._left_endpose_override: np.ndarray | None = None
        self._right_endpose_override: np.ndarray | None = None

        logger.info(
            "X-VLA policy: url=%s domain_id=%d steps=%d gripper_threshold=%.3f model_path=%s",
            self._act_url,
            self._domain_id,
            self._steps,
            self._gripper_threshold,
            self._model_path or "(unspecified)",
        )

        self._wait_for_server()

    @classmethod
    def load(cls, **kwargs: Any) -> XVLAPolicy:
        host = str(kwargs.pop("host", "127.0.0.1"))
        port = int(kwargs.pop("port", 8010))
        domain_id = int(kwargs.pop("domain_id", 6))
        steps = int(kwargs.pop("steps", 10))
        request_timeout = float(kwargs.pop("request_timeout", 180.0))
        server_timeout = float(kwargs.pop("server_timeout", 300.0))
        gripper_threshold = float(kwargs.pop("gripper_threshold", 0.7))
        model_path = str(kwargs.pop("model_path", ""))

        session = requests.Session()
        return cls(
            session=session,
            host=host,
            port=port,
            domain_id=domain_id,
            steps=steps,
            request_timeout=request_timeout,
            server_timeout=server_timeout,
            gripper_threshold=gripper_threshold,
            model_path=model_path,
        )

    def reset(
        self,
        instruction: str,
        task_env: Any,
        initial_obs: Mapping[str, Any],
    ) -> None:
        del task_env
        self._instruction = str(instruction)
        self._action_queue.clear()
        endpose = initial_obs["endpose"]
        self._left_endpose_override = np.asarray(
            endpose["left_endpose"],
            dtype=np.float32,
        ).copy()
        self._right_endpose_override = np.asarray(
            endpose["right_endpose"],
            dtype=np.float32,
        ).copy()

    def step(self, task_env: Any, observation: Mapping[str, Any]) -> None:
        if not self._action_queue:
            action_chunk = self._query_action_chunk(observation)
            for action in self._decode_action_chunk(action_chunk):
                self._action_queue.append(action)

        if not self._action_queue:
            raise RuntimeError("X-VLA returned an empty action chunk.")

        action = self._action_queue.popleft()
        task_env.take_action(action, action_type="ee")
        # Mirror the official RoboTwin client: future proprio uses the last
        # commanded EE pose rather than the environment-measured pose.
        self._left_endpose_override = np.asarray(action[:7], dtype=np.float32).copy()
        self._right_endpose_override = np.asarray(action[8:15], dtype=np.float32).copy()

    def close(self) -> None:
        self._session.close()

    def _wait_for_server(self) -> None:
        logger.info("Waiting for X-VLA server at %s ...", self._health_url)
        start = time.time()
        while True:
            if time.time() - start > self._server_timeout:
                raise TimeoutError(
                    f"Failed to connect to X-VLA server within {self._server_timeout}s: {self._health_url}"
                )
            try:
                response = self._session.get(self._health_url, timeout=5.0)
                if response.ok:
                    logger.info("X-VLA server is reachable.")
                    return
            except requests.RequestException:
                pass
            time.sleep(2.0)

    def _query_action_chunk(self, observation: Mapping[str, Any]) -> np.ndarray:
        cameras = observation["observation"]
        head_view = cameras["head_camera"]["rgb"]
        left_view = cameras["left_camera"]["rgb"]
        right_view = cameras["right_camera"]["rgb"]
        # Official client records front view but does not send it to /act.
        _front_camera_fallback(observation)

        endpose = observation["endpose"]
        left_ee = (
            self._left_endpose_override
            if self._left_endpose_override is not None
            else np.asarray(endpose["left_endpose"], dtype=np.float32)
        )
        right_ee = (
            self._right_endpose_override
            if self._right_endpose_override is not None
            else np.asarray(endpose["right_endpose"], dtype=np.float32)
        )
        left_grip = _bool_gripper_to_official(float(endpose["left_gripper"]))
        right_grip = _bool_gripper_to_official(float(endpose["right_gripper"]))

        abs_eef = np.concatenate(
            [
                left_ee[:3],
                _quat_to_rotate6d(left_ee[3:]),
                np.array([left_grip], dtype=np.float64),
                right_ee[:3],
                _quat_to_rotate6d(right_ee[3:]),
                np.array([right_grip], dtype=np.float64),
            ],
            axis=0,
        ).astype(np.float32)

        payload = {
            "domain_id": self._domain_id,
            "proprio": _json_numpy_compat_dumps(abs_eef),
            "language_instruction": self._instruction,
            "image0": _json_numpy_compat_dumps(np.asarray(head_view, dtype=np.uint8)),
            "image1": _json_numpy_compat_dumps(np.asarray(left_view, dtype=np.uint8)),
            "image2": _json_numpy_compat_dumps(np.asarray(right_view, dtype=np.uint8)),
            "steps": self._steps,
        }

        response = self._session.post(
            self._act_url,
            json=payload,
            timeout=self._request_timeout,
        )
        response.raise_for_status()
        body = response.json()
        if "action" not in body:
            raise KeyError(f"X-VLA response missing 'action': keys={list(body.keys())}")
        action_chunk = np.asarray(body["action"], dtype=np.float64)
        if action_chunk.ndim == 1:
            action_chunk = action_chunk[None, :]
        if action_chunk.ndim != 2 or action_chunk.shape[1] < 20:
            raise ValueError(f"Unexpected X-VLA action shape: {action_chunk.shape}")
        return action_chunk

    def _decode_action_chunk(self, action_chunk: np.ndarray) -> list[np.ndarray]:
        rollout_actions: list[np.ndarray] = []
        for action in action_chunk:
            left_xyz = action[0:3]
            left_quat = _rotate6d_to_quat(action[3:9])
            left_grip = np.array(
                [_pred_gripper_to_ee(action[9], self._gripper_threshold)],
                dtype=np.float64,
            )

            right_xyz = action[10:13]
            right_quat = _rotate6d_to_quat(action[13:19])
            right_grip = np.array(
                [_pred_gripper_to_ee(action[19], self._gripper_threshold)],
                dtype=np.float64,
            )

            rollout_action = np.concatenate(
                [left_xyz, left_quat, left_grip, right_xyz, right_quat, right_grip],
                axis=0,
            ).astype(np.float64)
            rollout_actions.append(rollout_action)
        return rollout_actions


__all__ = ["XVLAPolicy"]
