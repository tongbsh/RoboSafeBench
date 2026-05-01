"""
X-VLA HTTP client and observation helpers shared with
``scripts/x-vla/source_libero_eval/run_libero_xvla_eval.py``.

This module is the single source of truth for X-VLA /act payloads, proprio
initialisation from the simulator controller, and (T,10)→7D action conversion
as used in the verified source-LIBERO evaluation script.
"""

from __future__ import annotations

import collections
import logging
import time
from typing import Deque, Optional

import json_numpy
import numpy as np
import requests
import robosuite.utils.transform_utils as T

json_numpy.patch()

logger = logging.getLogger(__name__)

EPS = 1e-6


def mat_to_rot6d(R: np.ndarray) -> np.ndarray:
    """First two columns of a 3×3 rotation matrix → 6-dim representation."""
    if R.ndim == 2:
        return np.concatenate([R[:3, 0], R[:3, 1]], axis=-1)
    if R.ndim == 3:
        return np.concatenate([R[:, :3, 0], R[:, :3, 1]], axis=-1)
    raise ValueError("R must be (3,3) or (N,3,3)")


def rot6d_to_axis_angle(r6d: np.ndarray) -> np.ndarray:
    """6D rotation → axis-angle (3-dim), matching LiberoAbsActionProcessor."""
    single = r6d.ndim == 1
    if single:
        r6d = r6d[None, :]

    a1, a2 = r6d[:, :3], r6d[:, 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + EPS)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + EPS)
    b3 = np.cross(b1, b2, axis=-1)
    rmat = np.stack([b1, b2, b3], axis=-1)  # (N, 3, 3)

    aa_list = []
    for i in range(rmat.shape[0]):
        quat = T.mat2quat(rmat[i])
        aa_list.append(T.quat2axisangle(quat))

    aa = np.stack(aa_list, axis=0)
    return aa[0] if single else aa


def get_flipped_agentview(obs: dict) -> np.ndarray:
    """Third-person camera, flipped to match X-VLA training preprocessing."""
    img = obs["agentview_image"]
    return img[::-1, ::-1]


def get_wrist_image(obs: dict) -> Optional[np.ndarray]:
    img = obs.get("robot0_eye_in_hand_image")
    return np.array(img) if img is not None else None


def dummy_action() -> list:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def get_ee_proprio(env) -> np.ndarray:
    """
    Read absolute EEF state from the robot controller and format it as the
    20-dim proprio vector expected by X-VLA:
        [pos(3), rot6d(6), grip(1)]  ×2  (second half initialised to zeros)
    """
    robot = env.env.robots[0]
    ee_pos = robot.controller.ee_pos
    ee_rot6d = mat_to_rot6d(robot.controller.ee_ori_mat)
    state = np.concatenate([ee_pos, ee_rot6d, [0.0]])
    return np.concatenate([state, np.zeros(10, dtype=np.float32)])


class XVLAClient:
    """
    Thin HTTP client for the X-VLA FastAPI inference server.

    Manages:
      - Server readiness polling (X-VLA has no /healthz, so we probe /act).
      - Recurrent proprio state tracking across action chunks.
      - Action-chunk queue so that the caller can execute each returned step
        before re-querying when the queue is empty.
    """

    def __init__(
        self,
        host: str,
        port: int,
        domain_id: int = 3,
        steps: int = 10,
        timeout: int = 300,
    ):
        self._act_url = f"http://{host}:{port}/act"
        self._domain_id = domain_id
        self._steps = steps
        self._proprio: Optional[np.ndarray] = None
        self._action_queue: Deque[np.ndarray] = collections.deque()
        self._wait_for_server(timeout)

    def _wait_for_server(self, timeout: int) -> None:
        deadline = time.monotonic() + timeout
        logger.info("Waiting for X-VLA server at %s …", self._act_url)
        while time.monotonic() < deadline:
            try:
                r = requests.post(self._act_url, json={}, timeout=5)
                logger.info("X-VLA server ready (HTTP %d).", r.status_code)
                return
            except requests.ConnectionError:
                pass
            except requests.RequestException:
                logger.info("X-VLA server ready (responded to probe).")
                return
            time.sleep(5)
        raise TimeoutError(
            f"X-VLA server did not become ready within {timeout}s."
        )

    def reset(self) -> None:
        self._proprio = None
        self._action_queue.clear()

    def query(
        self,
        main_img: np.ndarray,
        wrist_img: Optional[np.ndarray],
        task: str,
        env,
    ) -> None:
        if self._proprio is None:
            self._proprio = get_ee_proprio(env)

        payload: dict = {
            "image0": json_numpy.dumps(main_img),
            "language_instruction": task,
            "proprio": json_numpy.dumps(self._proprio),
            "domain_id": self._domain_id,
            "steps": self._steps,
        }
        if wrist_img is not None:
            payload["image1"] = json_numpy.dumps(wrist_img)

        try:
            resp = requests.post(self._act_url, json=payload, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"X-VLA server request failed: {exc}") from exc

        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"X-VLA server error: {data['error']}")

        actions = np.array(data["action"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]

        self._proprio[:9] = actions[-1, :9].copy()

        for i in range(len(actions)):
            pos = actions[i, :3]
            aa = rot6d_to_axis_angle(actions[i, 3:9])
            grip = 1.0 if actions[i, 9] > 0.5 else -1.0
            self._action_queue.append(
                np.concatenate([pos, aa, [grip]]).astype(np.float32)
            )

    def has_actions(self) -> bool:
        return len(self._action_queue) > 0

    def pop_action(self) -> np.ndarray:
        return self._action_queue.popleft()
