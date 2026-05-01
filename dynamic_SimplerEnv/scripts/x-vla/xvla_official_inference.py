# -----------------------------------------------------------------------------
# X-VLA WidowX — bridge between upstream X-VLA HTTP API and dynamic_SimplerEnv.
#
# Use with main_inference.py:
#   --policy-model xvla_official --ckpt-path ignored --policy-setup widowx_bridge
#
# Env:
#   XVLA_SERVER_HOST / XVLA_SERVER_PORT / XVLA_DOMAIN_ID / XVLA_STEPS / XVLA_GRIPPER_THRESH
#   XVLA_EXEC_MODE:
#     simpler_delta (default) — 保留官方 WidowX client 的 chunk/proprio 记忆语义，但在
#       bundled delta-EE SimplerEnv 里把预测的绝对 EE 目标转换成当前步 delta。
#     upstream_literal — 给官方 absolute-EE SimplerEnv 用，尽量逐字复现上游
#       WidowX client 的 7D 语义（xyz + euler(xyz)+bias + binary gripper）。
#   XVLA_WIDOWX_POS_DECODE: abs_diff (default) | raw  （raw=直接把 pred[:3] 当平移增量）
#   XVLA_WIDOWX_ROT_DECODE: official_euler (default) | rel_so3 | rot6d_axangle
# -----------------------------------------------------------------------------

from __future__ import annotations

import collections
import logging
import math
import os
import time
from typing import Any, Optional, Sequence

import cv2 as cv
import json_numpy
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

# --- 6D rotation (与上游 client / X-VLA datasets/utils 一致) ---


def rotate6d_to_euler_xyz(v6: np.ndarray) -> np.ndarray:
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError(f"Last dimension must be 6, got {v6.shape[-1]}")
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)
    return R.from_matrix(rot_mats).as_euler("xyz")


def rot6d_to_matrix(v6: np.ndarray) -> np.ndarray:
    """(6,) → (3,3) 旋转矩阵。"""
    v6 = np.asarray(v6, dtype=np.float64).reshape(6)
    a1 = v6[0:5:2]
    a2 = v6[1:6:2]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2_raw = a2 - np.dot(b1, a2) * b1
    b2 = b2_raw / (np.linalg.norm(b2_raw) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def quat_wxyz_to_rot6d(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_xyzw = np.roll(np.asarray(quat_wxyz, dtype=np.float64), -1)
    m = R.from_quat(quat_xyzw).as_matrix()
    return np.concatenate([m[:, 0], m[:, 1]])


def proprio_20_from_eef(eef_pos: np.ndarray) -> np.ndarray:
    """SimplerEnv eef_pos [xyz, wxyz, grip_open] → X-VLA 20 维 proprio（单臂 + 填充）。"""
    eef = np.asarray(eef_pos, dtype=np.float32)
    pos = eef[:3]
    rot6d = quat_wxyz_to_rot6d(eef[3:7]).astype(np.float32)
    grip = eef[7:8]
    left10 = np.concatenate([pos, rot6d, grip])
    return np.concatenate([left10, np.zeros(10, dtype=np.float32)])


_OFFICIAL_PROPRIO_TAIL = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
_EULER_BIAS = np.array([0.0, math.pi / 2.0, 0.0], dtype=np.float64)


def official_proprio_20_from_eef(eef_pos: np.ndarray) -> np.ndarray:
    """Official WidowX client proprio: pos(3) + fixed tail(7) + zeros(10)."""
    pos = np.asarray(eef_pos[:3], dtype=np.float32)
    proprio_10 = np.concatenate([pos, _OFFICIAL_PROPRIO_TAIL])
    return np.concatenate([proprio_10, np.zeros(10, dtype=np.float32)])


class XVLAOfficialBridgeInference:
    """X-VLA HTTP 客户端，接入 DyniSafety / maniskill2 评估循环。"""

    def __init__(
        self,
        saved_model_path: str = "",
        policy_setup: str = "widowx_bridge",
        action_scale: float = 1.0,
        domain_id: int = 0,
        steps: int = 10,
    ) -> None:
        if policy_setup != "widowx_bridge":
            raise NotImplementedError(
                "xvla_official only supports policy_setup='widowx_bridge'."
            )

        host = os.environ.get("XVLA_SERVER_HOST", "127.0.0.1")
        port = int(os.environ.get("XVLA_SERVER_PORT", "8010"))
        self._url = f"http://{host}:{port}/act"
        self._domain_id = int(os.environ.get("XVLA_DOMAIN_ID", str(domain_id)))
        self._steps = int(os.environ.get("XVLA_STEPS", str(steps)))
        self._gripper_thresh = float(os.environ.get("XVLA_GRIPPER_THRESH", "0.95"))
        self._exec_mode = os.environ.get("XVLA_EXEC_MODE", "simpler_delta").strip().lower()
        self._pos_decode = os.environ.get("XVLA_WIDOWX_POS_DECODE", "abs_diff").strip().lower()
        self._rot_decode = os.environ.get("XVLA_WIDOWX_ROT_DECODE", "official_euler").strip().lower()

        self.policy_setup = policy_setup
        self.action_scale = float(action_scale)
        self.task_description: Optional[str] = None

        self._action_plan: collections.deque = collections.deque()
        self._proprio_20: Optional[np.ndarray] = None

        logger.info(
            "XVLAOfficialBridgeInference: server=%s domain_id=%d infer_steps=%d "
            "gripper_thresh=%.3f action_scale=%s exec_mode=%s pos=%s rot=%s",
            self._url,
            self._domain_id,
            self._steps,
            self._gripper_thresh,
            self.action_scale,
            self._exec_mode,
            self._pos_decode,
            self._rot_decode,
        )
        self._wait_for_server()

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self._action_plan.clear()
        self._proprio_20 = None

    def step(
        self,
        image: np.ndarray,
        task_description: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)

        assert image.dtype == np.uint8

        eef_pos = kwargs.get("eef_pos", None)

        if self._exec_mode == "upstream_literal":
            return self._step_upstream_literal(image, task_description, eef_pos)
        if self._exec_mode == "simpler_delta":
            return self._step_simpler_delta(image, task_description, eef_pos)

        raise ValueError(
            "XVLA_EXEC_MODE must be simpler_delta or upstream_literal "
            f"(got {self._exec_mode!r})"
        )

    def _step_simpler_delta(
        self,
        image: np.ndarray,
        task_description: Optional[str],
        eef_pos: Any,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Official chunk/proprio semantics + absolute-to-delta execution."""
        if self._proprio_20 is None:
            if eef_pos is not None and len(eef_pos) >= 3:
                self._proprio_20 = official_proprio_20_from_eef(np.asarray(eef_pos))
            else:
                self._proprio_20 = np.zeros(20, dtype=np.float32)

        if not self._action_plan:
            payload = {
                "proprio": json_numpy.dumps(self._proprio_20),
                "language_instruction": task_description or self.task_description or "",
                "image0": json_numpy.dumps(image),
                "domain_id": self._domain_id,
                "steps": self._steps,
            }
            self._post_act(payload)

        action_pred = np.asarray(self._action_plan.popleft(), dtype=np.float64)
        if action_pred.shape[-1] < 10:
            raise ValueError(f"X-VLA action row too short: shape={action_pred.shape}")
        self._proprio_20 = np.asarray(self._proprio_20, dtype=np.float32)
        self._proprio_20[:10] = action_pred[:10].astype(np.float32)

        grip_cmd = 1.0 if float(action_pred[9]) < self._gripper_thresh else -1.0

        rot6 = action_pred[3:9]
        if eef_pos is None or len(eef_pos) < 8:
            delta_p = np.asarray(action_pred[:3], dtype=np.float64)
            delta_rv = self._decode_delta_rotation(rot6, None)
            logger.warning(
                "xvla_official: missing eef_pos; cannot form delta — using model raw (likely wrong)."
            )
        else:
            eef = np.asarray(eef_pos, dtype=np.float64)
            pred_p = action_pred[:3].astype(np.float64)
            if self._pos_decode == "raw":
                delta_p = pred_p
            else:
                delta_p = pred_p - eef[:3]
            delta_rv = self._decode_delta_rotation(rot6, eef)

        raw_action = {
            "pred_abs_pos": action_pred[:3].copy(),
            "world_vector": delta_p.copy(),
            "rot_axangle_raw": delta_rv.copy(),
            "open_gripper": np.array([float(action_pred[9])], dtype=np.float64),
        }
        action: dict[str, np.ndarray] = {
            "world_vector": delta_p * self.action_scale,
            "rot_axangle": delta_rv * self.action_scale,
            "gripper": np.array([grip_cmd], dtype=np.float64),
            "terminate_episode": np.array([0.0]),
        }
        return raw_action, action

    def _official_euler_biased(self, rot6: np.ndarray) -> np.ndarray:
        return (rotate6d_to_euler_xyz(rot6) + _EULER_BIAS).astype(np.float64)

    def _target_rotation(self, rot6: np.ndarray) -> R:
        mode = self._rot_decode
        if mode == "official_euler":
            return R.from_euler("xyz", self._official_euler_biased(rot6))
        if mode in {"rel_so3", "rot6d_axangle"}:
            return R.from_matrix(rot6d_to_matrix(rot6))
        raise ValueError(
            f"XVLA_WIDOWX_ROT_DECODE must be official_euler|rel_so3|rot6d_axangle (got {mode!r})"
        )

    def _decode_delta_rotation(
        self, rot6: np.ndarray, eef: Optional[np.ndarray]
    ) -> np.ndarray:
        target_rot = self._target_rotation(rot6)
        if self._rot_decode == "rot6d_axangle":
            return target_rot.as_rotvec().astype(np.float64)
        if eef is None or len(eef) < 7:
            return target_rot.as_rotvec().astype(np.float64)
        quat_xyzw = np.roll(eef[3:7], -1)
        r_cur = R.from_quat(quat_xyzw)
        return (target_rot * r_cur.inv()).as_rotvec().astype(np.float64)

    def _step_upstream_literal(
        self,
        image: np.ndarray,
        task_description: Optional[str],
        eef_pos: Any,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """逐字复现官方 WidowX client，供 absolute-EE SimplerEnv 使用。"""
        if self._proprio_20 is None:
            if eef_pos is not None and len(eef_pos) >= 3:
                self._proprio_20 = official_proprio_20_from_eef(np.asarray(eef_pos))
            else:
                self._proprio_20 = np.zeros(20, dtype=np.float32)

        if not self._action_plan:
            payload = {
                "proprio": json_numpy.dumps(self._proprio_20),
                "language_instruction": task_description or self.task_description or "",
                "image0": json_numpy.dumps(image),
                "domain_id": self._domain_id,
                "steps": self._steps,
            }
            self._post_act(payload)

        action_pred = np.asarray(self._action_plan.popleft(), dtype=np.float32)
        if action_pred.shape[-1] < 10:
            raise ValueError(f"X-VLA action row too short: shape={action_pred.shape}")

        self._proprio_20 = np.asarray(self._proprio_20, dtype=np.float32)
        self._proprio_20[:10] = action_pred[:10]

        euler_biased = self._official_euler_biased(action_pred[3:9])
        grip_cmd = 1.0 if float(action_pred[9]) < self._gripper_thresh else -1.0
        pos3 = action_pred[:3].astype(np.float64)

        raw_action = {
            "world_vector": pos3.copy(),
            "rot_euler_xyz_biased": euler_biased.copy(),
            "open_gripper": np.array([float(action_pred[9])], dtype=np.float64),
        }
        action = {
            "world_vector": pos3 * self.action_scale,
            "rot_axangle": euler_biased * self.action_scale,
            "gripper": np.array([grip_cmd], dtype=np.float64),
            "terminate_episode": np.array([0.0]),
        }
        return raw_action, action

    def _post_act(self, payload: dict) -> None:
        try:
            resp = requests.post(self._url, json=payload, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"X-VLA server request failed: {e}") from e
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"X-VLA server error:\n{data['error']}")
        action_seq = np.asarray(data["action"], dtype=np.float32)
        if action_seq.ndim == 1:
            action_seq = action_seq.reshape(1, -1)
        self._action_plan.extend(action_seq.tolist())

    def visualize_epoch(
        self,
        predicted_raw_actions: Sequence[dict],
        images: Sequence[np.ndarray],
        save_path: str,
    ) -> None:
        h, w = 224, 224
        images_r = [cv.resize(img, (w, h), interpolation=cv.INTER_AREA) for img in images]
        labels = ["x", "y", "z", "rx", "ry", "rz", "grasp"]
        img_strip = np.concatenate(np.array(images_r[::3]), axis=1)

        figure_layout = [["image"] * len(labels), labels]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        rows = []
        for a in predicted_raw_actions:
            if "rot_axangle_raw" in a:
                rot = a["rot_axangle_raw"]
            else:
                rot = a["rot_euler_xyz_biased"]
            rows.append(
                np.concatenate([a["world_vector"], rot, a["open_gripper"]], axis=-1)
            )
        pred_actions = np.array(rows)
        for dim, label in enumerate(labels):
            axs[label].plot(pred_actions[:, dim], label="predicted action")
            axs[label].set_title(label)
            axs[label].set_xlabel("Time in one episode")
        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
        plt.close(fig)

    def _wait_for_server(self, timeout: int = 300) -> None:
        health_url = self._url.replace("/act", "/healthz")
        deadline = time.monotonic() + timeout
        logger.info("Waiting for X-VLA server at %s …", health_url)
        while time.monotonic() < deadline:
            try:
                r = requests.get(health_url, timeout=5)
                if r.status_code in (200, 404, 405):
                    logger.info("X-VLA server is reachable.")
                    return
            except requests.RequestException:
                pass
            time.sleep(5)
        raise TimeoutError(
            f"X-VLA server did not become ready within {timeout}s. "
            "Start it with: bash dynamic_SimplerEnv/scripts/x-vla/start_xvla_server.sh"
        )
