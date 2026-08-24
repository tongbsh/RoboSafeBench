"""Thin official-protocol X-VLA HTTP client for RoboTwin evaluation.

This client mirrors ``external/model_repos/X-VLA/evaluation/robotwin-2.0/client.py``
without depending on the ``RoboTwinPolicy`` interface used by
``dynamic_RoboTwin/main_inference.py``. It only knows how to:

1. serialize RoboTwin observations into the official ``/act`` payload, and
2. fetch the predicted action chunk from the running X-VLA server.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import requests

from .xvla_policy import (
    _bool_gripper_to_official,
    _json_numpy_compat_dumps,
    _quat_to_rotate6d,
)


class XVLAOfficialClient:
    """HTTP client that matches the official RoboTwin-2.0 contract."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        domain_id: int,
        steps: int,
        request_timeout: float,
        server_timeout: float,
        model_path: str = "",
    ) -> None:
        self._session = requests.Session()
        self._act_url = f"http://{host}:{port}/act"
        self._health_url = f"http://{host}:{port}/openapi.json"
        self._domain_id = int(domain_id)
        self._steps = int(steps)
        self._request_timeout = float(request_timeout)
        self._server_timeout = float(server_timeout)
        self._model_path = str(model_path)
        self._instruction = ""
        self._wait_for_server()

    @classmethod
    def load(cls, **kwargs: Any) -> XVLAOfficialClient:
        return cls(
            host=str(kwargs.pop("host", "127.0.0.1")),
            port=int(kwargs.pop("port", 8010)),
            domain_id=int(kwargs.pop("domain_id", 6)),
            steps=int(kwargs.pop("steps", 10)),
            request_timeout=float(kwargs.pop("request_timeout", 180.0)),
            server_timeout=float(kwargs.pop("server_timeout", 300.0)),
            model_path=str(kwargs.pop("model_path", "")),
        )

    @property
    def steps(self) -> int:
        return self._steps

    def close(self) -> None:
        self._session.close()

    def set_instruction(self, instruction: str) -> None:
        self._instruction = str(instruction)

    def step(self, observation: Mapping[str, Any]) -> np.ndarray:
        cameras = observation["observation"]
        endpose = observation["endpose"]

        left_ee = np.asarray(endpose["left_endpose"], dtype=np.float32)
        right_ee = np.asarray(endpose["right_endpose"], dtype=np.float32)
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
            "image0": _json_numpy_compat_dumps(np.asarray(cameras["head_camera"]["rgb"], dtype=np.uint8)),
            "image1": _json_numpy_compat_dumps(np.asarray(cameras["left_camera"]["rgb"], dtype=np.uint8)),
            "image2": _json_numpy_compat_dumps(np.asarray(cameras["right_camera"]["rgb"], dtype=np.uint8)),
            "steps": self._steps,
        }

        response = self._session.post(
            self._act_url,
            json=payload,
            timeout=self._request_timeout,
        )
        if not response.ok:
            try:
                body = response.text.strip()
            except Exception:
                body = "<unavailable>"
            raise requests.HTTPError(
                f"X-VLA request failed with status={response.status_code}: {body}",
                response=response,
            )

        body = response.json()
        if "action" not in body:
            raise KeyError(f"X-VLA response missing 'action': keys={list(body.keys())}")

        action_chunk = np.asarray(body["action"], dtype=np.float64)
        if action_chunk.ndim == 1:
            action_chunk = action_chunk[None, :]
        if action_chunk.ndim != 2 or action_chunk.shape[1] < 20:
            raise ValueError(f"Unexpected X-VLA action shape: {action_chunk.shape}")
        return action_chunk

    def _wait_for_server(self) -> None:
        start = time.time()
        while True:
            if time.time() - start > self._server_timeout:
                raise TimeoutError(
                    f"Failed to connect to X-VLA server within {self._server_timeout}s: {self._health_url}"
                )
            try:
                response = self._session.get(self._health_url, timeout=5.0)
                if response.ok:
                    return
            except requests.RequestException:
                pass
            time.sleep(2.0)


__all__ = ["XVLAOfficialClient"]
