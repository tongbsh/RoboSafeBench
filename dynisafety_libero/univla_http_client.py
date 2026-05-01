"""
HTTP client for :mod:`deploy_univla_dynisafety_server` (UniVLA FastAPI).

Used by ``scripts/univla/run_dynisafety_univla_eval.py`` from the benchmark conda env.
"""

from __future__ import annotations

import logging
import time
import numpy as np
import requests

logger = logging.getLogger(__name__)


class UniVLAHTTPClient:
    """Sequential UniVLA inference: call :meth:`reset` at each episode start, then :meth:`act` each step."""

    def __init__(
        self,
        host: str,
        port: int,
        timeout: int = 300,
        act_timeout: int = 180,
    ):
        self._base = f"http://{host}:{port}"
        self._timeout = timeout
        self._act_timeout = act_timeout
        self._wait_for_server()

    def _wait_for_server(self) -> None:
        deadline = time.monotonic() + self._timeout
        health = f"{self._base}/health"
        logger.info("Waiting for UniVLA server at %s …", self._base)
        while time.monotonic() < deadline:
            try:
                r = requests.get(health, timeout=5)
                if r.status_code == 200:
                    logger.info("UniVLA server ready.")
                    return
            except requests.RequestException:
                pass
            time.sleep(1.0)
        raise RuntimeError(f"UniVLA server at {self._base} did not become ready within {self._timeout}s")

    def reset(self) -> None:
        r = requests.post(f"{self._base}/reset", json={}, timeout=30)
        r.raise_for_status()

    def act(self, image_u8: np.ndarray, task: str, proprio: np.ndarray) -> np.ndarray:
        """image_u8: HxWx3 uint8, already 180° flipped like training; proprio matches prepare_observation state."""
        payload = {
            "image": np.asarray(image_u8, dtype=np.uint8).tolist(),
            "task": task,
            "proprio": np.asarray(proprio, dtype=np.float64).tolist(),
        }
        r = requests.post(f"{self._base}/act", json=payload, timeout=self._act_timeout)
        r.raise_for_status()
        data = r.json()
        return np.asarray(data["action"], dtype=np.float32)
