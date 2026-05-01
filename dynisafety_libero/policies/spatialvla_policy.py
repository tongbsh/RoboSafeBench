"""SpatialVLA policy adapter for DyniSafety-LIBERO evaluation.

Connects to a running SpatialVLA FastAPI server (vlaModelHome/SpatialVLA/server.py)
via HTTP and wraps it as a VLAPolicy compatible with the benchmark runner.

Start the server first (in the spatialvla conda env):
    bash dynisafety_libero/scripts/start_spatialvla_server.sh \\
        --model-path IPEC-COMMUNITY/spatialvla-4b-224-pt \\
        --unnorm-key bridge_orig/1.0.0 \\
        --port 8112

Then run the benchmark (in the openvla-oft conda env):
    python dynisafety_libero/run_dynisafety_eval.py \\
        --model_family spatialvla \\
        --server_host localhost \\
        --server_port 8112 \\
        --task_suite_name safelibero_spatial

Action format note
------------------
SpatialVLA outputs 7-DoF actions [delta_xyz(3) | delta_rpy(3) | gripper(1/0)]
via its action tokeniser.  The gripper value is 0.0 or 1.0 (binary, decoded
from tokens).  No additional post-processing is applied — the server's
processor.decode_actions() handles un-normalisation using the dataset
statistics embedded in the checkpoint.

unnorm_key note
---------------
The unnorm_key determines which dataset's action statistics are used for
un-normalisation.  For LIBERO evaluation, use the key that matches the
checkpoint's training data (e.g. "libero_spatial" for a LIBERO-fine-tuned
model, or "bridge_orig/1.0.0" for zero-shot Bridge evaluation).
Override per-policy via SPATIALVLA_UNNORM_KEY env var.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import json_numpy
import numpy as np
import requests

from .base_policy import VLAPolicy

logger = logging.getLogger(__name__)


class SpatialVLAPolicy(VLAPolicy):
    """HTTP client policy for the SpatialVLA inference server.

    Parameters
    ----------
    cfg:
        DyniSafety-LIBERO ``GenerateConfig`` with at minimum:
        - ``server_host`` (default "localhost")
        - ``server_port`` (default 8112)

    Environment variables (optional overrides):
        SPATIALVLA_SERVER_HOST   host of the SpatialVLA server
        SPATIALVLA_SERVER_PORT   port of the SpatialVLA server
        SPATIALVLA_UNNORM_KEY    unnorm_key sent per-request (overrides server default)
    """

    def __init__(self, cfg) -> None:
        host = os.environ.get("SPATIALVLA_SERVER_HOST",
                              getattr(cfg, "server_host", "localhost"))
        port = int(os.environ.get("SPATIALVLA_SERVER_PORT",
                                  getattr(cfg, "server_port", 8112)))
        self._url = f"http://{host}:{port}/act"
        self._unnorm_key: Optional[str] = os.environ.get("SPATIALVLA_UNNORM_KEY") or None

        logger.info("SpatialVLAPolicy: server at %s", self._url)
        self._wait_for_server()

    # ------------------------------------------------------------------
    # VLAPolicy interface
    # ------------------------------------------------------------------

    @property
    def action_chunk_size(self) -> int:
        # SpatialVLA's action_chunk_size is determined by the processor;
        # typically 1 for zero-shot (token decoding), more for fine-tuned.
        # The server returns however many the processor predicts.
        return 1

    def predict_actions(self, obs: Dict[str, Any], task_description: str) -> List[np.ndarray]:
        """Send observation to SpatialVLA server and return action list.

        Parameters
        ----------
        obs:
            Dict with keys ``full_image`` (uint8 H×W×3) and ``state``.
            Only ``full_image`` is sent (SpatialVLA uses RGB only).
        task_description:
            Natural-language task instruction.

        Returns
        -------
        List[np.ndarray]
            List of shape-(7,) delta action arrays [xyz, rpy, gripper].
            Gripper is converted from server's {0=close, 1=open} to
            LIBERO env convention {-1=open, +1=close}.
        """
        payload: Dict[str, Any] = {
            "image": json_numpy.dumps(obs["full_image"]),
            "task": task_description,
        }
        if self._unnorm_key is not None:
            payload["unnorm_key"] = self._unnorm_key

        try:
            resp = requests.post(self._url, json=payload, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"SpatialVLA server request failed: {e}") from e

        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"SpatialVLA server error:\n{data['error']}")

        # actions: nested list (chunk_size, 7)
        actions = np.array(data["actions"], dtype=np.float32)  # (chunk, 7)

        # Gripper convention fix:
        # SpatialVLA decode_actions passes gripper through as {0.0=close, 1.0=open}
        # (the gripper dimension has mask=False in processor statistics so it is
        # NOT unnormalized).
        #
        # LIBERO env convention (verified from robot_utils.invert_gripper_action):
        #   -1 = open gripper,  +1 = close gripper
        # (opposite of robosuite native; dummy_action uses -1 = open during stabilisation)
        #
        # So: model 1.0(open) → env -1.0(open),  model 0.0(close) → env +1.0(close)
        actions[:, -1] = np.where(actions[:, -1] > 0.5, -1.0, 1.0)

        return [actions[i] for i in range(len(actions))]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wait_for_server(self, timeout: int = 300) -> None:
        health_url = self._url.replace("/act", "/healthz")
        deadline = time.monotonic() + timeout
        logger.info("Waiting for SpatialVLA server at %s …", health_url)
        while time.monotonic() < deadline:
            try:
                r = requests.get(health_url, timeout=5)
                if r.status_code == 200:
                    logger.info("SpatialVLA server is ready.")
                    return
            except requests.RequestException:
                pass
            time.sleep(5)
        raise TimeoutError(
            f"SpatialVLA server did not become ready within {timeout}s. "
            "Make sure to start it with start_spatialvla_server.sh first."
        )
