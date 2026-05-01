"""X-VLA policy adapter for DyniSafety-LIBERO evaluation.

Connects to a running X-VLA FastAPI server (vlaModelHome/X-VLA/deploy.py)
via HTTP and wraps it as a VLAPolicy compatible with the benchmark runner.

Start the server first (in the XVLA conda env):
    bash dynisafety_libero/scripts/start_xvla_server.sh \\
        --model-path 2toINF/X-VLA-Libero \\
        --port 8010

Then run the benchmark (in the openvla-oft conda env):
    python dynisafety_libero/run_dynisafety_eval.py \\
        --model_family xvla \\
        --server_host localhost \\
        --server_port 8010 \\
        --task_suite_name safelibero_spatial

Action format: (T, 10) → 7D conversion
---------------------------------------
X-VLA uses the EE6D action space with ABSOLUTE EEF targets.
Each row of the server's action output is:
    Dims  0:3  → absolute EEF position xyz (3D)
    Dims  3:9  → EEF rotation in 6D representation (Zhou et al., 2019)
    Dim   9    → gripper openness after sigmoid ∈ (0, 1)

NOTE: run_dynisafety_eval.py sets robot.controller.use_delta=False for
model_family="xvla" immediately after the warm-up phase so the env
controller interprets these as absolute targets (not relative deltas).

Conversion to 7D [pos_xyz | axangle | gripper]:
    1. 6D rotation → rotation matrix (Gram-Schmidt orthonormalisation)
    2. Rotation matrix → axis-angle via scipy.spatial.transform.Rotation
    3. Concatenate: [pos3, axangle3, grip1]

proprio format (LIBERO, domain_id=3)
-------------------------------------
X-VLA expects proprio in 20D: [pos3, rot6d6, grip1, zeros10].
The gripper slot is fixed to 0.0 (matching X-VLA's libero_client.py).

Recurrent proprio tracking
--------------------------
X-VLA was trained with recurrent proprio: after each server call, the
proprio for the NEXT call is updated from the LAST predicted action
(pos3 + rot6d6) rather than re-read from the simulator.  Only the very
first call per episode initialises proprio from the actual robot state.

gripper convention
------------------
After sigmoid, grip ∈ (0, 1).  Binarised to {-1, +1} for LIBERO env.step(),
consistent with X-VLA's existing libero_client.py behaviour.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import json_numpy
import numpy as np
import requests
from scipy.spatial.transform import Rotation

from .base_policy import VLAPolicy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _rot6d_to_axangle(rot6d: np.ndarray) -> np.ndarray:
    """Convert a 6D rotation representation to axis-angle (3D rotvec).

    The 6D representation (Zhou et al., 2019) encodes [col0, col1] of the
    rotation matrix.  col2 is recovered via the cross product.

    Parameters
    ----------
    rot6d : np.ndarray, shape (6,)

    Returns
    -------
    np.ndarray, shape (3,)  — axis-angle / rotation vector
    """
    a1, a2 = rot6d[:3], rot6d[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2_raw = a2 - np.dot(b1, a2) * b1
    b2 = b2_raw / (np.linalg.norm(b2_raw) + 1e-8)
    b3 = np.cross(b1, b2)
    rot_matrix = np.stack([b1, b2, b3], axis=1)  # (3, 3), columns = b1, b2, b3
    return Rotation.from_matrix(rot_matrix).as_rotvec()  # (3,)


def _axangle_to_rot6d(axangle: np.ndarray) -> np.ndarray:
    """Convert axis-angle (3D rotvec) to 6D rotation representation.

    Returns the first two columns of the rotation matrix concatenated: shape (6,).
    """
    rot_matrix = Rotation.from_rotvec(axangle).as_matrix()  # (3, 3)
    return np.concatenate([rot_matrix[:, 0], rot_matrix[:, 1]])  # (6,)


def _action_row_to_7d(action_row: np.ndarray) -> np.ndarray:
    """Convert a single X-VLA action row (>=10D) to 7D LIBERO format.

    Parameters
    ----------
    action_row : np.ndarray, shape (10,) or (20,)
        Only the first 10 dims are used: [pos3, rot6d6, grip1].

    Returns
    -------
    np.ndarray, shape (7,) — [pos_xyz(3), axangle(3), gripper(1)]
    """
    pos = action_row[:3]
    rot6d = action_row[3:9]
    grip_raw = action_row[9]
    axangle = _rot6d_to_axangle(rot6d)
    # Binarize gripper to {-1, +1} as expected by LIBERO env.step().
    # Matches the original X-VLA libero_client.py convention:
    #   >0.5 → 1.0 (close),  ≤0.5 → -1.0 (open)
    grip = np.array([1.0 if grip_raw > 0.5 else -1.0])
    return np.concatenate([pos, axangle, grip])  # (7,)


def _obs_state_to_proprio_20d(state: np.ndarray) -> np.ndarray:
    """Convert dynisafety_libero obs["state"] (8D) to X-VLA proprio (20D).

    obs["state"] layout: [pos3, axangle3, gripper_qpos2]
    X-VLA proprio layout: [pos3, rot6d6, grip1, zeros10]

    Gripper is fixed to 0.0 (matching X-VLA's libero_client.py).
    """
    pos = state[:3]
    axangle = state[3:6]

    rot6d = _axangle_to_rot6d(axangle)
    proprio_10d = np.concatenate([pos, rot6d, [0.0]])   # grip = 0.0
    return np.concatenate([proprio_10d, np.zeros(10, dtype=np.float32)])


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class XVLAPolicy(VLAPolicy):
    """HTTP client policy for the X-VLA inference server.

    Maintains recurrent proprio tracking across server calls within an
    episode, matching the behaviour of X-VLA's original libero_client.py.

    Call :meth:`reset` at the start of each episode to clear the recurrent
    state (``run_dynisafety_eval.py`` already does this).

    Parameters
    ----------
    cfg:
        DyniSafety-LIBERO ``GenerateConfig`` with at minimum:
        - ``server_host`` (default "localhost")
        - ``server_port`` (default 8010)

    Environment variables (optional overrides):
        XVLA_SERVER_HOST   host of the X-VLA server
        XVLA_SERVER_PORT   port of the X-VLA server
        XVLA_DOMAIN_ID     domain id sent to the server (default 3 for LIBERO)
        XVLA_STEPS         denoising steps (default 10)
    """

    def __init__(self, cfg) -> None:
        host = os.environ.get("XVLA_SERVER_HOST",
                              getattr(cfg, "server_host", "localhost"))
        port = int(os.environ.get("XVLA_SERVER_PORT",
                                  getattr(cfg, "server_port", 8010)))
        self._url = f"http://{host}:{port}/act"
        self._domain_id: int = int(os.environ.get("XVLA_DOMAIN_ID", "3"))
        self._steps: int = int(os.environ.get("XVLA_STEPS", "10"))

        # Recurrent proprio state (cleared by reset() between episodes).
        self._proprio: Optional[np.ndarray] = None

        # Actual chunk size returned by the server (discovered on first call).
        self._last_chunk_size: int = 30  # default; updated after first response

        logger.info(
            "XVLAPolicy: server at %s  domain_id=%d  steps=%d",
            self._url, self._domain_id, self._steps,
        )
        self._wait_for_server()

    # ------------------------------------------------------------------
    # VLAPolicy interface
    # ------------------------------------------------------------------

    @property
    def action_chunk_size(self) -> int:
        # Return the full server chunk size so the runner executes all actions
        # before re-querying, matching X-VLA's libero_client.py behaviour.
        return self._last_chunk_size

    def reset(self) -> None:
        """Clear recurrent state.  Call at the start of each episode."""
        self._proprio = None

    def predict_actions(self, obs: Dict[str, Any], task_description: str) -> List[np.ndarray]:
        """Send observation to X-VLA server and return the full action chunk.

        Proprio handling (matching X-VLA libero_client.py):
          - First call per episode: initialised from obs["state"] with
            gripper fixed to 0.0.
          - Subsequent calls: pos and rot6d (dims 0:9) are taken from the
            LAST predicted action of the previous chunk (recurrent tracking).

        Parameters
        ----------
        obs:
            Dict with ``full_image`` (uint8 H*W*3), ``wrist_image``
            (uint8 H*W*3), and ``state`` (float 8D).
        task_description:
            Natural-language task instruction.

        Returns
        -------
        List[np.ndarray]
            Full list of shape-(7,) action arrays from the server response.
        """
        # --- Recurrent proprio ------------------------------------------
        if self._proprio is None:
            # First call: initialise from actual robot state (grip = 0.0).
            self._proprio = _obs_state_to_proprio_20d(
                obs["state"].astype(np.float32)
            )

        # --- Wrist image: undo the 180° flip applied by prepare_observation ---
        # X-VLA's original libero_client.py does NOT flip the wrist image,
        # only the agentview.  prepare_observation flips both, so reverse it.
        wrist_img = obs["wrist_image"][::-1, ::-1].copy()

        payload = {
            "image0":               json_numpy.dumps(obs["full_image"]),
            "image1":               json_numpy.dumps(wrist_img),
            "proprio":              json_numpy.dumps(self._proprio),
            "language_instruction": task_description,
            "domain_id":            self._domain_id,
            "steps":                self._steps,
        }

        try:
            resp = requests.post(self._url, json=payload, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"X-VLA server request failed: {e}") from e

        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"X-VLA server error:\n{data['error']}")

        # Server returns {"action": [[floats] * T]}  — shape (T, 10) or (T, 20)
        raw_actions = np.array(data["action"], dtype=np.float32)
        if raw_actions.ndim == 1:
            raw_actions = raw_actions[None, :]

        self._last_chunk_size = len(raw_actions)

        # Recurrent update: track the last predicted absolute EEF
        # (pos3 + rot6d6) for the next server call.
        self._proprio[:9] = raw_actions[-1, :9].copy()

        # Convert all actions to 7D and return the full chunk.
        return [_action_row_to_7d(raw_actions[i]) for i in range(len(raw_actions))]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wait_for_server(self, timeout: int = 300) -> None:
        # X-VLA's server has a GET /healthz check via the same FastAPI app.
        # Try a lightweight GET on the base URL; fall back if no healthz.
        health_url = self._url.replace("/act", "/healthz")
        deadline = time.monotonic() + timeout
        logger.info("Waiting for X-VLA server at %s …", health_url)
        while time.monotonic() < deadline:
            try:
                r = requests.get(health_url, timeout=5)
                if r.status_code in (200, 404, 405):
                    # 404/405 means the server is up but healthz route not defined
                    logger.info("X-VLA server is reachable.")
                    return
            except requests.RequestException:
                pass
            time.sleep(5)
        raise TimeoutError(
            f"X-VLA server did not become ready within {timeout}s. "
            "Make sure to start it with start_xvla_server.sh first."
        )
