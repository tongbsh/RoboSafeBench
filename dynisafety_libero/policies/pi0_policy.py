"""π₀ / π₀.₅ (openpi) policy adapter for DyniSafety-LIBERO evaluation.

This module connects to a running openpi WebSocket policy server and wraps it
as a :class:`VLAPolicy` compatible with the DyniSafety-LIBERO benchmark runner.

Because openpi uses JAX and is managed by ``uv``, its dependencies are
incompatible with the ``openvla-oft`` conda environment used to run the
benchmark.  The solution is a **server-client split**:

* **Server** (openpi env): start once before running the benchmark::

      cd vlaModelHome/openpi
      # with a local checkpoint:
      uv run scripts/serve_policy.py policy:checkpoint \\
          --policy.config pi0_libero \\
          --policy.dir /path/to/checkpoints/pi0_libero/my_exp/20000 \\
          --port 8000
      # or auto-download from GCS:
      uv run scripts/serve_policy.py policy:checkpoint \\
          --policy.config pi0_libero \\
          --policy.dir gs://openpi-assets/checkpoints/pi0_libero \\
          --port 8000

* **Client** (openvla-oft env): this file, instantiated by the benchmark::

      python run_dynisafety_eval.py \\
          --model_family pi0 \\
          --server_host localhost \\
          --server_port 8000 \\
          --task_suite_name safelibero_spatial

Observation format mapping
--------------------------
The benchmark provides::

    obs = {
        "full_image":  uint8 (H, W, 3),   # agentview, already flipped 180°
        "wrist_image": uint8 (H, W, 3),   # wrist cam, already flipped 180°
        "state":       float64 (8,),      # pos(3)+axis_angle(3)+gripper_qpos(2)
    }

openpi/LIBERO server expects::

    {
        "observation/image":       uint8 (H, W, 3),
        "observation/wrist_image": uint8 (H, W, 3),
        "observation/state":       float (8,),
        "prompt":                  str,
    }

The ``state`` vector layout is identical, so only key renaming is needed.

Action format
-------------
openpi returns ``result["actions"]`` with shape ``(chunk_size, 7)`` where
the 7 dimensions are ``[delta_xyz(3), delta_axis_angle(3), gripper(1)]`` in
the LIBERO convention — already de-normalised and ready for ``env.step()``.

Unlike OpenVLA-OFT, openpi does **not** require gripper normalisation or
sign-inversion post-processing; the model outputs the correct LIBERO gripper
convention directly.

Path resolution
---------------
The module locates ``openpi-client`` relative to the workspace root.
Override with the ``OPENPI_ROOT`` environment variable::

    OPENPI_ROOT=/custom/path/to/openpi python -m dynisafety_libero.run_dynisafety_eval ...
"""

import logging
import os
import sys
from pathlib import Path
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Locate openpi-client and inject it onto sys.path if not already importable.
# Layout assumed by default:
#
#   <workspace>/
#   ├── dynisafety_libero/       ← this file lives 2 levels up
#   └── vlaModelHome/
#       └── openpi/
#           └── packages/
#               └── openpi-client/src/
#
# Override with:  export OPENPI_ROOT=/path/to/openpi
# ---------------------------------------------------------------------------
_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
_OPENPI_ROOT = Path(
    os.environ.get("OPENPI_ROOT", str(_WORKSPACE_ROOT / "vlaModelHome" / "openpi"))
)
_OPENPI_CLIENT_SRC = _OPENPI_ROOT / "packages" / "openpi-client" / "src"

if not _OPENPI_ROOT.exists():
    raise RuntimeError(
        f"openpi root not found at '{_OPENPI_ROOT}'. "
        "Clone the repo there or set the OPENPI_ROOT environment variable."
    )

if str(_OPENPI_CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(_OPENPI_CLIENT_SRC))

from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: E402

from .base_policy import VLAPolicy  # noqa: E402

logger = logging.getLogger(__name__)


class Pi0Policy(VLAPolicy):
    """Wraps an openpi π₀ / π₀.₅ server for DyniSafety-LIBERO evaluation.

    The policy communicates with a running ``serve_policy.py`` WebSocket
    server over the openpi msgpack protocol.  The server must be started
    separately in the openpi conda/uv environment before launching the
    benchmark.

    Parameters
    ----------
    cfg:
        DyniSafety-LIBERO ``GenerateConfig`` dataclass (or any object with
        ``server_host`` and ``server_port`` attributes).  The policy reads:

        * ``cfg.server_host`` (default ``"localhost"``)
        * ``cfg.server_port`` (default ``8000``)

        All other config fields are ignored — the server determines the
        model, checkpoint, and action chunk size.
    """

    def __init__(self, cfg) -> None:
        host = getattr(cfg, "server_host", "localhost")
        port = getattr(cfg, "server_port", 8000)

        logger.info(
            "Pi0Policy: connecting to openpi server at ws://%s:%d …", host, port
        )
        self._client = WebsocketClientPolicy(host=host, port=port)

        # Try to read chunk size from server metadata (key varies by config).
        metadata = self._client.get_server_metadata()
        logger.info("Pi0Policy: server metadata = %s", metadata)

        # openpi's LIBERO config outputs 50-step chunks; the benchmark
        # controls how many steps are actually executed via num_open_loop_steps.
        # We expose the full chunk to let the benchmark manage the queue.
        # The server sends empty metadata for pi0_libero, so fall back to
        # action_horizon=50 (hardcoded in pi0_config.Pi0Config).
        chunk = metadata.get("action_chunk_size", metadata.get("chunk_size", None))
        self._chunk_size: int = int(chunk) if chunk is not None else 50

    # ------------------------------------------------------------------
    # VLAPolicy interface
    # ------------------------------------------------------------------

    @property
    def action_chunk_size(self) -> int:
        return self._chunk_size

    def predict_actions(self, obs: dict, task_description: str) -> List[np.ndarray]:
        """Send observation to the openpi server and return an action chunk.

        Parameters
        ----------
        obs:
            Dict with keys ``full_image``, ``wrist_image`` (uint8 numpy,
            already flipped by the benchmark), and ``state`` (float numpy).
        task_description:
            Natural-language task instruction.

        Returns
        -------
        List[np.ndarray]
            List of shape-``(7,)`` actions for one inference call.
            Each action is execution-ready (LIBERO convention, no additional
            gripper post-processing required).
        """
        element = {
            "observation/image":       obs["full_image"],
            "observation/wrist_image": obs["wrist_image"],
            "observation/state":       obs["state"].astype(np.float32),
            "prompt":                  task_description,
        }

        result = self._client.infer(element)
        actions = result["actions"]  # shape (chunk_size, 7)

        if actions.ndim == 1:
            # Scalar chunk — unlikely but guard against it.
            return [actions]

        return [actions[i] for i in range(len(actions))]
