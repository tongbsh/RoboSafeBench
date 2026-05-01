"""OpenVLA / OFT policy adapter for DyniSafety-LIBERO evaluation.

This module encapsulates *all* openvla-oft–specific code so the benchmark
runner (run_dynisafety_eval.py) has zero direct dependency on ``prismatic``
or the openvla experiment utilities.

Path resolution
---------------
The module locates the ``openvla-oft`` repository relative to the workspace
root (the directory that contains both ``dynisafety_libero/`` and
``openvla-oft/``).  Override with the ``OPENVLA_ROOT`` environment variable::

    OPENVLA_ROOT=/custom/path/to/openvla-oft python -m dynisafety_libero.run_dynisafety_eval ...

Adding a new model
------------------
You do **not** need to touch this file.  Create a new file (e.g.
``policies/pi0_policy.py``) that subclasses :class:`VLAPolicy`, then call
``register_policy("pi0", Pi0Policy)`` and pass ``--model_family pi0`` at
runtime.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Locate and inject openvla-oft onto sys.path before importing its modules.
# The layout assumed by default:
#
#   <workspace>/
#   ├── dynisafety_libero/          ← this file lives 2 levels up from here
#   └── openvla-oft/                ← default location of the VLA repo
#
# Override with:  export OPENVLA_ROOT=/path/to/openvla-oft
# ---------------------------------------------------------------------------
_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent  # dynisafety_libero/policies/ → workspace
_OPENVLA_ROOT = Path(os.environ.get("OPENVLA_ROOT", str(_WORKSPACE_ROOT / "vlaModelHome" / "openvla-oft")))

if not _OPENVLA_ROOT.exists():
    raise RuntimeError(
        f"openvla-oft root not found at '{_OPENVLA_ROOT}'. "
        "Clone the repo there or set the OPENVLA_ROOT environment variable."
    )

if str(_OPENVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(_OPENVLA_ROOT))

# ---------------------------------------------------------------------------
# Now safe to import openvla-oft modules.
# ---------------------------------------------------------------------------
from experiments.robot.openvla_utils import (  # noqa: E402
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (  # noqa: E402
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK  # noqa: E402

from .base_policy import VLAPolicy  # noqa: E402

logger = logging.getLogger(__name__)

# SafeLIBERO suite → vanilla-LIBERO un-norm key fallback map
_SAFELIBERO_TO_LIBERO_KEY = {
    "safelibero_spatial": "libero_spatial",
    "safelibero_object": "libero_object",
    "safelibero_goal": "libero_goal",
    "safelibero_long": "libero_10",
}


class OpenVLAPolicy(VLAPolicy):
    """Wraps an OpenVLA / OFT checkpoint for DyniSafety-LIBERO evaluation."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg

        # Validate openvla-specific config fields
        assert cfg.pretrained_checkpoint, "pretrained_checkpoint must not be empty!"
        if "image_aug" in str(cfg.pretrained_checkpoint):
            assert cfg.center_crop, "Expecting center_crop=True for image-aug trained model!"
        assert not (cfg.load_in_8bit and cfg.load_in_4bit), \
            "Cannot use both 8-bit and 4-bit quantization!"

        self.model = get_model(cfg)
        self._resolve_unnorm_key()

        self.processor = get_processor(cfg)
        self.proprio_projector = (
            get_proprio_projector(cfg, self.model.llm_dim, proprio_dim=8)
            if cfg.use_proprio else None
        )
        self.action_head = (
            get_action_head(cfg, self.model.llm_dim)
            if (cfg.use_l1_regression or cfg.use_diffusion) else None
        )
        self.noisy_action_projector = (
            get_noisy_action_projector(cfg, self.model.llm_dim)
            if cfg.use_diffusion else None
        )

    # ------------------------------------------------------------------
    # VLAPolicy interface
    # ------------------------------------------------------------------

    @property
    def action_chunk_size(self) -> int:
        return NUM_ACTIONS_CHUNK

    def predict_actions(self, obs: dict, task_description: str) -> List[np.ndarray]:
        """Resize images, run OpenVLA inference, post-process, return action list.

        Parameters
        ----------
        obs:
            Dict with keys ``full_image``, ``wrist_image`` (uint8 numpy, already
            flipped by the benchmark), and ``state`` (float numpy).
        task_description:
            Natural-language task instruction.

        Returns
        -------
        List[np.ndarray]
            List of shape-``(7,)`` post-processed actions (one per chunk step).
        """
        resize_size = get_image_resize_size(self.cfg)

        model_obs = {
            "full_image": resize_image_for_policy(obs["full_image"], resize_size),
            "wrist_image": resize_image_for_policy(obs["wrist_image"], resize_size),
            "state": obs["state"],
        }

        raw_actions = get_action(
            self.cfg,
            self.model,
            model_obs,
            task_description,
            processor=self.processor,
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            noisy_action_projector=self.noisy_action_projector,
            use_film=self.cfg.use_film,
        )

        # Normalise to list-of-arrays regardless of whether the model returned
        # a single array (1-step) or a list (chunked).
        if isinstance(raw_actions, np.ndarray) and raw_actions.ndim == 1:
            raw_actions = [raw_actions]

        return [self._postprocess(a) for a in raw_actions]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _postprocess(self, action: np.ndarray) -> np.ndarray:
        """Normalise gripper and invert sign (openvla convention)."""
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
        return action

    def _resolve_unnorm_key(self) -> None:
        """Determine and set ``cfg.unnorm_key``, falling back to LIBERO keys."""
        unnorm_key = self.cfg.task_suite_name

        if unnorm_key not in self.model.norm_stats:
            if f"{unnorm_key}_no_noops" in self.model.norm_stats:
                unnorm_key = f"{unnorm_key}_no_noops"

        if unnorm_key not in self.model.norm_stats:
            libero_key = _SAFELIBERO_TO_LIBERO_KEY.get(self.cfg.task_suite_name)
            if libero_key and libero_key in self.model.norm_stats:
                logger.warning("Falling back to LIBERO unnorm key '%s'.", libero_key)
                unnorm_key = libero_key
            elif libero_key and f"{libero_key}_no_noops" in self.model.norm_stats:
                unnorm_key = f"{libero_key}_no_noops"
                logger.warning("Falling back to LIBERO unnorm key '%s'.", unnorm_key)

        assert unnorm_key in self.model.norm_stats, (
            f"Action un-norm key '{unnorm_key}' not found in model.norm_stats. "
            f"Available: {list(self.model.norm_stats.keys())}"
        )
        self.cfg.unnorm_key = unnorm_key
