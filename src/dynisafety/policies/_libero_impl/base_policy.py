"""Abstract base class for VLA policies used in DyniSafety-LIBERO evaluation."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class VLAPolicy(ABC):
    """Protocol that all VLA policies must implement for the DyniSafety benchmark.

    The benchmark runner calls :meth:`predict_actions` once per action chunk and
    manages the open-loop queue itself.  Implementations are responsible for:

    * Resizing / tokenising the incoming images.
    * Running model inference.
    * Post-processing actions (gripper normalisation, sign inversion, …).

    This is the legacy LIBERO-side interface, kept only for the two
    implementations still reached through ``dynisafety.policies.adapters``.
    New models must implement the ``Policy`` protocol in
    ``dynisafety.policies.base`` instead — see ``docs/user/adding_a_model.md``.
    """

    @abstractmethod
    def predict_actions(self, obs: dict, task_description: str) -> List[np.ndarray]:
        """Return a list of ready-to-execute actions for one inference call.

        Parameters
        ----------
        obs:
            Dictionary with at minimum:

            * ``"full_image"``  – ``(H, W, 3)`` uint8 numpy array (agentview,
              already flipped 180°).
            * ``"wrist_image"`` – ``(H, W, 3)`` uint8 numpy array (wrist cam,
              already flipped 180°).
            * ``"state"``       – ``(proprio_dim,)`` float numpy array
              ``[eef_pos(3) | eef_axis_angle(3) | gripper_qpos(2)]``.

        task_description:
            Natural-language instruction string.

        Returns
        -------
        List[np.ndarray]
            One or more shape-``(7,)`` float arrays.  Each must be
            execution-ready: gripper action already normalised to ``[-1, +1]``
            and inverted if the model convention requires it.
        """

    @property
    def action_chunk_size(self) -> int:
        """Number of actions returned per :meth:`predict_actions` call.

        The benchmark uses this only for a consistency warning against
        ``cfg.num_open_loop_steps``.  Override when the chunk size is fixed
        and known at init time (e.g. ``return NUM_ACTIONS_CHUNK``).
        """
        return 1
