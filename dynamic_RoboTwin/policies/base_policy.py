"""Abstract base class for DyniSafety-RoboTwin policy adapters.

Each adapter encapsulates everything model-specific:
  * loading the checkpoint / connecting to a remote inference server,
  * shaping the RoboTwin observation (head + wrist cameras + joint state +
    instruction) into the model's input dictionary,
  * predicting an action chunk and unrolling it on the env via
    ``task_env.take_action(...)``,
  * deciding whether the action is dispatched as ``qpos`` (InternVLA-A1) or
    ``ee`` (lingbot-va).

Keeping all of this inside the policy means
:mod:`dynamic_RoboTwin.main_inference` stays model-agnostic - it only owns
the env/episode loop and the dynisafety wrapper plumbing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class RoboTwinPolicy(ABC):
    """Common interface every DyniSafety-RoboTwin VLA adapter must implement.

    The :mod:`dynamic_RoboTwin.main_inference` driver uses the adapter as
    follows::

        policy = SomePolicy.load(**policy_kwargs)
        ...
        for episode in episodes:
            policy.reset(instruction, TASK_ENV, initial_obs)
            while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
                obs = TASK_ENV.get_obs()
                policy.step(TASK_ENV, obs)
                if TASK_ENV.eval_success:
                    break

    Adapters therefore own their own action-chunk buffering and chose how
    often to run model inference.
    """

    name: str = ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def load(cls, **kwargs: Any) -> "RoboTwinPolicy":
        """Construct an adapter instance.

        Concrete adapters should accept the kwargs they actually need and
        ignore unknown ones (so a single ``--policy_kwargs`` JSON map can be
        fed to multiple models).
        """

    def close(self) -> None:
        """Release any external resources (sockets, GPU memory).

        Default is a no-op so adapters that do not need cleanup can ignore
        it.
        """

    # ------------------------------------------------------------------
    # Per-episode hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def reset(
        self,
        instruction: str,
        task_env: Any,
        initial_obs: Mapping[str, Any],
    ) -> None:
        """Called once at the start of every episode.

        Adapters typically use this hook to clear their action-chunk
        buffer, push the language instruction to the underlying model, and
        record any initial state needed to reconstruct delta actions.
        """

    @abstractmethod
    def step(self, task_env: Any, observation: Mapping[str, Any]) -> None:
        """Advance the env by exactly ONE simulator step.

        Adapters MUST call ``task_env.take_action(action, action_type=...)``
        exactly once per invocation (matching the cadence of the existing
        InternVLA-A1 / lingbot-va reference code). They may skip running
        full model inference on most calls and instead pop a precomputed
        action from their own buffer.
        """


__all__ = ["RoboTwinPolicy"]
