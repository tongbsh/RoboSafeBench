"""LeRobot π₀-fast local adapter (SimplerEnv)."""

from __future__ import annotations

from dynisafety.policies.adapters._common import (
    actions_to_chunk,
    ensure_repo_on_path,
    require_extra,
    require_image,
)
from dynisafety.policies.adapters.spatialvla import _simpler_dict_to_7d
from dynisafety.policies.base import (
    ActionChunk,
    EpisodeContext,
    ModelConfig,
    Observation,
)


class LeRobotPiFastAdapter:
    """In-process LeRobot PI0Policy path (not openpi websocket)."""

    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg
        ensure_repo_on_path(str(require_extra(cfg, "repo_root")))
        from simpler_env.policies.lerobotpi.pi0_or_fast import (  # noqa: PLC0415
            LerobotPiFastInference,
        )

        self._impl = LerobotPiFastInference(
            saved_model_path=cfg.checkpoint,
            policy_setup=str(require_extra(cfg, "policy_setup")),
            action_scale=float(require_extra(cfg, "action_scale")),
        )

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> LeRobotPiFastAdapter:
        if cfg.transport != "local":
            raise ValueError("LeRobotPiFastAdapter requires transport='local'")
        return cls(cfg)

    def reset(self, ctx: EpisodeContext) -> None:
        self._impl.reset(ctx.instruction)

    def act(self, obs: Observation) -> ActionChunk:
        raw, action = self._impl.step(require_image(obs, "primary"), obs.instruction)
        del raw
        return actions_to_chunk(
            _simpler_dict_to_7d(action).reshape(1, -1), space=self._cfg.action_space
        )

    def close(self) -> None:
        closer = getattr(self._impl, "close", None)
        if callable(closer):
            closer()
