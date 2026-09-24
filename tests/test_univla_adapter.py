"""UniVLA HTTP adapter: per-slot session_id (shared-server isolation)."""

from __future__ import annotations

from typing import Any

import numpy as np

from dynisafety.policies.adapters.univla import UniVLAAdapter
from dynisafety.policies.base import EpisodeContext, ModelConfig, Observation


def _cfg() -> ModelConfig:
    return ModelConfig(
        name="univla",
        checkpoint="/tmp/univla-fake",
        action_space="delta_ee_7d",
        transport="http",
        adapter="dynisafety.policies.adapters.univla:UniVLAAdapter",
        chunk_size=1,
        observation_images=("primary",),
        observation_needs_state=True,
        backends=("libero",),
        max_jobs_per_gpu=6,
        server_env="univla",
        server_root=None,
        server_launch=None,
        server_default_port=8020,
        server_health_path="/health",
        server_startup_timeout_s=600,
        extra={
            "act_timeout_s": 180,
            "reset_path": "/reset",
            "act_path": "/act",
            "server_host": "127.0.0.1",
        },
    )


class _FakeTransport:
    def __init__(self) -> None:
        self.resets: list[dict[str, Any] | None] = []
        self.infers: list[dict[str, Any]] = []

    def reset(self, payload=None) -> None:
        self.resets.append(dict(payload) if payload is not None else None)

    def infer(self, payload):
        self.infers.append(dict(payload))
        return {"action": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]]}

    def close(self) -> None:
        return None


def _ctx(*, episode_index: int = 0) -> EpisodeContext:
    return EpisodeContext(
        instruction="pick up the black bowl",
        task_id=0,
        task_name="spatial_0",
        episode_index=episode_index,
        seed=7,
        level=1,
        backend="libero",
        suite="safelibero_spatial",
        extra={},
    )


def _obs() -> Observation:
    return Observation(
        images={"primary": np.zeros((64, 64, 3), dtype=np.uint8)},
        state=np.zeros(8, dtype=np.float32),
        instruction="pick up the black bowl",
        step=0,
        extra={},
    )


def test_univla_reset_and_act_share_a_stable_session_id() -> None:
    transport = _FakeTransport()
    adapter = UniVLAAdapter(_cfg(), transport)  # type: ignore[arg-type]
    adapter.reset(_ctx(episode_index=0))
    adapter.reset(_ctx(episode_index=1))
    chunk = adapter.act(_obs())
    assert chunk.actions.shape == (1, 7)
    assert len(transport.resets) == 2
    sid0 = transport.resets[0]["session_id"]
    sid1 = transport.resets[1]["session_id"]
    assert sid0 and sid0 == sid1
    assert transport.infers[0]["session_id"] == sid0
    adapter.close()


def test_univla_eval_slots_do_not_share_session_ids() -> None:
    t_a = _FakeTransport()
    t_b = _FakeTransport()
    a = UniVLAAdapter(_cfg(), t_a)  # type: ignore[arg-type]
    b = UniVLAAdapter(_cfg(), t_b)  # type: ignore[arg-type]
    a.reset(_ctx())
    b.reset(_ctx())
    assert t_a.resets[0]["session_id"] != t_b.resets[0]["session_id"]
    a.close()
    b.close()
