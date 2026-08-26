"""RoboTwin expert-check cache + build_observation raw_obs reuse (no sapien)."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from dynisafety.backends.robotwin.env import build_observation, expert_check_setup
from dynisafety.backends.robotwin.expert_cache import (
    cache_key,
    load_entry,
    resolve_expert_check_cache_dir,
    store_entry,
)


def _sample_raw_obs() -> dict[str, Any]:
    return {
        "observation": {
            "head_camera": {"rgb": np.zeros((240, 320, 3), dtype=np.uint8)},
            "left_camera": {"rgb": np.ones((240, 320, 3), dtype=np.uint8)},
            "right_camera": {"rgb": np.full((240, 320, 3), 2, dtype=np.uint8)},
        },
        "endpose": {
            "left_endpose": np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]),
            "right_endpose": np.array([0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0]),
            "left_gripper": 0.0,
            "right_gripper": 1.0,
        },
        "joint_action": {"vector": np.arange(14, dtype=np.float64)},
    }


class _CountingEnv:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls = 0

    def get_obs(self) -> dict[str, Any]:
        self.calls += 1
        return self.payload


def test_build_observation_reuses_raw_obs_without_get_obs() -> None:
    payload = _sample_raw_obs()
    env = _CountingEnv(payload)
    handle = SimpleNamespace(task_env=env)
    obs = build_observation(
        handle,  # type: ignore[arg-type]
        instruction="handover block",
        step=4,
        needs_state=True,
        raw_obs=payload,
    )
    assert env.calls == 0
    assert obs.step == 4
    assert obs.instruction == "handover block"
    assert obs.images["primary"].shape == (240, 320, 3)
    assert "left" in obs.images and "right" in obs.images
    assert obs.state is not None and obs.state.shape == (14,)
    assert obs.extra["endpose"]["left_gripper"] == 0.0


def test_build_observation_fetches_when_raw_obs_omitted() -> None:
    payload = _sample_raw_obs()
    env = _CountingEnv(payload)
    handle = SimpleNamespace(task_env=env)
    obs = build_observation(
        handle,  # type: ignore[arg-type]
        instruction="lift pot",
        step=0,
        needs_state=True,
    )
    assert env.calls == 1
    assert obs.images["primary"].shape == (240, 320, 3)


class _ScriptEnv:
    """Accept the first seed >= ``accept_at``; count play_once calls."""

    def __init__(self, *, accept_at: int) -> None:
        self.accept_at = accept_at
        self.play_calls = 0
        self.setup_seeds: list[int] = []
        self.plan_success = False

    def setup_demo(self, **kwargs: Any) -> None:
        self.setup_seeds.append(int(kwargs["seed"]))

    def play_once(self) -> dict[str, Any]:
        self.play_calls += 1
        seed = self.setup_seeds[-1]
        self.plan_success = seed >= self.accept_at
        time.sleep(0.05)
        return {"info": {"demo_seed": seed}}

    def check_success(self) -> bool:
        return self.plan_success

    def close_env(self) -> None:
        return None


def _handle(env: _ScriptEnv, *, task_name: str = "handover_block") -> SimpleNamespace:
    return SimpleNamespace(
        task_env=env,
        task=SimpleNamespace(task_name=task_name),
        task_args={"task_config": "demo_clean"},
        episode_info=None,
    )


def _run_check(
    handle: SimpleNamespace,
    *,
    seed: int,
    episode_index: int = 0,
    cache_dir: Any = None,
) -> Any:
    return expert_check_setup(
        handle,  # type: ignore[arg-type]
        seed=seed,
        episode_index=episode_index,
        enabled=True,
        max_attempts=20,
        timeout_s=30.0,
        log_every=10,
        cache_dir=cache_dir,
    )


def test_expert_check_disabled_skips_play() -> None:
    env = _ScriptEnv(accept_at=0)
    handle = _handle(env)
    out = expert_check_setup(
        handle,  # type: ignore[arg-type]
        seed=16000,
        episode_index=0,
        enabled=False,
        max_attempts=20,
        timeout_s=30.0,
        log_every=10,
        cache_dir=None,
    )
    assert out.accepted_seed == 16000
    assert out.cache_hit is False
    assert env.play_calls == 0


def test_expert_check_cache_hit_skips_second_play_once(tmp_path: Any) -> None:
    env = _ScriptEnv(accept_at=16000)
    handle = _handle(env)
    first = _run_check(handle, seed=16000, cache_dir=tmp_path)
    assert first.cache_hit is False
    assert first.accepted_seed == 16000
    assert env.play_calls == 1
    assert first.episode_info["demo_seed"] == 16000

    second = _run_check(handle, seed=16000, cache_dir=tmp_path)
    assert second.cache_hit is True
    assert second.accepted_seed == 16000
    assert env.play_calls == 1
    assert second.episode_info["demo_seed"] == 16000


def test_expert_check_cache_stores_rejected_then_accepted(tmp_path: Any) -> None:
    env = _ScriptEnv(accept_at=16002)
    handle = _handle(env)
    first = _run_check(handle, seed=16000, episode_index=2, cache_dir=tmp_path)
    assert first.cache_hit is False
    assert first.accepted_seed == 16002
    assert env.play_calls == 3

    env2 = _ScriptEnv(accept_at=99999)
    handle2 = _handle(env2)
    second = _run_check(handle2, seed=16000, episode_index=2, cache_dir=tmp_path)
    assert second.cache_hit is True
    assert second.accepted_seed == 16002
    assert env2.play_calls == 0


def test_expert_check_cache_miss_on_different_start_seed(tmp_path: Any) -> None:
    env = _ScriptEnv(accept_at=0)
    handle = _handle(env)
    _run_check(handle, seed=16000, cache_dir=tmp_path)
    _run_check(handle, seed=16001, cache_dir=tmp_path)
    assert env.play_calls == 2


def test_expert_check_without_cache_dir_always_plays(tmp_path: Any) -> None:
    del tmp_path
    env = _ScriptEnv(accept_at=16000)
    handle = _handle(env)
    _run_check(handle, seed=16000, cache_dir=None)
    _run_check(handle, seed=16000, cache_dir=None)
    assert env.play_calls == 2


def test_expert_check_parallel_second_waits_for_cache(tmp_path: Any) -> None:
    env = _ScriptEnv(accept_at=16000)
    errors: list[BaseException] = []
    results: list[Any] = []

    def _worker() -> None:
        try:
            handle = _handle(env)
            results.append(_run_check(handle, seed=16000, cache_dir=tmp_path))
        except BaseException as exc:  # noqa: BLE001 — collect for the parent thread
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert env.play_calls == 1
    assert {out.accepted_seed for out in results} == {16000}
    assert sum(1 for out in results if out.cache_hit) == 1


def test_cache_key_and_roundtrip(tmp_path: Any) -> None:
    key = cache_key(
        task_name="handover_block",
        task_config="demo_clean",
        start_seed=16000,
        episode_index=3,
    )
    assert key == "handover_block__demo_clean__s16000__e3"
    store_entry(
        tmp_path,
        key,
        accepted_seed=16004,
        episode_info={"ok": True},
        task_name="handover_block",
        task_config="demo_clean",
        start_seed=16000,
        episode_index=3,
    )
    loaded = load_entry(tmp_path, key)
    assert loaded is not None
    assert loaded.accepted_seed == 16004
    assert loaded.episode_info == {"ok": True}


def test_resolve_cache_dir_env_override(tmp_path: Any) -> None:
    custom = tmp_path / "shared"
    resolved = resolve_expert_check_cache_dir(
        tmp_path / "ws",
        env={"DYNI_EXPERT_CHECK_CACHE_DIR": str(custom)},
    )
    assert resolved == custom


def test_resolve_cache_dir_workspace_layout(tmp_path: Any) -> None:
    ws = tmp_path / "ws"
    resolved = resolve_expert_check_cache_dir(ws, env={})
    assert resolved == ws / "old_cache" / "results_legacy" / "_cache" / "robotwin_expert_check"


def test_expert_check_requires_task_config(tmp_path: Any) -> None:
    env = _ScriptEnv(accept_at=0)
    handle = SimpleNamespace(
        task_env=env,
        task=SimpleNamespace(task_name="handover_block"),
        task_args={},
        episode_info=None,
    )
    with pytest.raises(ValueError, match="task_config"):
        _run_check(handle, seed=1, cache_dir=tmp_path)
