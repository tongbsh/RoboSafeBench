"""Cross-package rng_key contract: Env construction × Infer deterministic check.

A same-function-twice test would not have caught the original bug (Env sha256
vs Infer blake2b). This file goes through Env's ObsMsg builder and Infer's
``_check_obs_rng`` on opposite sides of the process split.
"""

from __future__ import annotations

import hashlib
import inspect
import threading
from pathlib import Path

import numpy as np
import pytest

from dynisafety.parallel_eval.channel import Channel
from dynisafety.parallel_eval.config import load_parallel_eval_config
from dynisafety.parallel_eval.env_worker import SlotEpisodeJob, _obs_msg, serve_env_slot
from dynisafety.parallel_eval.infer_worker import _check_obs_rng
from dynisafety.parallel_eval.protocol import (
    PROPRIO_DIM,
    ActMsg,
    ObsMsg,
    from_wire,
    mix_rng_key,
    to_wire,
)
from dynisafety.policies.base import Observation
from test_parallel_eval_env import (
    _GRIPPER_THRESHOLD,
    _cfg_bound_resolved,
    _FakeTaskEnv,
    _handle,
    _valid_actions_20,
)

# Pre-move blake2b(digest_size=8) value. Changing the hash or field order
# would silently reshuffle x1 and break the P0 bit-alignment gate.
_GOLDEN_KEY = 15913000601788401206
_RNG_SEED = 20260825
_CFG_SEED = 7  # RunConfig.seed in the env-test fixture; must stay != rng_seed


def _infer_cfg(*, rng_seed: int):
    return load_parallel_eval_config(
        {
            "extra": {
                "parallel_eval": {
                    "enabled": True,
                    "num_envs": 1,
                    "pipeline_stage_num": 1,
                    "batch_window_ms": 15,
                    "infer_conda_env": "XVLA",
                    "env_conda_env": "internvla_a1",
                    "infer_device": "cpu",
                    "env_cuda_visible_devices": "0",
                    "channel_host": "127.0.0.1",
                    "channel_port": 0,
                    "channel_timeout_s": 2.0,
                    "torch_dtype": "float32",
                    "deterministic_rng": True,
                    "rng_seed": rng_seed,
                }
            }
        }
    )


def _old_sha256_key(*, episode_uid: str, step: int, run_seed: int) -> int:
    """The Env-side algorithm that made P0 raise under deterministic_rng."""
    payload = f"{episode_uid}\0{int(step)}\0{int(run_seed)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def test_env_obs_rng_key_passes_infer_deterministic_check(tmp_path: Path) -> None:
    """Env-built ObsMsg.rng_key must be the value Infer's check expects.

    ``cfg.seed`` is deliberately not ``rng_seed``. The original Env worker
    hashed ``cfg.seed`` with sha256; that path must not satisfy Infer.
    """
    assert inspect.signature(_obs_msg).parameters["rng_seed"].default is inspect.Parameter.empty
    assert _CFG_SEED != _RNG_SEED

    observation = Observation(
        images={
            "primary": np.zeros((4, 4, 3), dtype=np.uint8),
            "left": np.full((4, 4, 3), 11, dtype=np.uint8),
            "right": np.full((4, 4, 3), 22, dtype=np.uint8),
        },
        state=None,
        instruction="place the cup",
        step=7,
        extra={},
    )
    built = _obs_msg(
        observation,
        slot_id=0,
        episode_uid="ep-0",
        proprio=np.zeros(PROPRIO_DIM, dtype=np.float32),
        domain_id=6,
        denoising_steps=4,
        rng_seed=_RNG_SEED,
    )
    assert isinstance(built, ObsMsg)
    assert built.rng_key == _GOLDEN_KEY
    assert built.rng_key == mix_rng_key(rng_seed=_RNG_SEED, episode_uid="ep-0", step=7)
    assert built.rng_key != mix_rng_key(rng_seed=_CFG_SEED, episode_uid="ep-0", step=7)
    assert built.rng_key != _old_sha256_key(episode_uid="ep-0", step=7, run_seed=_RNG_SEED)
    assert built.rng_key != _old_sha256_key(episode_uid="ep-0", step=7, run_seed=_CFG_SEED)

    infer_cfg = _infer_cfg(rng_seed=_RNG_SEED)
    _check_obs_rng(built, infer_cfg)

    stale = ObsMsg(
        slot_id=built.slot_id,
        episode_uid=built.episode_uid,
        step=built.step,
        images=built.images,
        proprio=built.proprio,
        instruction=built.instruction,
        domain_id=built.domain_id,
        denoising_steps=built.denoising_steps,
        rng_key=_old_sha256_key(episode_uid="ep-0", step=7, run_seed=_RNG_SEED),
    )
    with pytest.raises(ValueError, match="rng_key"):
        _check_obs_rng(stale, infer_cfg)

    env = _FakeTaskEnv()
    handle = _handle(tmp_path, env)
    cfg, bound, resolved = _cfg_bound_resolved(tmp_path, chunk_size=2, max_steps=2)
    assert int(cfg.seed) == _CFG_SEED
    job = SlotEpisodeJob(episode_uid="ep-0", episode_index=0, start_seed=11)
    listener = Channel.bind(host="127.0.0.1", port=0, timeout_s=5.0)
    err: list[BaseException] = []

    def _worker() -> None:
        try:
            serve_env_slot(
                slot_id=0,
                host=listener.host,
                port=listener.port,
                timeout_s=5.0,
                handle=handle,
                cfg=cfg,
                bound=bound,
                resolved_data=resolved,
                workspace=tmp_path,
                observation_needs_state=False,
                warmup_steps=None,
                post_warmup_space=None,
                gripper_threshold=_GRIPPER_THRESHOLD,
                domain_id=6,
                denoising_steps=10,
                rng_seed=_RNG_SEED,
                jobs=[job],
            )
        except BaseException as exc:  # noqa: BLE001
            err.append(exc)

    thread = threading.Thread(target=_worker)
    thread.start()
    try:
        with listener.accept() as peer:
            from_wire(peer.get())  # worker_ready
            from_wire(peer.get())  # episode_begin
            live = from_wire(peer.get())
            assert isinstance(live, ObsMsg)
            assert live.episode_uid == "ep-0"
            expected = mix_rng_key(
                rng_seed=_RNG_SEED,
                episode_uid=live.episode_uid,
                step=live.step,
            )
            used_cfg_seed = mix_rng_key(
                rng_seed=_CFG_SEED,
                episode_uid=live.episode_uid,
                step=live.step,
            )
            assert live.rng_key == expected
            assert live.rng_key != used_cfg_seed
            assert live.rng_key != _old_sha256_key(
                episode_uid=live.episode_uid,
                step=live.step,
                run_seed=_CFG_SEED,
            )
            _check_obs_rng(live, infer_cfg)
            peer.put(
                to_wire(
                    ActMsg(
                        slot_id=0,
                        episode_uid="ep-0",
                        step=live.step,
                        actions=_valid_actions_20(4),
                    )
                )
            )
            from_wire(peer.get())  # episode_end
    finally:
        thread.join(timeout=5.0)
        listener.close()

    assert err == []
