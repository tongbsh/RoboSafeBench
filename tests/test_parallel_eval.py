"""Contract tests for parallel_eval (no weights, no simulator)."""

from __future__ import annotations

import threading
from collections.abc import Iterator

import numpy as np
import pytest

from dynisafety.parallel_eval.channel import (
    Channel,
    ChannelClosed,
    ChannelError,
    ChannelTimeout,
)
from dynisafety.parallel_eval.config import REQUIRED_KEYS, load_parallel_eval_config
from dynisafety.parallel_eval.protocol import (
    ACTION_DIM,
    PROPRIO_DIM,
    ActMsg,
    ObsMsg,
    ctrl_episode_begin,
    ctrl_episode_end,
    ctrl_error,
    ctrl_shutdown,
    ctrl_worker_ready,
    from_wire,
    gather_by_slot_ids,
    group_by_slot,
    index_by_slot,
    mix_rng_key,
    to_wire,
    ungroup_by_slot,
    validate_payload,
)
from dynisafety.policies.base import ModelConfig


def _obs(*, slot_id: int, step: int, episode_uid: str = "ep-a") -> ObsMsg:
    images = [
        np.zeros((6, 8, 3), dtype=np.uint8),
        np.full((6, 8, 3), 17, dtype=np.uint8),
        np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3),
    ]
    proprio = np.linspace(-1.0, 1.0, PROPRIO_DIM, dtype=np.float32)
    return ObsMsg(
        slot_id=slot_id,
        episode_uid=episode_uid,
        step=step,
        images=images,
        proprio=proprio,
        instruction="place the cup",
        domain_id=6,
        denoising_steps=10,
        rng_key=99,
    )


def _act(*, slot_id: int, step: int, episode_uid: str = "ep-a") -> ActMsg:
    actions = np.arange(3 * ACTION_DIM, dtype=np.float32).reshape(3, ACTION_DIM)
    return ActMsg(
        slot_id=slot_id,
        episode_uid=episode_uid,
        step=step,
        actions=actions,
    )


def _parallel_eval_block() -> dict[str, object]:
    return {
        "enabled": False,
        "num_envs": 2,
        "pipeline_stage_num": 1,
        "batch_window_ms": 15,
        "infer_conda_env": "XVLA",
        "env_conda_env": "internvla_a1",
        "infer_device": "cuda:0",
        "env_cuda_visible_devices": "0",
        "channel_host": "127.0.0.1",
        "channel_port": 0,
        "channel_timeout_s": 120,
        "torch_dtype": "float32",
        "deterministic_rng": False,
        "rng_seed": 0,
    }


def _model_config(extra: dict[str, object]) -> ModelConfig:
    return ModelConfig(
        name="xvla_official_parallel",
        checkpoint="/ckpt",
        action_space="abs_ee_16d",
        transport="local",
        adapter="xvla",
        chunk_size=30,
        observation_images=("primary", "left", "right"),
        observation_needs_state=True,
        backends=("robotwin",),
        max_jobs_per_gpu=1,
        server_env=None,
        server_root=None,
        server_launch=None,
        server_default_port=None,
        server_health_path=None,
        server_startup_timeout_s=None,
        extra=extra,
    )


@pytest.fixture
def tcp_pair() -> Iterator[tuple[Channel, Channel]]:
    timeout_s = 2.0
    listener = Channel.bind(host="127.0.0.1", port=0, timeout_s=timeout_s)
    accepted: list[Channel] = []

    def _accept() -> None:
        accepted.append(listener.accept())

    thread = threading.Thread(target=_accept, daemon=True)
    thread.start()
    client = Channel.connect(
        host=listener.host, port=listener.port, timeout_s=timeout_s
    )
    thread.join(timeout=timeout_s)
    if not accepted:
        client.close()
        listener.close()
        raise AssertionError("listener.accept() did not return a channel")
    server = accepted[0]
    try:
        yield server, client
    finally:
        client.close()
        server.close()
        listener.close()


def test_mix_rng_key_golden_value_is_stable() -> None:
    """Pre-move blake2b value. Changing the algorithm would break P0 x1 alignment."""
    assert mix_rng_key(rng_seed=20260825, episode_uid="ep-0", step=7) == 15913000601788401206


def test_validate_payload_rejects_custom_class_with_path() -> None:
    class _Bad:
        pass

    payload = {"nested": {"images": [np.zeros(2, dtype=np.uint8), _Bad()]}}
    with pytest.raises(TypeError, match=r"nested\.images\[1\]"):
        validate_payload(payload)


def test_validate_payload_rejects_torch_like_tensor_with_path() -> None:
    torch_tensor = type("Tensor", (), {"__module__": "torch"})()
    payload = {"actions": torch_tensor}
    with pytest.raises(TypeError, match=r"actions.*torch"):
        validate_payload(payload)


def test_validate_payload_accepts_numpy_and_scalars() -> None:
    validate_payload(
        {
            "slot_id": 0,
            "ok": True,
            "score": 1.5,
            "name": "obs",
            "empty": None,
            "arr": np.arange(4, dtype=np.float32),
            "views": [np.zeros((2, 2), dtype=np.uint8)],
        }
    )


def test_wire_roundtrip_preserves_numpy_dtype_shape_values() -> None:
    obs = _obs(slot_id=3, step=4)
    restored = from_wire(to_wire(obs))
    assert isinstance(restored, ObsMsg)
    assert restored.slot_id == obs.slot_id
    assert restored.episode_uid == obs.episode_uid
    assert restored.step == obs.step
    assert restored.instruction == obs.instruction
    assert restored.domain_id == obs.domain_id
    assert restored.denoising_steps == obs.denoising_steps
    assert restored.rng_key == obs.rng_key
    assert restored.proprio.dtype == np.float32
    assert restored.proprio.shape == (PROPRIO_DIM,)
    assert np.array_equal(restored.proprio, obs.proprio)
    assert len(restored.images) == len(obs.images)
    for got, src in zip(restored.images, obs.images, strict=True):
        assert got.dtype == np.uint8
        assert got.shape == src.shape
        assert np.array_equal(got, src)

    act = _act(slot_id=3, step=4)
    restored_act = from_wire(to_wire(act))
    assert isinstance(restored_act, ActMsg)
    assert restored_act.actions.dtype == np.float32
    assert restored_act.actions.shape == act.actions.shape
    assert np.array_equal(restored_act.actions, act.actions)


def test_ctrl_messages_roundtrip() -> None:
    ready = from_wire(to_wire(ctrl_worker_ready(slot_id=1)))
    begin = from_wire(to_wire(ctrl_episode_begin(slot_id=1, episode_uid="ep-a")))
    end = from_wire(to_wire(ctrl_episode_end(slot_id=1, episode_uid="ep-a")))
    stop = from_wire(to_wire(ctrl_shutdown(reason="done")))
    err = from_wire(
        to_wire(
            ctrl_error(
                reason="infer failed",
                traceback="Traceback (most recent call last):\n  boom",
                slot_id=1,
                episode_uid="ep-a",
            )
        )
    )
    assert ready.kind == "worker_ready" and ready.slot_id == 1
    assert begin.kind == "episode_begin" and begin.episode_uid == "ep-a"
    assert end.kind == "episode_end"
    assert stop.kind == "shutdown" and stop.reason == "done"
    assert err.kind == "error"
    assert err.traceback is not None and "boom" in err.traceback


def test_group_and_ungroup_by_slot_id() -> None:
    msgs = [
        _obs(slot_id=2, step=0),
        _obs(slot_id=0, step=0),
        _obs(slot_id=2, step=1),
        _obs(slot_id=1, step=0),
    ]
    grouped = group_by_slot(msgs)
    assert list(grouped) == [2, 0, 1]
    assert [item.step for item in grouped[2]] == [0, 1]
    flat = ungroup_by_slot(grouped)
    assert [item.slot_id for item in flat] == [2, 2, 0, 1]

    unique = [msgs[0], msgs[1], msgs[3]]
    indexed = index_by_slot(unique)
    assert gather_by_slot_ids(indexed, [2, 0, 1]) == unique
    with pytest.raises(ValueError, match="duplicate slot_id 2"):
        index_by_slot(msgs)
    with pytest.raises(KeyError, match="slot_id 7"):
        gather_by_slot_ids(indexed, [2, 7])


def test_channel_tcp_roundtrip_preserves_order(tcp_pair: tuple[Channel, Channel]) -> None:
    server, client = tcp_pair
    payloads = [
        {"seq": index, "arr": np.array([index, index + 1], dtype=np.float32)}
        for index in range(5)
    ]
    for payload in payloads:
        client.put(payload)
    got = [server.get() for _ in payloads]
    assert [item["seq"] for item in got] == list(range(5))
    for src, dst in zip(payloads, got, strict=True):
        assert dst["arr"].dtype == src["arr"].dtype
        assert np.array_equal(dst["arr"], src["arr"])

    server.put(to_wire(_obs(slot_id=0, step=2)))
    restored = from_wire(client.get())
    assert isinstance(restored, ObsMsg)
    assert restored.step == 2
    assert np.array_equal(restored.proprio, _obs(slot_id=0, step=2).proprio)


def test_channel_get_timeout_is_fail_loud() -> None:
    timeout_s = 0.25
    listener = Channel.bind(host="127.0.0.1", port=0, timeout_s=timeout_s)
    accepted: list[Channel] = []

    def _accept() -> None:
        accepted.append(listener.accept())

    thread = threading.Thread(target=_accept, daemon=True)
    thread.start()
    client = Channel.connect(
        host=listener.host, port=listener.port, timeout_s=timeout_s
    )
    thread.join(timeout=2.0)
    try:
        with pytest.raises(ChannelTimeout, match="timed out"):
            client.get()
    finally:
        client.close()
        for channel in accepted:
            channel.close()
        listener.close()


def test_channel_put_rejects_unsafe_payload(tcp_pair: tuple[Channel, Channel]) -> None:
    class _Bad:
        pass

    _server, client = tcp_pair
    with pytest.raises(TypeError, match=r"<root>"):
        client.put(_Bad())


def test_channel_closed_peer_is_fail_loud(tcp_pair: tuple[Channel, Channel]) -> None:
    server, client = tcp_pair
    server.close()
    with pytest.raises(ChannelClosed):
        client.get()


def test_config_missing_field_names_key() -> None:
    with pytest.raises(ValueError, match="extra"):
        load_parallel_eval_config({})
    with pytest.raises(ValueError, match="parallel_eval"):
        load_parallel_eval_config({"extra": {}})

    for key in ("channel_timeout_s", "num_envs", "infer_conda_env"):
        block = _parallel_eval_block()
        block.pop(key)
        with pytest.raises(ValueError, match=key):
            load_parallel_eval_config({"extra": {"parallel_eval": block}})


def test_config_loads_from_resolved_and_model_config() -> None:
    block = _parallel_eval_block()
    loaded = load_parallel_eval_config({"extra": {"parallel_eval": block}})
    assert loaded.enabled is False
    assert loaded.num_envs == 2
    assert loaded.channel_host == "127.0.0.1"
    assert loaded.channel_port == 0
    assert loaded.channel_timeout_s == 120.0
    assert loaded.infer_conda_env == "XVLA"
    assert loaded.deterministic_rng is False
    assert set(REQUIRED_KEYS) <= set(block)

    from_model = load_parallel_eval_config(
        _model_config({"parallel_eval": _parallel_eval_block()})
    )
    assert from_model.torch_dtype == "float32"
    assert from_model.env_conda_env == "internvla_a1"
