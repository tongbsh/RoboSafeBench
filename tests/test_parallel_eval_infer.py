"""Infer-side parallel_eval tests (fake model; no 879M weights)."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import numpy as np
import pytest
import torch

from dynisafety.parallel_eval.channel import (
    Channel,
    ChannelError,
    poll_ready,
)
from dynisafety.parallel_eval.config import REQUIRED_KEYS, load_parallel_eval_config
from dynisafety.parallel_eval.infer_worker import serve_inference
from dynisafety.parallel_eval.protocol import (
    ACTION_DIM,
    PROPRIO_DIM,
    ActMsg,
    CtrlMsg,
    ObsMsg,
    ctrl_error,
    ctrl_shutdown,
    ctrl_worker_ready,
    from_wire,
    to_wire,
)
from dynisafety.parallel_eval.xvla_local import (
    XvlaRuntime,
    forward_obs_batch,
    generate_actions_with_noise,
    load_xvla,
    mix_rng_key,
    nested_pil_views,
    slot_noise,
    stack_slot_noise,
    unique_denoising_steps,
)

_FAKE_NUM_ACTIONS = 4


def _obs(
    *,
    slot_id: int,
    step: int = 0,
    episode_uid: str = "ep-a",
    rng_key: int = 99,
    denoising_steps: int = 4,
    proprio: np.ndarray | None = None,
    instruction: str = "place the cup",
) -> ObsMsg:
    images = [
        np.zeros((6, 8, 3), dtype=np.uint8),
        np.full((6, 8, 3), 17, dtype=np.uint8),
        np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3),
    ]
    if proprio is None:
        proprio = np.linspace(-1.0, 1.0, PROPRIO_DIM, dtype=np.float32)
    return ObsMsg(
        slot_id=slot_id,
        episode_uid=episode_uid,
        step=step,
        images=images,
        proprio=np.asarray(proprio, dtype=np.float32),
        instruction=instruction,
        domain_id=6,
        denoising_steps=denoising_steps,
        rng_key=rng_key,
    )


def _parallel_eval_block(**overrides: object) -> dict[str, object]:
    block: dict[str, object] = {
        "enabled": True,
        "num_envs": 2,
        "pipeline_stage_num": 1,
        "batch_window_ms": 80,
        "infer_conda_env": "XVLA",
        "env_conda_env": "internvla_a1",
        "infer_device": "cpu",
        "env_cuda_visible_devices": "0",
        "channel_host": "127.0.0.1",
        "channel_port": 0,
        "channel_timeout_s": 2.0,
        "torch_dtype": "float32",
        "deterministic_rng": False,
        "rng_seed": 7,
    }
    block.update(overrides)
    return block


def _config(**overrides: object):
    return load_parallel_eval_config({"extra": {"parallel_eval": _parallel_eval_block(**overrides)}})


class _FakeActionSpace:
    dim_action = ACTION_DIM

    def preprocess(
        self, proprio: torch.Tensor, x_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return proprio, x_t

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        return action


class _FakeModel:
    num_actions = _FAKE_NUM_ACTIONS
    action_space = _FakeActionSpace()

    def eval(self) -> None:
        return None

    def forward_vlm(
        self,
        input_ids: torch.Tensor,
        image_input: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {"vlm_features": input_ids}

    def transformer(
        self,
        domain_id: torch.Tensor,
        action_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
        **enc: torch.Tensor,
    ) -> torch.Tensor:
        return action_with_noise + proprio.unsqueeze(1)


class _RecordingProcessor:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object]] = []

    def __call__(self, images: object, language_instruction: object) -> dict[str, torch.Tensor]:
        self.calls.append((images, language_instruction))
        if not isinstance(images, list) or not images:
            raise AssertionError("processor expected a non-empty nested image list")
        if not isinstance(images[0], (list, tuple)):
            raise AssertionError("processor received flat images (B=1 trap)")
        batch = len(images)
        if not isinstance(language_instruction, list) or len(language_instruction) != batch:
            raise AssertionError("processor expected List[str] aligned with image batch")
        return {
            "input_ids": torch.zeros(batch, 8, dtype=torch.long),
            "image_input": torch.zeros(batch, 3, 3, 8, 8, dtype=torch.float32),
            "image_mask": torch.ones(batch, 3, dtype=torch.bool),
        }


def _runtime() -> tuple[XvlaRuntime, _RecordingProcessor]:
    processor = _RecordingProcessor()
    runtime = XvlaRuntime(
        model=_FakeModel(),
        processor=processor,
        device="cpu",
        dtype=torch.float32,
    )
    return runtime, processor


def _start_serve(
    *,
    config,
    runtime: XvlaRuntime,
    listener,
) -> tuple[threading.Thread, list[BaseException]]:
    errors: list[BaseException] = []

    def _run() -> None:
        try:
            serve_inference(
                config=config,
                model_path="unused",
                repo_root="unused",
                processor_path=None,
                runtime=runtime,
                listener=listener,
                on_bound=None,
            )
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread, errors


def _connect_clients(listener, *, n: int) -> list[Channel]:
    return [
        Channel.connect(host=listener.host, port=listener.port, timeout_s=listener.timeout_s)
        for _ in range(n)
    ]


@pytest.fixture
def tcp_pair() -> Iterator[tuple[Channel, Channel]]:
    timeout_s = 2.0
    listener = Channel.bind(host="127.0.0.1", port=0, timeout_s=timeout_s)
    accepted: list[Channel] = []

    def _accept() -> None:
        accepted.append(listener.accept())

    thread = threading.Thread(target=_accept, daemon=True)
    thread.start()
    client = Channel.connect(host=listener.host, port=listener.port, timeout_s=timeout_s)
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


def test_poll_ready_timeout_returns_empty_not_error(tcp_pair: tuple[Channel, Channel]) -> None:
    server, _client = tcp_pair
    ready = poll_ready([server], timeout_s=0.05)
    assert ready == []


def test_poll_ready_timeout_s_is_keyword_only(tcp_pair: tuple[Channel, Channel]) -> None:
    server, _client = tcp_pair
    with pytest.raises(TypeError):
        poll_ready([server], 0.05)  # type: ignore[misc]


def test_poll_ready_returns_only_readable_channel() -> None:
    timeout_s = 2.0
    listener = Channel.bind(host="127.0.0.1", port=0, timeout_s=timeout_s)
    accepted: list[Channel] = []

    def _accept_two() -> None:
        accepted.append(listener.accept())
        accepted.append(listener.accept())

    thread = threading.Thread(target=_accept_two, daemon=True)
    thread.start()
    client_a = Channel.connect(host=listener.host, port=listener.port, timeout_s=timeout_s)
    client_b = Channel.connect(host=listener.host, port=listener.port, timeout_s=timeout_s)
    thread.join(timeout=timeout_s)
    try:
        assert len(accepted) == 2
        server_a, server_b = accepted
        client_b.put({"slot": 1})
        ready = poll_ready([server_a, server_b], timeout_s=0.5)
        assert ready == [server_b]
        assert server_b.get()["slot"] == 1
    finally:
        client_a.close()
        client_b.close()
        for channel in accepted:
            channel.close()
        listener.close()


def test_fileno_after_close_is_fail_loud(tcp_pair: tuple[Channel, Channel]) -> None:
    server, _client = tcp_pair
    server.close()
    with pytest.raises(ChannelError, match="closed"):
        server.fileno()


def test_config_requires_rng_seed() -> None:
    block = _parallel_eval_block()
    block.pop("rng_seed")
    with pytest.raises(ValueError, match="rng_seed"):
        load_parallel_eval_config({"extra": {"parallel_eval": block}})
    assert "rng_seed" in REQUIRED_KEYS


def test_mix_rng_key_is_stable_and_not_python_hash() -> None:
    first = mix_rng_key(rng_seed=7, episode_uid="ep-a", step=3)
    second = mix_rng_key(rng_seed=7, episode_uid="ep-a", step=3)
    assert first == second
    assert mix_rng_key(rng_seed=7, episode_uid="ep-a", step=4) != first
    assert mix_rng_key(rng_seed=8, episode_uid="ep-a", step=3) != first
    assert isinstance(first, int)
    assert first >= 0
    assert mix_rng_key(rng_seed=20260825, episode_uid="ep-0", step=7) == 15913000601788401206


def test_slot_noise_independent_of_batch_size_and_position() -> None:
    key = 123456
    solo = slot_noise(key, num_actions=30, dim_action=ACTION_DIM)
    obs_a = _obs(slot_id=0, rng_key=key)
    obs_b = _obs(slot_id=1, rng_key=key)
    stacked2 = stack_slot_noise(
        [obs_a, obs_b], num_actions=30, dim_action=ACTION_DIM
    )
    stacked4 = stack_slot_noise(
        [obs_a, obs_b, obs_a, obs_b], num_actions=30, dim_action=ACTION_DIM
    )
    assert solo.shape == (1, 30, ACTION_DIM)
    assert np.array_equal(stacked2[0], solo[0])
    assert np.array_equal(stacked2[1], solo[0])
    assert np.array_equal(stacked4[0], solo[0])
    assert np.array_equal(stacked4[3], solo[0])
    other = slot_noise(key + 1, num_actions=30, dim_action=ACTION_DIM)
    assert not np.array_equal(other, solo)


def test_nested_pil_views_is_one_inner_list_per_sample() -> None:
    one = nested_pil_views([_obs(slot_id=0)])
    assert len(one) == 1
    assert isinstance(one[0], list)
    assert len(one[0]) == 3
    two = nested_pil_views([_obs(slot_id=0), _obs(slot_id=1)])
    assert len(two) == 2
    assert isinstance(two[0], list) and isinstance(two[1], list)
    assert len(two[0]) == 3 and len(two[1]) == 3


def test_unique_denoising_steps_rejects_mix() -> None:
    assert unique_denoising_steps([_obs(slot_id=0), _obs(slot_id=1)]) == 4
    with pytest.raises(ValueError, match="denoising_steps"):
        unique_denoising_steps(
            [_obs(slot_id=0, denoising_steps=4), _obs(slot_id=1, denoising_steps=8)]
        )


def test_generate_actions_with_noise_validates_x1_and_is_deterministic() -> None:
    model = _FakeModel()
    batch = 2
    input_ids = torch.zeros(batch, 8, dtype=torch.long)
    image_input = torch.zeros(batch, 3, 3, 8, 8)
    image_mask = torch.ones(batch, 3, dtype=torch.bool)
    domain_id = torch.zeros(batch, dtype=torch.long)
    proprio = torch.zeros(batch, PROPRIO_DIM)
    x1 = torch.ones(batch, _FAKE_NUM_ACTIONS, ACTION_DIM)
    kwargs = dict(
        input_ids=input_ids,
        image_input=image_input,
        image_mask=image_mask,
        domain_id=domain_id,
        proprio=proprio,
        steps=3,
    )
    with pytest.raises(ValueError, match="x1.shape"):
        generate_actions_with_noise(
            model, **kwargs, x1=torch.ones(1, _FAKE_NUM_ACTIONS, ACTION_DIM)
        )
    first = generate_actions_with_noise(model, **kwargs, x1=x1)
    second = generate_actions_with_noise(model, **kwargs, x1=x1)
    assert first.shape == (batch, _FAKE_NUM_ACTIONS, ACTION_DIM)
    assert torch.equal(first, second)
    other = generate_actions_with_noise(model, **kwargs, x1=x1 * 2)
    assert not torch.equal(first, other)


def test_forward_obs_batch_uses_nested_images_and_numpy_out() -> None:
    runtime, processor = _runtime()
    msgs = [
        _obs(slot_id=0, proprio=np.full(PROPRIO_DIM, 0.1, dtype=np.float32)),
        _obs(slot_id=1, proprio=np.full(PROPRIO_DIM, 0.7, dtype=np.float32)),
    ]
    x1 = stack_slot_noise(
        msgs, num_actions=_FAKE_NUM_ACTIONS, dim_action=ACTION_DIM
    )
    rows = forward_obs_batch(runtime, msgs, x1=x1)
    assert len(processor.calls) == 1
    images, texts = processor.calls[0]
    assert isinstance(images, list) and isinstance(images[0], list)
    assert len(images) == 2
    assert texts == ["place the cup", "place the cup"]
    assert len(rows) == 2
    for row in rows:
        assert isinstance(row, np.ndarray)
        assert row.dtype == np.float32
        assert row.shape == (_FAKE_NUM_ACTIONS, ACTION_DIM)
    assert not np.array_equal(rows[0], rows[1])


def test_load_xvla_rejects_non_float32() -> None:
    with pytest.raises(ValueError, match="float32"):
        load_xvla(
            repo_root="unused",
            model_path="unused",
            processor_path=None,
            device="cpu",
            torch_dtype="bfloat16",
        )


def test_serve_window_does_not_wait_for_all_slots() -> None:
    runtime, _processor = _runtime()
    config = _config(num_envs=2, batch_window_ms=80, channel_timeout_s=2.0)
    listener = Channel.bind(host="127.0.0.1", port=0, timeout_s=2.0)
    thread, errors = _start_serve(config=config, runtime=runtime, listener=listener)
    clients: list[Channel] = []
    try:
        clients = _connect_clients(listener, n=2)
        clients[0].put(to_wire(ctrl_worker_ready(slot_id=0)))
        clients[1].put(to_wire(ctrl_worker_ready(slot_id=1)))
        clients[0].put(to_wire(_obs(slot_id=0, step=1)))
        started = time.monotonic()
        reply = from_wire(clients[0].get())
        elapsed = time.monotonic() - started
        assert isinstance(reply, ActMsg)
        assert reply.slot_id == 0
        assert elapsed < 1.0
        clients[0].put(to_wire(ctrl_shutdown(reason="done")))
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert errors == []
    finally:
        for client in clients:
            client.close()
        listener.close()


def test_serve_routes_act_to_matching_channel() -> None:
    runtime, _processor = _runtime()
    config = _config(num_envs=2, batch_window_ms=200, channel_timeout_s=2.0)
    listener = Channel.bind(host="127.0.0.1", port=0, timeout_s=2.0)
    thread, errors = _start_serve(config=config, runtime=runtime, listener=listener)
    clients: list[Channel] = []
    try:
        clients = _connect_clients(listener, n=2)
        left = np.full(PROPRIO_DIM, 0.2, dtype=np.float32)
        right = np.full(PROPRIO_DIM, 0.9, dtype=np.float32)
        clients[0].put(to_wire(_obs(slot_id=0, proprio=left, rng_key=11)))
        clients[1].put(to_wire(_obs(slot_id=1, proprio=right, rng_key=22)))
        got0 = from_wire(clients[0].get())
        got1 = from_wire(clients[1].get())
        assert isinstance(got0, ActMsg) and isinstance(got1, ActMsg)
        assert got0.slot_id == 0 and got1.slot_id == 1
        assert not np.array_equal(got0.actions, got1.actions)
        clients[0].put(to_wire(ctrl_shutdown(reason="done")))
        thread.join(timeout=2.0)
        assert errors == []
    finally:
        for client in clients:
            client.close()
        listener.close()


def test_serve_peer_error_is_fail_loud() -> None:
    runtime, _processor = _runtime()
    config = _config(num_envs=1, batch_window_ms=80, channel_timeout_s=2.0)
    listener = Channel.bind(host="127.0.0.1", port=0, timeout_s=2.0)
    thread, errors = _start_serve(config=config, runtime=runtime, listener=listener)
    client: Channel | None = None
    try:
        client = Channel.connect(host=listener.host, port=listener.port, timeout_s=2.0)
        client.put(
            to_wire(
                ctrl_error(
                    reason="env crashed",
                    traceback="Traceback (most recent call last):\n  boom",
                    slot_id=0,
                    episode_uid="ep-a",
                )
            )
        )
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert errors
        assert "env crashed" in str(errors[0])
        assert "boom" in str(errors[0])
    finally:
        if client is not None:
            client.close()
        listener.close()


def test_serve_deterministic_rng_rejects_mismatched_key() -> None:
    runtime, _processor = _runtime()
    config = _config(
        num_envs=1,
        batch_window_ms=80,
        channel_timeout_s=2.0,
        deterministic_rng=True,
        rng_seed=7,
    )
    listener = Channel.bind(host="127.0.0.1", port=0, timeout_s=2.0)
    thread, errors = _start_serve(config=config, runtime=runtime, listener=listener)
    client: Channel | None = None
    try:
        client = Channel.connect(host=listener.host, port=listener.port, timeout_s=2.0)
        client.put(to_wire(_obs(slot_id=0, episode_uid="ep-a", step=3, rng_key=1)))
        reply = from_wire(client.get())
        assert isinstance(reply, CtrlMsg)
        assert reply.kind == "error"
        thread.join(timeout=2.0)
        assert errors
        assert "rng_key" in str(errors[0])
    finally:
        if client is not None:
            client.close()
        listener.close()


def test_serve_accepts_matching_deterministic_rng_key() -> None:
    runtime, _processor = _runtime()
    config = _config(
        num_envs=1,
        batch_window_ms=80,
        channel_timeout_s=2.0,
        deterministic_rng=True,
        rng_seed=7,
    )
    key = mix_rng_key(rng_seed=7, episode_uid="ep-a", step=3)
    listener = Channel.bind(host="127.0.0.1", port=0, timeout_s=2.0)
    thread, errors = _start_serve(config=config, runtime=runtime, listener=listener)
    client: Channel | None = None
    try:
        client = Channel.connect(host=listener.host, port=listener.port, timeout_s=2.0)
        client.put(to_wire(_obs(slot_id=0, episode_uid="ep-a", step=3, rng_key=key)))
        reply = from_wire(client.get())
        assert isinstance(reply, ActMsg)
        assert reply.step == 3
        client.put(to_wire(ctrl_shutdown(reason="done")))
        thread.join(timeout=2.0)
        assert errors == []
    finally:
        if client is not None:
            client.close()
        listener.close()


def test_serve_peer_close_does_not_kill_other_slot() -> None:
    """Env process exit closes TCP; Infer must keep serving remaining slots."""
    runtime, _processor = _runtime()
    config = _config(num_envs=2, batch_window_ms=80, channel_timeout_s=2.0)
    listener = Channel.bind(host="127.0.0.1", port=0, timeout_s=2.0)
    thread, errors = _start_serve(config=config, runtime=runtime, listener=listener)
    clients: list[Channel] = []
    try:
        clients = _connect_clients(listener, n=2)
        clients[0].put(to_wire(ctrl_worker_ready(slot_id=0)))
        clients[1].put(to_wire(ctrl_worker_ready(slot_id=1)))
        clients[0].close()
        clients[1].put(to_wire(_obs(slot_id=1, step=4)))
        reply = from_wire(clients[1].get())
        assert isinstance(reply, ActMsg)
        assert reply.slot_id == 1
        assert reply.step == 4
        assert thread.is_alive()
        clients[1].put(to_wire(ctrl_shutdown(reason="done")))
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert errors == []
    finally:
        for client in clients:
            client.close()
        listener.close()


def test_serve_all_peers_closed_exits_cleanly() -> None:
    runtime, _processor = _runtime()
    config = _config(num_envs=2, batch_window_ms=80, channel_timeout_s=2.0)
    listener = Channel.bind(host="127.0.0.1", port=0, timeout_s=2.0)
    thread, errors = _start_serve(config=config, runtime=runtime, listener=listener)
    clients: list[Channel] = []
    try:
        clients = _connect_clients(listener, n=2)
        clients[0].put(to_wire(ctrl_worker_ready(slot_id=0)))
        clients[1].put(to_wire(ctrl_worker_ready(slot_id=1)))
        clients[0].close()
        clients[1].close()
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert errors == []
    finally:
        for client in clients:
            client.close()
        listener.close()


def test_serve_peer_close_does_not_kill_other_slot() -> None:
    """Env process exit closes TCP; Infer must keep serving remaining slots."""
    runtime, _processor = _runtime()
    config = _config(num_envs=2, batch_window_ms=80, channel_timeout_s=2.0)
    listener = Channel.bind(host="127.0.0.1", port=0, timeout_s=2.0)
    thread, errors = _start_serve(config=config, runtime=runtime, listener=listener)
    clients: list[Channel] = []
    try:
        clients = _connect_clients(listener, n=2)
        clients[0].put(to_wire(ctrl_worker_ready(slot_id=0)))
        clients[1].put(to_wire(ctrl_worker_ready(slot_id=1)))
        clients[0].close()
        clients[1].put(to_wire(_obs(slot_id=1, step=4)))
        reply = from_wire(clients[1].get())
        assert isinstance(reply, ActMsg)
        assert reply.slot_id == 1
        assert reply.step == 4
        assert thread.is_alive()
        clients[1].put(to_wire(ctrl_shutdown(reason="done")))
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert errors == []
    finally:
        for client in clients:
            client.close()
        listener.close()


def test_serve_all_peers_closed_exits_cleanly() -> None:
    runtime, _processor = _runtime()
    config = _config(num_envs=2, batch_window_ms=80, channel_timeout_s=2.0)
    listener = Channel.bind(host="127.0.0.1", port=0, timeout_s=2.0)
    thread, errors = _start_serve(config=config, runtime=runtime, listener=listener)
    clients: list[Channel] = []
    try:
        clients = _connect_clients(listener, n=2)
        clients[0].put(to_wire(ctrl_worker_ready(slot_id=0)))
        clients[1].put(to_wire(ctrl_worker_ready(slot_id=1)))
        clients[0].close()
        clients[1].close()
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert errors == []
    finally:
        for client in clients:
            client.close()
        listener.close()
