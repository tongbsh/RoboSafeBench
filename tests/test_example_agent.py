"""Example-agent integration tests — no simulator, no GPU."""

from __future__ import annotations

import copy
import importlib.util
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

# Skip this module instead of aborting collection for the whole suite when the
# out-of-tree example package is absent (trimmed deployments ship src/ + tests/
# + configs/ without examples/).
pytest.importorskip(
    "dyni_example_agent",
    reason="example package not installed: pip install -e examples/example_agent",
)

from dyni_example_agent.http_adapter import ExampleAgentHttpAdapter  # noqa: E402
from dyni_example_agent.local_adapter import ExampleAgentAdapter  # noqa: E402

from dynisafety.policies.action_space import dim_for_space, validate_chunk  # noqa: E402
from dynisafety.policies.base import (  # noqa: E402
    ActionChunk,
    EpisodeContext,
    ModelConfig,
    Observation,
    model_config_from_resolved,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "configs" / "models"
CHECKPOINT = REPO_ROOT / "examples" / "example_agent" / "assets" / "mock_vla_weights.npz"
_SERVER_PATH = REPO_ROOT / "examples" / "example_agent" / "server" / "toy_vla_server.py"

_BACKENDS = ("libero", "simplerenv", "robotwin")
_STATE_DIM = {"libero": 8, "simplerenv": 7, "robotwin": 14}
_REQUIRED_EXTRAS = ("seed", "replan_steps", "action_scale", "feature_grid")


def _load_raw(stem: str) -> dict[str, Any]:
    path = MODELS_DIR / f"{stem}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _with_checkpoint(cfg: ModelConfig, **extra_updates: Any) -> ModelConfig:
    extra = dict(cfg.extra)
    extra.update(extra_updates)
    return replace(cfg, checkpoint=str(CHECKPOINT), extra=extra)


def _ctx(*, backend: str = "libero", episode_index: int = 0) -> EpisodeContext:
    return EpisodeContext(
        instruction="pick up the cup",
        task_id=0,
        task_name="t0",
        episode_index=episode_index,
        seed=0,
        level=0,
        backend=backend,
        suite="smoke",
        extra={},
    )


def _image(key: str) -> np.ndarray:
    codes = {"primary": 10, "wrist": 40, "left": 80, "right": 120}
    return np.full((16, 16, 3), codes.get(key, 1), dtype=np.uint8)


def _obs(cfg: ModelConfig, *, step: int = 0, backend: str = "libero") -> Observation:
    images = {key: _image(key) for key in cfg.observation_images}
    state = np.linspace(0.0, 1.0, _STATE_DIM[backend], dtype=np.float32)
    return Observation(
        images=images,
        state=state,
        instruction="pick up the cup",
        step=step,
        extra={},
    )


def _import_toy_server() -> Any:
    spec = importlib.util.spec_from_file_location("dyni_example_agent_toy_vla_server", _SERVER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load toy server from {_SERVER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wait_listening(host: str, port: int, *, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.02)
    raise RuntimeError(f"toy server did not listen on {host}:{port}")


@contextmanager
def _running_toy_server() -> Iterator[tuple[int, Any]]:
    module = _import_toy_server()
    server = module.build_server("127.0.0.1", 0, CHECKPOINT)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = int(server.server_address[1])
        _wait_listening("127.0.0.1", port)
        yield port, server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# YAML parse
# ---------------------------------------------------------------------------


def test_example_agent_yaml_parses() -> None:
    raw = _load_raw("example_agent")
    cfg = model_config_from_resolved(raw)
    assert cfg.name == "example_agent"
    assert cfg.action_space == "delta_ee_7d"
    assert cfg.chunk_size == 8
    assert cfg.transport == "local"
    assert cfg.observation_images == ("primary", "wrist")
    assert cfg.backends == ("libero", "simplerenv", "robotwin")


def test_example_agent_http_yaml_parses() -> None:
    raw = _load_raw("example_agent_http")
    cfg = model_config_from_resolved(raw)
    assert cfg.name == "example_agent_http"
    assert cfg.action_space == "delta_ee_7d"
    assert cfg.chunk_size == 8
    assert cfg.transport == "http"
    assert cfg.observation_images == ("primary", "wrist")
    assert cfg.backends == ("libero", "simplerenv", "robotwin")
    assert cfg.server_default_port == 8090
    assert cfg.server_health_path == "/health"


# ---------------------------------------------------------------------------
# Per-backend local act + validate_chunk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", _BACKENDS)
def test_local_adapter_act_per_backend(backend: str) -> None:
    raw = _load_raw("example_agent")
    cfg = _with_checkpoint(model_config_from_resolved(raw, backend=backend))
    policy = ExampleAgentAdapter.from_config(cfg)
    policy.reset(_ctx(backend=backend))
    chunk = policy.act(_obs(cfg, backend=backend))
    assert isinstance(chunk, ActionChunk)
    assert chunk.space == cfg.action_space
    want_rows = min(int(cfg.extra["replan_steps"]), cfg.chunk_size)
    assert chunk.actions.shape == (want_rows, dim_for_space(cfg.action_space))
    validate_chunk(chunk)
    policy.close()


def test_local_adapter_rejects_non_local_transport() -> None:
    raw = _load_raw("example_agent")
    cfg = _with_checkpoint(model_config_from_resolved(raw))
    cfg = replace(cfg, transport="http")
    with pytest.raises(ValueError, match="transport='local'"):
        ExampleAgentAdapter.from_config(cfg)


# ---------------------------------------------------------------------------
# Chunk modes
# ---------------------------------------------------------------------------


def test_chunk_mode_open_loop_returns_full_horizon() -> None:
    raw = _load_raw("example_agent")
    cfg = _with_checkpoint(model_config_from_resolved(raw, backend="libero"))
    assert int(cfg.extra["replan_steps"]) == cfg.chunk_size
    policy = ExampleAgentAdapter.from_config(cfg)
    policy.reset(_ctx())
    assert policy._backbone is not None
    chunk = policy.act(_obs(cfg))
    assert chunk.actions.shape == (cfg.chunk_size, 7)
    assert policy._backbone.plan_calls == 1
    policy.act(_obs(cfg, step=cfg.chunk_size))
    assert policy._backbone.plan_calls == 2
    policy.close()


def test_chunk_mode_replan_one_returns_single_row() -> None:
    raw = _load_raw("example_agent")
    raw = copy.deepcopy(raw)
    raw["extra"]["replan_steps"] = 1
    cfg = _with_checkpoint(model_config_from_resolved(raw, backend="libero"))
    policy = ExampleAgentAdapter.from_config(cfg)
    policy.reset(_ctx())
    assert policy._backbone is not None
    for step in range(cfg.chunk_size):
        chunk = policy.act(_obs(cfg, step=step))
        assert chunk.actions.shape == (1, 7)
        validate_chunk(chunk)
        assert policy._backbone.plan_calls == 1
    policy.act(_obs(cfg, step=cfg.chunk_size))
    assert policy._backbone.plan_calls == 2
    policy.close()


# ---------------------------------------------------------------------------
# Failure modes RAISE
# ---------------------------------------------------------------------------


def test_missing_primary_image_raises() -> None:
    raw = _load_raw("example_agent")
    cfg = _with_checkpoint(model_config_from_resolved(raw, backend="libero"))
    policy = ExampleAgentAdapter.from_config(cfg)
    policy.reset(_ctx())
    obs = Observation(
        images={"wrist": _image("wrist")},
        state=np.zeros(8, dtype=np.float32),
        instruction="pick up the cup",
        step=0,
        extra={},
    )
    with pytest.raises(KeyError, match="primary"):
        policy.act(obs)
    policy.close()


def test_missing_state_raises() -> None:
    raw = _load_raw("example_agent")
    cfg = _with_checkpoint(model_config_from_resolved(raw, backend="libero"))
    policy = ExampleAgentAdapter.from_config(cfg)
    policy.reset(_ctx())
    obs = Observation(
        images={key: _image(key) for key in cfg.observation_images},
        state=None,
        instruction="pick up the cup",
        step=0,
        extra={},
    )
    with pytest.raises(ValueError, match="state"):
        policy.act(obs)
    policy.close()


@pytest.mark.parametrize("key", _REQUIRED_EXTRAS)
def test_missing_required_extra_raises(key: str) -> None:
    raw = copy.deepcopy(_load_raw("example_agent"))
    raw["extra"].pop(key)
    cfg = _with_checkpoint(model_config_from_resolved(raw))
    with pytest.raises(ValueError, match=key):
        ExampleAgentAdapter.from_config(cfg)


def test_bool_rejected_as_int_extra() -> None:
    raw = copy.deepcopy(_load_raw("example_agent"))
    raw["extra"]["seed"] = True
    cfg = _with_checkpoint(model_config_from_resolved(raw))
    with pytest.raises(ValueError, match="seed"):
        ExampleAgentAdapter.from_config(cfg)


# ---------------------------------------------------------------------------
# HTTP round trip
# ---------------------------------------------------------------------------


def test_http_round_trip() -> None:
    raw = _load_raw("example_agent_http")
    base = model_config_from_resolved(raw)
    with _running_toy_server() as (port, _server):
        cfg = _with_checkpoint(
            replace(base, server_startup_timeout_s=5),
            server_port=port,
        )
        policy = ExampleAgentHttpAdapter.from_config(cfg)
        try:
            policy.reset(_ctx())
            chunk = policy.act(_obs(cfg))
            assert chunk.space == cfg.action_space
            want_rows = min(int(cfg.extra["replan_steps"]), cfg.chunk_size)
            assert chunk.actions.shape == (want_rows, dim_for_space(cfg.action_space))
            validate_chunk(chunk)
        finally:
            policy.close()


def test_http_requeries_only_when_buffer_drains() -> None:
    raw = copy.deepcopy(_load_raw("example_agent_http"))
    raw["extra"]["replan_steps"] = 1
    base = model_config_from_resolved(raw)
    with _running_toy_server() as (port, server):
        cfg = _with_checkpoint(
            replace(base, server_startup_timeout_s=5),
            server_port=port,
        )
        policy = ExampleAgentHttpAdapter.from_config(cfg)
        try:
            policy.reset(_ctx())
            for step in range(cfg.chunk_size):
                chunk = policy.act(_obs(cfg, step=step))
                assert chunk.actions.shape == (1, 7)
                validate_chunk(chunk)
                assert server.inference_state.act_hits == 1
            policy.act(_obs(cfg, step=cfg.chunk_size))
            assert server.inference_state.act_hits == 2
        finally:
            policy.close()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_different_episode_index_produces_different_actions() -> None:
    raw = _load_raw("example_agent")
    cfg_a = _with_checkpoint(model_config_from_resolved(raw, backend="libero"))
    cfg_b = _with_checkpoint(model_config_from_resolved(copy.deepcopy(raw), backend="libero"))
    policy_a = ExampleAgentAdapter.from_config(cfg_a)
    policy_b = ExampleAgentAdapter.from_config(cfg_b)
    try:
        policy_a.reset(_ctx(episode_index=0))
        policy_b.reset(_ctx(episode_index=1))
        obs = _obs(cfg_a)
        chunk_a = policy_a.act(obs)
        chunk_b = policy_b.act(obs)
        assert chunk_a.actions.shape == chunk_b.actions.shape
        assert not np.array_equal(chunk_a.actions, chunk_b.actions)
    finally:
        policy_a.close()
        policy_b.close()


def test_independent_adapters_are_bit_identical() -> None:
    raw = _load_raw("example_agent")
    cfg_a = _with_checkpoint(model_config_from_resolved(raw, backend="libero"))
    cfg_b = _with_checkpoint(model_config_from_resolved(copy.deepcopy(raw), backend="libero"))
    policy_a = ExampleAgentAdapter.from_config(cfg_a)
    policy_b = ExampleAgentAdapter.from_config(cfg_b)
    try:
        policy_a.reset(_ctx())
        policy_b.reset(_ctx())
        obs = _obs(cfg_a)
        chunk_a = policy_a.act(obs)
        chunk_b = policy_b.act(obs)
        np.testing.assert_array_equal(chunk_a.actions, chunk_b.actions)
    finally:
        policy_a.close()
        policy_b.close()
