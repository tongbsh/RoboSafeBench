"""One-shot self-check for the example-agent integration.

Run: ``python examples/example_agent/verify.py``
"""

from __future__ import annotations

import copy
import importlib.util
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_EXAMPLE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _EXAMPLE_ROOT.parents[1]
_SRC = _EXAMPLE_ROOT / "src"
for _path in (_SRC, _REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from dyni_example_agent.http_adapter import ExampleAgentHttpAdapter  # noqa: E402
from dyni_example_agent.local_adapter import ExampleAgentAdapter  # noqa: E402

from dynisafety.policies.action_space import dim_for_space, validate_chunk  # noqa: E402
from dynisafety.policies.base import (  # noqa: E402
    EpisodeContext,
    ModelConfig,
    Observation,
    model_config_from_resolved,
)

CHECKPOINT = _EXAMPLE_ROOT / "assets" / "mock_vla_weights.npz"
MODELS_DIR = _REPO_ROOT / "configs" / "models"
_SERVER_PATH = _EXAMPLE_ROOT / "server" / "toy_vla_server.py"

_BACKENDS = ("libero", "simplerenv", "robotwin")
_STATE_DIM = {"libero": 8, "simplerenv": 7, "robotwin": 14}
_EVAL_JOBS = (
    ("libero", "safelibero_spatial"),
    ("simplerenv", "bridge"),
    ("robotwin", "main20"),
)


def _load_raw(stem: str) -> dict[str, Any]:
    raw = yaml.safe_load((MODELS_DIR / f"{stem}.yaml").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"{stem}.yaml root must be a mapping")
    return raw


def _with_checkpoint(cfg: ModelConfig, **extra_updates: Any) -> ModelConfig:
    extra = dict(cfg.extra)
    extra.update(extra_updates)
    return replace(cfg, checkpoint=str(CHECKPOINT), extra=extra)


def _ctx(backend: str = "libero") -> EpisodeContext:
    return EpisodeContext(
        instruction="pick up the cup",
        task_id=0,
        task_name="t0",
        episode_index=0,
        seed=0,
        level=0,
        backend=backend,
        suite="smoke",
        extra={},
    )


def _image(key: str) -> np.ndarray:
    codes = {"primary": 10, "wrist": 40, "left": 80, "right": 120}
    return np.full((16, 16, 3), codes.get(key, 1), dtype=np.uint8)


def _obs(cfg: ModelConfig, *, backend: str, step: int = 0) -> Observation:
    return Observation(
        images={key: _image(key) for key in cfg.observation_images},
        state=np.linspace(0.0, 1.0, _STATE_DIM[backend], dtype=np.float32),
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


def _check_backend(backend: str) -> None:
    raw = _load_raw("example_agent")
    cfg = _with_checkpoint(model_config_from_resolved(raw, backend=backend))
    policy = ExampleAgentAdapter.from_config(cfg)
    try:
        policy.reset(_ctx(backend))
        chunk = policy.act(_obs(cfg, backend=backend))
        if chunk.space != cfg.action_space:
            raise AssertionError(f"space {chunk.space!r} != {cfg.action_space!r}")
        want = min(int(cfg.extra["replan_steps"]), cfg.chunk_size)
        if chunk.actions.shape != (want, dim_for_space(cfg.action_space)):
            raise AssertionError(
                f"shape {chunk.actions.shape} != {(want, dim_for_space(cfg.action_space))}"
            )
        validate_chunk(chunk)
    finally:
        policy.close()


def _check_chunk_modes() -> None:
    raw = _load_raw("example_agent")
    open_loop = _with_checkpoint(model_config_from_resolved(raw, backend="libero"))
    policy = ExampleAgentAdapter.from_config(open_loop)
    try:
        policy.reset(_ctx())
        chunk = policy.act(_obs(open_loop, backend="libero"))
        if chunk.actions.shape[0] != open_loop.chunk_size:
            raise AssertionError("open-loop mode must return chunk_size rows")
    finally:
        policy.close()

    raw_one = copy.deepcopy(raw)
    raw_one["extra"]["replan_steps"] = 1
    one = _with_checkpoint(model_config_from_resolved(raw_one, backend="libero"))
    policy = ExampleAgentAdapter.from_config(one)
    try:
        policy.reset(_ctx())
        assert policy._backbone is not None
        for step in range(one.chunk_size):
            chunk = policy.act(_obs(one, backend="libero", step=step))
            if chunk.actions.shape != (1, 7):
                raise AssertionError(f"replan-1 must return (1, 7), got {chunk.actions.shape}")
            if policy._backbone.plan_calls != 1:
                raise AssertionError("replan-1 must plan only on an empty buffer")
        policy.act(_obs(one, backend="libero", step=one.chunk_size))
        if policy._backbone.plan_calls != 2:
            raise AssertionError("replan-1 must re-plan every chunk_size calls")
    finally:
        policy.close()


def _check_http() -> None:
    raw = _load_raw("example_agent_http")
    base = model_config_from_resolved(raw)
    module = _import_toy_server()
    server = module.build_server("127.0.0.1", 0, CHECKPOINT)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = int(server.server_address[1])
        _wait_listening("127.0.0.1", port)
        cfg = _with_checkpoint(replace(base, server_startup_timeout_s=5), server_port=port)
        policy = ExampleAgentHttpAdapter.from_config(cfg)
        try:
            policy.reset(_ctx())
            chunk = policy.act(_obs(cfg, backend="libero"))
            if chunk.space != cfg.action_space:
                raise AssertionError(f"http space {chunk.space!r}")
            want_rows = min(int(cfg.extra["replan_steps"]), cfg.chunk_size)
            want = (want_rows, dim_for_space(cfg.action_space))
            if chunk.actions.shape != want:
                raise AssertionError(f"http shape {chunk.actions.shape} != {want}")
            validate_chunk(chunk)
        finally:
            policy.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _cli_argv() -> list[str] | None:
    found = shutil.which("dynisafety")
    if found:
        return [found]
    try:
        import dynisafety.cli  # noqa: F401
    except ImportError:
        return None
    return [sys.executable, "-m", "dynisafety.cli"]


def _run_cli(argv: list[str]) -> None:
    proc = subprocess.run(argv, cwd=_REPO_ROOT, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"exit {proc.returncode}: {detail[-400:]}")


def main() -> int:
    rows: list[tuple[str, str]] = []

    def record(name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — table must show every row
            rows.append((name, f"FAIL ({type(exc).__name__}: {exc})"))
        else:
            rows.append((name, "PASS"))

    for backend in _BACKENDS:
        record(f"offline {backend}", lambda b=backend: _check_backend(b))
    record("chunk modes", _check_chunk_modes)
    record("http round-trip", _check_http)

    cli = _cli_argv()
    if cli is None:
        for backend, suite in _EVAL_JOBS:
            rows.append((f"eval {backend} --dry-run", "SKIP (dynisafety CLI unavailable)"))
        rows.append(("sweep smoke_example_agent --dry-run", "SKIP (dynisafety CLI unavailable)"))
    else:
        for backend, suite in _EVAL_JOBS:
            record(
                f"eval {backend} --dry-run",
                lambda b=backend, s=suite: _run_cli(
                    [
                        *cli,
                        "eval",
                        "--backend",
                        b,
                        "--model",
                        "example_agent",
                        "--suite",
                        s,
                        "--level",
                        "0",
                        "--task",
                        "0",
                        "--dry-run",
                    ]
                ),
            )
        record(
            "sweep smoke_example_agent --dry-run",
            lambda: _run_cli([*cli, "sweep", "--suite", "smoke_example_agent", "--dry-run"]),
        )

    width = max(len(name) for name, _ in rows)
    print(f"{'check':<{width}}  result")
    print(f"{'-' * width}  ------")
    failed = False
    for name, result in rows:
        print(f"{name:<{width}}  {result}")
        if result.startswith("FAIL"):
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
