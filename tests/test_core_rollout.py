"""Pure-logic tests for ``dynisafety.core.rollout`` (no simulator)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dynisafety.config.schema import MISSING, OutputConfig, RunConfig
from dynisafety.core.episode import SCHEMA_VERSION
from dynisafety.core.rollout import (
    ActionChunkQueue,
    BoundRun,
    build_episode_record,
    episode_video_dir,
    require_bound,
    resolve_save_video,
)


def test_require_bound_returns_frozen_run() -> None:
    bound = require_bound(
        backend="libero",
        config_hash="abc123",
        action_space="delta_ee_7d",
        chunk_size=8,
    )
    assert bound == BoundRun(
        config_hash="abc123",
        action_space="delta_ee_7d",
        chunk_size=8,
    )
    assert bound.config_hash == "abc123"
    assert bound.action_space == "delta_ee_7d"
    assert bound.chunk_size == 8


@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        (
            {"config_hash": None, "action_space": "delta_ee_7d", "chunk_size": 4},
            "config_hash",
        ),
        (
            {"config_hash": "h", "action_space": None, "chunk_size": 4},
            "action_space",
        ),
        (
            {"config_hash": "h", "action_space": "delta_ee_7d", "chunk_size": None},
            "chunk_size",
        ),
    ],
)
def test_require_bound_missing_field_raises(kwargs: dict[str, object], needle: str) -> None:
    """Unbound fields raise RuntimeError, as the backends did before this helper."""
    with pytest.raises(RuntimeError, match="bind_run") as exc:
        require_bound(backend="robocasa", **kwargs)  # type: ignore[arg-type]
    msg = str(exc.value)
    assert "robocasa" in msg
    assert needle in msg


def test_require_bound_unbound_error_is_not_a_value_error() -> None:
    """tests/test_backends_robocasa.py asserts RuntimeError; ValueError would pass it silently."""
    with pytest.raises(RuntimeError):
        require_bound(
            backend="robocasa", config_hash=None, action_space="delta_ee_7d", chunk_size=4
        )
    assert not issubclass(RuntimeError, ValueError)


@pytest.mark.parametrize("chunk_size", [0, -1, -8])
def test_require_bound_non_positive_chunk_size_raises(chunk_size: int) -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        require_bound(
            backend="robotwin",
            config_hash="h",
            action_space="qpos_14d",
            chunk_size=chunk_size,
        )


@pytest.mark.parametrize("chunk_size", [0, -3])
def test_action_chunk_queue_rejects_non_positive_chunk_size(chunk_size: int) -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        ActionChunkQueue(chunk_size)


def test_action_chunk_queue_refill_truncates_and_preserves_order() -> None:
    queue = ActionChunkQueue(3)
    assert queue.is_empty
    assert len(queue) == 0

    converted = np.arange(20, dtype=np.float32).reshape(5, 4)
    n_take = queue.refill(converted)
    assert n_take == 3
    assert not queue.is_empty
    assert len(queue) == 3

    first = queue.popleft()
    second = queue.popleft()
    third = queue.popleft()
    assert first == pytest.approx(converted[0].astype(np.float64))
    assert second == pytest.approx(converted[1].astype(np.float64))
    assert third == pytest.approx(converted[2].astype(np.float64))
    assert first.dtype == np.float64
    assert queue.is_empty


def test_action_chunk_queue_refill_shorter_and_equal_chunk_size() -> None:
    short = ActionChunkQueue(8)
    converted = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.int32)
    assert short.refill(converted) == 2
    assert len(short) == 2
    row = short.popleft()
    assert row.dtype == np.float64
    assert row == pytest.approx(np.array([1.0, 2.0], dtype=np.float64))

    exact = ActionChunkQueue(2)
    assert exact.refill(np.ones((2, 3), dtype=np.float64)) == 2
    assert len(exact) == 2


def test_action_chunk_queue_clear_and_empty_popleft() -> None:
    queue = ActionChunkQueue(4)
    queue.refill(np.zeros((2, 3), dtype=np.float64))
    queue.clear()
    assert queue.is_empty
    assert len(queue) == 0
    with pytest.raises(IndexError):
        queue.popleft()


def test_action_chunk_queue_refill_requires_empty_queue() -> None:
    """A non-empty refill would drop un-executed actions; make that explicit."""
    queue = ActionChunkQueue(4)
    queue.refill(np.zeros((3, 2), dtype=np.float64))
    with pytest.raises(RuntimeError, match="empty queue"):
        queue.refill(np.ones((2, 2), dtype=np.float64))
    queue.clear()
    assert queue.refill(np.ones((2, 2), dtype=np.float64)) == 2


def test_resolve_save_video_four_cases() -> None:
    assert resolve_save_video(RunConfig()) is False
    assert RunConfig().output is MISSING

    missing_flag = RunConfig(output=OutputConfig())
    assert missing_flag.output.save_video is MISSING
    assert resolve_save_video(missing_flag) is False

    assert resolve_save_video(RunConfig(output=OutputConfig(save_video=True))) is True
    assert resolve_save_video(RunConfig(output=OutputConfig(save_video=False))) is False


def test_episode_video_dir_joins_shared_layout() -> None:
    path = episode_video_dir(
        results_dir="/tmp/results",
        suite="robocasa_nav",
        level=2,
        task_id=0,
        task_name="navigate_to_the_sink.",
    )
    assert path == Path("/tmp/results/videos/robocasa_nav_level2/task0_navigate_to_the_sink.")
    via_path = episode_video_dir(
        results_dir=Path("artifacts/out"),
        suite="bridge",
        level=0,
        task_id=3,
        task_name="bridge_carrot",
    )
    assert via_path == Path("artifacts/out/videos/bridge_level0/task3_bridge_carrot")


def _metrics(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "success": True,
        "success_step": 8,
        "steps": 10,
        "contact_steps": 2,
        "contact_events": 1,
        "first_contact_step": 3,
        "contact_partners": {"link_a": 2},
        "peak_force_n": 1.5,
        "peak_impulse_ns": 0.01,
        "cumulative_impulse_ns": 0.02,
        "force_equiv_dt_s": 0.05,
        "max_tilt_deg": 12.0,
        "toppled": False,
    }
    base.update(overrides)
    return base


def _record_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "metrics": _metrics(),
        "run_id": "libero-dummy-l1-s0",
        "config_hash": "abc123def456",
        "backend": "libero",
        "suite": "smoke",
        "model": "dummy",
        "checkpoint": "/ckpt",
        "level": 1,
        "episode_index": 0,
        "task_id": 0,
        "task_name": "task0",
        "instruction": "pick up the object",
        "seed": 7,
        "max_steps": 50,
        "steps": 10,
        "termination": "success",
        "wall_time_s": 1.25,
        "video_path": None,
        "extra": {"note": "unit"},
    }
    kwargs.update(overrides)
    return kwargs


def test_build_episode_record_maps_metrics_and_identity() -> None:
    rec = build_episode_record(**_record_kwargs())  # type: ignore[arg-type]
    assert rec.schema_version == SCHEMA_VERSION
    assert rec.run_id == "libero-dummy-l1-s0"
    assert rec.config_hash == "abc123def456"
    assert rec.backend == "libero"
    assert rec.suite == "smoke"
    assert rec.task_id == 0
    assert rec.task_name == "task0"
    assert rec.instruction == "pick up the object"
    assert rec.model == "dummy"
    assert rec.checkpoint == "/ckpt"
    assert rec.level == 1
    assert rec.seed == 7
    assert rec.episode_index == 0
    assert rec.steps == 10
    assert rec.max_steps == 50
    assert rec.termination == "success"
    assert rec.wall_time_s == pytest.approx(1.25)
    assert rec.success is True
    assert rec.success_step == 8
    assert rec.contact_steps == 2
    assert rec.contact_events == 1
    assert rec.first_contact_step == 3
    assert rec.contact_partners == {"link_a": 2}
    assert rec.peak_force_n == pytest.approx(1.5)
    assert rec.peak_impulse_ns == pytest.approx(0.01)
    assert rec.cumulative_impulse_ns == pytest.approx(0.02)
    assert rec.force_equiv_dt_s == pytest.approx(0.05)
    assert rec.max_tilt_deg == pytest.approx(12.0)
    assert rec.toppled is False
    assert rec.diagnostics is None
    assert rec.video_path is None
    assert rec.extra == {"note": "unit"}


def test_build_episode_record_seed_is_explicit_argument() -> None:
    metrics = _metrics()
    rec = build_episode_record(
        **_record_kwargs(metrics=metrics, seed=4242)  # type: ignore[arg-type]
    )
    assert rec.seed == 4242
    assert "seed" not in metrics


def test_build_episode_record_forwards_diagnostics() -> None:
    diag = {"behavior_class": "not_triggered", "disturbance_triggered": False}
    rec = build_episode_record(**_record_kwargs(diagnostics=diag))  # type: ignore[arg-type]
    assert rec.diagnostics == diag
    assert rec.diagnostics is not diag


def test_build_episode_record_tilt_fields_missing_are_none() -> None:
    metrics = _metrics()
    del metrics["max_tilt_deg"]
    del metrics["toppled"]
    rec = build_episode_record(**_record_kwargs(metrics=metrics))  # type: ignore[arg-type]
    assert rec.max_tilt_deg is None
    assert rec.toppled is None
