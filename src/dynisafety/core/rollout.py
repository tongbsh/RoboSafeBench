"""Engine-agnostic rollout primitives (stdlib + numpy only).

Composable helpers for bind checks, action-chunk queues, video-dir layout,
and EpisodeRecord assembly. Backends keep their own loop bodies — this
module does not own control flow and is not a template-method base class.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dynisafety.config.schema import MISSING, RunConfig
from dynisafety.core.episode import SCHEMA_VERSION, EpisodeRecord, format_checkpoint_label

# Whitelist extra keys (aggregate may read these; report/tables must not).
SETUP_INVALID_KEY = "setup_invalid"
SETUP_INVALID_REASON_KEY = "setup_invalid_reason"


def setup_invalid_extra(reason: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Mark an episode as not a valid trial of the requested disturbance level.

    Used when L1–L3 assembly fails (no visible obstacle, bind miss, no L3
    target). The requested ``level`` stays on the record for coverage; metrics
    exclude these rows so they cannot inflate ``sr`` or deflate ``cr``.
    """
    out = dict(extra or {})
    out[SETUP_INVALID_KEY] = True
    out[SETUP_INVALID_REASON_KEY] = str(reason)
    return out


@dataclass(frozen=True)
class BoundRun:
    """Fields that ``bind_run`` must populate before ``rollout``."""

    config_hash: str
    action_space: str
    chunk_size: int


def require_bound(
    *,
    backend: str,
    config_hash: str | None,
    action_space: str | None,
    chunk_size: int | None,
) -> BoundRun:
    """Return a ``BoundRun`` or raise if ``bind_run`` has not been called.

    Unbound fields raise ``RuntimeError``, matching what the backends raised
    before this helper existed. A bound but out-of-range ``chunk_size`` is a
    bad value rather than a missing binding, so it raises ``ValueError``.
    """
    if config_hash is None:
        raise RuntimeError(f"{backend}.bind_run(...) must be called before rollout: config_hash")
    if action_space is None:
        raise RuntimeError(f"{backend}.bind_run(...) must be called before rollout: action_space")
    if chunk_size is None:
        raise RuntimeError(f"{backend}: chunk_size not bound; call bind_run()")
    if int(chunk_size) < 1:
        raise ValueError(f"{backend}: chunk_size must be >= 1, got {chunk_size!r}")
    return BoundRun(
        config_hash=str(config_hash),
        action_space=str(action_space),
        chunk_size=int(chunk_size),
    )


class ActionChunkQueue:
    """FIFO of converted action rows; each ``refill`` takes at most ``chunk_size``.

    Matches the four-backend enqueue loop: ``min(chunk_size, n_rows)`` then
    ``np.asarray(row, dtype=np.float64)``. ``popleft`` on an empty queue
    raises ``IndexError`` (same as ``collections.deque``).
    """

    def __init__(self, chunk_size: int) -> None:
        if int(chunk_size) < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size!r}")
        self._chunk_size = int(chunk_size)
        self._queue: deque[np.ndarray] = deque()

    def __len__(self) -> int:
        return len(self._queue)

    @property
    def is_empty(self) -> bool:
        return not self._queue

    def refill(self, converted: np.ndarray) -> int:
        """Enqueue up to ``chunk_size`` leading rows. Returns the count appended.

        The queue must be empty. Backends refill only after draining, and a
        non-empty refill has no correct meaning: the previous ``deque(maxlen=
        chunk_size)`` silently dropped un-executed actions off the left.
        """
        if self._queue:
            raise RuntimeError(
                f"refill requires an empty queue, {len(self._queue)} action(s) still pending"
            )
        n_take = min(self._chunk_size, int(converted.shape[0]))
        for row in converted[:n_take]:
            self._queue.append(np.asarray(row, dtype=np.float64))
        return n_take

    def popleft(self) -> np.ndarray:
        return self._queue.popleft()

    def clear(self) -> None:
        self._queue.clear()


def resolve_save_video(cfg: RunConfig) -> bool:
    """True only when ``cfg.output.save_video`` is present and true."""
    return bool(
        cfg.output is not MISSING and cfg.output.save_video is not MISSING and cfg.output.save_video
    )


def run_dir_layout(
    results_root: Path | str,
    *,
    backend: str,
    suite: str,
    level: int,
    run_id: str,
) -> Path:
    """``{out}/eval/{backend}/{suite}/l{level}/{run_id}`` from ``configs/paths.yaml``.

    ``results_root`` is the one I/O root (``paths.out`` / ``--output-root``).
    ``runner.loop.make_run_dir`` delegates here so backends reconstructing the
    path from ``cfg`` cannot drift from the runner.
    """
    from dynisafety.config.paths import load_paths

    return load_paths().with_out(results_root).eval_dir(
        backend=str(backend),
        suite=str(suite),
        level=int(level),
        run_id=str(run_id),
    )


def run_dir_from_cfg(cfg: RunConfig) -> Path | None:
    """Run directory implied by ``cfg``, or ``None`` if it cannot be derived.

    Returns ``None`` rather than raising: a missing ``run_id`` / ``results_dir``
    means diagnostics traces are skipped, which must never abort a rollout.
    """
    if cfg.output is MISSING or cfg.output.results_dir is MISSING:
        return None
    if cfg.run_id is MISSING or not cfg.run_id:
        return None
    if cfg.level is MISSING:
        return None
    if cfg.backend is MISSING or cfg.suite is MISSING:
        return None
    return run_dir_layout(
        str(cfg.output.results_dir),
        backend=str(cfg.backend),
        suite=str(cfg.suite),
        level=int(cfg.level),
        run_id=str(cfg.run_id),
    )


def episode_video_dir(
    *,
    results_dir: str | Path,
    backend: str,
    suite: str,
    level: int,
    run_id: str,
    task_id: int = 0,
    task_name: str = "",
) -> Path:
    """``{out}/eval/{backend}/{suite}/l{level}/{run_id}/videos``.

    ``task_id`` / ``task_name`` are unused (filenames stay the caller's job).
    """
    del task_id, task_name
    from dynisafety.config.paths import load_paths

    return load_paths().with_out(results_dir).videos_dir(
        backend=str(backend),
        suite=str(suite),
        level=int(level),
        run_id=str(run_id),
    )


def build_episode_record(
    *,
    metrics: Mapping[str, Any],
    run_id: str,
    config_hash: str,
    backend: str,
    suite: str,
    model: str,
    checkpoint: str,
    level: int,
    episode_index: int,
    task_id: int,
    task_name: str,
    instruction: str,
    seed: int,
    max_steps: int,
    steps: int,
    termination: str,
    wall_time_s: float,
    video_path: str | None,
    extra: Mapping[str, Any],
    diagnostics: Mapping[str, Any] | None = None,
) -> EpisodeRecord:
    """Assemble an ``EpisodeRecord`` from finalized metrics plus identity fields.

    ``success`` / ``success_step`` come from ``metrics`` (all four backends
    read them after ``finalize``). ``steps`` and ``seed`` are explicit so a
    caller can pass values that are not ``cfg.seed`` / ``metrics['steps']``.
    ``max_tilt_deg`` and ``toppled`` use ``.get`` (absent → ``None``).

    ``diagnostics`` is ``BehaviorDiagnostics.to_json_dict()`` for backends that
    have been wired for behavior diagnostics; ``None`` everywhere else, which
    tables must render as n/a rather than 0.
    """
    return EpisodeRecord(
        schema_version=SCHEMA_VERSION,
        run_id=str(run_id),
        config_hash=str(config_hash),
        backend=str(backend),
        suite=str(suite),
        task_id=int(task_id),
        task_name=str(task_name),
        instruction=str(instruction),
        model=str(model),
        checkpoint=format_checkpoint_label(checkpoint, suite=str(suite)),
        level=int(level),
        seed=int(seed),
        episode_index=int(episode_index),
        steps=int(steps),
        max_steps=int(max_steps),
        termination=str(termination),
        wall_time_s=float(wall_time_s),
        success=bool(metrics["success"]),
        success_step=metrics["success_step"],
        contact_steps=int(metrics["contact_steps"]),
        contact_events=int(metrics["contact_events"]),
        first_contact_step=metrics["first_contact_step"],
        contact_partners=dict(metrics["contact_partners"]),
        peak_force_n=float(metrics["peak_force_n"]),
        peak_impulse_ns=float(metrics["peak_impulse_ns"]),
        cumulative_impulse_ns=float(metrics["cumulative_impulse_ns"]),
        force_equiv_dt_s=float(metrics["force_equiv_dt_s"]),
        max_tilt_deg=metrics.get("max_tilt_deg"),
        toppled=metrics.get("toppled"),
        video_path=video_path,
        extra=dict(extra),
        diagnostics=dict(diagnostics) if diagnostics is not None else None,
    )
