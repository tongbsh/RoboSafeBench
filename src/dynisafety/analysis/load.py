"""Read-only loaders that bind each episode to its run directory.

Uses the existing aggregate JSONL parser. Never opens a backend or writes
into the results tree.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dynisafety.core.aggregate import discover_episode_files, load_episodes
from dynisafety.core.episode import EpisodeRecord
from dynisafety.diagnostics.reader import read_trace
from dynisafety.diagnostics.schema import Trace
from dynisafety.diagnostics.writer import TRACES_DIRNAME, episode_trace_filename

_SWEEP_DIRNAME = "_sweep"
# Safety-layer arms are named ``{base}_oracle_...``. Ranking them against the
# bare VLA compares a filter to a policy. Their own pre-onset is also not a
# counterfactual null: L2/L3 already show the parked obstacle, so a shield
# that stops on sight inflates the fitted scale.
_SAFETY_LAYER_MARKER = "_oracle_"


def is_safety_layer_model(model: str) -> bool:
    return _SAFETY_LAYER_MARKER in str(model)


def null_parent_model(model: str) -> str | None:
    """Bare VLA whose fitted null a safety-layer arm should inherit.

    ``pi05_oracle_swr_aegis`` → ``pi05``. ``None`` means fit this model's own
    pre-onset (it is not a safety-layer name).
    """
    name = str(model)
    if _SAFETY_LAYER_MARKER not in name:
        return None
    parent = name.split(_SAFETY_LAYER_MARKER, 1)[0]
    return parent or None


@dataclass(frozen=True)
class BoundEpisode:
    """One ``EpisodeRecord`` plus the run directory it was loaded from."""

    episode: EpisodeRecord
    run_dir: Path
    jsonl_path: Path


def load_bound_episodes(
    results_root: Path | str,
    *,
    models: Sequence[str] | None = None,
    exclude_models: Sequence[str] | None = None,
) -> list[BoundEpisode]:
    """Load every ``episodes.jsonl`` under ``results_root``.

    Skips ``_sweep`` bookkeeping trees (same rule as the calibrator).

    ``models`` / ``exclude_models`` filter on ``EpisodeRecord.model``. The
    results tree is laid out by suite and level, not by model, so an unfiltered
    load pools every arm that ever wrote into it — including safety-layer arms
    such as ``pi05_oracle_swr_aegis``. Ranking those against the base policies
    compares a policy with a filter bolted on against policies without one.
    ``report`` does not need this because its coverage grid names the models.
    """
    keep = None if models is None else {str(m) for m in models}
    drop = set() if exclude_models is None else {str(m) for m in exclude_models}
    if keep is not None and (keep & drop):
        raise ValueError(f"models and exclude_models overlap: {sorted(keep & drop)}")

    root = Path(results_root)
    paths = [
        path
        for path in discover_episode_files(root)
        if _SWEEP_DIRNAME not in path.parts
    ]
    bound: list[BoundEpisode] = []
    for path in paths:
        run_dir = path.parent
        for ep in load_episodes((path,)):
            name = str(ep.model)
            if keep is not None and name not in keep:
                continue
            if name in drop:
                continue
            bound.append(BoundEpisode(episode=ep, run_dir=run_dir, jsonl_path=path))
    return bound


def behavior_class_of(ep: EpisodeRecord) -> str | None:
    """Return the 9-class label, or ``None`` when diagnostics are missing."""
    diag = ep.diagnostics
    if not isinstance(diag, dict):
        return None
    raw = diag.get("behavior_class")
    if raw is None:
        return None
    return str(getattr(raw, "value", raw))


def diagnostics_triggered(ep: EpisodeRecord) -> bool | None:
    diag = ep.diagnostics
    if not isinstance(diag, dict) or "disturbance_triggered" not in diag:
        return None
    return bool(diag["disturbance_triggered"])


def resolve_trace_path(bound: BoundEpisode) -> Path | None:
    """Best-effort path to the episode trace. ``None`` if nothing is on disk."""
    ep = bound.episode
    diag = ep.diagnostics if isinstance(ep.diagnostics, dict) else {}
    recorded = diag.get("trace_path") if diag else None
    candidates: list[Path] = []
    if recorded:
        raw = Path(str(recorded))
        candidates.append(raw if raw.is_absolute() else bound.run_dir / raw)
        candidates.append(bound.run_dir / TRACES_DIRNAME / raw.name)
    traces_dir = bound.run_dir / TRACES_DIRNAME
    idx = int(ep.episode_index)
    candidates.append(traces_dir / episode_trace_filename(idx, compressed=True))
    candidates.append(traces_dir / episode_trace_filename(idx, compressed=False))
    seen: set[Path] = set()
    for cand in candidates:
        resolved = cand.resolve() if cand.exists() else cand
        if resolved in seen:
            continue
        seen.add(resolved)
        if cand.is_file():
            return cand
    return None


def load_trace(bound: BoundEpisode, *, strict: bool = True) -> Trace | None:
    """Read the episode trace, or ``None`` when the file is absent."""
    path = resolve_trace_path(bound)
    if path is None:
        return None
    return read_trace(path, strict=strict)


def resolve_video_path(bound: BoundEpisode) -> Path | None:
    raw = bound.episode.video_path
    if not raw:
        return None
    path = Path(str(raw))
    if path.is_file():
        return path
    nested = bound.run_dir / path
    if nested.is_file():
        return nested
    return None


def fingerprint_results(results_root: Path | str) -> str:
    """SHA-256 of completed-run ``episodes.jsonl`` and ``traces/`` files.

    Only completed cells enter analysis (same rule as ``discover_episode_files``).
    A live incomplete sweep next door must not trip the mutation guard.
    Used by tests to prove this package never mutates the results tree.
    """
    root = Path(results_root)
    digest = hashlib.sha256()
    files: list[Path] = []
    if root.is_dir():
        for jsonl in discover_episode_files(root):
            files.append(jsonl)
            traces = jsonl.parent / "traces"
            if traces.is_dir():
                files.extend(p for p in traces.rglob("*") if p.is_file())
    for path in sorted(files, key=lambda p: str(p)):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def episode_source_row(bound: BoundEpisode) -> dict[str, Any]:
    """Stable identity fields for CSV rows."""
    ep = bound.episode
    return {
        "run_dir": str(bound.run_dir),
        "run_id": ep.run_id,
        "backend": ep.backend,
        "model": ep.model,
        "suite": ep.suite,
        "task_id": int(ep.task_id),
        "task_name": ep.task_name,
        "level": int(ep.level),
        "seed": int(ep.seed),
        "episode_index": int(ep.episode_index),
    }


def dump_jsonl_identity(path: Path) -> list[dict[str, Any]]:
    """Parse JSONL as raw dicts (tests use this; not for metrics)."""
    rows: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def group_key_tuple(ep: EpisodeRecord, group_cols: Sequence[str]) -> tuple[Any, ...]:
    values: list[Any] = []
    for col in group_cols:
        if not hasattr(ep, col):
            raise KeyError(f"unknown EpisodeRecord field {col!r}")
        values.append(getattr(ep, col))
    return tuple(values)
