"""Offline accepted-seed sequence for parallel eval (plan 0.7).

``_seed_cursor`` is sequential: episode k's search start is
``next_candidate_seed`` of episode k-1's accepted seed. Parallel workers
cannot reconstruct that chain, so this module walks it once and writes
``(episode_index, start_seed, accepted_seed)``.

Expert-check cache keys on the **search start**, not the accepted seed
(``expert_cache.cache_key(..., start_seed=...)``). ``begin`` still calls
``expert_check_setup(seed=...)`` and treats that argument as the start.
Jobs therefore put ``start_seed`` into ``SlotEpisodeJob.start_seed`` so
the cache hits and the accepted result cannot drift. The on-disk JSON
still stores the ``(episode_index, start_seed, accepted_seed)`` triple.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dynisafety.backends.robotwin.seed import episode_search_start, next_candidate_seed

WARM_SCHEMA_VERSION = 1

AcceptFn = Callable[[int, int], int]


@dataclass(frozen=True)
class WarmEpisode:
    """One cursor step. ``start_seed`` is the cache key; ``accepted_seed`` is the demo."""

    episode_index: int
    start_seed: int
    accepted_seed: int

    def __post_init__(self) -> None:
        if isinstance(self.episode_index, bool) or not isinstance(self.episode_index, int):
            raise TypeError(f"episode_index must be int, got {self.episode_index!r}")
        if self.episode_index < 0:
            raise ValueError(f"episode_index must be >= 0, got {self.episode_index}")
        if isinstance(self.start_seed, bool) or not isinstance(self.start_seed, int):
            raise TypeError(f"start_seed must be int, got {self.start_seed!r}")
        if isinstance(self.accepted_seed, bool) or not isinstance(self.accepted_seed, int):
            raise TypeError(f"accepted_seed must be int, got {self.accepted_seed!r}")


@dataclass(frozen=True)
class WarmDocument:
    """On-disk seed-warm artifact. Schema version is required so a format change fails loud."""

    schema: int
    task: str
    suite: str
    level: int
    run_seed: int
    seed_stride: int
    episodes: tuple[WarmEpisode, ...]

    def __post_init__(self) -> None:
        if self.schema != WARM_SCHEMA_VERSION:
            raise ValueError(
                f"seed-warm schema {self.schema} != {WARM_SCHEMA_VERSION}"
            )
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError(f"task must be a non-empty str, got {self.task!r}")
        if not isinstance(self.suite, str) or not self.suite.strip():
            raise ValueError(f"suite must be a non-empty str, got {self.suite!r}")
        if isinstance(self.level, bool) or not isinstance(self.level, int):
            raise TypeError(f"level must be int, got {self.level!r}")
        if isinstance(self.run_seed, bool) or not isinstance(self.run_seed, int):
            raise TypeError(f"run_seed must be int, got {self.run_seed!r}")
        if isinstance(self.seed_stride, bool) or not isinstance(self.seed_stride, int):
            raise TypeError(f"seed_stride must be int, got {self.seed_stride!r}")
        if self.seed_stride <= 0:
            raise ValueError(f"seed_stride must be > 0, got {self.seed_stride}")


def make_episode_uid(
    *,
    task: str,
    level: int,
    run_seed: int,
    episode_index: int,
) -> str:
    """Deterministic uid. ``ObsMsg.rng_key`` hashes this; a rename changes x1."""
    if not isinstance(task, str) or not task.strip():
        raise ValueError(f"task must be a non-empty str, got {task!r}")
    if isinstance(level, bool) or not isinstance(level, int):
        raise TypeError(f"level must be int, got {level!r}")
    if isinstance(run_seed, bool) or not isinstance(run_seed, int):
        raise TypeError(f"run_seed must be int, got {run_seed!r}")
    if isinstance(episode_index, bool) or not isinstance(episode_index, int):
        raise TypeError(f"episode_index must be int, got {episode_index!r}")
    if episode_index < 0:
        raise ValueError(f"episode_index must be >= 0, got {episode_index}")
    return f"{task.strip()}__l{int(level)}__s{int(run_seed)}__e{int(episode_index)}"


def walk_seed_cursor(
    *,
    run_seed: int,
    seed_stride: int,
    n_episodes: int,
    accept: AcceptFn,
) -> list[WarmEpisode]:
    """Replay the old ``_seed_cursor`` chain. ``accept(start, index) -> accepted``."""
    if isinstance(n_episodes, bool) or not isinstance(n_episodes, int):
        raise TypeError(f"n_episodes must be int, got {n_episodes!r}")
    if n_episodes < 0:
        raise ValueError(f"n_episodes must be >= 0, got {n_episodes}")
    cursor = episode_search_start(run_seed=int(run_seed), seed_stride=int(seed_stride))
    rows: list[WarmEpisode] = []
    for index in range(n_episodes):
        accepted = int(accept(int(cursor), int(index)))
        rows.append(
            WarmEpisode(
                episode_index=index,
                start_seed=int(cursor),
                accepted_seed=accepted,
            )
        )
        cursor = next_candidate_seed(accepted)
    return rows


def warm_with_handle(
    handle: Any,
    *,
    n_episodes: int,
    run_seed: int,
    seed_stride: int,
    enabled: bool,
    max_attempts: int,
    timeout_s: float,
    log_every: int,
    cache_dir: Path | None,
) -> list[WarmEpisode]:
    """Serial ``expert_check_setup``. Writes the existing flock cache as a side effect."""
    from dynisafety.backends.robotwin.env import expert_check_setup

    def _accept(start_seed: int, episode_index: int) -> int:
        outcome = expert_check_setup(
            handle,
            seed=start_seed,
            episode_index=episode_index,
            enabled=enabled,
            max_attempts=max_attempts,
            timeout_s=timeout_s,
            log_every=log_every,
            cache_dir=cache_dir,
        )
        return int(outcome.accepted_seed)

    return walk_seed_cursor(
        run_seed=run_seed,
        seed_stride=seed_stride,
        n_episodes=n_episodes,
        accept=_accept,
    )


def build_warm_document(
    *,
    task: str,
    suite: str,
    level: int,
    run_seed: int,
    seed_stride: int,
    episodes: Sequence[WarmEpisode],
) -> WarmDocument:
    return WarmDocument(
        schema=WARM_SCHEMA_VERSION,
        task=str(task),
        suite=str(suite),
        level=int(level),
        run_seed=int(run_seed),
        seed_stride=int(seed_stride),
        episodes=tuple(episodes),
    )


def write_warm_document(document: WarmDocument, path: Path | str) -> None:
    """Temp file + flush/fsync + ``os.replace`` so a crash cannot tear the dest."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": int(document.schema),
        "task": document.task,
        "suite": document.suite,
        "level": int(document.level),
        "run_seed": int(document.run_seed),
        "seed_stride": int(document.seed_stride),
        "episodes": [
            {
                "episode_index": row.episode_index,
                "start_seed": row.start_seed,
                "accepted_seed": row.accepted_seed,
            }
            for row in document.episodes
        ],
    }
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, dest)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def load_warm_document(path: Path | str) -> WarmDocument:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError(f"seed-warm JSON root must be a mapping, got {type(raw)!r}")
    if "schema" not in raw:
        raise ValueError("seed-warm JSON missing required key 'schema'")
    episodes_raw = raw.get("episodes")
    if not isinstance(episodes_raw, list):
        raise TypeError("seed-warm JSON 'episodes' must be a list")
    episodes = tuple(
        WarmEpisode(
            episode_index=_require_int(row, "episode_index"),
            start_seed=_require_int(row, "start_seed"),
            accepted_seed=_require_int(row, "accepted_seed"),
        )
        for row in episodes_raw
    )
    return WarmDocument(
        schema=_require_int(raw, "schema"),
        task=_require_str(raw, "task"),
        suite=_require_str(raw, "suite"),
        level=_require_int(raw, "level"),
        run_seed=_require_int(raw, "run_seed"),
        seed_stride=_require_int(raw, "seed_stride"),
        episodes=episodes,
    )


def jobs_from_warm_document(document: WarmDocument) -> list[Any]:
    """``SlotEpisodeJob.start_seed`` is the search start (cache-key alignment)."""
    from dynisafety.parallel_eval.env_worker import SlotEpisodeJob

    jobs = []
    for row in document.episodes:
        jobs.append(
            SlotEpisodeJob(
                episode_uid=make_episode_uid(
                    task=document.task,
                    level=document.level,
                    run_seed=document.run_seed,
                    episode_index=row.episode_index,
                ),
                episode_index=row.episode_index,
                start_seed=row.start_seed,
            )
        )
    return jobs


def jobs_from_warm_path(path: Path | str) -> list[Any]:
    return jobs_from_warm_document(load_warm_document(path))


def default_warm_path(
    *,
    results_dir: Path,
    task: str,
    suite: str,
    level: int,
    run_seed: int,
    n_episodes: int,
) -> Path:
    """Derived cache location. No hardcoded workspace roots."""
    stem = (
        f"{task.strip()}__{suite.strip()}__l{int(level)}"
        f"__s{int(run_seed)}__n{int(n_episodes)}.json"
    )
    return Path(results_dir) / "_cache" / "robotwin_seed_warm" / stem


def _require_int(data: Mapping[str, Any], key: str) -> int:
    if key not in data:
        raise ValueError(f"seed-warm JSON missing required key {key!r}")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"seed-warm JSON {key!r} must be int, got {value!r}")
    return value


def _require_str(data: Mapping[str, Any], key: str) -> str:
    if key not in data:
        raise ValueError(f"seed-warm JSON missing required key {key!r}")
    value = data[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"seed-warm JSON {key!r} must be a non-empty str, got {value!r}")
    return value
