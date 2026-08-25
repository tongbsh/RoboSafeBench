"""On-disk cache for RoboTwin expert-check accepted seeds.

Expert-check is independent of disturbance level and of the policy. L1/L2/L3
and different models with the same ``(task, task_config, start_seed,
episode_index)`` used to each replay ``play_once``. This module stores the
first successful search so later jobs skip that demo.

Layout: ``{workspace}/results/_cache/robotwin_expert_check/`` unless
``DYNI_EXPERT_CHECK_CACHE_DIR`` is set. One JSON file per key; writers take
an exclusive flock so a parallel job waits and then hits instead of
re-running the demo.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_SCHEMA = 1
_SAFE_TOKEN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class ExpertCacheEntry:
    accepted_seed: int
    episode_info: dict[str, Any]


def resolve_expert_check_cache_dir(
    workspace: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the shared expert-check cache directory."""
    env_map = os.environ if env is None else env
    raw = str(env_map.get("DYNI_EXPERT_CHECK_CACHE_DIR", "")).strip()
    if raw:
        return Path(raw)
    return Path(workspace) / "results" / "_cache" / "robotwin_expert_check"


def cache_key(
    *,
    task_name: str,
    task_config: str,
    start_seed: int,
    episode_index: int,
) -> str:
    """Stable filename stem for one expert-check lookup."""
    return (
        f"{_safe_token(task_name)}__{_safe_token(task_config)}"
        f"__s{int(start_seed)}__e{int(episode_index)}"
    )


def _safe_token(value: str) -> str:
    token = _SAFE_TOKEN.sub("_", str(value).strip()).strip("._")
    if not token:
        raise ValueError(f"expert-check cache token empty after sanitize: {value!r}")
    return token


def cache_path(cache_dir: Path, key: str) -> Path:
    return Path(cache_dir) / f"{key}.json"


@contextmanager
def exclusive_key_lock(cache_dir: Path, key: str) -> Iterator[None]:
    """Block until this process owns *key* (creates ``cache_dir`` if needed)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / f"{key}.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        _flock_exclusive(handle.fileno())
        try:
            yield
        finally:
            _flock_unlock(handle.fileno())


def load_entry(cache_dir: Path, key: str) -> ExpertCacheEntry | None:
    path = cache_path(cache_dir, key)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, Mapping):
        return None
    if int(raw.get("schema", -1)) != _SCHEMA:
        return None
    accepted = raw.get("accepted_seed")
    info = raw.get("episode_info")
    if isinstance(accepted, bool) or not isinstance(accepted, int):
        return None
    if not isinstance(info, dict):
        return None
    return ExpertCacheEntry(accepted_seed=int(accepted), episode_info=dict(info))


def store_entry(
    cache_dir: Path,
    key: str,
    *,
    accepted_seed: int,
    episode_info: Mapping[str, Any],
    task_name: str,
    task_config: str,
    start_seed: int,
    episode_index: int,
) -> None:
    """Atomically replace the JSON record for *key*."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": _SCHEMA,
        "task_name": str(task_name),
        "task_config": str(task_config),
        "start_seed": int(start_seed),
        "episode_index": int(episode_index),
        "accepted_seed": int(accepted_seed),
        "episode_info": _jsonable(episode_info),
        "stored_unix_s": time.time(),
    }
    path = cache_path(cache_dir, key)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _flock_exclusive(fd: int) -> None:
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_EX)


def _flock_unlock(fd: int) -> None:
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)
