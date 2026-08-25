"""Crash-safe per-episode trace files under ``{run_dir}/traces/``.

Layout (docs/user/behavior_diagnostics.md §4.5)::

    {run_dir}/traces/
    ├── ep0000.jsonl.gz     # line 0 = TraceHeader, then one StepRecord per line
    ├── ep0001.jsonl.gz
    └── index.json          # episode_index -> {filename, n_steps, sha256}

``jsonl_gz`` is the default (streamable, no extra dependency). ``jsonl`` is
uncompressed for debugging. Each trace is written to a pid-suffixed temp file,
``flush`` + ``os.fsync``'d, then ``os.replace``'d into place. ``index.json`` is
updated the same way so a crash cannot publish a half-written index.

``sha256`` hashes the on-disk bytes of the finished trace file (gzip-compressed
when that is the format) so an open-source drop can be integrity-checked.

When ``diagnostics.trace.save`` is False this class is a no-op: ``write``
returns None and creates no files. Rollout can always construct a writer and
unconditionally call ``write`` without branching.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from dynisafety.config.schema import MISSING, DiagnosticsConfig, DiagnosticsTraceConfig
from dynisafety.diagnostics.schema import Trace

TRACES_DIRNAME = "traces"
INDEX_FILENAME = "index.json"
_VALID_FORMATS = frozenset({"jsonl_gz", "jsonl"})


def _as_trace_cfg(
    cfg: DiagnosticsTraceConfig | DiagnosticsConfig | None,
) -> DiagnosticsTraceConfig | None:
    if cfg is None:
        return None
    if isinstance(cfg, DiagnosticsTraceConfig):
        return cfg
    if isinstance(cfg, DiagnosticsConfig):
        inner = cfg.trace
        if isinstance(inner, DiagnosticsTraceConfig):
            return inner
    raise TypeError(
        "cfg must be DiagnosticsTraceConfig, DiagnosticsConfig, or None, "
        f"got {type(cfg)!r}"
    )


def _cfg_value(value: Any, fallback: Any) -> Any:
    return fallback if value is MISSING or value is None else value


def episode_trace_filename(episode_index: int, *, compressed: bool) -> str:
    suffix = "jsonl.gz" if compressed else "jsonl"
    return f"ep{int(episode_index):04d}.{suffix}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_line(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _write_bytes_durable(path: Path, chunks: list[bytes], *, compressed: bool) -> None:
    """Write ``chunks`` via temp + fsync + replace so a crash cannot tear the dest."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as raw:
            if compressed:
                with gzip.GzipFile(
                    filename="", mode="wb", fileobj=raw, mtime=0
                ) as gz:
                    for chunk in chunks:
                        gz.write(chunk)
            else:
                for chunk in chunks:
                    raw.write(chunk)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    try:
        with tmp.open("w", encoding="utf-8") as fp:
            fp.write(data)
            fp.write("\n")
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


class TraceWriter:
    """Write one ``Trace`` per episode and incrementally update ``index.json``.

    ``save=False`` (from ``cfg``) makes every method a no-op except construction.
    """

    def __init__(
        self,
        run_dir: Path | str,
        cfg: DiagnosticsTraceConfig | DiagnosticsConfig | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        trace_cfg = _as_trace_cfg(cfg)
        if trace_cfg is None:
            self._save = True
            self._format = "jsonl_gz"
        else:
            self._save = bool(_cfg_value(trace_cfg.save, True))
            fmt = str(_cfg_value(trace_cfg.format, "jsonl_gz"))
            if fmt not in _VALID_FORMATS:
                raise ValueError(
                    f"diagnostics.trace.format must be one of {sorted(_VALID_FORMATS)}, "
                    f"got {fmt!r}"
                )
            self._format = fmt
        self._compressed = self._format == "jsonl_gz"
        self._traces_dir = self.run_dir / TRACES_DIRNAME
        self._index_path = self._traces_dir / INDEX_FILENAME

    @property
    def save(self) -> bool:
        return self._save

    @property
    def format(self) -> str:
        return self._format

    def write(self, episode_index: int, trace: Trace) -> str | None:
        """Persist ``trace`` as ``traces/epXXXX.jsonl[.gz]``.

        Returns the path relative to ``run_dir`` (posix), for
        ``EpisodeRecord.diagnostics.trace_path``. Returns None when save is off.

        The N-th write does not rewrite earlier episode files; only that file
        and ``index.json`` change.
        """
        if not self._save:
            return None
        ep = int(episode_index)
        if ep != int(trace.header.episode_index):
            raise ValueError(
                f"write episode_index={ep} != header.episode_index="
                f"{trace.header.episode_index}"
            )
        self._traces_dir.mkdir(parents=True, exist_ok=True)
        filename = episode_trace_filename(ep, compressed=self._compressed)
        dest = self._traces_dir / filename
        chunks = [_json_line(trace.header.to_json_dict())]
        chunks.extend(_json_line(step.to_json_dict()) for step in trace.steps)
        _write_bytes_durable(dest, chunks, compressed=self._compressed)
        index = self._read_index_raw()
        index[str(ep)] = {
            "filename": filename,
            "n_steps": int(trace.header.n_steps),
            "sha256": _sha256_file(dest),
        }
        _write_json_atomic(self._index_path, index)
        return f"{TRACES_DIRNAME}/{filename}"

    def _read_index_raw(self) -> dict[str, Any]:
        if not self._index_path.exists():
            return {}
        with self._index_path.open(encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, dict):
            raise ValueError(f"{self._index_path} must be a JSON object")
        return data


__all__ = [
    "INDEX_FILENAME",
    "TRACES_DIRNAME",
    "TraceWriter",
    "episode_trace_filename",
]
