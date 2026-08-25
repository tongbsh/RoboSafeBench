"""Offline readers for traces written by :class:`TraceWriter`.

Schema mismatches fail loudly: ``trace_schema_version`` is checked against
this package **before** ``TraceHeader`` fields are parsed, then missing or
extra keys are wrapped the same way. Both raise :class:`TraceSchemaVersionError`
pointing at ``docs/user/behavior_diagnostics.md``. Silent reinterpretation of a
different contract would corrupt every downstream diagnostic.

Truncated files (process killed mid-write) raise :class:`TruncatedTraceError`
by default. ``strict=False`` keeps a complete JSONL *prefix* and repairs
``header.n_steps``. That mode is opt-in because a repaired trace is no longer
the episode the header claimed to describe; gzip framing is not recovered
(partial gzip is not a defined prefix of records).
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, TextIO

from dynisafety.diagnostics.schema import (
    TRACE_SCHEMA_VERSION,
    StepRecord,
    Trace,
    TraceHeader,
)
from dynisafety.diagnostics.writer import INDEX_FILENAME, TRACES_DIRNAME


class TraceSchemaVersionError(ValueError):
    """``trace_schema_version`` does not match this package's frozen contract."""

    def __init__(self, found: str, path: Path, *, reason: str | None = None) -> None:
        if reason is None:
            message = (
                f"trace_schema_version {found!r} in {path} does not match "
                f"this package ({TRACE_SCHEMA_VERSION!r}). Refusing to read a "
                f"foreign contract. See docs/user/behavior_diagnostics.md §4 "
                f"(TraceHeader / StepRecord)."
            )
        else:
            message = (
                f"trace header in {path} does not satisfy this package's "
                f"trace schema contract (version {TRACE_SCHEMA_VERSION!r}): "
                f"{reason}. The on-disk header is from a different schema "
                f"revision (trace_schema_version={found!r}; missing or extra "
                f"TraceHeader keys). Refusing to read a foreign contract. "
                f"See docs/user/behavior_diagnostics.md §4 (TraceHeader / StepRecord)."
            )
        super().__init__(message)
        self.found = found
        self.path = path


class TruncatedTraceError(ValueError):
    """A trace file ended mid-record (crash) or has fewer steps than its header."""

    def __init__(self, message: str, path: Path) -> None:
        super().__init__(message)
        self.path = path


def _is_gzip_path(path: Path) -> bool:
    name = path.name
    return name.endswith(".jsonl.gz") or path.suffix == ".gz"


def _open_text(path: Path) -> TextIO:
    if _is_gzip_path(path):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _read_lines(path: Path) -> list[str]:
    try:
        with _open_text(path) as fp:
            return fp.read().splitlines()
    except (EOFError, gzip.BadGzipFile) as exc:
        raise TruncatedTraceError(
            f"{path}: failed to decompress the gzip stream ({type(exc).__name__}: "
            f"{exc}). A crash mid-write leaves gzip members unrecoverable; "
            f"re-run the episode. See docs/user/behavior_diagnostics.md §4.5.",
            path,
        ) from exc


def _parse_trace_header(payload: dict[str, Any], path: Path) -> TraceHeader:
    """Check ``trace_schema_version`` before the strict field parse.

    ``TraceHeader.from_json_dict`` raises ``KeyError`` on missing/unknown keys.
    An old file that also lacks new fields (e.g. ``stride``) would otherwise
    surface that KeyError instead of a version/contract mismatch. Version is
    therefore validated on the raw dict first; a subsequent key mismatch is
    re-raised as :class:`TraceSchemaVersionError`.
    """
    found = payload.get("trace_schema_version")
    if found is None or str(found) != TRACE_SCHEMA_VERSION:
        raise TraceSchemaVersionError(
            "<missing>" if found is None else str(found),
            path,
        )
    try:
        return TraceHeader.from_json_dict(payload)
    except KeyError as exc:
        detail = exc.args[0] if exc.args else str(exc)
        raise TraceSchemaVersionError(str(found), path, reason=str(detail)) from exc


def read_trace(path: Path | str, *, strict: bool = True) -> Trace:
    """Load one episode ``Trace`` from a ``.jsonl`` or ``.jsonl.gz`` file.

    ``strict=True`` (default) raises :class:`TruncatedTraceError` on an
    incomplete last line or when ``header.n_steps`` does not match the number
    of complete step rows. ``strict=False`` drops a trailing incomplete JSONL
    line and rewrites ``n_steps`` to the kept count. Gzip corruption always
    raises — there is no safe prefix of a broken gzip member.
    """
    dest = Path(path)
    if not dest.is_file():
        raise FileNotFoundError(f"trace file not found: {dest}")
    lines = _read_lines(dest)
    if not lines:
        raise TruncatedTraceError(
            f"{dest}: file is empty (no TraceHeader). See "
            f"docs/user/behavior_diagnostics.md §4.",
            dest,
        )

    records: list[dict[str, Any]] = []
    last = len(lines) - 1
    for i, raw in enumerate(lines):
        text = raw.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            if (not strict) and i == last:
                break
            raise TruncatedTraceError(
                f"{dest}: line {i + 1} is not valid JSON ({exc}). "
                f"The file may have been truncated (process killed mid-write). "
                f"Pass strict=False to keep the complete prefix. "
                f"See docs/user/behavior_diagnostics.md §4.5.",
                dest,
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{dest}: line {i + 1} must be a JSON object")
        records.append(payload)

    if not records:
        raise TruncatedTraceError(
            f"{dest}: no complete JSON objects (header missing). See "
            f"docs/user/behavior_diagnostics.md §4.",
            dest,
        )

    header = _parse_trace_header(records[0], dest)
    steps = tuple(StepRecord.from_json_dict(row) for row in records[1:])
    if header.n_steps != len(steps):
        if strict:
            raise TruncatedTraceError(
                f"{dest}: header.n_steps={header.n_steps} but found "
                f"{len(steps)} complete step records. The file may have been "
                f"truncated (process killed mid-write). Pass strict=False to "
                f"keep the complete prefix and repair n_steps. See "
                f"docs/user/behavior_diagnostics.md §4.5.",
                dest,
            )
        header = replace(header, n_steps=len(steps))
    return Trace(header=header, steps=steps)


def read_index(run_dir: Path | str) -> dict[int, dict[str, Any]]:
    """Return ``episode_index -> {filename, n_steps, sha256}`` from ``traces/index.json``."""
    path = Path(run_dir) / TRACES_DIRNAME / INDEX_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"trace index not found: {path}")
    with path.open(encoding="utf-8") as fp:
        raw = json.load(fp)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a JSON object mapping episode_index to entries")
    out: dict[int, dict[str, Any]] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"{path}: entry {key!r} must be a JSON object")
        out[int(key)] = dict(value)
    return out


def iter_traces(run_dir: Path | str, *, strict: bool = True) -> Iterator[Trace]:
    """Yield traces for a run, ordered by ``episode_index``.

    Prefers ``traces/index.json`` (source of truth after a durable write).
    If the index is missing, falls back to ``traces/ep*.jsonl*`` so a leftover
    directory can still be inspected.
    """
    root = Path(run_dir)
    traces_dir = root / TRACES_DIRNAME
    index_path = traces_dir / INDEX_FILENAME
    if index_path.is_file():
        index = read_index(root)
        for ep in sorted(index):
            filename = Path(str(index[ep]["filename"])).name
            yield read_trace(traces_dir / filename, strict=strict)
        return
    if not traces_dir.is_dir():
        return
    files = sorted(
        p
        for p in traces_dir.iterdir()
        if p.is_file() and (p.name.endswith(".jsonl") or p.name.endswith(".jsonl.gz"))
    )
    for path in files:
        yield read_trace(path, strict=strict)


__all__ = [
    "TraceSchemaVersionError",
    "TruncatedTraceError",
    "iter_traces",
    "read_index",
    "read_trace",
]
