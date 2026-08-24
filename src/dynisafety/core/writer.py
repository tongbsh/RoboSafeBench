"""Run output writer: crash-safe episode JSONL + manifest/config/summary."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, TextIO

import yaml

from dynisafety.core.episode import EpisodeRecord

_MANIFEST_KEYS = (
    "git_sha",
    "git_dirty",
    "submodules",
    "python_version",
    "packages",
    "gpu_name",
    "config_hash",
    "schema_version",
)


class RunWriter:
    """Write `results/l{level}/<run_id>/` artifacts with crash-safe episode appends."""

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir)
        self._episodes_path = self.run_dir / "episodes.jsonl"
        self._fp: TextIO | None = None
        self._created = False

    def create(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "videos").mkdir(exist_ok=True)
        # Ensure JSONL exists; do not truncate an existing recoverable file.
        if not self._episodes_path.exists():
            self._episodes_path.touch()
        run_log = self.run_dir / "run.log"
        if not run_log.exists():
            run_log.touch()
        if self._fp is None:
            # line-buffered text mode; durability still relies on flush+fsync
            self._fp = self._episodes_path.open("a", encoding="utf-8", buffering=1)
        self._created = True

    def write_resolved_config(self, config: Mapping[str, Any] | str | Path) -> None:
        self._ensure_created()
        path = self.run_dir / "resolved_config.yaml"
        if isinstance(config, (str, Path)):
            src = Path(config)
            path.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            return
        with path.open("w", encoding="utf-8") as fp:
            yaml.safe_dump(dict(config), fp, sort_keys=False, allow_unicode=True)

    def write_manifest(self, manifest: Mapping[str, Any]) -> None:
        self._ensure_created()
        missing = [k for k in _MANIFEST_KEYS if k not in manifest]
        if missing:
            raise KeyError(f"manifest missing required keys: {missing}")
        path = self.run_dir / "manifest.json"
        self._write_json_atomic(path, dict(manifest))

    def append_episode(self, record: EpisodeRecord) -> None:
        """Append one EpisodeRecord as a single JSONL line; crash-safe.

        After this returns, the line is flushed and fsync'd so a crash before
        the next append cannot tear previously written episodes.
        """
        self._ensure_created()
        assert self._fp is not None
        line = json.dumps(record.to_json_dict(), ensure_ascii=False, separators=(",", ":"))
        self._fp.write(line + "\n")
        self._fp.flush()
        os.fsync(self._fp.fileno())

    def write_summary(self, summary: Mapping[str, Any]) -> None:
        self._ensure_created()
        path = self.run_dir / "summary.json"
        self._write_json_atomic(path, dict(summary))

    def close(self) -> None:
        if self._fp is not None:
            self._fp.flush()
            try:
                os.fsync(self._fp.fileno())
            except OSError:
                pass
            self._fp.close()
            self._fp = None

    def __enter__(self) -> RunWriter:
        self.create()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _ensure_created(self) -> None:
        if not self._created or self._fp is None:
            self.create()

    @staticmethod
    def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
        # Include pid so two writers never share the same .tmp name.
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        data = json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True)
        with tmp.open("w", encoding="utf-8") as fp:
            fp.write(data)
            fp.write("\n")
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp, path)


def build_manifest(
    *,
    config_hash: str,
    schema_version: str,
    **probe_overrides: Any,
) -> dict[str, Any]:
    """Build a manifest dict with frozen keys; overrides win for tests."""
    probed: dict[str, Any] = {
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "submodules": _git_submodules(),
        "python_version": sys.version.split()[0],
        "packages": _package_versions(),
        "gpu_name": _gpu_name(),
        "config_hash": config_hash,
        "schema_version": schema_version,
    }
    probed.update(probe_overrides)
    missing = [k for k in _MANIFEST_KEYS if k not in probed]
    if missing:
        raise KeyError(f"manifest missing required keys after build: {missing}")
    return {k: probed[k] for k in _MANIFEST_KEYS}


def _run_git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _git_sha() -> str | None:
    return _run_git("rev-parse", "HEAD")


def _git_dirty() -> bool | None:
    out = _run_git("status", "--porcelain")
    if out is None:
        return None
    return bool(out)


def _git_submodules() -> dict[str, str | None]:
    out = _run_git("submodule", "status", "--recursive")
    if out is None:
        return {}
    result: dict[str, str | None] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        # format: [ +-U]<sha> <path> ...
        parts = line.split()
        if len(parts) < 2:
            continue
        sha = parts[0].lstrip("+-U")
        name = parts[1]
        result[name] = sha or None
    return result


def _package_versions() -> dict[str, str | None]:
    # Dist name for PyYAML is "PyYAML"; record under canonical key "pyyaml".
    mapping = {
        "torch": "torch",
        "numpy": "numpy",
        "pandas": "pandas",
        "pyyaml": "PyYAML",
    }
    versions: dict[str, str | None] = {}
    for key, dist in mapping.items():
        try:
            versions[key] = importlib_metadata.version(dist)
        except importlib_metadata.PackageNotFoundError:
            versions[key] = None
    return versions


def _gpu_name() -> str | None:
    try:
        import torch  # type: ignore
    except Exception:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return str(torch.cuda.get_device_name(0))
    except Exception:
        return None
