"""Load ``configs/paths.yaml`` and join the one I/O root with layout templates.

``catalog`` / ``kinds`` are documentation; this module ignores them.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import yaml

from dynisafety.config.resolve import (
    default_third_party_root,
    find_workspace_root,
    resolve_placeholders,
)

_LAYOUT_KEYS = ("eval", "videos", "sweep", "logs", "reports", "analysis")


@dataclass(frozen=True)
class PathsConfig:
    """Resolved workspace I/O map. ``out`` is the single root."""

    out: Path
    eval: str
    videos: str
    sweep: str
    logs: str
    reports: str
    analysis: str
    save_video: bool
    consumers: Mapping[str, Mapping[str, str]]

    def with_out(self, out: Path | str) -> PathsConfig:
        return replace(self, out=Path(out))

    def layout_template(self, key: str) -> str:
        if key not in _LAYOUT_KEYS:
            raise KeyError(f"unknown layout {key!r}; expected one of {_LAYOUT_KEYS}")
        return str(getattr(self, key))

    def layout_root(self, key: str) -> Path:
        """Static prefix of a layout template (before the first ``{``)."""
        prefix = self.layout_template(key).split("{", 1)[0].rstrip("/")
        return self.out / prefix if prefix else self.out

    def eval_dir(
        self,
        *,
        backend: str,
        suite: str,
        level: int,
        run_id: str,
    ) -> Path:
        rel = self.eval.format(
            backend=backend,
            suite=suite,
            level=int(level),
            run_id=run_id,
        )
        return self.out / rel

    def eval_parent(self, *, backend: str, suite: str, level: int) -> Path:
        """Directory that contains run dirs for one (backend, suite, level)."""
        rel = self.eval.replace("/{run_id}", "").format(
            backend=backend,
            suite=suite,
            level=int(level),
        )
        return self.out / rel.rstrip("/")

    def videos_dir(
        self,
        *,
        backend: str,
        suite: str,
        level: int,
        run_id: str,
    ) -> Path:
        rel = self.videos.format(
            backend=backend,
            suite=suite,
            level=int(level),
            run_id=run_id,
        )
        return self.out / rel

    def sweep_dir(self, suite: str) -> Path:
        return self.out / self.sweep.format(suite=suite)

    def logs_dir(self, kind: str, name: str) -> Path:
        return self.out / self.logs.format(kind=kind, name=name)

    def reports_dir(self, grid: str) -> Path:
        return self.out / self.reports.format(grid=grid)

    def analysis_dir(self) -> Path:
        return self.out / self.analysis

    def consumer_dir(self, name: str, *, side: str) -> Path:
        if name not in self.consumers:
            raise KeyError(f"unknown consumer {name!r}")
        raw = str(self.consumers[name][side])
        if raw in _LAYOUT_KEYS:
            return self.layout_root(raw)
        return self.out / raw


def load_paths(
    *,
    workspace: Path | None = None,
    path: Path | None = None,
) -> PathsConfig:
    """Read ``configs/paths.yaml`` (or ``path``) and resolve placeholders."""
    workspace = Path(workspace) if workspace is not None else find_workspace_root()
    yaml_path = Path(path) if path is not None else workspace / "configs" / "paths.yaml"
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{yaml_path} must be a mapping")
    data = resolve_placeholders(
        raw,
        workspace=workspace,
        third_party=default_third_party_root(workspace),
    )
    roots = data.get("roots")
    layout = data.get("layout")
    if not isinstance(roots, dict) or not isinstance(layout, dict):
        raise ValueError(f"{yaml_path} must have roots: and layout:")
    missing = [k for k in _LAYOUT_KEYS if k not in layout]
    if missing:
        raise ValueError(f"{yaml_path} layout missing {missing}")
    if "out" not in roots:
        raise ValueError(f"{yaml_path} roots.out is required")
    flags = data.get("flags") if isinstance(data.get("flags"), dict) else {}
    consumers_raw = data.get("consumers") if isinstance(data.get("consumers"), dict) else {}
    consumers: dict[str, dict[str, str]] = {}
    for cname, spec in consumers_raw.items():
        if not isinstance(spec, dict):
            continue
        consumers[str(cname)] = {str(k): str(v) for k, v in spec.items()}
    return PathsConfig(
        out=Path(str(roots["out"])).expanduser(),
        eval=str(layout["eval"]),
        videos=str(layout["videos"]),
        sweep=str(layout["sweep"]),
        logs=str(layout["logs"]),
        reports=str(layout["reports"]),
        analysis=str(layout["analysis"]),
        save_video=bool(flags.get("save_video", False)),
        consumers=consumers,
    )


def _print_cmd(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dynisafety.config.paths")
    parser.add_argument(
        "what",
        choices=(
            "out",
            "eval-root",
            "logs",
            "reports-root",
            "analysis-root",
            "sweep-root",
            "consumer-read",
            "consumer-write",
        ),
    )
    parser.add_argument("args", nargs="*")
    ns = parser.parse_args(argv)
    paths = load_paths()
    what = ns.what
    extra = list(ns.args)
    if what == "out":
        print(paths.out)
        return 0
    if what == "eval-root":
        print(paths.layout_root("eval"))
        return 0
    if what == "logs":
        if len(extra) != 2:
            print("usage: python -m dynisafety.config.paths logs KIND NAME", file=sys.stderr)
            return 2
        print(paths.logs_dir(extra[0], extra[1]))
        return 0
    if what == "reports-root":
        print(paths.layout_root("reports"))
        return 0
    if what == "analysis-root":
        print(paths.layout_root("analysis"))
        return 0
    if what == "sweep-root":
        suite = extra[0] if extra else ""
        print(paths.sweep_dir(suite) if suite else paths.layout_root("sweep"))
        return 0
    if what in {"consumer-read", "consumer-write"}:
        if len(extra) != 1:
            print(f"usage: python -m dynisafety.config.paths {what} NAME", file=sys.stderr)
            return 2
        side = "read" if what == "consumer-read" else "write"
        print(paths.consumer_dir(extra[0], side=side))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(_print_cmd())
