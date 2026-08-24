"""``dynisafety config`` subcommands: print / diff / validate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from dynisafety.config.loader import (
    ConfigValidationError,
    LoadRequest,
    ResolvedConfig,
    load_config,
    validate_config,
)
from dynisafety.config.schema import KNOWN_BACKENDS


def _add_selector_args(p: argparse.ArgumentParser, *, required_core: bool) -> None:
    """Shared selectors. No hyperparameter ``default=`` for business values."""
    p.add_argument(
        "--backend",
        required=required_core,
        help="Backend id (" + "|".join(KNOWN_BACKENDS) + ")",
    )
    p.add_argument("--level", type=int, required=required_core, help="Disturbance level (validated 0..3)")
    p.add_argument("--suite", default=None, help="Suite layer name (optional)")
    p.add_argument("--task", default=None, help="Task YAML stem (optional)")
    p.add_argument("--model", default=None, help="Model YAML stem (optional)")
    p.add_argument("--config", type=Path, default=None, help="Extra experiment YAML")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Override dotted key (repeatable)",
    )


def register_config_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``config`` with print / diff / validate."""
    cfg = subparsers.add_parser("config", help="Inspect and validate layered configs")
    cfg_sub = cfg.add_subparsers(dest="config_command", required=True)

    p_print = cfg_sub.add_parser("print", help="Print resolved YAML with # from: provenance")
    _add_selector_args(p_print, required_core=True)
    p_print.set_defaults(config_handler=cmd_config_print)

    p_val = cfg_sub.add_parser("validate", help="Validate resolved config")
    _add_selector_args(p_val, required_core=True)
    p_val.set_defaults(config_handler=cmd_config_validate)

    p_diff = cfg_sub.add_parser("diff", help="Diff two resolved configs")
    _add_selector_args(p_diff, required_core=True)
    p_diff.add_argument("--vs-backend", required=True, help="Right-hand backend")
    p_diff.add_argument("--vs-level", type=int, required=True, help="Right-hand level")
    p_diff.add_argument("--vs-suite", default=None)
    p_diff.add_argument("--vs-task", default=None)
    p_diff.add_argument("--vs-model", default=None)
    p_diff.add_argument("--vs-config", type=Path, default=None)
    p_diff.add_argument(
        "--vs-set",
        dest="vs_overrides",
        action="append",
        default=None,
        metavar="KEY=VALUE",
    )
    p_diff.set_defaults(config_handler=cmd_config_diff)


def _request_from_args(args: argparse.Namespace) -> LoadRequest:
    return LoadRequest(
        backend=args.backend,
        level=args.level,
        suite=args.suite,
        task=args.task,
        model=args.model,
        config_file=args.config,
        overrides=tuple(args.overrides or ()),
    )


def _vs_request_from_args(args: argparse.Namespace) -> LoadRequest:
    return LoadRequest(
        backend=args.vs_backend,
        level=args.vs_level,
        suite=args.vs_suite,
        task=args.vs_task,
        model=args.vs_model,
        config_file=args.vs_config,
        overrides=tuple(args.vs_overrides or ()),
    )


def _yaml_scalar(value: Any) -> str:
    text = yaml.safe_dump(
        value,
        default_flow_style=True,
        allow_unicode=True,
        width=10_000,
    )
    # PyYAML may append a document-end marker ("..."); strip it so leaf lines stay single-line.
    lines = [ln for ln in text.splitlines() if ln.strip() != "..."]
    return "\n".join(lines).strip()


def format_yaml_with_provenance(data: dict[str, Any], provenance: dict[str, str]) -> str:
    """Block YAML; each leaf line ends with ``  # from: <source>``."""
    lines: list[str] = []

    def src_for(path: str) -> str:
        if path in provenance:
            return provenance[path]
        # List/map open children inherit the nearest annotated ancestor source.
        parts = path.split(".")
        for i in range(len(parts) - 1, 0, -1):
            parent = ".".join(parts[:i])
            if parent in provenance:
                return provenance[parent]
        return "unknown"

    def emit_mapping(obj: dict[str, Any], path: str, indent: int) -> None:
        pad = "  " * indent
        if not obj:
            lines.append(f"{pad}{{}}  # from: {src_for(path) if path else 'unknown'}")
            return
        for key, value in obj.items():
            child = f"{path}.{key}" if path else str(key)
            if isinstance(value, dict):
                lines.append(f"{pad}{key}:")
                emit_mapping(value, child, indent + 1)
                continue
            if isinstance(value, list):
                emit_list(value, child, indent, key)
                continue
            lines.append(f"{pad}{key}: {_yaml_scalar(value)}  # from: {src_for(child)}")

    def emit_list(value: list[Any], path: str, indent: int, key: str | None) -> None:
        pad = "  " * indent
        src = src_for(path)
        if key is not None:
            if value and all(not isinstance(x, (dict, list)) for x in value):
                lines.append(f"{pad}{key}: {_yaml_scalar(value)}  # from: {src}")
                return
            lines.append(f"{pad}{key}:")
            item_indent = indent + 1
        else:
            item_indent = indent
        ipad = "  " * item_indent
        for idx, item in enumerate(value):
            item_path = f"{path}.{idx}"
            if isinstance(item, dict):
                keys = list(item.keys())
                if not keys:
                    lines.append(f"{ipad}- {{}}  # from: {src_for(item_path)}")
                    continue
                first = keys[0]
                rest = keys[1:]
                first_val = item[first]
                first_path = f"{item_path}.{first}"
                if isinstance(first_val, dict):
                    lines.append(f"{ipad}- {first}:")
                    emit_mapping(first_val, first_path, item_indent + 2)
                elif isinstance(first_val, list):
                    # rare
                    lines.append(f"{ipad}- {first}: {_yaml_scalar(first_val)}  # from: {src_for(first_path)}")
                else:
                    lines.append(
                        f"{ipad}- {first}: {_yaml_scalar(first_val)}  # from: {src_for(first_path)}"
                    )
                for rk in rest:
                    rv = item[rk]
                    rpath = f"{item_path}.{rk}"
                    rpad = "  " * (item_indent + 1)
                    if isinstance(rv, dict):
                        lines.append(f"{rpad}{rk}:")
                        emit_mapping(rv, rpath, item_indent + 2)
                    elif isinstance(rv, list):
                        if rv and all(not isinstance(x, (dict, list)) for x in rv):
                            lines.append(f"{rpad}{rk}: {_yaml_scalar(rv)}  # from: {src_for(rpath)}")
                        else:
                            lines.append(f"{rpad}{rk}:")
                            emit_list(rv, rpath, item_indent + 2, key=None)
                    else:
                        lines.append(f"{rpad}{rk}: {_yaml_scalar(rv)}  # from: {src_for(rpath)}")
            elif isinstance(item, list):
                lines.append(f"{ipad}- {_yaml_scalar(item)}  # from: {src_for(item_path)}")
            else:
                lines.append(f"{ipad}- {_yaml_scalar(item)}  # from: {src_for(item_path)}")

    if not isinstance(data, dict):
        raise TypeError("resolved config root must be a mapping")
    emit_mapping(data, "", 0)
    return "\n".join(lines) + "\n"


def format_resolved(rc: ResolvedConfig) -> str:
    header = [
        f"# config_hash: {rc.config_hash}",
        f"# warnings: {len(rc.warnings)}",
    ]
    for w in rc.warnings:
        header.append(f"# warning: {w}")
    return "\n".join(header) + "\n" + format_yaml_with_provenance(rc.data, rc.provenance)


def cmd_config_print(args: argparse.Namespace) -> int:
    try:
        rc = load_config(_request_from_args(args))
    except ConfigValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    sys.stdout.write(format_resolved(rc))
    return 0


def cmd_config_validate(args: argparse.Namespace) -> int:
    try:
        rc = load_config(_request_from_args(args))
    except ConfigValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    issues = validate_config(rc.data)
    warnings = [i for i in issues if i.severity == "warning"]
    print(f"OK config_hash={rc.config_hash}")
    for issue in warnings:
        sug = f" (did you mean {issue.suggestion!r}?)" if issue.suggestion else ""
        print(f"WARN {issue.path}: {issue.message}{sug}", file=sys.stderr)
    return 0


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, path))
        return out
    if prefix:
        out[prefix] = obj
    return out


class _MissingSentinel:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<missing>"


_MISSING = _MissingSentinel()


def cmd_config_diff(args: argparse.Namespace) -> int:
    try:
        left = load_config(_request_from_args(args))
        right = load_config(_vs_request_from_args(args))
    except ConfigValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    flat_l = _flatten(left.data)
    flat_r = _flatten(right.data)
    keys = sorted(set(flat_l) | set(flat_r))
    diffs = 0
    for key in keys:
        lv = flat_l.get(key, _MISSING)
        rv = flat_r.get(key, _MISSING)
        if lv != rv:
            diffs += 1
            print(f"{key}:")
            print(f"  - {_yaml_scalar(lv) if lv is not _MISSING else '<missing>'}")
            print(f"  + {_yaml_scalar(rv) if rv is not _MISSING else '<missing>'}")
    print(f"# left_hash={left.config_hash} right_hash={right.config_hash} diffs={diffs}")
    return 0 if diffs == 0 else 1


def dispatch_config(args: argparse.Namespace) -> int:
    handler = getattr(args, "config_handler", None)
    if handler is None:
        print("dynisafety config: missing subcommand (print|diff|validate)", file=sys.stderr)
        return 2
    return int(handler(args))
