"""Path / env placeholder resolution for config trees."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_MAX_PLACEHOLDER_DEPTH = 16


class ConfigResolveError(ValueError):
    """Raised when a placeholder cannot be resolved."""


def find_workspace_root(start: Path | None = None) -> Path:
    """Walk upward for a repo root containing ``pyproject.toml`` and ``src/dynisafety``.

    If ``start`` is outside the repo (e.g. a temp resolved_config.yaml), fall back to cwd.
    """
    roots: list[Path] = []
    if start is not None:
        roots.append(Path(start).resolve())
    cwd = Path.cwd().resolve()
    if cwd not in roots:
        roots.append(cwd)

    tried: list[str] = []
    for cur in roots:
        tried.append(str(cur))
        for candidate in (cur, *cur.parents):
            if (candidate / "pyproject.toml").is_file() and (
                candidate / "src" / "dynisafety"
            ).is_dir():
                return candidate
    raise ConfigResolveError(
        "Could not find dynisafety workspace root from "
        + " | ".join(tried)
        + " (need pyproject.toml and src/dynisafety/)"
    )


def default_third_party_root(workspace: Path) -> Path:
    """Convention path ``<workspace>/third_party`` (need not exist yet)."""
    return workspace / "third_party"


def _next_placeholder(value: str, start: int) -> tuple[int, int, str] | None:
    """Locate the next ``${...}`` with **balanced** braces.

    Returns ``(open_idx, close_idx_exclusive, inner_expr)`` or ``None``.
    Brace matching (rather than a regex) is what makes nested defaults such as
    ``${env:VAR|${workspace}/sub}`` work.
    """
    open_idx = value.find("${", start)
    if open_idx == -1:
        return None
    depth = 0
    for idx in range(open_idx + 1, len(value)):
        char = value[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return open_idx, idx + 1, value[open_idx + 2 : idx]
    raise ConfigResolveError(f"Unbalanced '${{' in {value!r}")


def _resolve_string(
    value: str,
    *,
    workspace: Path,
    third_party: Path,
    env: Mapping[str, str],
    _depth: int = 0,
) -> str:
    """Resolve ``${workspace}`` / ``${third_party}`` / ``${env:VAR[|default]}``.

    ``${env:VAR|default}`` uses the environment value when set, otherwise the
    default — which may itself contain placeholders.
    """
    if "${" not in value:
        return value
    if _depth > _MAX_PLACEHOLDER_DEPTH:
        raise ConfigResolveError(
            f"Placeholder nesting exceeded {_MAX_PLACEHOLDER_DEPTH} levels in {value!r}"
        )

    out: list[str] = []
    pos = 0
    while True:
        found = _next_placeholder(value, pos)
        if found is None:
            out.append(value[pos:])
            break
        open_idx, close_idx, inner = found
        out.append(value[pos:open_idx])
        out.append(
            _resolve_expr(
                inner.strip(),
                workspace=workspace,
                third_party=third_party,
                env=env,
                depth=_depth,
            )
        )
        pos = close_idx
    return "".join(out)


def _resolve_expr(
    expr: str,
    *,
    workspace: Path,
    third_party: Path,
    env: Mapping[str, str],
    depth: int,
) -> str:
    if expr == "workspace":
        return str(workspace)
    if expr == "third_party":
        return str(third_party)
    if expr.startswith("env:"):
        rest = expr[4:]
        var, sep, default = rest.partition("|")
        var = var.strip()
        if not var:
            raise ConfigResolveError("Empty ${env:} placeholder")
        if var in env:
            return env[var]
        if sep:
            return _resolve_string(
                default,
                workspace=workspace,
                third_party=third_party,
                env=env,
                _depth=depth + 1,
            )
        raise ConfigResolveError(
            f"Environment variable {var!r} is not set (${{env:{var}}}) and no "
            "default was given; use ${env:VAR|default}"
        )
    raise ConfigResolveError(
        f"Unknown placeholder ${{{expr}}}; allowed: ${{workspace}}, "
        "${third_party}, ${env:VAR}, ${env:VAR|default}"
    )


def resolve_placeholders(
    obj: Any,
    *,
    workspace: Path,
    third_party: Path,
    env: Mapping[str, str] | None = None,
) -> Any:
    """Recursively resolve ``${workspace}`` / ``${third_party}`` / ``${env:VAR}`` in strings."""
    env_map: Mapping[str, str] = os.environ if env is None else env
    if isinstance(obj, str):
        return _resolve_string(obj, workspace=workspace, third_party=third_party, env=env_map)
    if isinstance(obj, list):
        return [
            resolve_placeholders(v, workspace=workspace, third_party=third_party, env=env_map)
            for v in obj
        ]
    if isinstance(obj, tuple):
        return tuple(
            resolve_placeholders(v, workspace=workspace, third_party=third_party, env=env_map)
            for v in obj
        )
    if isinstance(obj, dict):
        return {
            k: resolve_placeholders(v, workspace=workspace, third_party=third_party, env=env_map)
            for k, v in obj.items()
        }
    return obj
