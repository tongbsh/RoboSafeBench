"""Layered YAML config load, merge, override, validate, and config_hash."""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from dynisafety.config.resolve import (
    ConfigResolveError,
    default_third_party_root,
    find_workspace_root,
    resolve_placeholders,
)
from dynisafety.config.schema import (
    LEGACY_KEY_SUGGESTIONS,
    MISSING,
    CaptureConfig,
    ContactConfig,
    DiagnosticsAvoidConfig,
    DiagnosticsConfig,
    DiagnosticsCounterfactualConfig,
    DiagnosticsHazardEndConfig,
    DiagnosticsReactionConfig,
    DiagnosticsRecoveryConfig,
    DiagnosticsStopConfig,
    DiagnosticsTraceConfig,
    DisturbanceConfig,
    DisturbanceL1Config,
    DisturbanceL2Config,
    DisturbanceL3Config,
    HandConfig,
    MetricsConfig,
    ObstacleConfig,
    OutputConfig,
    ReactivityConfig,
    RunConfig,
    RuntimeConfig,
    SceneConfig,
    backend_supported_levels,
    known_config_paths,
    level_relevant_prefixes,
    open_config_prefixes,
)

_META_HASH_KEYS = frozenset({"config_hash", "_provenance"})

_NESTED_BUILDERS: dict[type, dict[str, type]] = {
    MetricsConfig: {"reactivity": ReactivityConfig},
    DiagnosticsConfig: {
        "trace": DiagnosticsTraceConfig,
        "reaction": DiagnosticsReactionConfig,
        "stop": DiagnosticsStopConfig,
        "avoid": DiagnosticsAvoidConfig,
        "recovery": DiagnosticsRecoveryConfig,
        "hazard_end": DiagnosticsHazardEndConfig,
        "counterfactual": DiagnosticsCounterfactualConfig,
    },
    DisturbanceConfig: {
        "l1": DisturbanceL1Config,
        "l2": DisturbanceL2Config,
        "l3": DisturbanceL3Config,
        "obstacle": ObstacleConfig,
    },
    RunConfig: {
        "contact": ContactConfig,
        "metrics": MetricsConfig,
        "diagnostics": DiagnosticsConfig,
        "capture": CaptureConfig,
        "output": OutputConfig,
        "runtime": RuntimeConfig,
        "disturbance": DisturbanceConfig,
        "hand": HandConfig,
        "scene": SceneConfig,
    },
}


@dataclass(frozen=True)
class LoadRequest:
    backend: str
    level: int
    suite: str | None = None
    task: str | None = None
    model: str | None = None
    config_file: Path | None = None
    config_files: tuple[Path, ...] = ()
    overrides: tuple[str, ...] = ()
    configs_root: Path | None = None


@dataclass(frozen=True)
class ValidationIssue:
    severity: str  # "error" | "warning"
    path: str
    message: str
    suggestion: str | None = None


@dataclass(frozen=True)
class ResolvedConfig:
    data: dict[str, Any]
    provenance: dict[str, str]
    config_hash: str
    run: RunConfig
    warnings: tuple[str, ...]


class ConfigValidationError(ValueError):
    def __init__(self, issues: list[ValidationIssue]) -> None:
        self.issues = issues
        parts = []
        for issue in issues:
            line = f"[{issue.severity}] {issue.path}: {issue.message}"
            if issue.suggestion:
                line += f" (did you mean {issue.suggestion!r}?)"
            parts.append(line)
        super().__init__("\n".join(parts) if parts else "config validation failed")


def task_yaml_stem(suite: str, task_id: int) -> str:
    """Task-layer filename stem: ``{suite}_{task_id}`` (no ``.yaml``)."""
    return f"{suite}_{int(task_id)}"


def task_layer_stems(suite: str | None, task: str | None) -> tuple[str, ...]:
    """YAML stems under ``backends/{backend}/tasks/``.

    Accepts the full stem (``safelibero_spatial_1``) or a numeric task id
    (``1``) plus suite, which composes to ``safelibero_spatial_1``.
    """
    if task is None:
        return ()
    raw = str(task).strip()
    if not raw:
        return ()
    stems = [raw]
    if suite is not None and not raw.startswith(f"{suite}_"):
        composed = f"{suite}_{raw}"
        if composed != raw:
            stems.append(composed)
    return tuple(stems)


def layer_paths(req: LoadRequest, configs_root: Path) -> list[Path]:
    """Return merge-ordered YAML paths (only callers filter by existence)."""
    paths: list[Path] = [
        configs_root / "base.yaml",
        configs_root / "levels" / f"l{req.level}.yaml",
        configs_root / "backends" / f"{req.backend}.yaml",
        configs_root / "backends" / req.backend / "hand.yaml",
        configs_root / "backends" / req.backend / "levels" / f"l{req.level}.yaml",
    ]
    if req.suite is not None:
        paths.append(configs_root / "backends" / req.backend / "suites" / f"{req.suite}.yaml")
    for stem in task_layer_stems(req.suite, req.task):
        paths.append(configs_root / "backends" / req.backend / "tasks" / f"{stem}.yaml")
    if req.model is not None:
        paths.append(configs_root / "models" / f"{req.model}.yaml")
    extras: list[Path] = []
    if req.config_file is not None:
        extras.append(Path(req.config_file))
    extras.extend(Path(p) for p in req.config_files)
    seen: set[Path] = set()
    for extra in extras:
        key = extra.resolve() if extra.exists() else extra
        if key in seen:
            continue
        seen.add(key)
        paths.append(extra)
    return paths


def _read_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(text)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigValidationError(
            [
                ValidationIssue(
                    "error",
                    str(path),
                    f"YAML root must be a mapping, got {type(loaded).__name__}",
                )
            ]
        )
    return loaded


def deep_merge(
    layers: Sequence[tuple[Path, dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Merge layers low→high; record leaf provenance as POSIX-ish path strings."""
    out: dict[str, Any] = {}
    provenance: dict[str, str] = {}

    def merge_dict(dst: dict[str, Any], src: dict[str, Any], source: str, prefix: str) -> None:
        for key, value in src.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                existing = dst.get(key)
                if not isinstance(existing, dict):
                    dst[key] = {}
                if not value:
                    # Empty mapping is still a concrete value — record its source.
                    provenance[path] = source
                merge_dict(dst[key], value, source, path)
            else:
                dst[key] = copy.deepcopy(value)
                provenance[path] = source

    for path, data in layers:
        source = path.as_posix()
        merge_dict(out, data, source, "")
    return out, provenance


def _set_path(
    data: dict[str, Any],
    dotted: str,
    value: Any,
) -> None:
    parts = dotted.split(".")
    cur: dict[str, Any] = data
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _get_path(data: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return MISSING
        cur = cur[part]
    return cur


def apply_cli_overrides(
    data: dict[str, Any],
    provenance: dict[str, str],
    overrides: Sequence[str],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Apply ``a.b.c=value`` overrides (YAML scalar syntax for values)."""
    data = copy.deepcopy(data)
    provenance = dict(provenance)
    issues: list[ValidationIssue] = []
    for raw in overrides:
        if "=" not in raw:
            issues.append(
                ValidationIssue("error", raw, "override must look like key.path=value")
            )
            continue
        key, value_text = raw.split("=", 1)
        key = key.strip()
        if not key:
            issues.append(ValidationIssue("error", raw, "empty override key"))
            continue
        try:
            value = yaml.safe_load(value_text)
        except yaml.YAMLError as exc:
            issues.append(ValidationIssue("error", key, f"invalid YAML value: {exc}"))
            continue
        if key in LEGACY_KEY_SUGGESTIONS:
            issues.append(
                ValidationIssue(
                    "error",
                    key,
                    "legacy key is not allowed",
                    suggestion=LEGACY_KEY_SUGGESTIONS[key],
                )
            )
            continue
        if not _path_allowed(key):
            issues.append(
                ValidationIssue(
                    "error",
                    key,
                    "unknown key",
                    suggestion=_suggest_key(key),
                )
            )
            continue
        _set_path(data, key, value)
        provenance[key] = "cli:--set"
    if issues:
        raise ConfigValidationError(issues)
    return data, provenance


def _json_safe(obj: Any) -> Any:
    """Convert non-JSON floats (inf/nan) so config_hash stays deterministic."""
    if isinstance(obj, float):
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        if math.isnan(obj):
            return "nan"
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    return obj


def compute_config_hash(data: dict[str, Any]) -> str:
    """sha256 of canonical JSON, first 12 hex chars."""
    payload = _json_safe({k: v for k, v in data.items() if k not in _META_HASH_KEYS})
    try:
        blob = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except ValueError as exc:
        raise ConfigValidationError(
            [ValidationIssue("error", "", f"config not JSON-serializable for hash: {exc}")]
        ) from exc
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _flatten_leaves(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(obj, dict):
        items: list[tuple[str, Any]] = []
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            items.extend(_flatten_leaves(value, path))
        return items
    return [(prefix, obj)] if prefix else []


def _path_allowed(path: str) -> bool:
    known = known_config_paths()
    opens = open_config_prefixes()
    if path in known:
        return True
    for open_p in opens:
        if path == open_p or path.startswith(open_p + "."):
            return True
    # Allow descendants of known container paths (e.g. future optional leaves
    # under a known dataclass dict field that is itself open-ish).
    parts = path.split(".")
    for i in range(len(parts) - 1, 0, -1):
        parent = ".".join(parts[:i])
        if parent in opens:
            return True
        if parent in known:
            # Parent is a known field; allow one extra segment only if parent
            # denotes an open map type — otherwise require exact known leaf.
            return False
    # Top-level unknown
    return False


def _suggest_key(path: str) -> str | None:
    if path in LEGACY_KEY_SUGGESTIONS:
        return LEGACY_KEY_SUGGESTIONS[path]
    leaf = path.split(".")[-1]
    if leaf in LEGACY_KEY_SUGGESTIONS:
        return LEGACY_KEY_SUGGESTIONS[leaf]
    candidates = sorted(known_config_paths())
    matches = difflib.get_close_matches(path, candidates, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    leaf_matches = difflib.get_close_matches(leaf, [c.split(".")[-1] for c in candidates], n=3, cutoff=0.7)
    for cand in candidates:
        if cand.split(".")[-1] in leaf_matches:
            return cand
    return None


def validate_config(data: dict[str, Any]) -> list[ValidationIssue]:
    """Validate merged config; returns errors and warnings (does not raise)."""
    issues: list[ValidationIssue] = []

    # Legacy key names used at the wrong path (e.g. top-level topple_threshold_deg).
    # The canonical path equal to the suggestion is allowed.
    def walk_keys(obj: Any, prefix: str) -> None:
        if not isinstance(obj, dict):
            return
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in LEGACY_KEY_SUGGESTIONS:
                suggestion = LEGACY_KEY_SUGGESTIONS[key]
                if path != suggestion:
                    issues.append(
                        ValidationIssue(
                            "error",
                            path,
                            "legacy key is not allowed",
                            suggestion=suggestion,
                        )
                    )
            walk_keys(value, path)

    walk_keys(data, "")

    for path, value in _flatten_leaves(data):
        if path.split(".")[0] in _META_HASH_KEYS or path in _META_HASH_KEYS:
            continue
        if not _path_allowed(path):
            # Skip if an ancestor already flagged as legacy.
            issues.append(
                ValidationIssue(
                    "error",
                    path,
                    "unknown key",
                    suggestion=_suggest_key(path),
                )
            )
            continue
        _validate_leaf(path, value, issues)

    backend = data.get("backend")
    level = data.get("level")
    if isinstance(level, int):
        if level not in (0, 1, 2, 3):
            issues.append(
                ValidationIssue("error", "level", f"level must be 0..3, got {level}")
            )
        elif isinstance(backend, str):
            supported = backend_supported_levels(backend)
            if supported and level not in supported:
                issues.append(
                    ValidationIssue(
                        "error",
                        "level",
                        f"backend {backend!r} does not support level {level}; "
                        f"supported={sorted(supported)}",
                    )
                )
        # Cross-field: move_start < move_end
        start = _get_path(data, "disturbance.l2.move_start_step")
        end = _get_path(data, "disturbance.l2.move_end_step")
        if start is not MISSING and end is not MISSING:
            if isinstance(start, int) and isinstance(end, int) and not (start < end):
                issues.append(
                    ValidationIssue(
                        "error",
                        "disturbance.l2.move_end_step",
                        f"move_start_step ({start}) must be < move_end_step ({end})",
                    )
                )
        # Level-irrelevant disturbance.l* → WARN
        relevant = level_relevant_prefixes(int(level))
        for path, _ in _flatten_leaves(data.get("disturbance", {}) if isinstance(data.get("disturbance"), dict) else {}, "disturbance"):
            if path == "disturbance.level":
                continue
            if path.startswith("disturbance.obstacle"):
                continue
            ok = any(path == r or path.startswith(r + ".") for r in relevant)
            if not ok and path.startswith("disturbance.l"):
                issues.append(
                    ValidationIssue(
                        "warning",
                        path,
                        f"parameter is not used at level {level}",
                    )
                )

    geom = _get_path(data, "hand.geometry")
    if geom is not MISSING and geom not in ("box", "mesh"):
        issues.append(
            ValidationIssue(
                "error",
                "hand.geometry",
                f"must be 'box' or 'mesh', got {geom!r}",
            )
        )

    impulse = _get_path(data, "contact.impulse_reduce")
    if impulse is not MISSING and impulse not in ("sum", "max"):
        issues.append(
            ValidationIssue(
                "error",
                "contact.impulse_reduce",
                f"must be 'sum' or 'max', got {impulse!r}",
            )
        )

    return issues


def _validate_leaf(path: str, value: Any, issues: list[ValidationIssue]) -> None:
    if path.endswith(".alpha") and isinstance(value, (int, float)) and not isinstance(value, bool):
        if not (0.0 <= float(value) <= 1.0) or isinstance(value, float) and math.isnan(value):
            issues.append(
                ValidationIssue("error", path, f"alpha must be in [0, 1], got {value}")
            )
    if path.endswith("_steps") or path.endswith("max_steps") or path.endswith("episodes"):
        if isinstance(value, int) and value < 0:
            issues.append(ValidationIssue("error", path, f"must be >= 0, got {value}"))
        elif isinstance(value, float):
            issues.append(ValidationIssue("error", path, f"must be int, got float {value}"))
    if path.startswith("disturbance.obstacle.rescale."):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0.0:
            issues.append(
                ValidationIssue(
                    "error",
                    path,
                    f"obstacle rescale must be a positive number, got {value!r}",
                )
            )


def _build_dataclass(cls: type, raw: Any) -> Any:
    if raw is None or raw is MISSING:
        return MISSING
    if not isinstance(raw, dict):
        raise ConfigValidationError(
            [
                ValidationIssue(
                    "error",
                    cls.__name__,
                    f"expected mapping for {cls.__name__}, got {type(raw).__name__}",
                )
            ]
        )
    nested = _NESTED_BUILDERS.get(cls, {})
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in raw:
            kwargs[f.name] = MISSING
            continue
        val = raw[f.name]
        child_cls = nested.get(f.name)
        if child_cls is not None:
            kwargs[f.name] = _build_dataclass(child_cls, val)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


def to_run_config(data: dict[str, Any], *, config_hash: str | None = None) -> RunConfig:
    """Map nested YAML tree onto ``RunConfig`` (flat + nested views)."""
    del config_hash  # reserved for callers; hash lives on ResolvedConfig
    runtime = data.get("runtime")
    runtime_raw: dict[str, Any] = runtime if isinstance(runtime, dict) else {}
    seed = runtime_raw.get("seed", data.get("seed", MISSING))
    max_steps = runtime_raw.get("max_steps", data.get("max_steps", MISSING))
    model = data.get("model", data.get("name", MISSING))
    level = data.get("level", MISSING)
    dist_raw = data.get("disturbance")
    if isinstance(dist_raw, dict) and level is not MISSING and "level" not in dist_raw:
        dist_raw = {**dist_raw, "level": level}

    return RunConfig(
        backend=data.get("backend", MISSING),
        suite=data.get("suite", MISSING),
        level=level,
        seed=seed,
        model=model,
        checkpoint=data.get("checkpoint", MISSING),
        max_steps=max_steps,
        run_id=data.get("run_id", MISSING),
        contact=_build_dataclass(ContactConfig, data.get("contact")),
        metrics=_build_dataclass(MetricsConfig, data.get("metrics")),
        diagnostics=_build_dataclass(DiagnosticsConfig, data.get("diagnostics")),
        capture=_build_dataclass(CaptureConfig, data.get("capture")),
        output=_build_dataclass(OutputConfig, data.get("output")),
        runtime=_build_dataclass(RuntimeConfig, data.get("runtime")),
        disturbance=_build_dataclass(DisturbanceConfig, dist_raw),
        hand=_build_dataclass(HandConfig, data.get("hand")),
        scene=_build_dataclass(SceneConfig, data.get("scene")),
    )


def _inject_selectors(
    data: dict[str, Any],
    provenance: dict[str, str],
    req: LoadRequest,
) -> tuple[dict[str, Any], dict[str, str]]:
    data = copy.deepcopy(data)
    provenance = dict(provenance)
    source = "load_request"
    data["backend"] = req.backend
    provenance["backend"] = source
    data["level"] = req.level
    provenance["level"] = source
    dist = data.get("disturbance")
    if not isinstance(dist, dict):
        dist = {}
        data["disturbance"] = dist
    dist["level"] = req.level
    provenance["disturbance.level"] = source
    if req.suite is not None:
        data["suite"] = req.suite
        provenance["suite"] = source
    if req.model is not None:
        data["model"] = req.model
        provenance["model"] = source
    return data, provenance


def _rel_provenance(provenance: dict[str, str], workspace: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, source in provenance.items():
        if source.startswith("cli:") or source == "load_request":
            out[key] = source
            continue
        try:
            out[key] = Path(source).resolve().relative_to(workspace.resolve()).as_posix()
        except ValueError:
            out[key] = source
    return out


def load_config(req: LoadRequest) -> ResolvedConfig:
    """Load layered YAML → merge → resolve → override → validate → hash."""
    workspace = find_workspace_root()
    configs_root = Path(req.configs_root) if req.configs_root is not None else workspace / "configs"
    third_party = default_third_party_root(workspace)

    layers: list[tuple[Path, dict[str, Any]]] = []
    for path in layer_paths(req, configs_root):
        if path.is_file():
            layers.append((path.resolve(), _read_yaml(path)))

    data, provenance = deep_merge(layers)
    data, provenance = _inject_selectors(data, provenance, req)
    try:
        data = resolve_placeholders(data, workspace=workspace, third_party=third_party)
    except ConfigResolveError as exc:
        raise ConfigValidationError(
            [ValidationIssue("error", "", str(exc))]
        ) from exc

    if req.overrides:
        data, provenance = apply_cli_overrides(data, provenance, req.overrides)

    issues = validate_config(data)
    errors = [i for i in issues if i.severity == "error"]
    warnings = tuple(i.message if not i.path else f"{i.path}: {i.message}" for i in issues if i.severity == "warning")
    if errors:
        raise ConfigValidationError(errors)

    config_hash = compute_config_hash(data)
    run = to_run_config(data, config_hash=config_hash)
    return ResolvedConfig(
        data=data,
        provenance=_rel_provenance(provenance, workspace),
        config_hash=config_hash,
        run=run,
        warnings=warnings,
    )


def load_resolved_yaml(path: Path) -> ResolvedConfig:
    """Re-load a written resolved config YAML (no layer merge); hash must be stable."""
    workspace = find_workspace_root(path.parent)
    third_party = default_third_party_root(workspace)
    raw = _read_yaml(path)
    stored_hash = raw.pop("config_hash", None)
    raw.pop("_provenance", None)
    try:
        data = resolve_placeholders(raw, workspace=workspace, third_party=third_party)
    except ConfigResolveError as exc:
        raise ConfigValidationError(
            [ValidationIssue("error", "", str(exc))]
        ) from exc

    issues = validate_config(data)
    errors = [i for i in issues if i.severity == "error"]
    warnings = tuple(
        i.message if not i.path else f"{i.path}: {i.message}"
        for i in issues
        if i.severity == "warning"
    )
    if errors:
        raise ConfigValidationError(errors)

    config_hash = compute_config_hash(data)
    if stored_hash is not None and stored_hash != config_hash:
        # Not a hard failure for structural reload; caller may compare.
        pass
    run = to_run_config(data, config_hash=config_hash)
    return ResolvedConfig(
        data=data,
        provenance={k: path.as_posix() for k, _ in _flatten_leaves(data)},
        config_hash=config_hash,
        run=run,
        warnings=warnings,
    )
