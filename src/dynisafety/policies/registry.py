"""Policy model registry data plane (M4).

``POLICY_REGISTRY`` is owned by ``cli/eval.py`` (S2 D5=B). This module injects
entries via :func:`populate_policies` and never imports ``dynisafety.cli``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dynisafety.core.registry import Registry, RegistryLoadError
from dynisafety.policies.base import ModelConfig, Policy, model_config_from_resolved

# New-stack exclusion (S2 D9=A).
_EXCLUDED_MODEL_IDS: frozenset[str] = frozenset({"memoryvla"})


@dataclass(frozen=True)
class ModelStatus:
    name: str
    state: str  # ready | missing_dependency | missing_checkpoint | load_error
    detail: str
    backends: tuple[str, ...]
    transport: str
    adapter: str


@dataclass
class _ModelDecl:
    name: str
    yaml_path: Path
    raw: dict[str, Any]
    adapter: str = ""
    transport: str = ""
    backends: tuple[str, ...] = ()
    checkpoint: str = ""
    parse_error: str | None = None


_DECLS: dict[str, _ModelDecl] = {}


def _find_configs_root(explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        return explicit
    cwd = Path.cwd().resolve()
    for parent in (cwd, *cwd.parents):
        candidate = parent / "configs"
        if (candidate / "models").is_dir():
            return candidate
    return None


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(text)
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{path}: YAML root must be a mapping, got {type(loaded).__name__}")
    return dict(loaded)


def _checkpoint_string(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, Mapping) and isinstance(raw.get("default"), str):
        return str(raw["default"])
    return ""


def _looks_like_local_path(checkpoint: str) -> bool:
    if not checkpoint or checkpoint.startswith("${"):
        return True
    path = Path(checkpoint).expanduser()
    if path.is_absolute():
        return True
    if "/" in checkpoint and not checkpoint.startswith(".") and "://" not in checkpoint:
        return Path(checkpoint).exists()
    return True


def _checkpoint_missing(checkpoint: str) -> bool:
    if not checkpoint or checkpoint.startswith("${"):
        return True
    if not _looks_like_local_path(checkpoint):
        return False
    return not Path(checkpoint).expanduser().exists()


def load_model_decls(models_dir: Path) -> dict[str, _ModelDecl]:
    """Scan ``models_dir/*.yaml`` into declarations (no adapter import)."""
    if not models_dir.is_dir():
        raise FileNotFoundError(f"models dir not found: {models_dir}")
    decls: dict[str, _ModelDecl] = {}
    for path in sorted(models_dir.glob("*.yaml")):
        name = path.stem
        if name in _EXCLUDED_MODEL_IDS:
            continue
        try:
            raw = _load_yaml_mapping(path)
        except Exception as exc:  # noqa: BLE001 — recorded as load_error in available()
            decls[name] = _ModelDecl(
                name=name,
                yaml_path=path,
                raw={},
                parse_error=f"yaml_read_error: {type(exc).__name__}: {exc}",
            )
            continue
        if "name" not in raw:
            raw = {**raw, "name": name}
        adapter = raw.get("adapter") if isinstance(raw.get("adapter"), str) else ""
        transport = raw.get("transport") if isinstance(raw.get("transport"), str) else ""
        backends_raw = raw.get("backends")
        backends: tuple[str, ...] = ()
        if isinstance(backends_raw, (list, tuple)):
            backends = tuple(str(b) for b in backends_raw if isinstance(b, str))
        decls[name] = _ModelDecl(
            name=name,
            yaml_path=path,
            raw=raw,
            adapter=adapter,
            transport=transport,
            backends=backends,
            checkpoint=_checkpoint_string(raw.get("checkpoint")),
        )
    return decls


def refresh_decls(*, configs_root: Path | None = None) -> dict[str, _ModelDecl]:
    """Reload YAML declarations into the module catalog."""
    global _DECLS
    root = _find_configs_root(configs_root)
    if root is None:
        _DECLS = {}
        return _DECLS
    _DECLS = load_model_decls(root / "models")
    return _DECLS


def register_model(registry: Registry[Any], name: str, adapter_cls: type) -> None:
    """Register *adapter_cls* under *name*. Loud on duplicate. No warn-skip."""
    if name in _EXCLUDED_MODEL_IDS:
        raise ValueError(f"model {name!r} is excluded from the new-stack registry (D9=A)")
    if not isinstance(adapter_cls, type):
        raise TypeError(f"adapter_cls must be a class, got {type(adapter_cls)!r}")
    registry.register(name)(adapter_cls)


def populate_policies(
    registry: Registry[Any],
    *,
    configs_root: Path | None = None,
    models_dir: Path | None = None,
) -> None:
    """Inject YAML-declared adapters into *registry* (eval-owned singleton).

    Import failures are **not** warned-and-skipped: they remain visible via
    :func:`available` as ``missing_dependency`` / ``load_error``. Successful
    imports are registered with :meth:`Registry.load_entry`.
    """
    global _DECLS
    if models_dir is not None:
        _DECLS = load_model_decls(models_dir)
    else:
        refresh_decls(configs_root=configs_root)

    for name, decl in _DECLS.items():
        if decl.parse_error or not decl.adapter:
            continue
        if name in registry.available():
            continue
        try:
            registry.load_entry(name, decl.adapter)
        except RegistryLoadError:
            continue


def create_policy(cfg: ModelConfig, registry: Registry[Any]) -> Policy:
    """Instantiate the registered adapter for ``cfg.name``."""
    cls = registry.get(cfg.name)
    return cls.from_config(cfg)  # type: ignore[no-any-return, attr-defined]


def list_model_ids(*, configs_root: Path | None = None) -> list[str]:
    """Sorted declared model ids (from YAML catalog; excludes memoryvla)."""
    if not _DECLS:
        refresh_decls(configs_root=configs_root)
    return sorted(_DECLS)


def _probe_adapter(adapter: str) -> tuple[str | None, str]:
    """Try importing adapter path. Returns ``(state_or_None_if_ok, detail)``."""
    probe: Registry[Any] = Registry("policy-probe")
    try:
        probe.load_entry("__probe__", adapter)
    except RegistryLoadError as exc:
        if exc.missing_dependencies:
            return (
                "missing_dependency",
                f"missing dependency: {', '.join(exc.missing_dependencies)} "
                f"({type(exc.__cause__).__name__ if exc.__cause__ else 'Error'}: "
                f"{exc.__cause__ or exc})",
            )
        return ("load_error", str(exc))
    except Exception as exc:  # noqa: BLE001
        return ("load_error", f"{type(exc).__name__}: {exc}")
    return (None, "")


def available(
    *,
    registry: Registry[Any] | None = None,
    configs_root: Path | None = None,
) -> list[ModelStatus]:
    """Report every declared model's readiness. Never silently omits a stem.

    Per S2 D6=A: failures appear as status rows; callers may still exit 0 when
    listing.
    """
    if not _DECLS or configs_root is not None:
        refresh_decls(configs_root=configs_root)

    statuses: list[ModelStatus] = []
    for name in sorted(_DECLS):
        decl = _DECLS[name]
        adapter = decl.adapter or ""
        transport = decl.transport or ""
        backends = decl.backends

        if decl.parse_error:
            statuses.append(
                ModelStatus(
                    name=name,
                    state="load_error",
                    detail=decl.parse_error,
                    backends=backends,
                    transport=transport,
                    adapter=adapter,
                )
            )
            continue

        cfg: ModelConfig | None = None
        parse_detail = ""
        try:
            cfg = model_config_from_resolved(decl.raw)
            adapter = cfg.adapter
            transport = cfg.transport
            backends = cfg.backends
        except Exception as exc:  # noqa: BLE001 — incomplete placeholder YAMLs
            parse_detail = f"config_error: {type(exc).__name__}: {exc}"

        if not adapter:
            statuses.append(
                ModelStatus(
                    name=name,
                    state="load_error",
                    detail=parse_detail or "missing adapter: field in model YAML",
                    backends=backends,
                    transport=transport,
                    adapter=adapter,
                )
            )
            continue

        load_state: str | None = None
        load_detail = ""
        registered = registry is not None and name in registry.available()
        if not registered:
            load_state, load_detail = _probe_adapter(adapter)

        checkpoint = cfg.checkpoint if cfg is not None else decl.checkpoint
        if load_state is None and _checkpoint_missing(checkpoint):
            statuses.append(
                ModelStatus(
                    name=name,
                    state="missing_checkpoint",
                    detail=f"checkpoint not found: {checkpoint!r}",
                    backends=backends,
                    transport=transport,
                    adapter=adapter,
                )
            )
            continue

        if load_state is not None:
            detail = load_detail
            if parse_detail:
                detail = f"{parse_detail}; {detail}"
            statuses.append(
                ModelStatus(
                    name=name,
                    state=load_state,
                    detail=detail,
                    backends=backends,
                    transport=transport,
                    adapter=adapter,
                )
            )
            continue

        if parse_detail:
            statuses.append(
                ModelStatus(
                    name=name,
                    state="load_error",
                    detail=parse_detail,
                    backends=backends,
                    transport=transport,
                    adapter=adapter,
                )
            )
            continue

        statuses.append(
            ModelStatus(
                name=name,
                state="ready",
                detail="",
                backends=backends,
                transport=transport,
                adapter=adapter,
            )
        )
    return statuses
