"""Generic name→type registry with loud import failures (no silent skip)."""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")

_MISSING_MOD_RE = re.compile(
    r"No module named ['\"]([^'\"]+)['\"]|"
    r"cannot import name ['\"]([^'\"]+)['\"]"
)


class RegistryLoadError(ImportError):
    """Raised when a registry entry cannot be imported."""

    def __init__(
        self,
        *,
        kind: str,
        name: str,
        import_path: str,
        missing_dependencies: list[str],
        cause: BaseException,
    ) -> None:
        self.kind = kind
        self.name = name
        self.import_path = import_path
        self.missing_dependencies = list(missing_dependencies)
        self.__cause__ = cause
        miss = ", ".join(missing_dependencies) if missing_dependencies else "(unknown)"
        super().__init__(
            f"Failed to load {kind} {name!r} from {import_path!r}. "
            f"Missing dependencies: {miss}. "
            f"Original error: {type(cause).__name__}: {cause}"
        )


def _parse_import_path(import_path: str) -> tuple[str, str]:
    """Split ``package.module:Attr`` or ``package.module.Attr`` into (module, attr)."""
    if ":" in import_path:
        module_name, attr = import_path.split(":", 1)
        if not module_name or not attr:
            raise ValueError(f"invalid import_path: {import_path!r}")
        return module_name, attr
    if "." not in import_path:
        raise ValueError(
            f"invalid import_path {import_path!r}; expected 'module.path:Attr' or 'module.path.Attr'"
        )
    module_name, attr = import_path.rsplit(".", 1)
    return module_name, attr


def _missing_dependencies_from(exc: BaseException) -> list[str]:
    names: list[str] = []
    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, ModuleNotFoundError) and getattr(cur, "name", None):
            names.append(str(cur.name))
        else:
            match = _MISSING_MOD_RE.search(str(cur))
            if match:
                names.append(match.group(1) or match.group(2))
        cur = cur.__cause__ or cur.__context__
    # unique, preserve order
    out: list[str] = []
    for n in names:
        if n and n not in out:
            out.append(n)
    return out


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def decorator(cls: type[T]) -> type[T]:
            if name in self._items:
                raise KeyError(f"{self._kind} {name!r} already registered")
            self._items[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        if name not in self._items:
            avail = ", ".join(self.available()) or "(none)"
            raise KeyError(f"unknown {self._kind} {name!r}; available: {avail}")
        return self._items[name]

    def available(self) -> list[str]:
        return sorted(self._items)

    def load_entry(self, name: str, import_path: str) -> type[T]:
        """Import ``import_path``, register under ``name``, return the type.

        Never warns and skips: import failures raise :class:`RegistryLoadError`
        listing missing dependency module names.
        """
        if name in self._items:
            return self._items[name]
        module_name, attr = _parse_import_path(import_path)
        try:
            module = importlib.import_module(module_name)
            obj = module
            for part in attr.split("."):
                obj = getattr(obj, part)
        except Exception as exc:  # noqa: BLE001 — re-raised as RegistryLoadError
            missing = _missing_dependencies_from(exc)
            raise RegistryLoadError(
                kind=self._kind,
                name=name,
                import_path=import_path,
                missing_dependencies=missing,
                cause=exc,
            ) from exc
        if not isinstance(obj, type):
            raise TypeError(
                f"{self._kind} entry {name!r} from {import_path!r} is not a class: {type(obj)!r}"
            )
        self._items[name] = obj  # type: ignore[assignment]
        return obj  # type: ignore[return-value]
