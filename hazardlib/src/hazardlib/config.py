"""YAML loading for ``configs/hazard/``.

* Duplicate mapping keys are an error (PyYAML would silently keep the last).
* ``strict=True`` rejects any leaf equal to ``"PLACEHOLDER"`` (a value nobody
  has decided yet) or ``"unknown"`` (a benchmark fact nobody has probed yet),
  naming every offending path. ``strict=False`` loads such files unchanged.
* Gotchas of YAML 1.1 as implemented by PyYAML: quote versions and commit ids
  (``"3.8"``, ``"797142c"``), write exponents with a dot (``1.0e-3``, not
  ``1e-3``), and avoid bare ``yes/no/on/off``.

Numeric defaults never live in code; this module only locates and reads files.
"""
import os
from pathlib import Path
from typing import Any, List, Optional, Union

import yaml

from ._records import MARKERS, UNKNOWN, SpecError, find_markers
from .spec.benchmark_descriptor import DECLARATIVE_UNKNOWN_FIELDS, BenchmarkDescriptor

CONFIG_DIR_ENV = "HAZARDLIB_CONFIG_DIR"
_SHOW_AT_MOST = 20


class ConfigError(SpecError):
    """A config file is missing, malformed, or (strict) still unresolved."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: yaml.SafeLoader, node: yaml.Node, deep: bool = False) -> Any:
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in seen
        except TypeError:
            continue  # unhashable key: let SafeLoader raise its own error below
        if duplicate:
            raise yaml.constructor.ConstructorError(
                None, None, "duplicate mapping key %r" % (key,), key_node.start_mark)
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def default_config_dir() -> Path:
    """``$HAZARDLIB_CONFIG_DIR`` if set, else the first ``configs/hazard`` found
    walking up from this file (editable install) and then from the cwd."""
    env = os.environ.get(CONFIG_DIR_ENV)
    if env:
        path = Path(env)
        if not path.is_dir():
            raise ConfigError("%s=%s is not a directory" % (CONFIG_DIR_ENV, env))
        return path
    searched = []  # type: List[str]
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for base in [start] + list(start.parents):
            candidate = base / "configs" / "hazard"
            searched.append(str(candidate))
            if candidate.is_dir():
                return candidate
    raise ConfigError("configs/hazard not found (set %s). Looked in: %s"
                      % (CONFIG_DIR_ENV, ", ".join(searched)))


def _check_strict(data: Any, where: str, declarative: Any = ()) -> None:
    """``declarative``: dotted paths whose ``unknown`` is a declaration, not an open probe (BD, D15 a)."""
    found = [(p, m) for p, m in find_markers(data, MARKERS) if not (m == UNKNOWN and p in declarative)]
    if not found:
        return
    shown = ["%s=%s" % (p, m) for p, m in found[:_SHOW_AT_MOST]]
    more = len(found) - len(shown)
    raise ConfigError("%s: %d unresolved value(s) with strict=True: %s%s"
                      % (where, len(found), "; ".join(shown),
                         " ... and %d more" % more if more > 0 else ""))


def load_yaml(path: Union[str, "os.PathLike[str]"], strict: bool = False) -> Any:
    """Read one YAML file. See the module docstring for ``strict``."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError("cannot read %s: %s" % (path, exc)) from None
    try:
        data = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ConfigError("%s: invalid YAML: %s" % (path, exc)) from None
    if strict:
        _check_strict(data, str(path))
    return data


def load_config(name: str, strict: bool = False, config_dir: Optional[Union[str, Path]] = None) -> Any:
    """Load ``<config_dir>/<name>.yaml`` (e.g. ``"families"``, ``"gates"``)."""
    base = Path(config_dir) if config_dir is not None else default_config_dir()
    return load_yaml(base / (name + ".yaml"), strict=strict)


def load_bd(bench: Union[str, "os.PathLike[str]"], strict: bool = False,
            config_dir: Optional[Union[str, Path]] = None) -> BenchmarkDescriptor:
    """Load a BenchmarkDescriptor: a benchmark name (``"calvin"`` ->
    ``<config_dir>/bd/calvin.yaml``) or an explicit ``.yaml`` path."""
    text = str(bench)
    if text.endswith((".yaml", ".yml")) or os.sep in text:
        path = Path(text)
    else:
        base = Path(config_dir) if config_dir is not None else default_config_dir()
        path = base / "bd" / (text + ".yaml")
    data = load_yaml(path, strict=False)
    if strict:
        _check_strict(data, str(path), DECLARATIVE_UNKNOWN_FIELDS)
    try:
        return BenchmarkDescriptor.from_dict(data, path=path.name, strict=strict)
    except SpecError as exc:
        raise ConfigError("%s: %s" % (path, exc)) from None
