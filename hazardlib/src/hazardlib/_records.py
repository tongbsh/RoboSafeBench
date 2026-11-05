"""Internal machinery shared by ``hazardlib.host`` and ``hazardlib.spec``.

Plain annotated dataclasses become *records* by mixing in :class:`Record`:

* ``to_dict()``   -> JSON/YAML-able plain structure (lists, dicts, str, int,
  float, bool, None). Non-finite floats are rejected.
* ``from_dict()`` -> strict inverse. Unknown keys and missing required keys are
  errors; values are type-checked against the dataclass annotations; the
  record's own ``validate()`` runs last. Error messages start with a dotted
  path such as ``HazardSpec.agent.half_extents_m[1]``.
* ``hash()``      -> sha256 of the canonical JSON of ``to_dict()``
  (see :mod:`hazardlib.hashing`).

Supported annotations: ``str, int, float, bool, Any, Optional[X], List[X],
Dict[str, X]`` and other ``Record`` subclasses. Nothing here carries a numeric
default (rule: thresholds live in ``configs/hazard/*.yaml`` only).

Python >= 3.8: annotations are evaluated objects (no ``X | Y``, no
``list[int]``).
"""
import dataclasses
import enum
import functools
import math
import numbers
from typing import (Any, ClassVar, Dict, List, Mapping, Optional, Tuple, Union,
                    get_args, get_origin, get_type_hints)

import numpy as np

UNKNOWN = "unknown"
PLACEHOLDER = "PLACEHOLDER"
MARKERS = (UNKNOWN, PLACEHOLDER)


class SpecError(ValueError):
    """A spec object or config file failed validation."""


def is_unknown(value: Any) -> bool:
    return isinstance(value, str) and value == UNKNOWN


def join_path(path: str, key: Any) -> str:
    if isinstance(key, int) and not isinstance(key, bool):
        return "%s[%d]" % (path, key)
    return "%s.%s" % (path, key) if path else str(key)


def describe_value(value: Any) -> str:
    return "%s %r" % (type(value).__name__, value)


# --------------------------------------------------------------------------
# plain-structure conversion
# --------------------------------------------------------------------------

def to_plain(obj: Any, path: str = "") -> Any:
    """Convert records / numpy / tuples to plain JSON-able Python values."""
    if isinstance(obj, Record):
        return obj.to_dict()
    if isinstance(obj, enum.Enum):
        return to_plain(obj.value, path)
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, numbers.Integral):
        return int(obj)
    if isinstance(obj, numbers.Real):
        number = float(obj)
        if not math.isfinite(number):
            raise SpecError("%s: non-finite number %r is not allowed"
                            % (path or "<root>", number))
        return number
    if isinstance(obj, np.ndarray):
        return to_plain(obj.tolist(), path)
    if isinstance(obj, (list, tuple)):
        return [to_plain(v, join_path(path, i)) for i, v in enumerate(obj)]
    if isinstance(obj, Mapping):
        out = {}
        for key, val in obj.items():
            if not isinstance(key, str):
                raise SpecError("%s: mapping keys must be strings, got %s"
                                % (path or "<root>", describe_value(key)))
            out[key] = to_plain(val, join_path(path, key))
        return out
    raise SpecError("%s: unsupported value type %s"
                    % (path or "<root>", type(obj).__name__))


def find_markers(obj: Any, markers: Tuple[str, ...] = MARKERS,
                 path: str = "") -> List[Tuple[str, str]]:
    """Return ``(dotted_path, marker)`` for every leaf equal to a marker string."""
    found = []  # type: List[Tuple[str, str]]
    if isinstance(obj, str):
        if obj in markers:
            found.append((path or "<root>", obj))
    elif isinstance(obj, Mapping):
        for key, val in obj.items():
            found.extend(find_markers(val, markers, join_path(path, key)))
    elif isinstance(obj, (list, tuple)):
        for i, val in enumerate(obj):
            found.extend(find_markers(val, markers, join_path(path, i)))
    return found


# --------------------------------------------------------------------------
# validation helpers (raise SpecError)
# --------------------------------------------------------------------------

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SpecError(message)


def check_nonempty_str(name: str, value: Any) -> None:
    require(isinstance(value, str) and value.strip() != "",
            "%s must be a non-empty string, got %s" % (name, describe_value(value)))


def check_vec(name: str, value: Any, length: int) -> None:
    require(isinstance(value, (list, tuple)) and len(value) == length,
            "%s must be a list of %d numbers, got %s" % (name, length, describe_value(value)))
    for i, item in enumerate(value):
        require(isinstance(item, numbers.Real) and not isinstance(item, bool)
                and math.isfinite(item),
                "%s[%d] must be a finite number, got %s" % (name, i, describe_value(item)))


def check_positive(name: str, value: Any) -> None:
    require(isinstance(value, numbers.Real) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0,
            "%s must be a finite number > 0, got %s" % (name, describe_value(value)))


def check_finite(name: str, value: Any) -> None:
    require(isinstance(value, numbers.Real) and not isinstance(value, bool)
            and math.isfinite(value),
            "%s must be a finite number, got %s" % (name, describe_value(value)))


# --------------------------------------------------------------------------
# type-driven coercion
# --------------------------------------------------------------------------

def choice(options: Tuple[str, ...], **kwargs: Any) -> Any:
    """``dataclasses.field`` whose value must be one of ``options``."""
    return dataclasses.field(metadata={"choices": tuple(options)}, **kwargs)


@functools.lru_cache(maxsize=None)
def _hints(cls: type) -> Dict[str, Any]:
    return get_type_hints(cls)


def coerce(tp: Any, value: Any, path: str, allow_unknown: bool) -> Any:
    if allow_unknown and is_unknown(value):
        return UNKNOWN
    if tp is Any:
        return to_plain(value, path)
    origin = get_origin(tp)
    if origin is Union:
        members = get_args(tp)
        inner = [a for a in members if a is not type(None)]
        if value is None:
            if len(inner) == len(members):
                raise SpecError("%s: None is not allowed here" % path)
            return None
        if len(inner) != 1:
            raise TypeError("unsupported Union annotation %r" % (tp,))
        return coerce(inner[0], value, path, allow_unknown)
    if origin is list:
        if not isinstance(value, (list, tuple, np.ndarray)):
            raise SpecError("%s: expected a list, got %s" % (path, describe_value(value)))
        (item_tp,) = get_args(tp)
        items = value.tolist() if isinstance(value, np.ndarray) else value
        return [coerce(item_tp, v, join_path(path, i), allow_unknown)
                for i, v in enumerate(items)]
    if origin is dict:
        if not isinstance(value, Mapping):
            raise SpecError("%s: expected a mapping, got %s" % (path, describe_value(value)))
        _, val_tp = get_args(tp)
        out = {}
        for key, val in value.items():
            if not isinstance(key, str):
                raise SpecError("%s: mapping keys must be strings, got %s"
                                % (path, describe_value(key)))
            out[key] = coerce(val_tp, val, join_path(path, key), allow_unknown)
        return out
    if isinstance(tp, type) and issubclass(tp, Record):
        return tp.from_dict(value, path=path)
    if tp is bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        raise SpecError("%s: expected a bool, got %s" % (path, describe_value(value)))
    if tp is int:
        if isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_)):
            return int(value)
        raise SpecError("%s: expected an int, got %s" % (path, describe_value(value)))
    if tp is float:
        if isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_)):
            number = float(value)
            if not math.isfinite(number):
                raise SpecError("%s: non-finite number %r is not allowed" % (path, number))
            return number
        hint = ""
        if isinstance(value, str):
            hint = " (YAML 1.1 needs a dot in exponent notation, e.g. 1.0e-3)"
        raise SpecError("%s: expected a number, got %s%s" % (path, describe_value(value), hint))
    if tp is str:
        if isinstance(value, str):
            return value
        hint = ""
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            hint = " (quote versions/identifiers in YAML, e.g. \"3.8\")"
        raise SpecError("%s: expected a string, got %s%s" % (path, describe_value(value), hint))
    raise TypeError("unsupported annotation %r at %s" % (tp, path))


class Record:
    """Mixin for dataclasses; see module docstring."""

    #: When True, the string ``"unknown"`` is accepted in place of any value
    #: (used by BenchmarkDescriptor; ``strict`` loading then rejects it).
    ALLOW_UNKNOWN: ClassVar[bool] = False

    def validate(self) -> None:
        """Cross-field checks. Raise :class:`SpecError`. Default: nothing."""

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: to_plain(getattr(self, f.name), f.name)
                for f in dataclasses.fields(self)}

    @classmethod
    def from_dict(cls, data: Any, path: Optional[str] = None) -> Any:
        path = cls.__name__ if path is None else path
        if not isinstance(data, Mapping):
            raise SpecError("%s: expected a mapping, got %s" % (path, describe_value(data)))
        flds = {f.name: f for f in dataclasses.fields(cls)}
        extra = sorted(str(k) for k in data if k not in flds)
        if extra:
            raise SpecError("%s: unexpected key(s) %s (allowed: %s)"
                            % (path, ", ".join(extra), ", ".join(flds)))
        required = [n for n, f in flds.items()
                    if f.default is dataclasses.MISSING
                    and f.default_factory is dataclasses.MISSING]  # type: ignore
        missing = [n for n in required if n not in data]
        if missing:
            raise SpecError("%s: missing required key(s) %s" % (path, ", ".join(missing)))
        hints = _hints(cls)
        kwargs = {}
        for name, fld in flds.items():
            if name not in data:
                continue
            fpath = join_path(path, name)
            value = coerce(hints[name], data[name], fpath, cls.ALLOW_UNKNOWN)
            options = fld.metadata.get("choices")
            if (options is not None and value is not None
                    and not (cls.ALLOW_UNKNOWN and is_unknown(value))
                    and value not in options):
                raise SpecError("%s: %r is not one of %s" % (fpath, value, list(options)))
            kwargs[name] = value
        obj = cls(**kwargs)
        try:
            obj.validate()
        except SpecError as exc:
            raise SpecError("%s: %s" % (path, exc)) from None
        return obj

    def hash(self) -> str:
        from .hashing import sha256_hex
        return sha256_hex(self.to_dict())
