"""``HazardEventLog``: one record per episode (00 §6, rule R14), written as JSON Lines.

Fields are the 00 §6 minimum set plus ``schema_version``, ``attribution_ambiguous`` (needed so that the
D9 d "ambiguous" count can be aggregated from logs) and ``first_scene_contact_t_s`` (D10 a: the first contact
with a ``scene`` partner, e.g. landing; official contacts are robot side only). Every field except
``schema_version`` and ``instance_id``
is optional: **a quantity that was not measured is ``None`` (JSON ``null``), never 0.**

Validation: finite numbers only (no NaN / inf, also at write time), non-negative where physical (durations,
speeds, impulses, forces, separations, TTC, counts), strictly positive mass, real bools, enum members. Enum
fields accept the member or its string value and are written as strings. Reading rejects unknown keys and
unsupported schema versions; keys that are absent read back as ``None``.
"""
import dataclasses
import json
import numbers
import os
from typing import Any, Dict, List, Optional, Union

from ..host import PartnerClass
from .contact_graph import ConstraintState
from .invalid import InvalidClass
from .proximity import SeparationKind

SCHEMA_VERSION = "1"

_TIME_FIELDS = ("onset_t_s", "released_t_s", "first_official_contact_t_s", "first_scene_contact_t_s")
_NON_NEGATIVE_FIELDS = ("v_rel_normal_mps", "impulse_analytic_ns", "engine_peak_force_n",
                        "contact_duration_s", "min_separation_m", "ttc_min_s")
_BOOL_FIELDS = ("exposure_low", "toppled", "attribution_ambiguous")
_ENUM_FIELDS = {"partner_class": PartnerClass, "constraint_state": ConstraintState,
                "separation_kind": SeparationKind, "invalid_class": InvalidClass}


def _number(name: str, value: Any) -> float:
    if isinstance(value, (bool, str, bytes)) or not isinstance(value, numbers.Real):
        raise ValueError("%s must be a number or None, got %r" % (name, value))
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError("%s must be finite, got %r" % (name, value))
    return number


@dataclasses.dataclass(eq=False)
class HazardEventLog:
    schema_version: str
    instance_id: str
    onset_t_s: Optional[float] = None
    released_t_s: Optional[float] = None
    first_official_contact_t_s: Optional[float] = None
    first_scene_contact_t_s: Optional[float] = None
    partner_class: Optional[PartnerClass] = None
    v_rel_normal_mps: Optional[float] = None
    hazard_mass_kg: Optional[float] = None
    impulse_analytic_ns: Optional[float] = None
    engine_peak_force_n: Optional[float] = None
    force_source: Optional[str] = None
    contact_duration_s: Optional[float] = None
    constraint_state: Optional[ConstraintState] = None
    min_separation_m: Optional[float] = None
    separation_kind: Optional[SeparationKind] = None
    ttc_min_s: Optional[float] = None
    exposure_low: Optional[bool] = None
    toppled: Optional[bool] = None
    kinematic_overlap_count: Optional[int] = None
    invalid_class: Optional[InvalidClass] = None
    attribution_ambiguous: Optional[bool] = None

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str) or not self.schema_version:
            raise ValueError("schema_version must be a non-empty string, got %r" % (self.schema_version,))
        if not isinstance(self.instance_id, str) or not self.instance_id:
            raise ValueError("instance_id must be a non-empty string, got %r" % (self.instance_id,))
        for name in _TIME_FIELDS:
            self._set_number(name, minimum=None, strictly=False)
        for name in _NON_NEGATIVE_FIELDS:
            self._set_number(name, minimum=0.0, strictly=False)
        self._set_number("hazard_mass_kg", minimum=0.0, strictly=True)
        for name in _BOOL_FIELDS:
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ValueError("%s must be a bool or None, got %r" % (name, value))
        for name, enum_type in _ENUM_FIELDS.items():
            value = getattr(self, name)
            if value is not None:
                try:
                    setattr(self, name, enum_type(value))
                except ValueError:
                    raise ValueError("%s=%r is not one of %s"
                                     % (name, value, [m.value for m in enum_type])) from None
        if self.force_source is not None and (not isinstance(self.force_source, str) or not self.force_source):
            raise ValueError("force_source must be a non-empty string or None, got %r" % (self.force_source,))
        count = self.kinematic_overlap_count
        if count is not None:
            if isinstance(count, bool) or not isinstance(count, numbers.Integral) or count < 0:
                raise ValueError("kinematic_overlap_count must be an int >= 0 or None, got %r" % (count,))
            self.kinematic_overlap_count = int(count)

    def _set_number(self, name: str, minimum: Optional[float], strictly: bool) -> None:
        value = getattr(self, name)
        if value is None:
            return
        number = _number(name, value)
        if minimum is not None and (number <= minimum if strictly else number < minimum):
            raise ValueError("%s must be %s %r, got %r" % (name, ">" if strictly else ">=", minimum, number))
        setattr(self, name, number)

    @classmethod
    def new(cls, instance_id: str, **fields: Any) -> "HazardEventLog":
        """A log with the current :data:`SCHEMA_VERSION`; unspecified fields are ``None``."""
        return cls(schema_version=SCHEMA_VERSION, instance_id=instance_id, **fields)

    def to_dict(self) -> Dict[str, Any]:
        out = {}  # type: Dict[str, Any]
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            out[field.name] = value.value if hasattr(value, "value") and field.name in _ENUM_FIELDS else value
        return out

    @classmethod
    def from_dict(cls, data: Any) -> "HazardEventLog":
        if not isinstance(data, dict):
            raise ValueError("a HazardEventLog record must be a JSON object, got %r" % (type(data).__name__,))
        names = [f.name for f in dataclasses.fields(cls)]
        extra = sorted(str(k) for k in data if k not in names)
        if extra:
            raise ValueError("unexpected key(s) %s (allowed: %s)" % (", ".join(extra), ", ".join(names)))
        for required in ("schema_version", "instance_id"):
            if required not in data:
                raise ValueError("missing required key %s" % required)
        if data["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported schema_version %r (this reader supports %r)"
                             % (data["schema_version"], SCHEMA_VERSION))
        return cls(**data)

    def to_json_line(self) -> str:
        try:
            return json.dumps(self.to_dict(), allow_nan=False, separators=(",", ":"))
        except ValueError as exc:
            raise ValueError("cannot serialise record %r: %s" % (self.instance_id, exc)) from None


_Path = Union[str, "os.PathLike[str]"]


def write_jsonl(path: _Path, logs: List[HazardEventLog]) -> None:
    """Write ``logs`` to ``path`` (replacing it), one JSON object per line."""
    lines = [log.to_json_line() for log in logs]
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line + "\n")


def append_jsonl(path: _Path, log: HazardEventLog) -> None:
    """Append one record (one episode) to ``path``."""
    line = log.to_json_line()
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


def read_jsonl(path: _Path) -> List[HazardEventLog]:
    """Read every record of a JSONL file; blank lines are skipped, errors name the line number."""
    out = []  # type: List[HazardEventLog]
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                out.append(HazardEventLog.from_dict(json.loads(line)))
            except ValueError as exc:
                raise ValueError("%s line %d: %s" % (path, number, exc)) from None
    return out
