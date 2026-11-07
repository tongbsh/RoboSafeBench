"""Contact events and partner classification (00 §6, rule R8).

``PartnerClass`` and ``ContactObs`` are the T01 types from :mod:`hazardlib.host`; they are imported, not
redefined. Contact normals follow the project-wide convention (D9 a): the normal points **from the partner
toward the hazard**, i.e. the direction in which the partner pushes the hazard.
"""
import dataclasses
import enum
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from .._numeric import finite_float, non_negative_float, vec3
from ..host import ContactObs, PartnerClass

#: ``robot_contact_bodies.include`` may carry this category token (01 §2 example). It is not a body-name
#: pattern: the grasped objects are identified by name through ``held_object_names`` instead.
HELD_OBJECT_TOKEN = "held_object"

#: Match kinds of a ``robot_contact_bodies`` entry (D10 f), written ``<kind>:<text>``.
MATCH_KINDS = ("exact", "prefix", "substr")

_MATCHERS = {
    "exact": lambda name, text: name == text,
    "prefix": lambda name, text: name.startswith(text),
    "substr": lambda name, text: text in name,
}


class EventKind(str, enum.Enum):
    """``dynamic``: any contact after the hazard became dynamic, scene partners included; an *official
    contact* (D10 a) is a ``dynamic`` event with a robot-side partner (D12 f). ``kinematic_overlap``: an
    overlap of a kinematically driven hazard with the robot side, which is never an official contact (R8)."""
    DYNAMIC = "dynamic"
    KINEMATIC_OVERLAP = "kinematic_overlap"


_ROBOT_SIDE = (PartnerClass.ROBOT_LINK, PartnerClass.HELD_OBJECT)


@dataclasses.dataclass(eq=False)
class ContactEvent:
    """One contact event of a hazard.

    ``normal`` (partner -> hazard) and ``v_normal_mps`` (closing normal speed at the contact) are ``None``
    when unknown; ``impulse_ns`` is the engine's impulse if it reports one. A ``kinematic_overlap`` must be
    with the robot side (``robot_link`` or ``held_object``): R8 records only those.
    """
    kind: EventKind
    time_s: float
    partner_name: str
    partner_class: PartnerClass
    position: Any
    normal: Optional[Any] = None
    v_normal_mps: Optional[float] = None
    impulse_ns: Optional[float] = None

    def __post_init__(self) -> None:
        try:
            self.kind = EventKind(self.kind)
        except ValueError:
            raise ValueError("kind %r is not one of %s" % (self.kind, [k.value for k in EventKind])) from None
        try:
            self.partner_class = PartnerClass(self.partner_class)
        except ValueError:
            raise ValueError("partner_class %r is not one of %s"
                             % (self.partner_class, [c.value for c in PartnerClass])) from None
        self.time_s = finite_float("time_s", self.time_s)
        if not isinstance(self.partner_name, str) or not self.partner_name:
            raise ValueError("partner_name must be a non-empty string, got %r" % (self.partner_name,))
        self.position = vec3("position", self.position)
        if self.normal is not None:
            self.normal = vec3("normal", self.normal)
        if self.v_normal_mps is not None:
            self.v_normal_mps = non_negative_float("v_normal_mps", self.v_normal_mps)
        if self.impulse_ns is not None:
            self.impulse_ns = non_negative_float("impulse_ns", self.impulse_ns)
        if self.kind is EventKind.KINEMATIC_OVERLAP and self.partner_class not in _ROBOT_SIDE:
            raise ValueError("a kinematic_overlap must be with the robot side (robot_link or held_object), "
                             "got partner_class %r" % (self.partner_class.value,))

    @classmethod
    def from_obs(cls, obs: ContactObs, kind: EventKind, v_normal_mps: Optional[float]) -> "ContactEvent":
        """Event from a host contact observation; ``v_normal_mps`` is computed by the caller (the core
        recomputes it from kinematic data, 01 §3.1) and may be ``None``."""
        return cls(kind=kind, time_s=obs.time_s, partner_name=obs.partner_name,
                   partner_class=obs.partner_class, position=obs.position, normal=obs.normal,
                   v_normal_mps=v_normal_mps, impulse_ns=obs.impulse_ns)


def classify_partner(partner_name: str, robot_include: Sequence[str], robot_exclude: Sequence[str],
                     held_object_names: Sequence[str], hazard_names: Sequence[str]) -> PartnerClass:
    """Class of a contact partner (00 §6 "robot side").

    Order of the rules:

    1. ``partner_name`` is one of ``hazard_names`` (exact) -> ``hazard``;
    2. it is one of ``held_object_names`` (exact) -> ``held_object`` (a grasped object counts as robot side);
    3. it matches a ``robot_exclude`` entry (cameras, sensors) -> ``scene`` (not robot side);
    4. it matches a ``robot_include`` entry -> ``robot_link``;
    5. otherwise ``scene``.

    Every include / exclude entry names its match kind (D10 f): ``exact:<name>``, ``prefix:<start>`` or
    ``substr:<fragment>``, all case-sensitive; an entry without one is rejected, so a bare fragment such as
    ``arm`` cannot silently claim ``armchair``. The token ``held_object`` in ``robot_include`` is a category,
    not a pattern, and is skipped. There must be at least one real include entry, so a misconfigured BD
    cannot silently classify every body as scene. ``held_object_names`` may be empty (nothing grasped).
    """
    if not isinstance(partner_name, str) or not partner_name:
        raise ValueError("partner_name must be a non-empty string, got %r" % (partner_name,))
    includes = _match_entries("robot_include", robot_include, skip_token=True)
    excludes = _match_entries("robot_exclude", robot_exclude, skip_token=False)
    if not includes:
        raise ValueError("robot_include has no body-name entry (only the %r token or nothing)"
                         % (HELD_OBJECT_TOKEN,))
    if partner_name in set(_names("hazard_names", hazard_names)):
        return PartnerClass.HAZARD
    if partner_name in set(_names("held_object_names", held_object_names)):
        return PartnerClass.HELD_OBJECT
    if any(match(partner_name, text) for match, text in excludes):
        return PartnerClass.SCENE
    if any(match(partner_name, text) for match, text in includes):
        return PartnerClass.ROBOT_LINK
    return PartnerClass.SCENE


def _names(name: str, values: Sequence[str]) -> List[str]:
    if isinstance(values, str):
        raise ValueError("%s must be a sequence of strings, not a single string %r" % (name, values))
    out = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError("%s contains an invalid entry %r (need non-empty strings)" % (name, value))
        out.append(value)
    return out


def _match_entries(name: str, values: Sequence[str],
                   skip_token: bool) -> List[Tuple[Callable[[str, str], bool], str]]:
    out = []
    for value in _names(name, values):
        if skip_token and value == HELD_OBJECT_TOKEN:
            continue
        kind, sep, text = value.partition(":")
        if not sep or kind not in _MATCHERS:
            raise ValueError("%s entry %r has no match-kind prefix (need one of %s)"
                             % (name, value, ", ".join(k + ":" for k in MATCH_KINDS)))
        if not text:
            raise ValueError("%s entry %r has an empty pattern after %r" % (name, value, kind + ":"))
        out.append((_MATCHERS[kind], text))
    return out
