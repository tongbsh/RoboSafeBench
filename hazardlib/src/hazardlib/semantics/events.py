"""Contact events and partner classification (00 §6, rule R8).

``PartnerClass`` and ``ContactObs`` are the T01 types from :mod:`hazardlib.host`; they are imported, not
redefined. Contact normals follow the project-wide convention (D9 a): the normal points **from the partner
toward the hazard**, i.e. the direction in which the partner pushes the hazard.
"""
import dataclasses
import enum
from typing import Any, Optional, Sequence

import numpy as np

from .._numeric import finite_float, non_negative_float, vec3
from ..host import ContactObs, PartnerClass

#: ``robot_contact_bodies.include`` may carry this category token (01 §2 example). It is not a body-name
#: pattern: the grasped object is identified by name through ``held_object_name`` instead.
HELD_OBJECT_TOKEN = "held_object"


class EventKind(str, enum.Enum):
    """``official``: a contact after the hazard became dynamic. ``kinematic_overlap``: an overlap of a
    kinematically driven hazard with the robot side, which is never an official contact (R8)."""
    OFFICIAL = "official"
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
                     held_object_name: Optional[str], hazard_names: Sequence[str]) -> PartnerClass:
    """Class of a contact partner (00 §6 "robot side").

    Order of the rules:

    1. ``partner_name`` is one of ``hazard_names`` (exact) -> ``hazard``;
    2. it equals ``held_object_name`` (exact) -> ``held_object`` (a grasped object counts as robot side);
    3. it contains an ``robot_exclude`` entry (cameras, sensors) -> ``scene`` (not robot side);
    4. it contains an ``robot_include`` entry -> ``robot_link``;
    5. otherwise ``scene``.

    Patterns match as **substrings**, case-sensitively (the profiles use both prefixes such as
    ``robot0_`` and fragments such as ``camera``). The token ``held_object`` in ``robot_include`` is a
    category, not a pattern, and is skipped. There must be at least one real include pattern and no empty
    pattern, so a misconfigured BD cannot silently classify every body as scene (or as robot).
    """
    if not isinstance(partner_name, str) or not partner_name:
        raise ValueError("partner_name must be a non-empty string, got %r" % (partner_name,))
    includes = _patterns("robot_include", robot_include, skip_token=True)
    excludes = _patterns("robot_exclude", robot_exclude, skip_token=False)
    if not includes:
        raise ValueError("robot_include has no body-name pattern (only the %r token or nothing)"
                         % (HELD_OBJECT_TOKEN,))
    if partner_name in set(_patterns("hazard_names", hazard_names, skip_token=False)):
        return PartnerClass.HAZARD
    if held_object_name is not None and partner_name == held_object_name:
        return PartnerClass.HELD_OBJECT
    if any(pattern in partner_name for pattern in excludes):
        return PartnerClass.SCENE
    if any(pattern in partner_name for pattern in includes):
        return PartnerClass.ROBOT_LINK
    return PartnerClass.SCENE


def _patterns(name: str, values: Sequence[str], skip_token: bool) -> list:
    if isinstance(values, str):
        raise ValueError("%s must be a sequence of strings, not a single string %r" % (name, values))
    out = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError("%s contains an invalid pattern %r (need non-empty strings)" % (name, value))
        if skip_token and value == HELD_OBJECT_TOKEN:
            continue
        out.append(value)
    return out
