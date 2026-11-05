"""HazardSpec: the simulator-independent hazard specification (paradigm §2, §4)."""
import dataclasses
from typing import Any, Dict, List, Optional

from .._records import (Record, check_nonempty_str, check_positive, check_vec, choice,
                        require)

TIMING_MODES = ("nominal_time", "phase_event")


@dataclasses.dataclass
class AgentSpec(Record):
    """The hazard body. ``kind``/``shape`` are open vocabularies (the family set
    is still [开放] in the paradigm); the adapter decides what it can build."""
    kind: str
    shape: str
    half_extents_m: List[float]
    mass_kg: float

    def validate(self) -> None:
        check_nonempty_str("kind", self.kind)
        check_nonempty_str("shape", self.shape)
        require(len(self.half_extents_m) >= 1, "half_extents_m must not be empty")
        for i, h in enumerate(self.half_extents_m):
            check_positive("half_extents_m[%d]" % i, h)
        check_positive("mass_kg", self.mass_kg)


@dataclasses.dataclass
class MotionSpec(Record):
    """Motion primitive name + parameters (primitives are defined by T02)."""
    primitive: str
    params: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def validate(self) -> None:
        check_nonempty_str("primitive", self.primitive)


@dataclasses.dataclass
class PhaseRef(Record):
    """A point inside a TaskContext phase: ``frac`` in [0, 1] of its duration."""
    phase: str
    frac: float

    def validate(self) -> None:
        check_nonempty_str("phase", self.phase)
        require(0.0 <= self.frac <= 1.0, "frac must lie in [0, 1], got %r" % (self.frac,))


@dataclasses.dataclass
class ExposureSpec(Record):
    """Where the hazard meets the task. ``anchor`` names what it is anchored to
    (open vocabulary, e.g. the nominal EEF path or an entity role);
    ``anchor_phase`` is only needed when the anchor is time-based."""
    anchor: str
    offset_m: List[float]
    anchor_phase: Optional[PhaseRef] = None

    def validate(self) -> None:
        check_nonempty_str("anchor", self.anchor)
        check_vec("offset_m", self.offset_m, 3)


@dataclasses.dataclass
class TimingSpec(Record):
    mode: str = choice(TIMING_MODES)  # type: ignore[assignment]


@dataclasses.dataclass
class ReleaseSpec(Record):
    """How the hazard is handed to dynamics (open vocabulary; the paradigm's
    default scheme is ``dynamic_release``, defined by T03/T04)."""
    mode: str

    def validate(self) -> None:
        check_nonempty_str("mode", self.mode)


@dataclasses.dataclass
class HazardSpec(Record):
    """One entry of the human-written hazard library.

    ``intensity`` holds the family's intensity vector (axis name -> number or
    list); its schema is family-specific and defined in ``configs/hazard``.
    """
    spec_version: str
    id: str
    family: str
    agent: AgentSpec
    motion: MotionSpec
    exposure: ExposureSpec
    timing: TimingSpec
    intensity: Dict[str, Any]
    release: ReleaseSpec

    def validate(self) -> None:
        check_nonempty_str("spec_version", self.spec_version)
        check_nonempty_str("id", self.id)
        check_nonempty_str("family", self.family)
