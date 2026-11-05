"""BenchmarkDescriptor (BD): what a benchmark can provide, declared once in YAML.

Field set mirrors docs/design/01_integration_rules.md §2 exactly. Every key is
required (omitting one cannot hide an unknown); any value may be the string
``"unknown"`` unless ``strict=True`` (an unknown must be removed by a probe
before a BD may be used in a formal experiment, §2). Closed vocabularies are
checked; open ones (``engine.name``, ``success_oracle.kind`` …) only for type.
"""
import dataclasses
from typing import Any, Dict, List, Optional, Tuple

from .._records import (UNKNOWN, Record, SpecError, check_nonempty_str, choice,
                        find_markers, is_unknown, require)
from ..host import CapabilityError, Tier
from .task_context import ENTITY_SOURCES

LOOP_OWNERS = ("user_script", "policy_side", "env_internal")
SPAWN_MODES = ("runtime", "pooled", "preload_xml")
CONTACT_TIERS = ("A", "B", "C")
DISTANCE_KINDS = ("geometric", "surrogate", "none")
REPRO_LEVELS = ("exact", "statistical", "unknown")


class TierUndeterminedError(CapabilityError):
    """The Tier cannot be derived because tier-relevant fields are ``unknown``."""

    def __init__(self, message: str, fields: List[str]) -> None:
        super().__init__(message)
        self.fields = fields


class _BDRecord(Record):
    ALLOW_UNKNOWN = True


@dataclasses.dataclass
class BenchmarkInfo(_BDRecord):
    repo: str
    commits: Dict[str, str]
    license: str


@dataclasses.dataclass
class PythonInfo(_BDRecord):
    required: str
    env_name: str


@dataclasses.dataclass
class EngineInfo(_BDRecord):
    name: str
    version: str


@dataclasses.dataclass
class Timebase(_BDRecord):
    physics_dt_s: float
    control_dt_s: float
    substeps_per_action: int

    def validate(self) -> None:
        for name in ("physics_dt_s", "control_dt_s"):
            value = getattr(self, name)
            require(is_unknown(value) or value > 0, "%s must be > 0, got %r" % (name, value))
        require(is_unknown(self.substeps_per_action) or self.substeps_per_action >= 1,
                "substeps_per_action must be >= 1, got %r" % (self.substeps_per_action,))


@dataclasses.dataclass
class ParallelInfo(_BDRecord):
    num_envs: int
    batched: bool

    def validate(self) -> None:
        require(is_unknown(self.num_envs) or self.num_envs >= 1,
                "num_envs must be >= 1, got %r" % (self.num_envs,))


@dataclasses.dataclass
class RobotContactBodies(_BDRecord):
    include: List[str]
    exclude: List[str]


@dataclasses.dataclass
class StateAccess(_BDRecord):
    eef: bool
    gripper: bool
    base: bool
    objects: bool
    fixtures: bool
    support_surfaces: bool


@dataclasses.dataclass
class SuccessOracle(_BDRecord):
    kind: str
    readonly: bool


@dataclasses.dataclass
class DeterminismInfo(_BDRecord):
    same_seed_repro: str = choice(REPRO_LEVELS)  # type: ignore[assignment]


@dataclasses.dataclass
class InvalidSampleMapping(_BDRecord):
    setup_invalid: str
    sim_unstable: str


@dataclasses.dataclass
class ObservationIsolation(_BDRecord):
    hazard_visible_in_rgb: bool
    hazard_in_state_vectors: bool


@dataclasses.dataclass
class PolicyInterface(_BDRecord):
    kind: str
    notes: str


#: (dotted BD field, required value) pairs that together make Tier A (01 §2).
TIER_A_REQUIREMENTS = (
    ("contact_tier", "A"),
    ("kinematic_pose_set", True),
    ("release_with_velocity", True),
    ("state_access.eef", True),
    ("state_access.objects", True),
)


@dataclasses.dataclass
class BenchmarkDescriptor(_BDRecord):
    name: str
    bd_version: str
    benchmark: BenchmarkInfo
    python: PythonInfo
    engine: EngineInfo
    timebase: Timebase
    parallel: ParallelInfo
    loop_owner: str = choice(LOOP_OWNERS)  # type: ignore[assignment]
    hook_point: str
    spawn_mode: str = choice(SPAWN_MODES)  # type: ignore[assignment]
    kinematic_pose_set: bool
    release_with_velocity: bool
    contact_tier: str = choice(CONTACT_TIERS)  # type: ignore[assignment]
    distance: str = choice(DISTANCE_KINDS)  # type: ignore[assignment]
    robot_contact_bodies: RobotContactBodies
    state_access: StateAccess
    task_entity_source: str = choice(ENTITY_SOURCES)  # type: ignore[assignment]
    success_oracle: SuccessOracle
    snapshot_restore: bool
    determinism: DeterminismInfo
    invalid_sample_mapping: InvalidSampleMapping
    observation_isolation: ObservationIsolation
    policy_interface: PolicyInterface

    # ------------------------------------------------------------------
    def validate(self) -> None:
        check_nonempty_str("name", self.name)
        check_nonempty_str("bd_version", self.bd_version)

    @classmethod
    def from_dict(cls, data: Any, path: Optional[str] = None, strict: bool = False) -> "BenchmarkDescriptor":
        """``strict=True`` additionally rejects every ``unknown`` (see
        :meth:`require_complete`)."""
        obj = super().from_dict(data, path=path)
        if strict:
            obj.require_complete()
        return obj

    # ------------------------------------------------------------------
    def unknown_fields(self) -> List[str]:
        """Dotted paths of every field still equal to ``"unknown"``."""
        return [p for p, _ in find_markers(self.to_dict(), (UNKNOWN,))]

    def require_complete(self) -> None:
        """Raise :class:`SpecError` listing every ``unknown`` field."""
        unknown = self.unknown_fields()
        if unknown:
            raise SpecError(
                "BenchmarkDescriptor %r still has %d unknown field(s); remove them with a probe "
                "before formal use: %s" % (self.name, len(unknown), ", ".join(unknown)))

    # ------------------------------------------------------------------
    def _lookup(self, dotted: str) -> Any:
        current = self  # type: Any
        for part in dotted.split("."):
            if is_unknown(current):
                return UNKNOWN
            current = getattr(current, part)
        return current

    def tier(self) -> Tier:
        """Derive the Tier from the declared capabilities (01 §2).

        * **A** iff every pair in :data:`TIER_A_REQUIREMENTS` holds.
        * **C** iff ``kinematic_pose_set`` is False: a hazard body cannot be
          placed/moved at all, so there is nothing to evaluate (the paradigm's
          "no physics" case, §10).
        * **B** otherwise (physics exists and the hazard can be placed, but
          contact information, release-to-dynamics or state access is missing).

        ``unknown`` is never treated as a capability. If the answer would depend
        on an ``unknown`` field, :class:`TierUndeterminedError` is raised (naming
        those fields) instead of guessing a tier; known failures are decisive
        only when they already settle A vs not-A and B vs C.
        """
        if self._lookup("kinematic_pose_set") is False:
            return Tier.C
        states = []
        for dotted, required in TIER_A_REQUIREMENTS:
            value = self._lookup(dotted)
            if is_unknown(value):
                states.append((dotted, None))
            else:
                states.append((dotted, value == required))
        if all(ok is True for _, ok in states):
            return Tier.A
        if any(ok is False for _, ok in states) and self._lookup("kinematic_pose_set") is True:
            return Tier.B
        pending = [d for d, ok in states if ok is None]
        raise TierUndeterminedError(
            "Tier of %r is undetermined; unknown tier-relevant field(s): %s"
            % (self.name, ", ".join(pending)), pending)
