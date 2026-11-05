"""TaskContext: what the probe learned about one (task, layout) (paradigm §2)."""
import dataclasses
import math
from typing import Any, Dict, List

from .._records import Record, check_nonempty_str, check_vec, require
from ..host import Surface
from .pose_sequence import PoseSequence

#: Where a task-relevant entity name came from (rule R3 discovery order).
ENTITY_SOURCES = ("static_labels", "named_refs", "oracle_config", "bddl_goal",
                  "probe_inferred", "annotation")


@dataclasses.dataclass
class EntityRef(Record):
    name: str
    source: str

    def validate(self) -> None:
        check_nonempty_str("name", self.name)
        require(self.source in ENTITY_SOURCES,
                "source %r is not one of %s" % (self.source, list(ENTITY_SOURCES)))


@dataclasses.dataclass
class Phase(Record):
    name: str
    t_start_s: float
    t_end_s: float

    def validate(self) -> None:
        check_nonempty_str("name", self.name)
        require(self.t_end_s >= self.t_start_s,
                "t_end_s=%r is before t_start_s=%r" % (self.t_end_s, self.t_start_s))


@dataclasses.dataclass
class Occupancy(Record):
    """Where the robot was, as a point cloud: ``points[i]`` (m, world) at
    ``time_s[i]``. May be empty."""
    points: List[List[float]]
    time_s: List[float]

    def validate(self) -> None:
        require(len(self.points) == len(self.time_s),
                "points and time_s must have equal length, got %d and %d"
                % (len(self.points), len(self.time_s)))
        for i, p in enumerate(self.points):
            check_vec("points[%d]" % i, p, 3)
        for i, t in enumerate(self.time_s):
            require(math.isfinite(t), "time_s[%d] must be finite" % i)


@dataclasses.dataclass
class AABB(Record):
    """World-frame axis-aligned box."""
    min_m: List[float]
    max_m: List[float]

    def validate(self) -> None:
        check_vec("min_m", self.min_m, 3)
        check_vec("max_m", self.max_m, 3)
        for axis in range(3):
            require(self.min_m[axis] <= self.max_m[axis],
                    "min_m[%d]=%r exceeds max_m[%d]=%r"
                    % (axis, self.min_m[axis], axis, self.max_m[axis]))


@dataclasses.dataclass
class TaskContext(Record):
    """Output of the probe for one (task, layout).

    ``entities`` maps a role (e.g. ``"target"``) to the entity name and where
    that name came from. ``phases`` must be sorted by ``t_start_s`` with unique
    names. ``provenance`` is free-form JSON (probe sources, hashes, seeds).
    """
    task_id: str
    layout_id: str
    entities: Dict[str, EntityRef]
    phases: List[Phase]
    nominal_ee_path: PoseSequence
    occupancy: Occupancy
    workspace_bounds: AABB
    support_surfaces: List[Surface]
    provenance: Dict[str, Any]

    def validate(self) -> None:
        check_nonempty_str("task_id", self.task_id)
        check_nonempty_str("layout_id", self.layout_id)
        for role in self.entities:
            check_nonempty_str("entities role", role)
        names = [p.name for p in self.phases]
        require(len(set(names)) == len(names), "phase names must be unique, got %s" % names)
        for i in range(1, len(self.phases)):
            require(self.phases[i].t_start_s >= self.phases[i - 1].t_start_s,
                    "phases must be sorted by t_start_s (%r before %r)"
                    % (self.phases[i - 1].name, self.phases[i].name))
