"""Probe traces (T06): what one probe run recorded, one JSON Lines file per run.

The layout follows the repository's diagnostics-trace convention (a header line, then one record per control
step), in seconds and SI units:

* line 1: ``{"probe_trace": <TraceHeader>}``
* every further line: ``{"step": <TraceStep>}``

``TraceStep`` fields: ``t_s`` (seconds from the episode start), ``eef_pos``, ``eef_quat`` (wxyz or null),
``eef_lin_vel_mps`` (or null), ``gripper`` (the opening the host reports, any unit, or null),
``held_object`` (or null), ``robot_bodies`` (robot-side collision body name -> position per the BD's
``robot_contact_bodies``, or null), ``object_poses`` (name -> ``{pos, quat}``), ``in_contact`` (an official
hazard contact happened during the step) and ``phase`` (the runtime's hazard phase, null when no hazard was
armed). A probe run is meant to be hazard-free; ``build_task_context`` drops runs that are not.

The header carries the task, layout, seed, the source (``kind`` in :data:`SOURCE_KINDS`, ``name``,
``under_test``), the success event (``success``, ``success_t_s``), the control step and the host's
support surfaces when it offers ``support_surfaces``. Rule R4: a source marked ``under_test`` is refused by
``build_task_context``.
"""
import dataclasses
import json
import os
from typing import Any, Dict, List, Optional, Union

from .._records import (Record, check_finite, check_nonempty_str, check_positive, check_vec, choice,
                        require)
from ..driving import REFERENCE_KINDS, EpisodeSession, describe_policy
from ..host import Surface

PROBE_TRACE_SCHEMA_VERSION = "1"
SOURCE_KINDS = REFERENCE_KINDS


@dataclasses.dataclass
class ProbeSource(Record):
    kind: str = choice(SOURCE_KINDS)  # type: ignore[assignment]
    name: str
    under_test: bool

    def validate(self) -> None:
        check_nonempty_str("name", self.name)


@dataclasses.dataclass
class ObjectPose(Record):
    pos: List[float]
    quat: List[float]

    def validate(self) -> None:
        check_vec("pos", self.pos, 3)
        check_vec("quat", self.quat, 4)


@dataclasses.dataclass
class TraceStep(Record):
    t_s: float
    eef_pos: List[float]
    eef_quat: Optional[List[float]]
    eef_lin_vel_mps: Optional[List[float]]
    gripper: Optional[float]
    held_object: Optional[str]
    robot_bodies: Optional[Dict[str, List[float]]]
    object_poses: Dict[str, ObjectPose]
    in_contact: bool
    phase: Optional[str]

    def validate(self) -> None:
        check_finite("t_s", self.t_s)
        check_vec("eef_pos", self.eef_pos, 3)
        if self.eef_quat is not None:
            check_vec("eef_quat", self.eef_quat, 4)
        if self.eef_lin_vel_mps is not None:
            check_vec("eef_lin_vel_mps", self.eef_lin_vel_mps, 3)
        if self.gripper is not None:
            check_finite("gripper", self.gripper)
        if self.robot_bodies is not None:
            for name, pos in self.robot_bodies.items():
                check_vec("robot_bodies[%r]" % name, pos, 3)


@dataclasses.dataclass
class TraceHeader(Record):
    schema_version: str
    task_id: str
    layout_id: str
    seed: int
    source: ProbeSource
    success: bool
    success_t_s: Optional[float]
    control_dt_s: float
    support_surfaces: Optional[List[Surface]]
    n_steps: int

    def validate(self) -> None:
        require(self.schema_version == PROBE_TRACE_SCHEMA_VERSION,
                "unsupported schema_version %r (supported: %r)" % (self.schema_version, PROBE_TRACE_SCHEMA_VERSION))
        check_nonempty_str("task_id", self.task_id)
        check_nonempty_str("layout_id", self.layout_id)
        check_positive("control_dt_s", self.control_dt_s)
        require(self.success_t_s is None or self.success, "success_t_s is set but success is false")


@dataclasses.dataclass
class ProbeTrace(Record):
    header: TraceHeader
    steps: List[TraceStep]

    def validate(self) -> None:
        require(len(self.steps) >= 2, "a probe trace needs at least 2 steps, got %d" % len(self.steps))
        require(self.header.n_steps == len(self.steps),
                "header.n_steps=%d but %d steps" % (self.header.n_steps, len(self.steps)))
        for i in range(1, len(self.steps)):
            require(self.steps[i].t_s > self.steps[i - 1].t_s,
                    "steps[%d].t_s=%r is not after steps[%d].t_s=%r"
                    % (i, self.steps[i].t_s, i - 1, self.steps[i - 1].t_s))


_PathLike = Union[str, "os.PathLike[str]"]


def write_probe_trace(path: _PathLike, trace: ProbeTrace) -> None:
    trace.validate()
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"probe_trace": trace.header.to_dict()}, allow_nan=False, sort_keys=True) + "\n")
        for step in trace.steps:
            handle.write(json.dumps({"step": step.to_dict()}, allow_nan=False, sort_keys=True) + "\n")


def read_probe_trace(path: _PathLike) -> ProbeTrace:
    header = None  # type: Any
    steps = []  # type: List[Any]
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise ValueError("%s line %d: invalid JSON: %s" % (path, number, exc)) from None
            if not isinstance(record, dict) or len(record) != 1:
                raise ValueError("%s line %d: expected one of {\"probe_trace\": ...} / {\"step\": ...}" % (path, number))
            if "probe_trace" in record:
                if number != 1 or header is not None:
                    raise ValueError("%s line %d: the header must be the first line" % (path, number))
                header = record["probe_trace"]
            elif "step" in record:
                if header is None:
                    raise ValueError("%s line %d: step before the header" % (path, number))
                steps.append(record["step"])
            else:
                raise ValueError("%s line %d: unknown record %s" % (path, number, sorted(record)))
    if header is None:
        raise ValueError("%s: empty probe trace" % (path,))
    return ProbeTrace.from_dict({"header": header, "steps": steps}, path=str(path))


def record_probe_trace(session: EpisodeSession, policy: Any, task_id: str, layout_id: str, seed: int,
                       horizon_s: float, env: int) -> ProbeTrace:
    """Run ``policy`` in a hazard-free control episode and record its probe trace. The run stops at the
    first success (the TaskContext only uses the trace up to it) or at ``horizon_s``."""
    source = describe_policy(policy)
    control_id = "probe:%s:%s:%s:%s:%d" % (source["kind"], source["name"], task_id, layout_id, seed)
    result = session.run_episode(env, None, policy, task_id, layout_id, seed, horizon_s, control_id=control_id,
                                 record_trace=True, stop_on_success=True)
    surfaces = None
    query = getattr(session.host, "support_surfaces", None)
    if callable(query):
        surfaces = [s.to_dict() for s in query(env)]
    header = {
        "schema_version": PROBE_TRACE_SCHEMA_VERSION, "task_id": task_id, "layout_id": layout_id, "seed": seed,
        "source": source, "success": result.success, "success_t_s": result.success_t_s,
        "control_dt_s": session.host.time_state(env).control_dt_s, "support_surfaces": surfaces,
        "n_steps": len(result.trace_steps),
    }
    return ProbeTrace.from_dict({"header": header, "steps": result.trace_steps})
