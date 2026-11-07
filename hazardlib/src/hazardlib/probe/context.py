"""Probe traces -> TaskContext (paradigm §2, §5.4; rules R3, R4). Pure function of its inputs.

Algorithm (every number comes from :class:`ProbeSettings`, i.e. ``configs/hazard/probe.yaml``):

1. **R4.** Any trace whose source is marked ``under_test``, or whose source name is listed in
   ``policies_under_test``, raises :class:`ProbeSourceError`.
2. **Usable runs.** Runs that did not succeed, or that are not hazard-free (a step with ``in_contact`` or a
   ``kinematic`` / ``released`` hazard phase), are dropped and counted by reason. Each run is cut at its
   success time (``success_t_s``, else its last step).
3. **Gripper events.** The opening is normalised per run by its largest magnitude; it closes when it drops
   below ``gripper_closed_below`` and opens again when it rises above ``gripper_open_above`` (hysteresis).
   An event time is the first trace sample past the threshold (resolution: one control step).
   The signature of a run is its initial state plus the event sequence; the most common signature wins
   (ties: the lexicographically smallest) and runs with another signature are dropped. At least
   ``min_consistent_runs`` runs from at least ``min_source_kinds`` source kinds must remain (R4).
4. **Phases.** Boundaries are the run start, the events and the success time; the nominal boundaries are their
   medians across runs. Segments with the gripper open are named ``approach``, ``approach_2``...; closed
   segments ``carry``, ``carry_2``...
5. **Nominal EEF path.** Inside each phase, ``path_samples_per_phase`` + 1 samples at equal phase-normalised
   times; at each sample the coordinate-wise median over the runs (linear interpolation of each run), placed
   at the nominal boundary times. Orientation: sign-aligned normalised mean of the runs' interpolated
   quaternions if every step has one, else the identity (recorded). Robot-side collision bodies get median
   paths the same way when every step of every run reports the same set (``provenance.robot_body_paths``).
6. **Occupancy** = the nominal EEF samples plus the nominal body samples, time-stamped.
7. **Entities** (R3 order): static declarations -> grasped objects (``held_object`` in at least
   ``entity_min_run_fraction`` of the runs) -> moved objects (displaced by more than
   ``moved_object_min_displacement_m`` in that many runs) -> annotations. Each keeps its source; an annotation
   that names a different entity for an existing role raises.
8. **Workspace**: the declared box if given (source ``declared``), else the bounding box of every EEF, robot
   body and object position of the used runs (source ``probe_hull``). Gate G0 adds its own margin.
9. **Support surfaces**: the host's (trace headers; they must agree across runs) else one estimated plane at the
   lowest task-object centre minus ``surface_below_object_m`` over the xy hull of objects and EEF (source
   ``probe_estimate``), else none.
10. **Nominal speed**: median EEF speed over the samples faster than ``moving_speed_min_mps`` (00 §7).

The probe-set description (sources, run counts, trace hashes, exclusions) goes into ``provenance`` together
with its hash, so it is part of the TaskContext hash.
"""
import dataclasses
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .._numeric import non_negative_float, positive_float, positive_int
from .._settings import ValueReader
from ..config import ConfigError, load_config
from ..hashing import sha256_hex
from ..kinematics.path import SampledPath
from ..runtime import KINEMATIC, RELEASED
from ..spec.task_context import TaskContext
from .trace import ProbeTrace

STATIC_ENTITY_SOURCES = ("static_labels", "named_refs", "oracle_config", "bddl_goal")
PROBE_KEYS = ("min_source_kinds", "gripper_closed_below", "gripper_open_above", "min_consistent_runs",
              "path_samples_per_phase", "moved_object_min_displacement_m", "entity_min_run_fraction",
              "moving_speed_min_mps", "surface_below_object_m")
_ACTIVE = (KINEMATIC, RELEASED)
_IDENTITY = (1.0, 0.0, 0.0, 0.0)


class ProbeSourceError(ValueError):
    """A probe source is (or is declared as) a policy under test (R4)."""


@dataclasses.dataclass(frozen=True)
class ProbeSettings:
    min_source_kinds: int
    gripper_closed_below: float
    gripper_open_above: float
    min_consistent_runs: int
    path_samples_per_phase: int
    moved_object_min_displacement_m: float
    entity_min_run_fraction: float
    moving_speed_min_mps: float
    surface_below_object_m: float

    def __post_init__(self) -> None:
        for name in ("min_source_kinds", "min_consistent_runs", "path_samples_per_phase"):
            object.__setattr__(self, name, positive_int(name, getattr(self, name)))
        for name in ("gripper_closed_below", "gripper_open_above", "moving_speed_min_mps", "surface_below_object_m"):
            object.__setattr__(self, name, non_negative_float(name, getattr(self, name)))
        for name in ("moved_object_min_displacement_m", "entity_min_run_fraction"):
            object.__setattr__(self, name, positive_float(name, getattr(self, name)))
        if not self.gripper_closed_below < self.gripper_open_above <= 1.0:
            raise ValueError("need gripper_closed_below < gripper_open_above <= 1 (normalised opening), got %r, %r"
                             % (self.gripper_closed_below, self.gripper_open_above))
        if self.entity_min_run_fraction > 1.0:
            raise ValueError("entity_min_run_fraction must be <= 1, got %r" % (self.entity_min_run_fraction,))

    @classmethod
    def from_config(cls, probe: Mapping[str, Any]) -> "ProbeSettings":
        reader = ValueReader("ProbeSettings")
        values = {key: reader.get("probe", probe, key) for key in PROBE_KEYS}
        reader.done()
        try:
            return cls(**values)
        except ValueError as exc:
            raise ConfigError("ProbeSettings: %s" % exc) from None

    @classmethod
    def load(cls, config_dir: Optional[Any] = None) -> "ProbeSettings":
        return cls.from_config(load_config("probe", config_dir=config_dir))

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# ------------------------------------------------------------------------------------------- per run

class _Run:
    def __init__(self, trace: ProbeTrace) -> None:
        header = trace.header
        self.trace = trace
        self.source = header.source
        end = header.success_t_s if header.success_t_s is not None else trace.steps[-1].t_s
        steps = [s for s in trace.steps if s.t_s <= end]
        if len(steps) < 2:
            raise ValueError("probe trace %s/%s seed %d has fewer than 2 steps before its success time"
                             % (header.task_id, header.layout_id, header.seed))
        self.steps = steps
        self.success_t = float(end)
        self.t = np.array([s.t_s for s in steps])
        self.pos = np.array([s.eef_pos for s in steps], dtype=np.float64)
        self.events = []  # type: List[Tuple[str, float]]
        self.initial = "open"

    def interp(self, values: np.ndarray, times: np.ndarray) -> np.ndarray:
        return np.stack([np.interp(times, self.t, values[:, k]) for k in range(values.shape[1])], axis=-1)

    def detect_events(self, closed_below: float, open_above: float) -> None:
        values = [(s.t_s, s.gripper) for s in self.steps if s.gripper is not None]
        if not values:
            return
        scale = max(abs(g) for _, g in values)
        if scale == 0.0:
            return
        state = "closed" if values[0][1] / scale < closed_below else "open"
        self.initial = state
        for t, g in values[1:]:
            level = g / scale
            if state == "open" and level < closed_below:
                state = "closed"
                self.events.append(("close", t))
            elif state == "closed" and level > open_above:
                state = "open"
                self.events.append(("open", t))

    @property
    def signature(self) -> Tuple[str, ...]:
        return (self.initial,) + tuple(kind for kind, _ in self.events)

    @property
    def boundaries(self) -> List[float]:
        return [float(self.t[0])] + [t for _, t in self.events] + [self.success_t]


def _hazard_free(trace: ProbeTrace) -> bool:
    return not any(s.in_contact or s.phase in _ACTIVE for s in trace.steps)


def _phase_names(initial: str, n_segments: int) -> List[str]:
    names, counts, state = [], {"open": 0, "closed": 0}, initial
    for _ in range(n_segments):
        counts[state] += 1
        base = "approach" if state == "open" else "carry"
        names.append(base if counts[state] == 1 else "%s_%d" % (base, counts[state]))
        state = "closed" if state == "open" else "open"
    return names


def _median_quats(runs: List[_Run], run_times: List[np.ndarray]) -> Optional[np.ndarray]:
    if not all(s.eef_quat is not None for r in runs for s in r.steps):
        return None
    stacks = []
    for run, times in zip(runs, run_times):
        quats = np.array([s.eef_quat for s in run.steps], dtype=np.float64)
        stacks.append(SampledPath(run.t, run.pos, quats).quat(times))
    ref = stacks[0]
    total = np.zeros_like(ref)
    for q in stacks:
        sign = np.where(np.sum(q * ref, axis=1) < 0, -1.0, 1.0)[:, None]
        total += sign * q
    norms = np.linalg.norm(total, axis=1, keepdims=True)
    if not np.all(norms > 0):
        return None
    return total / norms


def _body_paths(runs: List[_Run], run_times: List[np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
    names = None  # type: Optional[List[str]]
    for run in runs:
        for step in run.steps:
            if step.robot_bodies is None:
                return None
            keys = sorted(step.robot_bodies)
            if names is None:
                names = keys
            elif keys != names:
                return None
    if not names:
        return None
    out = {}
    for name in names:
        per_run = []
        for run, times in zip(runs, run_times):
            values = np.array([s.robot_bodies[name] for s in run.steps], dtype=np.float64)
            per_run.append(run.interp(values, times))
        out[name] = np.median(np.stack(per_run), axis=0)
    return out


def _speeds(run: _Run) -> np.ndarray:
    if all(s.eef_lin_vel_mps is not None for s in run.steps):
        return np.linalg.norm(np.array([s.eef_lin_vel_mps for s in run.steps], dtype=np.float64), axis=1)
    return np.linalg.norm(np.diff(run.pos, axis=0), axis=1) / np.diff(run.t)


def _entities(runs: List[_Run], settings: ProbeSettings, static_entities: Mapping[str, Mapping[str, str]],
              annotations: Mapping[str, str]) -> Tuple[Dict[str, Dict[str, str]], List[Dict[str, str]]]:
    entities = {}  # type: Dict[str, Dict[str, str]]
    discovery = []  # type: List[Dict[str, str]]

    def add(role: str, name: str, source: str, stage: str) -> None:
        entities[role] = {"name": name, "source": source}
        discovery.append({"role": role, "name": name, "source": source, "stage": stage})

    for role, ref in static_entities.items():
        if not isinstance(ref, Mapping) or set(ref) != {"name", "source"}:
            raise ValueError("static entity %r must be a mapping with exactly 'name' and 'source', got %r" % (role, ref))
        if ref["source"] not in STATIC_ENTITY_SOURCES:
            raise ValueError("static entity %r: source %r is not a static declaration (%s)"
                             % (role, ref["source"], list(STATIC_ENTITY_SOURCES)))
        add(role, ref["name"], ref["source"], "static")
    needed = settings.entity_min_run_fraction * len(runs)
    known = {ref["name"] for ref in entities.values()}

    grasp_times = {}  # type: Dict[str, List[float]]
    for run in runs:
        first = {}  # type: Dict[str, float]
        for step in run.steps:
            if step.held_object is not None and step.held_object not in first:
                first[step.held_object] = step.t_s
        for name, t in first.items():
            grasp_times.setdefault(name, []).append(t)
    grasped = sorted((float(np.median(ts)), name) for name, ts in grasp_times.items()
                     if len(ts) >= needed and name not in known)
    for index, (_, name) in enumerate(grasped):
        add("grasped_%d" % index, name, "probe_inferred", "grasped")
        known.add(name)

    move_times = {}  # type: Dict[str, List[float]]
    for run in runs:
        start = run.steps[0].object_poses
        for name, pose0 in start.items():
            p0 = np.array(pose0.pos)
            for step in run.steps:
                pose = step.object_poses.get(name)
                if pose is not None and float(np.linalg.norm(np.array(pose.pos) - p0)) > settings.moved_object_min_displacement_m:
                    move_times.setdefault(name, []).append(step.t_s)
                    break
    moved = sorted((float(np.median(ts)), name) for name, ts in move_times.items()
                   if len(ts) >= needed and name not in known)
    for index, (_, name) in enumerate(moved):
        add("moved_%d" % index, name, "probe_inferred", "moved")
        known.add(name)

    for role, name in annotations.items():
        if not isinstance(name, str) or not name:
            raise ValueError("annotation %r must name an entity, got %r" % (role, name))
        if role in entities:
            if entities[role]["name"] != name:
                raise ValueError("annotation %r=%r conflicts with %r from %s (R3: earlier sources win; fix the "
                                 "annotation)" % (role, name, entities[role]["name"], entities[role]["source"]))
            continue
        add(role, name, "annotation", "annotation")
    return entities, discovery


def _surfaces(runs: List[_Run], settings: ProbeSettings) -> Tuple[List[Dict[str, Any]], str]:
    reported = [r.trace.header.support_surfaces for r in runs]
    if any(s is not None for s in reported):
        dicts = [None if s is None else [x.to_dict() for x in s] for s in reported]
        if any(d != dicts[0] for d in dicts):
            raise ValueError("probe runs report different support surfaces; probe one (task, layout) at a time")
        return dicts[0], "host"
    objects = [pose.pos for r in runs for pose in r.steps[0].object_poses.values()]
    if not objects:
        return [], "none"
    objects_arr = np.array(objects, dtype=np.float64)
    top = float(objects_arr[:, 2].min()) - settings.surface_below_object_m
    xy = np.concatenate([objects_arr[:, :2]] + [r.pos[:, :2] for r in runs])
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    return [{"name": "probe_estimated_support", "min_m": [float(lo[0]), float(lo[1]), top],
             "max_m": [float(hi[0]), float(hi[1]), top], "source": "probe_estimate"}], "probe_estimate"


def build_task_context(traces: Sequence[ProbeTrace], settings: ProbeSettings,
                       static_entities: Optional[Mapping[str, Mapping[str, str]]] = None,
                       annotations: Optional[Mapping[str, str]] = None,
                       declared_workspace: Optional[Mapping[str, Sequence[float]]] = None,
                       policies_under_test: Sequence[str] = ()) -> TaskContext:
    """See the module docstring."""
    traces = list(traces)
    if not traces:
        raise ValueError("build_task_context needs at least one probe trace")
    if not isinstance(settings, ProbeSettings):
        raise TypeError("settings must be ProbeSettings, got %r" % (type(settings).__name__,))
    under_test = set(policies_under_test)
    for trace in traces:
        if not isinstance(trace, ProbeTrace):
            raise TypeError("traces must be ProbeTrace objects, got %r" % (type(trace).__name__,))
        source = trace.header.source
        if source.under_test or source.name in under_test:
            raise ProbeSourceError("probe source %s %r is a policy under test; R4 forbids it in the probe set"
                                   % (source.kind, source.name))
    keys = {(t.header.task_id, t.header.layout_id) for t in traces}
    if len(keys) != 1:
        raise ValueError("all probe traces must share one (task_id, layout_id), got %s" % sorted(keys))
    task_id, layout_id = keys.pop()

    excluded = {}  # type: Dict[str, int]
    runs = []  # type: List[_Run]
    for trace in traces:
        if not trace.header.success:
            excluded["not_successful"] = excluded.get("not_successful", 0) + 1
        elif not _hazard_free(trace):
            excluded["hazard_present"] = excluded.get("hazard_present", 0) + 1
        else:
            run = _Run(trace)
            run.detect_events(settings.gripper_closed_below, settings.gripper_open_above)
            runs.append(run)
    if not runs:
        raise ValueError("no successful hazard-free probe run (excluded: %s)" % excluded)

    counts = {}  # type: Dict[Tuple[str, ...], int]
    for run in runs:
        counts[run.signature] = counts.get(run.signature, 0) + 1
    signature = sorted(counts.items(), key=lambda item: (-item[1], repr(item[0])))[0][0]
    consistent = []
    for run in runs:
        bounds = run.boundaries
        if run.signature != signature:
            excluded["inconsistent_gripper_events"] = excluded.get("inconsistent_gripper_events", 0) + 1
        elif not all(b1 > b0 for b0, b1 in zip(bounds[:-1], bounds[1:])):
            excluded["degenerate_phase"] = excluded.get("degenerate_phase", 0) + 1
        else:
            consistent.append(run)
    if len(consistent) < settings.min_consistent_runs:
        raise ValueError("only %d consistent probe run(s), %d required (signature %s; excluded: %s)"
                         % (len(consistent), settings.min_consistent_runs, list(signature), excluded))
    kinds = sorted({r.source.kind for r in consistent})
    if len(kinds) < settings.min_source_kinds:
        raise ValueError("the consistent probe runs come from %d source kind(s) %s; R4 requires at least %d"
                         % (len(kinds), kinds, settings.min_source_kinds))

    boundary_matrix = np.array([r.boundaries for r in consistent])
    nominal_bounds = np.median(boundary_matrix, axis=0)
    names = _phase_names(signature[0], len(nominal_bounds) - 1)
    n = settings.path_samples_per_phase
    nominal_t, run_times = [], [[] for _ in consistent]  # type: List[np.ndarray], List[List[np.ndarray]]
    for k in range(len(names)):
        u = np.linspace(0.0, 1.0, n + 1)
        if k > 0:
            u = u[1:]
        nominal_t.append(nominal_bounds[k] + u * (nominal_bounds[k + 1] - nominal_bounds[k]))
        for i, run in enumerate(consistent):
            b = run.boundaries
            run_times[i].append(b[k] + u * (b[k + 1] - b[k]))
    times = np.concatenate(nominal_t)
    per_run_times = [np.concatenate(ts) for ts in run_times]
    positions = np.median(np.stack([r.interp(r.pos, ts) for r, ts in zip(consistent, per_run_times)]), axis=0)
    quats = _median_quats(consistent, per_run_times)
    quat_note = "median_of_probe_quaternions"
    if quats is None:
        quats = np.tile(np.array(_IDENTITY), (times.size, 1))
        quat_note = "identity: some probe step has no eef_quat"
    bodies = _body_paths(consistent, per_run_times)

    points = [positions]
    point_t = [times]
    if bodies is not None:
        for name in sorted(bodies):
            points.append(bodies[name])
            point_t.append(times)
    occupancy = {"points": np.concatenate(points).tolist(), "time_s": np.concatenate(point_t).tolist()}

    entities, discovery = _entities(consistent, settings, static_entities or {}, annotations or {})

    if declared_workspace is not None:
        workspace = {"min_m": [float(v) for v in declared_workspace["min_m"]],
                     "max_m": [float(v) for v in declared_workspace["max_m"]]}
        workspace_source = "declared"
    else:
        cloud = [r.pos for r in consistent]
        for run in consistent:
            for step in run.steps:
                if step.robot_bodies:
                    cloud.append(np.array(list(step.robot_bodies.values()), dtype=np.float64))
                if step.object_poses:
                    cloud.append(np.array([p.pos for p in step.object_poses.values()], dtype=np.float64))
        hull = np.concatenate(cloud)
        workspace = {"min_m": hull.min(axis=0).tolist(), "max_m": hull.max(axis=0).tolist()}
        workspace_source = "probe_hull"

    surfaces, surface_source = _surfaces(consistent, settings)

    speeds = np.concatenate([_speeds(r) for r in consistent])
    moving = speeds[speeds > settings.moving_speed_min_mps]
    speed_median = float(np.median(moving)) if moving.size else None

    sources = {}  # type: Dict[Tuple[str, str], Dict[str, Any]]
    for trace in traces:
        key = (trace.header.source.kind, trace.header.source.name)
        entry = sources.setdefault(key, {"kind": key[0], "name": key[1], "n_runs": 0, "n_used": 0})
        entry["n_runs"] += 1
    for run in consistent:
        sources[(run.source.kind, run.source.name)]["n_used"] += 1
    probe_set = {
        "task_id": task_id, "layout_id": layout_id,
        "sources": [sources[k] for k in sorted(sources)],
        "source_kinds_used": kinds,
        "reference_policies": sorted({k[1] for k in sources if k[0] == "external_policy"}),
        "n_runs_total": len(traces), "n_runs_used": len(consistent),
        "excluded": dict(sorted(excluded.items())),
        "trace_hashes": sorted(t.hash() for t in traces),
    }
    provenance = {
        "builder": "hazardlib.probe.build_task_context",
        "probe_set": probe_set,
        "probe_set_hash": sha256_hex(probe_set),
        "settings": settings.to_dict(),
        "phase_signature": list(signature),
        "signature_counts": [{"signature": list(s), "runs": c} for s, c in sorted(counts.items())],
        "success_times_s": sorted(r.success_t for r in consistent),
        "nominal_success_time_s": float(nominal_bounds[-1]),
        "nominal_path_aggregation": "per-phase normalised time, coordinate-wise median over runs",
        "nominal_quat": quat_note,
        "occupancy_content": "nominal_eef" if bodies is None else "nominal_eef_and_robot_bodies",
        "robot_body_paths": None if bodies is None else {k: v.tolist() for k, v in sorted(bodies.items())},
        "entity_discovery": discovery,
        "workspace_source": workspace_source,
        "support_surface_source": surface_source,
        "eef_speed_median_mps": speed_median,
    }
    data = {
        "task_id": task_id, "layout_id": layout_id,
        "entities": entities,
        "phases": [{"name": names[k], "t_start_s": float(nominal_bounds[k]), "t_end_s": float(nominal_bounds[k + 1])}
                   for k in range(len(names))],
        "nominal_ee_path": {"time_s": times.tolist(), "pos": positions.tolist(), "quat": quats.tolist()},
        "occupancy": occupancy,
        "workspace_bounds": workspace,
        "support_surfaces": surfaces,
        "provenance": provenance,
    }
    return TaskContext.from_dict(data)
