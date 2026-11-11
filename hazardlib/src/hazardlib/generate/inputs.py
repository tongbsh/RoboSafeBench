"""The two input variables of the perturbation paradigm (D20).

* **V1 - successful execution trajectories** (:class:`Trajectory` per reference demonstration, collected in a
  :class:`TrajectorySet` with the shared local robot surface point clouds): the robot geometry over time (EEF
  pose, collision-body poses), the poses of the task entities (the frames the interaction segments refer to; an
  articulated entity such as a drawer is a site frame, not necessarily a body), the held entity, the
  interaction segments (``approach_k`` / ``manipulate_k`` / ``retreat``, each with the entity concerned) and
  the control timing.
* **V2 - the 3D scene layout** (:class:`SceneLayout`, one per demonstration since every demonstration starts
  from its own layout): every non-robot body with its collision geometry as oriented boxes in the body frame,
  its role (``task_entity`` / ``neighbour`` / ``fixture`` / ``support``) and its world pose over time (one pose
  per control step, or a single pose for a body that never moves), the support surfaces and the measured
  workspace.

Both are plain data with ``save`` / ``load`` (``.npz`` arrays + ``.json`` metadata) and a content hash over
metadata and array bytes. No host, no simulator.
"""
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import quat_to_mat, world_boxes

SCHEMA = "generate-inputs-1"
ROLES = ("task_entity", "neighbour", "fixture", "support")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _hash(meta: Any, arrays: Dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256(_canonical(meta))
    for key in sorted(arrays):
        a = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(a.dtype).encode("ascii"))
        digest.update(_canonical(list(a.shape)))
        digest.update(a.tobytes())
    return digest.hexdigest()


def _write(folder: Path, stem: str, meta: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    npz, js = folder / (stem + ".npz"), folder / (stem + ".json")
    tmp = folder / (stem + ".tmp.npz")
    np.savez_compressed(str(tmp), **arrays)
    tmp.replace(npz)
    tmp_json = folder / (stem + ".json.tmp")
    tmp_json.write_text(json.dumps(meta, indent=1, sort_keys=True, allow_nan=False) + "\n")
    tmp_json.replace(js)


def _read(folder: Path, stem: str) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    meta = json.loads((Path(folder) / (stem + ".json")).read_text())
    if meta.get("schema") != SCHEMA:
        raise ValueError("%s/%s.json has schema %r, expected %r" % (folder, stem, meta.get("schema"), SCHEMA))
    with np.load(str(Path(folder) / (stem + ".npz"))) as data:
        return meta, {k: data[k] for k in data.files}


@dataclasses.dataclass
class Segment:
    """One interaction segment of a demonstration; ``start_index`` / ``end_index`` are control samples."""
    name: str
    kind: str
    entity: str
    start_index: int
    end_index: int
    t_start_s: float
    t_end_s: float

    @property
    def duration_s(self) -> float:
        return self.t_end_s - self.t_start_s

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Segment":
        return cls(name=data["name"], kind=data["kind"], entity=data["entity"], start_index=int(data["start_index"]),
                   end_index=int(data["end_index"]), t_start_s=float(data["t_start_s"]),
                   t_end_s=float(data["t_end_s"]))


@dataclasses.dataclass
class Trajectory:
    """V1 of one successful demonstration. Arrays are per control sample ``T`` (``time_s[k] = k dt``)."""
    demo: str
    control_dt_s: float
    time_s: np.ndarray            # (T,)
    eef_pos: np.ndarray           # (T, 3)
    eef_quat: np.ndarray          # (T, 4) wxyz
    body_names: List[str]         # robot collision bodies; keys of TrajectorySet.robot_points
    body_pos: np.ndarray          # (T, B, 3)
    body_quat: np.ndarray         # (T, B, 4)
    entity_names: List[str]       # task entities; the frames of the segments
    entity_pos: np.ndarray        # (T, E, 3)
    entity_quat: np.ndarray       # (T, E, 4)
    held: List[Optional[str]]     # held entity name per sample (None: nothing held)
    segments: List[Segment]

    def __post_init__(self) -> None:
        T = len(self.time_s)
        shapes = {"eef_pos": (T, 3), "eef_quat": (T, 4), "body_pos": (T, len(self.body_names), 3),
                  "body_quat": (T, len(self.body_names), 4), "entity_pos": (T, len(self.entity_names), 3),
                  "entity_quat": (T, len(self.entity_names), 4)}
        for key, shape in shapes.items():
            a = np.asarray(getattr(self, key), dtype=np.float64)
            if a.shape != shape:
                raise ValueError("Trajectory %s: %s has shape %s, expected %s" % (self.demo, key, a.shape, shape))
            setattr(self, key, a)
        self.time_s = np.asarray(self.time_s, dtype=np.float64)
        if len(self.held) != T:
            raise ValueError("Trajectory %s: held has %d entries for %d samples" % (self.demo, len(self.held), T))
        for s in self.segments:
            if s.entity not in self.entity_names:
                raise ValueError("Trajectory %s: segment %s refers to entity %r, not in %s"
                                 % (self.demo, s.name, s.entity, self.entity_names))
            if not 0 <= s.start_index < s.end_index < T:
                raise ValueError("Trajectory %s: segment %s has indices [%d, %d] outside 0..%d"
                                 % (self.demo, s.name, s.start_index, s.end_index, T - 1))

    @property
    def first_entity(self) -> Optional[str]:
        return self.segments[0].entity if self.segments else None

    def entity_frame(self, name: str, step: int) -> Tuple[np.ndarray, np.ndarray]:
        """``(origin (3,), rotation (3, 3))`` of an entity at a control sample (``0``: its initial frame)."""
        i = self.entity_names.index(name)
        return self.entity_pos[step, i], quat_to_mat(self.entity_quat[step, i])

    def _parts(self) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        meta = {"schema": SCHEMA, "kind": "trajectory", "demo": self.demo, "control_dt_s": self.control_dt_s,
                "body_names": list(self.body_names), "entity_names": list(self.entity_names),
                "held": list(self.held), "segments": [s.to_dict() for s in self.segments]}
        arrays = {k: getattr(self, k) for k in ("time_s", "eef_pos", "eef_quat", "body_pos", "body_quat",
                                                "entity_pos", "entity_quat")}
        return meta, arrays

    def content_hash(self) -> str:
        return _hash(*self._parts())

    def save(self, folder, stem: str = "v1") -> str:
        meta, arrays = self._parts()
        digest = _hash(meta, arrays)
        _write(Path(folder), stem, dict(meta, content_hash=digest), arrays)
        return digest

    @classmethod
    def load(cls, folder, stem: str = "v1") -> "Trajectory":
        meta, a = _read(Path(folder), stem)
        out = cls(demo=meta["demo"], control_dt_s=float(meta["control_dt_s"]), time_s=a["time_s"],
                  eef_pos=a["eef_pos"], eef_quat=a["eef_quat"], body_names=list(meta["body_names"]),
                  body_pos=a["body_pos"], body_quat=a["body_quat"], entity_names=list(meta["entity_names"]),
                  entity_pos=a["entity_pos"], entity_quat=a["entity_quat"], held=list(meta["held"]),
                  segments=[Segment.from_dict(s) for s in meta["segments"]])
        if out.content_hash() != meta["content_hash"]:
            raise ValueError("%s/%s: content hash mismatch (file edited or corrupted)" % (folder, stem))
        return out


@dataclasses.dataclass
class LayoutBody:
    """One non-robot body: collision boxes in the body frame and the body's world pose over time.

    ``pos`` / ``quat`` are ``(3,)`` / ``(4,)`` for a body that never moves, else ``(T, 3)`` / ``(T, 4)``."""
    label: str
    role: str
    box_centre: np.ndarray        # (K, 3) body frame
    box_rot: np.ndarray           # (K, 3, 3) body frame
    box_half: np.ndarray          # (K, 3)
    pos: np.ndarray
    quat: np.ndarray

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError("body %r: role %r is not one of %s" % (self.label, self.role, list(ROLES)))
        self.box_centre = np.asarray(self.box_centre, dtype=np.float64).reshape(-1, 3)
        self.box_rot = np.asarray(self.box_rot, dtype=np.float64).reshape(-1, 3, 3)
        self.box_half = np.asarray(self.box_half, dtype=np.float64).reshape(-1, 3)
        self.pos = np.asarray(self.pos, dtype=np.float64)
        self.quat = np.asarray(self.quat, dtype=np.float64)
        if self.pos.shape[:-1] != self.quat.shape[:-1] or self.pos.shape[-1] != 3 or self.quat.shape[-1] != 4:
            raise ValueError("body %r: pos %s and quat %s do not form a pose track" % (self.label, self.pos.shape,
                                                                                    self.quat.shape))

    @property
    def static(self) -> bool:
        return self.pos.ndim == 1

    def world_boxes_at(self, step: int):
        """``(centres (K, 3), rotations (K, 3, 3), halves (K, 3))`` at a control sample (clamped)."""
        if self.static:
            pos, quat = self.pos, self.quat
        else:
            k = min(max(int(step), 0), len(self.pos) - 1)
            pos, quat = self.pos[k], self.quat[k]
        return world_boxes(self.box_centre, self.box_rot, self.box_half, pos, quat_to_mat(quat))


@dataclasses.dataclass
class SceneLayout:
    """V2 of one demonstration's layout; ``n_steps`` control samples (the same timing as its V1)."""
    layout_id: str
    control_dt_s: float
    n_steps: int
    bodies: List[LayoutBody]
    support_surfaces: List[Dict[str, Any]]            # {name, min_m, max_m}
    workspace_measured_m: Dict[str, List[float]]      # {min, max}: robot + entity envelope of the task's demos

    def __post_init__(self) -> None:
        labels = [b.label for b in self.bodies]
        if len(set(labels)) != len(labels):
            raise ValueError("SceneLayout %s: duplicate body labels" % self.layout_id)
        for b in self.bodies:
            if not b.static and len(b.pos) != self.n_steps:
                raise ValueError("SceneLayout %s: body %r has %d poses for %d steps" % (self.layout_id, b.label,
                                                                                       len(b.pos), self.n_steps))

    def body(self, label: str) -> LayoutBody:
        for b in self.bodies:
            if b.label == label:
                return b
        raise KeyError(label)

    def by_role(self, *roles: str) -> List[LayoutBody]:
        return [b for b in self.bodies if b.role in roles]

    def boxes_at(self, step: int, roles: Optional[Sequence[str]] = None, exclude: Sequence[str] = ()):
        """All boxes of the bodies with a role in ``roles`` (all roles if ``None``), minus ``exclude`` labels,
        in the world at a control sample: ``(centres (N, 3), rotations (N, 3, 3), halves (N, 3), labels [N])``."""
        centres, rots, halves, labels = [], [], [], []
        for b in self.bodies:
            if (roles is not None and b.role not in roles) or b.label in exclude or not len(b.box_half):
                continue
            c, r, h = b.world_boxes_at(step)
            centres.append(c)
            rots.append(r)
            halves.append(h)
            labels.extend([b.label] * len(h))
        if not labels:
            return np.empty((0, 3)), np.empty((0, 3, 3)), np.empty((0, 3)), []
        return np.concatenate(centres), np.concatenate(rots), np.concatenate(halves), labels

    def _parts(self) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        meta = {"schema": SCHEMA, "kind": "layout", "layout_id": self.layout_id, "control_dt_s": self.control_dt_s,
                "n_steps": self.n_steps, "support_surfaces": self.support_surfaces,
                "workspace_measured_m": self.workspace_measured_m,
                "bodies": [{"label": b.label, "role": b.role, "static": b.static} for b in self.bodies]}
        arrays = {}
        for i, b in enumerate(self.bodies):
            for key in ("box_centre", "box_rot", "box_half", "pos", "quat"):
                arrays["b%d_%s" % (i, key)] = getattr(b, key)
        return meta, arrays

    def content_hash(self) -> str:
        return _hash(*self._parts())

    def save(self, folder, stem: str = "v2") -> str:
        meta, arrays = self._parts()
        digest = _hash(meta, arrays)
        _write(Path(folder), stem, dict(meta, content_hash=digest), arrays)
        return digest

    @classmethod
    def load(cls, folder, stem: str = "v2") -> "SceneLayout":
        meta, a = _read(Path(folder), stem)
        bodies = [LayoutBody(label=b["label"], role=b["role"], box_centre=a["b%d_box_centre" % i],
                             box_rot=a["b%d_box_rot" % i], box_half=a["b%d_box_half" % i], pos=a["b%d_pos" % i],
                             quat=a["b%d_quat" % i]) for i, b in enumerate(meta["bodies"])]
        out = cls(layout_id=meta["layout_id"], control_dt_s=float(meta["control_dt_s"]), n_steps=int(meta["n_steps"]),
                  bodies=bodies, support_surfaces=meta["support_surfaces"],
                  workspace_measured_m=meta["workspace_measured_m"])
        if out.content_hash() != meta["content_hash"]:
            raise ValueError("%s/%s: content hash mismatch (file edited or corrupted)" % (folder, stem))
        return out


@dataclasses.dataclass
class TrajectorySet:
    """V1 of a task: the successful demonstrations with their own layouts and the shared robot surface points.

    ``robot_points[body]`` are surface samples in the body's frame (``robot_point_spacing_m`` apart). The
    demonstrations are in coverage order (the order ``--max-demos`` truncates)."""
    task_id: str
    robot_points: Dict[str, np.ndarray]
    robot_point_spacing_m: float
    trajectories: List[Trajectory]
    layouts: Dict[str, SceneLayout]

    def __post_init__(self) -> None:
        for t in self.trajectories:
            if t.demo not in self.layouts:
                raise ValueError("TrajectorySet: no layout for demonstration %s" % t.demo)
            missing = [b for b in t.body_names if b not in self.robot_points]
            if missing:
                raise ValueError("TrajectorySet: demonstration %s has bodies without surface points: %s"
                                 % (t.demo, missing))

    def content_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(_canonical({"task_id": self.task_id, "spacing": self.robot_point_spacing_m}))
        digest.update(_hash({}, {k: self.robot_points[k] for k in self.robot_points}).encode("ascii"))
        for t in self.trajectories:
            digest.update(t.content_hash().encode("ascii"))
            digest.update(self.layouts[t.demo].content_hash().encode("ascii"))
        return digest.hexdigest()

    def first(self, n: Optional[int]) -> "TrajectorySet":
        keep = self.trajectories if n is None else self.trajectories[:n]
        return TrajectorySet(self.task_id, self.robot_points, self.robot_point_spacing_m, list(keep), self.layouts)

    def save_shared(self, folder) -> None:
        """``robot_points.npz`` and ``index.json`` (demonstrations and hashes); per-demo files are saved by
        :meth:`save_demo`."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(folder / "robot_points.npz"), **self.robot_points)

    def save_demo(self, folder, trajectory: Trajectory) -> Dict[str, str]:
        sub = Path(folder) / "demos" / trajectory.demo
        return {"v1": trajectory.save(sub), "v2": self.layouts[trajectory.demo].save(sub)}

    def save_index(self, folder, hashes: Dict[str, Dict[str, str]]) -> None:
        folder = Path(folder)
        index = {"schema": SCHEMA, "task_id": self.task_id, "robot_point_spacing_m": self.robot_point_spacing_m,
                 "demos": [t.demo for t in self.trajectories], "hashes": hashes}
        (folder / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True, allow_nan=False) + "\n")

    def save(self, folder) -> None:
        self.save_shared(folder)
        self.save_index(folder, {t.demo: self.save_demo(folder, t) for t in self.trajectories})

    @classmethod
    def load(cls, folder, max_demos: Optional[int] = None) -> "TrajectorySet":
        folder = Path(folder)
        index = json.loads((folder / "index.json").read_text())
        if index.get("schema") != SCHEMA:
            raise ValueError("%s/index.json has schema %r, expected %r" % (folder, index.get("schema"), SCHEMA))
        with np.load(str(folder / "robot_points.npz")) as data:
            points = {k: data[k] for k in data.files}
        demos = index["demos"] if max_demos is None else index["demos"][:max_demos]
        trajectories, layouts = [], {}
        for demo in demos:
            sub = folder / "demos" / demo
            t, layout = Trajectory.load(sub), SceneLayout.load(sub)
            if (t.content_hash(), layout.content_hash()) != (index["hashes"][demo]["v1"], index["hashes"][demo]["v2"]):
                raise ValueError("%s: demonstration %s does not match index.json" % (folder, demo))
            trajectories.append(t)
            layouts[demo] = layout
        return cls(index["task_id"], points, float(index["robot_point_spacing_m"]), trajectories, layouts)
