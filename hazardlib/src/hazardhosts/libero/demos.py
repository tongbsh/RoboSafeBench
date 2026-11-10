"""Original LIBERO layouts and joint-name state mapping.

``<suite>/demo_<k>[@<step>]`` selects an official demonstration state;
``init/<k>`` selects the original benchmark's evaluation initial state.
The host appends hazard joints to the original MJCF. Both kinds of original
state are mapped by joint name; only the added hazard joints remain unfilled
and are parked by the host. There are no SafeLIBERO obstacle or pedestal DOFs.

Demonstration states[k] precedes action k, while obs[k] follows it.
"""
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

JOINT_SIZES = {"free": (7, 6), "ball": (4, 3), "hinge": (1, 1), "slide": (1, 1)}
_DEMO = re.compile(r"^(?P<set>[A-Za-z0-9_]+)/demo_(?P<k>[0-9]+)(@(?P<j>[0-9]+))?$")
_INIT = re.compile(r"^init/(?P<k>[0-9]+)$")


class Layout(object):
    """A parsed layout id. ``kind`` is ``demo`` or ``init``; ``start`` the demonstration step (``demo`` only)."""

    def __init__(self, layout_id: str, kind: str, demo_set: Optional[str], index: int, start: int) -> None:
        self.layout_id = layout_id
        self.kind = kind
        self.demo_set = demo_set
        self.index = index
        self.start = start

    @property
    def episode(self) -> str:
        return "demo_%d" % self.index


def parse_layout(layout_id: str) -> Layout:
    if not isinstance(layout_id, str):
        raise ValueError("layout_id must be a string, got %r" % (layout_id,))
    m = _DEMO.match(layout_id)
    if m:
        return Layout(layout_id, "demo", m.group("set"), int(m.group("k")), int(m.group("j") or 0))
    m = _INIT.match(layout_id)
    if m:
        return Layout(layout_id, "init", None, int(m.group("k")), 0)
    raise ValueError("layout_id %r is neither '<demo_set>/demo_<k>[@<step>]' nor 'init/<k>'" % (layout_id,))


def plain_joint_layout(model_xml: str) -> List[Tuple[str, str, int, int]]:
    """``(name, type, nq, nv)`` of every joint of an MJCF string, in compile order (document order). A joint
    without ``type`` is a hinge unless a ``<default>`` sets the joint type, which is refused (it would make the
    document order ambiguous to read without compiling)."""
    root = ET.fromstring(model_xml)
    for default in root.iter("default"):
        for joint in default.iter("joint"):
            if "type" in joint.attrib:
                raise ValueError("model_file sets a default joint type; the plain joint layout cannot be read")
    out = []
    for el in root.find("worldbody").iter():
        if el.tag == "freejoint":
            kind = "free"
        elif el.tag == "joint":
            kind = el.attrib.get("type", "hinge")
        else:
            continue
        if kind not in JOINT_SIZES:
            raise ValueError("unsupported joint type %r of joint %r" % (kind, el.attrib.get("name")))
        name = el.attrib.get("name")
        if not name:
            raise ValueError("an unnamed joint in model_file: cannot map it by name")
        nq, nv = JOINT_SIZES[kind]
        out.append((name, kind, nq, nv))
    return out


class DemoFile(object):
    """One official ``*_demo.hdf5`` (opened read-only, lazily)."""

    def __init__(self, path: Path) -> None:
        import h5py
        if not path.is_file():
            raise FileNotFoundError("demonstration file %s does not exist" % path)
        self.path = path
        self._h5 = h5py.File(str(path), "r")
        self._layouts = {}  # type: Dict[str, List[Tuple[str, str, int, int]]]

    def episodes(self) -> List[str]:
        return sorted(self._h5["data"].keys(), key=lambda s: int(s.split("_")[1]))

    def episode(self, name: str) -> Any:
        if name not in self._h5["data"]:
            raise KeyError("%s has no episode %r" % (self.path.name, name))
        return self._h5["data"][name]

    def states(self, name: str) -> np.ndarray:
        return np.asarray(self.episode(name)["states"], dtype=np.float64)

    def actions(self, name: str) -> np.ndarray:
        return np.asarray(self.episode(name)["actions"], dtype=np.float64)

    def language(self) -> str:
        import json
        return str(json.loads(self._h5["data"].attrs["problem_info"])["language_instruction"])

    def joint_layout(self, name: str) -> List[Tuple[str, str, int, int]]:
        if name not in self._layouts:
            layout = plain_joint_layout(self.episode(name).attrs["model_file"])
            width = self.episode(name)["states"].shape[1]
            nq = sum(j[2] for j in layout)
            nv = sum(j[3] for j in layout)
            if 1 + nq + nv != width:
                raise ValueError("%s %s: joints of model_file give 1 + %d + %d values, states have %d"
                                 % (self.path.name, name, nq, nv, width))
            self._layouts[name] = layout
        return self._layouts[name]

    def close(self) -> None:
        self._h5.close()


def map_plain_state(plain_state: np.ndarray, plain_layout: List[Tuple[str, str, int, int]],
                    target_joints: Dict[str, Tuple[int, int, int, int]], target_width: int) -> Tuple[np.ndarray, List[str]]:
    """Copy ``plain_state`` (``[time, qpos, qvel]`` of the plain scene) into a zero state of the target scene by
    joint name. ``target_joints`` maps a joint name to ``(qposadr, nq, dofadr, nv)`` of the target model and
    ``target_width`` is ``1 + nq + nv`` of the target. Returns the state and the target joints that were not
    filled (the caller places them)."""
    plain_state = np.asarray(plain_state, dtype=np.float64)
    nq_plain = sum(j[2] for j in plain_layout)
    nv_plain = sum(j[3] for j in plain_layout)
    if plain_state.shape != (1 + nq_plain + nv_plain,):
        raise ValueError("original state width does not match its joint layout")
    nq_target = sum(v[1] for v in target_joints.values())
    out = np.zeros(target_width, dtype=np.float64)
    out[0] = plain_state[0]
    qpos_at, qvel_at = 1, 1 + nq_plain
    filled = set()
    for name, _kind, nq, nv in plain_layout:
        if name not in target_joints:
            raise KeyError("joint %r of the demonstration scene does not exist in the augmented LIBERO scene" % name)
        qadr, tnq, dadr, tnv = target_joints[name]
        if (tnq, tnv) != (nq, nv):
            raise ValueError("joint %r: sizes %s in the demonstration, %s in the augmented LIBERO scene"
                             % (name, (nq, nv), (tnq, tnv)))
        out[1 + qadr:1 + qadr + nq] = plain_state[qpos_at:qpos_at + nq]
        out[1 + nq_target + dadr:1 + nq_target + dadr + nv] = plain_state[qvel_at:qvel_at + nv]
        qpos_at += nq
        qvel_at += nv
        filled.add(name)
    return out, sorted(set(target_joints) - filled)
