"""Task roles of CALVIN, read statically from the oracle's task configuration (R3 source ``oracle_config``, H1 7).

``new_playtable_tasks.yaml`` declares every task as ``[method, *args]`` of ``calvin_env.envs.tasks.Tasks``, which
binds them with ``functools.partial``. The arguments are matched to the method's parameters
(``inspect.signature``) and every argument that names one scene entity becomes a role named after its parameter:

* a movable object (``block_red``) -> itself;
* a door (its name is the joint's) or a light -> ``<body>/<link>`` of the link the joint moves / the light is on;
* a body followed by the name of one of its links (``'table', 'drawer_link'``) -> ``<body>/<link>`` (the role is
  the body's parameter).

Entity names are keys of :meth:`CalvinHost.object_poses`. A list argument (``push_object_into``: any of three
blocks) names no single entity, and some tasks have no entity argument at all (``stack_objects``); both are left to
the probe's grasped / moved discovery and listed under ``unresolved``. No task name appears here (R2).

    python -m hazardhosts.calvin.roles --egl-device 2 --out-dir <dir> TASK [TASK ...]

writes ``<dir>/<task>.yaml`` (the ``hazardctl probe --entities`` file) and prints what each argument became.
"""
import argparse
import functools
import inspect
import os
import sys
from typing import Any, Dict, List, Mapping, Sequence, Tuple

ORACLE_SOURCE = "oracle_config"
ORACLE_PARAMS = ("start_info", "end_info")


class Scene:
    """What the oracle's arguments can name: movable objects, bodies with their links, doors and lights."""

    def __init__(self, movable: Sequence[str], links: Mapping[str, Sequence[str]], joints: Mapping[str, str],
                 lights: Mapping[str, str]) -> None:
        self.movable = set(movable)
        self.links = {body: set(names) for body, names in links.items()}
        self.joints = dict(joints)
        self.lights = dict(lights)

    @classmethod
    def of_host(cls, host: Any) -> "Scene":
        scene, bodies = host.env.scene, host.bodies
        links = {}  # type: Dict[str, List[str]]
        for obj in scene.fixed_objects:
            links[obj.name] = [name for (uid, _), name in bodies.link_name.items() if uid == obj.uid]
        joints = {door.name: "%s/%s" % (bodies.body_name[door.uid], bodies.link_name[(door.uid, door.joint_index)])
                  for door in scene.doors}
        lights = {light.name: "%s/%s" % (bodies.body_name[light.uid], light.link) for light in scene.lights}
        return cls([obj.name for obj in scene.movable_objects], links, joints, lights)


def bind(task: functools.partial) -> List[Tuple[str, Any]]:
    """(parameter, value) of the configured arguments of one oracle task, in order."""
    params = [name for name in inspect.signature(task.func).parameters if name not in ORACLE_PARAMS]
    if len(task.args) > len(params):
        raise ValueError("oracle task %s has %d arguments for parameters %s" % (task.func.__name__, len(task.args),
                                                                              params))
    return list(zip(params, task.args))


def oracle_entities(task: functools.partial, scene: Scene) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """``({role: {"name", "source"}}, [unresolved arguments])`` of one oracle task (see the module docstring)."""
    bound = bind(task)
    entities = {}  # type: Dict[str, Dict[str, str]]
    unresolved = []  # type: List[str]
    i = 0
    while i < len(bound):
        param, value = bound[i]
        i += 1
        if isinstance(value, str):
            if value in scene.movable:
                name = value
            elif value in scene.joints:
                name = scene.joints[value]
            elif value in scene.lights:
                name = scene.lights[value]
            elif value in scene.links and i < len(bound) and bound[i][1] in scene.links[value]:
                name = "%s/%s" % (value, bound[i][1])
                i += 1
            else:
                unresolved.append("%s=%r" % (param, value))
                continue
            entities[param] = {"name": name, "source": ORACLE_SOURCE}
        elif isinstance(value, (list, tuple)) or type(value).__name__ == "ListConfig":
            unresolved.append("%s=%s (one of several)" % (param, list(value)))
    return entities, unresolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tasks", nargs="+")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--egl-device", type=int, default=None)
    parser.add_argument("--dataset-dir", default=None)
    args = parser.parse_args()
    from hazardlib.pipeline_io import write_yaml
    from hazardhosts.calvin.host import CalvinHost
    host = CalvinHost(dataset_dir=args.dataset_dir, egl_device=args.egl_device)
    try:
        scene = Scene.of_host(host)
        os.makedirs(args.out_dir, exist_ok=True)
        for name in args.tasks:
            task = host.oracle.tasks[name]
            entities, unresolved = oracle_entities(task, scene)
            write_yaml(os.path.join(args.out_dir, "%s.yaml" % name), entities)
            print("%-26s %-18s %s  unresolved: %s" % (name, task.func.__name__,
                                                       {r: e["name"] for r, e in sorted(entities.items())},
                                                       unresolved or "-"))
    finally:
        host.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
