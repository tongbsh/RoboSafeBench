"""Probe: isolation of CALVIN's state vectors and task oracle from the hazard body (H1 item 3, R5, C7).

Every variant resets to the same layout (default: the conformance ``still`` scenario, red block lifted in the
gripper), sends ``hold_action`` for ``--steps`` control steps and records after every step ``robot_obs``,
``scene_obs``, the full ``env.get_info()`` (``robot_info`` and ``scene_info``, contact lists included) and the
oracle's verdict for every task (``Tasks.get_task_info(start_info, info)``). Each variant is compared with the
hazard-free reference run, value by value:

``parked``    a hazard allocated and parked for the whole run;
``ghost``     a kinematic hazard at the conformance free point (visible, collision-filtered);
``fall``      a hazard released at rest on the drop column (lands on the table, touches nothing else);
``robot``     a hazard released at rest above the TCP: it lands on the arm, so ``robot_info['contacts']`` lists it;
``block``     a hazard thrown at the held block from -y (``scenario.held_strike``): it touches the block.

For the touching variants it reports the steps at which the hazard appears in the contact lists and the tasks
whose verdict differs from the reference. Finally it parks the hazard of ``robot`` while it is in contact and
reads ``get_info()`` again before any physics step: contact lists are those of the last ``stepSimulation``, so a
just-parked hazard can still be listed (``stale``), which matters where an oracle start state is read right after
an episode end without a physics step in between (``evaluate_policy.rollout``).

    python -m hazardhosts.calvin.probes.oracle_isolation --egl-device 2 --steps 45
"""
import argparse
import sys

import numpy as np

from hazardlib.config import load_config
from hazardlib.spec.hazard_spec import AgentSpec

from hazardhosts.calvin.host import HOST_CONFIG, CalvinHost


def _flatten(value, prefix=""):
    out = {}
    if isinstance(value, dict):
        for k in sorted(value, key=str):
            out.update(_flatten(value[k], "%s/%s" % (prefix, k)))
    elif isinstance(value, (list, tuple)):
        out[prefix + "#len"] = len(value)
        for i, v in enumerate(value):
            out.update(_flatten(v, "%s[%d]" % (prefix, i)))
    elif isinstance(value, np.ndarray):
        out.update(_flatten(value.tolist(), prefix))
    else:
        out[prefix] = value
    return out


def _contact_uids(info):
    robot = {c[2] for c in info["robot_info"]["contacts"]}
    objects = {name: {c[2] for c in obj["contacts"]} for name, obj in info["scene_info"]["movable_objects"].items()}
    return robot, objects


def _run(host, task, layout, steps, place=None):
    """``place(host)`` puts the hazard in after the reset (``None``: hazard-free); returns the records and the
    hazard handle."""
    host.reset_episode(0, task, layout, 0)
    start = host._start_info
    handle = place(host) if place is not None else None
    rows = []
    for _ in range(steps):
        host.step(0, host.hold_action(0))
        obs = host.policy_observation(0)
        info = host.env.get_info()
        rows.append({"robot_obs": np.array(obs["robot_obs"]), "scene_obs": np.array(obs["scene_obs"]),
                     "info": _flatten(info), "contacts": _contact_uids(info),
                     "oracle": sorted(host.oracle.get_task_info(start, info))})
    return rows, handle


def _compare(ref, rows):
    out = {"robot_obs": 0.0, "scene_obs": 0.0, "info_keys": set(), "oracle_steps": [], "oracle_tasks": set()}
    for k, (a, b) in enumerate(zip(ref, rows)):
        out["robot_obs"] = max(out["robot_obs"], float(np.max(np.abs(a["robot_obs"] - b["robot_obs"]))))
        out["scene_obs"] = max(out["scene_obs"], float(np.max(np.abs(a["scene_obs"] - b["scene_obs"]))))
        keys = set(a["info"]).union(b["info"])
        out["info_keys"] |= {key for key in keys if a["info"].get(key) != b["info"].get(key)}
        if a["oracle"] != b["oracle"]:
            out["oracle_steps"].append(k)
            out["oracle_tasks"] |= set(a["oracle"]) ^ set(b["oracle"])
    return out


def _report(name, ref, rows, uid):
    diff = _compare(ref, rows)
    robot_steps = [k for k, r in enumerate(rows) if uid is not None and uid in r["contacts"][0]]
    object_steps = {}
    for k, r in enumerate(rows):
        for obj, uids in r["contacts"][1].items():
            if uid is not None and uid in uids:
                object_steps.setdefault(obj, []).append(k)
    sections = sorted({"/".join(part.split("[")[0] for part in key.split("/")[1:4]) for key in diff["info_keys"]})
    print("%-7s robot_obs max|diff| %.3g  scene_obs max|diff| %.3g  get_info: %d differing values in %s"
          % (name, diff["robot_obs"], diff["scene_obs"], len(diff["info_keys"]), sections))
    print("        hazard in robot_info['contacts'] at steps %s; in object contacts %s"
          % (_span(robot_steps), {o: _span(s) for o, s in sorted(object_steps.items())}))
    print("        oracle verdicts differ at %d step(s) %s, tasks %s"
          % (len(diff["oracle_steps"]), _span(diff["oracle_steps"]), sorted(diff["oracle_tasks"])))


def _span(steps):
    if not steps:
        return "none"
    return "%d..%d (%d)" % (steps[0], steps[-1], len(steps))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--egl-device", type=int, required=True)
    parser.add_argument("--steps", type=int, default=45)
    parser.add_argument("--radius", type=float, default=0.02)
    args = parser.parse_args()
    scenario = load_config(HOST_CONFIG, strict=True)["scenario"]
    task, layout = scenario["still"]["task_id"], scenario["still"]["layout_id"]
    agent = AgentSpec.from_dict({"kind": "probe", "shape": "sphere", "half_extents_m": [args.radius],
                                 "mass_kg": 0.05})
    host = CalvinHost(egl_device=args.egl_device)
    try:
        def put(pos, lin=(0.0, 0.0, 0.0), release=True):
            def place(h):
                handle = h.alloc_hazard(0, agent)
                h.set_hazard_pose(handle, list(pos), [1.0, 0.0, 0.0, 0.0])
                h.set_hazard_visible(handle, True)
                if release:
                    h.release_to_physics(handle, list(lin), [0.0, 0.0, 0.0])
                return handle
            return place

        def parked(h):
            return h.alloc_hazard(0, agent)

        ref, _ = _run(host, task, layout, args.steps)
        print("reference (%s, %s): oracle at the last step %s; robot contacts at the last step %s"
              % (task, layout, ref[-1]["oracle"], sorted(ref[-1]["contacts"][0])))
        host.reset_episode(0, task, layout, 0)
        tcp = np.array(host.robot_state(0).eef_pose.pos)
        block = np.array(host.object_poses(0)[scenario["held_strike"]["body"]].pos)
        strike = scenario["held_strike"]
        direction = np.array(strike["direction"], dtype=float)
        variants = [
            ("parked", parked),
            ("ghost", put(scenario["free_point_m"], release=False)),
            ("fall", put(scenario["drop_point_m"])),
            ("robot", put(tcp + np.array([0.0, 0.0, 0.40]))),
            ("block", put(block - direction * (strike["radius_m"] + args.radius + 0.005), lin=direction * 0.2)),
        ]
        for name, place in variants:
            rows, handle = _run(host, task, layout, args.steps, place)
            _report(name, ref, rows, None if handle is None else host.hazard_uid(handle))
            if handle is not None:
                host.free_hazard(handle)
            if name == "robot":
                host.reset_episode(0, task, layout, 0)
                handle = place(host)
                uid = host.hazard_uid(handle)
                listed = None
                for k in range(args.steps):
                    host.step(0, host.hold_action(0))
                    if uid in _contact_uids(host.env.get_info())[0]:
                        listed = k
                        break
                if listed is None:
                    print("        stale: the hazard never touched the robot, not tested")
                    continue
                host.free_hazard(handle)
                before = uid in _contact_uids(host.env.get_info())[0]
                host.p.performCollisionDetection(physicsClientId=host.cid)
                after_cd = uid in _contact_uids(host.env.get_info())[0]
                print("        stale: parked at step %d while touching the robot; get_info() right after parking "
                      "lists it: %s; after performCollisionDetection: %s" % (listed, before, after_cd))
    finally:
        host.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
