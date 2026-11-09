"""Probe: same-seed determinism of CALVIN episodes (BD ``determinism.same_seed_repro``, C8).

Runs the conformance motion (reset to the layout, replay ``--steps`` recorded frames as relative actions) several
times on one ``CalvinHost`` and compares every policy observation (``robot_obs``, ``scene_obs``, both camera
images) with the first run. Between the repeats it optionally runs another episode first (``--between``: a reset to
that layout and ``--steps`` hold actions), so that the repeats start from different simulator histories; with
``--drop-hazard X Y Z`` that episode also releases a new hazard body (a different size each time, so the hazard
pool grows) at that point to fall onto the scene, and ``--hazard`` allocates a parked hazard before the repeats.
``--no-rebuild`` replaces ``CalvinHost._rebuild_world`` by a no-op (the upstream ``env.reset`` alone);
``--restore bullet|storage`` then also restores the state saved after the first reset to a frame after every later
reset to it (``bullet``: PyBullet ``saveState`` / ``restoreState``; ``storage``: CALVIN's ``serialize`` /
``reset_from_storage``, through a pickle in a temporary directory).

    python -m hazardhosts.calvin.probes.determinism --egl-device 2 --task unstack_block --layout val/555000 \
        --steps 36 --repeats 3 --between lift_red_block_table:val/554085..554105
"""
import argparse
import os
import pickle
import sys
import tempfile
import time

import numpy as np

from hazardlib.spec.hazard_spec import AgentSpec

from hazardhosts.calvin.host import CalvinHost
from hazardhosts.calvin.policies import _scales, relative_actions


def _run(host, task, layout, actions):
    host.reset_episode(0, task, layout, 0)
    out = [host.policy_observation(0)]
    for action in actions:
        host.step(0, action)
        out.append(host.policy_observation(0))
    return [{k: np.array(v, copy=True) for k, v in obs.items()} for obs in out]


def _diff(a, b):
    worst = {}
    first = {}
    for k, (oa, ob) in enumerate(zip(a, b)):
        for key in oa:
            d = float(np.max(np.abs(oa[key].astype(np.float64) - ob[key].astype(np.float64))))
            if d > worst.get(key, 0.0):
                worst[key] = d
            if d > 0 and key not in first:
                first[key] = k
    return worst, first


def _sphere(radius):
    return AgentSpec.from_dict({"kind": "probe", "shape": "sphere", "half_extents_m": [radius], "mass_kg": 0.05})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--egl-device", type=int, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--steps", type=int, default=36)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--between", default=None, help="task:layout of an episode run before every repeat")
    parser.add_argument("--drop-hazard", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                        help="release a new hazard body at this point during every --between episode")
    parser.add_argument("--hazard", action="store_true", help="allocate a parked hazard before the repeats")
    parser.add_argument("--no-rebuild", action="store_true", help="skip the world rebuild in reset_episode")
    parser.add_argument("--restore", choices=["bullet", "storage"], default=None,
                        help="with --no-rebuild: save the state after the first reset to a frame and restore it "
                             "after every later reset to it; bullet: saveState / restoreState, storage: CALVIN's "
                             "serialize / reset_from_storage (BD snapshot_restore)")
    args = parser.parse_args()
    host = CalvinHost(egl_device=args.egl_device)
    if args.no_rebuild:
        host._rebuild_world = lambda: None
    if args.restore:
        if not args.no_rebuild:
            parser.error("--restore needs --no-rebuild (a rebuilt world invalidates saved states)")
        snapshots = {}
        reset_to = host._reset_to
        tmp = tempfile.mkdtemp(prefix="calvin_determinism_")

        def restoring_reset_to(split, frame):
            obs = reset_to(split, frame)
            if (split, frame) not in snapshots:
                if args.restore == "bullet":
                    snapshots[(split, frame)] = host.p.saveState(physicsClientId=host.cid)
                else:
                    snapshots[(split, frame)] = os.path.join(tmp, "%s_%d.pkl" % (split, frame))
                    data = host.env.serialize()
                    # reset_from_storage returns these keys of a recorded rollout file; serialize() has none
                    data.update(state_obs=None, done=False, info=None)
                    with open(snapshots[(split, frame)], "wb") as f:
                        pickle.dump(data, f)
                return obs
            if args.restore == "bullet":
                host.p.restoreState(stateId=snapshots[(split, frame)], physicsClientId=host.cid)
            else:
                host.env.reset_from_storage(snapshots[(split, frame)])
            return host.env.get_obs()
        host._reset_to = restoring_reset_to
    resets = []
    reset_episode = host.reset_episode

    def timed_reset(*a):
        t0 = time.perf_counter()
        reset_episode(*a)
        resets.append(time.perf_counter() - t0)
    host.reset_episode = timed_reset
    try:
        layout = host.dataset.layout(args.layout)
        actions = relative_actions(host.dataset, layout.split, layout.frame, args.steps, _scales(host.env))
        if args.hazard:
            host.alloc_hazard(0, _sphere(0.02))
        runs = []
        for i in range(args.repeats):
            if args.between:
                task, other = args.between.split(":", 1)
                host.reset_episode(0, task, other, 0)
                h = None
                if args.drop_hazard is not None:
                    h = host.alloc_hazard(0, _sphere(0.015 + 0.005 * i))
                    host.set_hazard_pose(h, args.drop_hazard, [1.0, 0.0, 0.0, 0.0])
                    host.set_hazard_visible(h, True)
                    host.release_to_physics(h, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
                for _ in range(args.steps):
                    host.step(0, host.hold_action(0))
                if h is not None:
                    print("between episode %d: hazard %s ended at %s" % (i, h.key, np.round(host.hazard_state(h).pos, 3)))
                    host.free_hazard(h)
            runs.append(_run(host, args.task, args.layout, actions))
            worst, first = _diff(runs[0], runs[i])
            print("repeat %d vs repeat 0: max |diff| %s; first differing control step %s"
                  % (i, {k: float("%.3g" % v) for k, v in sorted(worst.items())}, first))
            if i >= 2:
                worst, first = _diff(runs[1], runs[i])
                print("repeat %d vs repeat 1: max |diff| %s; first differing control step %s"
                      % (i, {k: float("%.3g" % v) for k, v in sorted(worst.items())}, first))
        print("reset_episode: %d calls, mean %.3f s, max %.3f s" % (len(resets), np.mean(resets), np.max(resets)))
    finally:
        host.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
