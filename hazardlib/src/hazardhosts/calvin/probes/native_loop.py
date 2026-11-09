"""Probe: 01 §3.3 item 14 on CALVIN's native evaluation loop (D13 j): every subtask episode calls
``runtime.end_episode`` exactly once and ``hazard_events.jsonl`` has one line per episode; no hazard is left.

Runs :class:`hazardhosts.calvin.native.NativeLoop` (upstream ``evaluate_sequence`` / ``rollout``) with a hold
model (``hold_action`` every step: no subtask succeeds, each runs the full ``EP_LEN``):

* part A: upstream ``evaluate_sequence`` on ``--sequences`` initial conditions (each stops after its first, failed,
  subtask: one episode per sequence);
* part B: the wrapped ``rollout`` called for 5 subtasks in a row in one env without a reset, as
  ``evaluate_sequence`` chains them after successes.

Episodes alternate between control episodes and the hazard instances of ``--instances`` (a gated JSONL of
``hazardctl gate``, e.g. the pilot's). After every episode the probe checks that every hazard body is parked and
that neither the robot's nor any scene object's contact list names a hazard. Needs runtime.yaml / semantics.yaml
values: run with ``$HAZARDLIB_CONFIG_DIR`` set to an assembled pilot config (``hazardhosts.calvin.pilot assemble``).

    python -m hazardhosts.calvin.probes.native_loop --egl-device 2 --instances gated.jsonl --log events.jsonl
"""
import collections
import json
import os
import sys

from hazardhosts.calvin.probes.common import probe_args

INITIAL_STATES = [
    {"led": 0, "lightbulb": 0, "slider": "right", "drawer": "closed", "red_block": "table",
     "blue_block": "slider_right", "pink_block": "slider_left", "grasped": 0},
    {"led": 1, "lightbulb": 1, "slider": "left", "drawer": "open", "red_block": "slider_left",
     "blue_block": "table", "pink_block": "table", "grasped": 0},
    {"led": 0, "lightbulb": 1, "slider": "left", "drawer": "closed", "red_block": "table",
     "blue_block": "table", "pink_block": "slider_right", "grasped": 0},
]
SEQUENCES = [
    ["open_drawer", "push_red_block_left", "turn_on_led", "lift_red_block_table", "place_in_slider"],
    ["turn_off_lightbulb", "close_drawer", "lift_blue_block_table", "stack_block", "move_slider_right"],
    ["move_slider_right", "turn_on_led", "rotate_red_block_left", "push_blue_block_right", "unstack_block"],
]
CHAIN = ["turn_on_lightbulb", "open_drawer", "lift_pink_block_table", "move_slider_left", "rotate_blue_block_right"]


class HoldModel:
    """CalvinBaseModel interface (``reset`` / ``step``): the host's ``hold_action`` every step."""

    def __init__(self, host):
        self.host = host

    def reset(self):
        pass

    def step(self, obs, goal):
        return self.host.hold_action(0)


def main() -> int:
    parser = probe_args(__doc__)
    parser.add_argument("--instances", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--sequences", type=int, default=3)
    args = parser.parse_args()
    from hazardlib.pipeline_io import read_records
    from hazardlib.runtime import HazardRuntime, Provenance, RuntimeConfig
    from hazardhosts.calvin.host import CalvinHost
    from hazardhosts.calvin.native import CALVIN_COMMIT, NativeLoop
    from hazardhosts.calvin.env import CALVIN_ENV_COMMIT

    if os.path.exists(args.log):
        os.remove(args.log)
    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    host = CalvinHost(dataset_dir=args.dataset_dir, egl_device=args.egl_device)
    instances, _ = read_records(args.instances)
    bd = host.describe()
    provenance = Provenance(bd_hash=bd.hash(), adapter_version=host.config["adapter_version"], manifest_hash=None,
                            benchmark_commits={"calvin": CALVIN_COMMIT, "calvin_env": CALVIN_ENV_COMMIT})
    runtime = HazardRuntime(host, RuntimeConfig.load(), provenance)
    queue = []
    for inst in instances:
        queue += [None, (inst, None)]
    picks = iter(queue)
    loop = NativeLoop(host, runtime, args.log, arm_next=lambda subtask: next(picks))
    hazard_uids = None
    residue = []

    def check_residue(where):
        uids = {h.uid for h in host._hazards}
        unparked = [h.key for h in host._hazards if not h.parked]
        info = host.env.get_info()
        named = [c for c in info["robot_info"]["contacts"] if c[2] in uids]
        for group in ("movable_objects", "fixed_objects"):
            for name, obj in info["scene_info"][group].items():
                named += [(name, c) for c in obj.get("contacts", []) if c[2] in uids]
        if unparked or named:
            residue.append((where, unparked, named))

    original = loop.rollout

    def rollout(*a, **k):
        ok = original(*a, **k)
        check_residue("after episode %d (%s)" % (loop.episodes, a[3]))
        return ok

    loop.upstream["rollout"] = rollout
    model = HoldModel(host)
    annotations = {task: [task.replace("_", " ")] for task in host.oracle.tasks}
    try:
        print("== part A: upstream evaluate_sequence, %d sequences" % args.sequences)
        for state, sequence in list(zip(INITIAL_STATES, SEQUENCES))[:args.sequences]:
            before = loop.episodes
            n = loop.evaluate_sequence(model, host.oracle, state, sequence, annotations)
            print("  sequence %s -> %d subtasks succeeded, %d episode(s)" % (sequence[0], n, loop.episodes - before))
        print("== part B: %d chained subtasks without a reset" % len(CHAIN))
        for subtask in CHAIN:
            rollout(host.env, model, host.oracle, subtask, annotations, collections.defaultdict(list), False)
    finally:
        host.close()
    with open(args.log) as fh:
        lines = [json.loads(line) for line in fh if line.strip()]
    for result, line in zip(loop.results, lines):
        print("  %-26s steps %3d upstream_success %-5s log task_success %-5s log %s" % (
            result["subtask"], result["steps"], result["upstream_success"], line["task_success"], line["instance_id"]))
    agree = all(r["upstream_success"] == line["task_success"] for r, line in zip(loop.results, lines))
    print("episodes begun %d, end_episode calls %d, hazard_events.jsonl lines %d" % (loop.episodes, loop.ended,
                                                                                   len(lines)))
    print("upstream verdict == log task_success in every episode: %s" % agree)
    print("hazard residue after an episode: %s" % (residue or "none"))
    ok = loop.episodes == loop.ended == len(lines) and agree and not residue
    print("ITEM 14 %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
