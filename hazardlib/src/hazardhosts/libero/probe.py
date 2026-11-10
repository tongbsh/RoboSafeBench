"""Read-only source inspection and real-engine H3 probes for original LIBERO."""
import argparse
import json
import os
from pathlib import Path

# Avoid a full BLAS thread team for the many tiny 3D transforms in G0.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

from .host import LiberoHost


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--replay", type=int, default=0)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--conformance", action="store_true")
    parser.add_argument("--select")
    parser.add_argument("--survey", action="store_true")
    parser.add_argument("--layout")
    parser.add_argument("--hold-survey", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--pilot-root")
    parser.add_argument("--tasks")
    parser.add_argument("--model")
    parser.add_argument("--checkpoint")
    parser.add_argument("--port", type=int)
    parser.add_argument("--client-src")
    parser.add_argument("--policy-smoke", action="store_true")
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--initial-compare", action="store_true")
    args = parser.parse_args()
    host = LiberoHost()
    output = Path(args.out)
    if args.suite != "libero_spatial":
        output = output / args.suite
    output.mkdir(parents=True, exist_ok=True)
    if args.initial_compare:
        host.close()
        from .parity import compare_initial_layouts
        indices = [args.task_index] if args.tasks is None else [int(i) for i in args.tasks.split(",")]
        compare_initial_layouts(output / "layouts", args.suite, indices, 3)
        return
    if args.parity:
        host.close()
        from .parity import compare_original
        indices = [args.task_index] if args.tasks is None else [int(i) for i in args.tasks.split(",")]
        for index in indices:
            compare_original(output / "parity", args.suite, index, 0)
        return
    if args.native:
        host.close()
        from .native import validate_native_loop
        validate_native_loop(Path(args.pilot_root) / "config", output / "native", 4, 10)
        return
    if args.prepare or args.evaluate:
        host.close()
        from .pilot import prepare, evaluate
        if not args.pilot_root:
            parser.error("--pilot-root is required")
        tasks = None if args.tasks is None else [int(i) for i in args.tasks.split(",")]
        if args.prepare:
            prepare(args.pilot_root, tasks)
        if args.evaluate:
            if not all((args.model, args.checkpoint, args.port, args.client_src)):
                parser.error("evaluation needs --model --checkpoint --port --client-src")
            evaluate(args.pilot_root, args.model, args.checkpoint, args.port, args.client_src, tasks)
        return
    if args.conformance:
        host.close()
        from hazardlib.conformance.suite import SuiteSession, run_suite
        from hazardlib.conformance.report import write_report
        session = SuiteSession.open("libero", [], None)
        try:
            select = None if args.select is None else args.select.split(",")
            report = run_suite(session, select)
            filename = "conformance.json" if args.select is None else "conformance_%s.json" % args.select.replace(",", "_")
            write_report(report, str(output / filename))
            for check in report["checks"]:
                print(check["id"], check["status"], check["reason"], flush=True)
            print("CONFORMANCE " + json.dumps(report["summary"]), flush=True)
            raise SystemExit(0 if report["verdict"] == "pass" else 1)
        finally:
            session.target.host.close()
    try:
        catalog = {suite: [t for t in host.task_ids() if t.startswith(suite + "/")]
                   for suite in host.config["suites"]}
        print(json.dumps({"suite_sizes": {k: len(v) for k, v in catalog.items()}}), flush=True)
        task = catalog[args.suite][args.task_index]
        host.reset_episode(0, task, args.layout or args.suite + "/demo_0", 0)
        info = {"task": task, "timebase": vars(host.time_state(0)), "gravity": host.m.opt.gravity.tolist(),
                "robot_groups": host.robot_groups, "pool": host.pool_table(),
                "surfaces": [s.to_dict() for s in host.support_surfaces(0)],
                "state_shapes": {k: list(v.shape) for k, v in host.policy_observation(0).items()},
                "eef": host.robot_state(0).eef_pose.pos.tolist(),
                "geometry_samples": {k: len(v) for k, v in host.robot_collision_points(0).items()}}
        print(json.dumps(info, indent=2), flush=True)
        (output / "scene.json").write_text(json.dumps(info, indent=2))
        (output / "catalog.json").write_text(json.dumps(catalog, indent=2))
        if args.policy_smoke:
            from .policies import DyniPolicyBridge
            from .video import Mp4FrameSink
            policy = DyniPolicyBridge(host, "configs/models/%s.yaml" % args.model, args.checkpoint,
                                      "127.0.0.1", args.port, args.client_src)
            sink = Mp4FrameSink(output / "smoke_video", 20, 18, "fast")
            sink.open("original-libero-%s-smoke" % args.model, {"pilot": True, "task": task})
            policy.reset(task, args.layout or args.suite + "/demo_0", 0)
            success = False
            try:
                for k in range(220):
                    obs = host.policy_observation(0)
                    sink.add("primary", obs["primary"], k * host.control_dt, None)
                    sink.add("wrist", obs["wrist"], k * host.control_dt, None)
                    action = policy.act(obs)
                    host.step(0, action)
                    if host.task_success(0):
                        success = True
                        break
                paths = sink.close()
                report = {"pilot": True, "task": task, "policy": policy.name,
                          "success": success, "steps": k + 1, "video": paths}
                (output / (args.model + "_smoke.json")).write_text(json.dumps(report, indent=2))
                print(json.dumps(report), flush=True)
            finally:
                policy.close()
        if args.hold_survey:
            actions = host._demo(task, args.suite).actions("demo_0")
            rows = []
            for count in (0, 2, 6, 14, 30, 50):
                host.reset_episode(0, task, args.suite + "/demo_0", 0)
                for a in actions[:count]:
                    host.step(0, a)
                speeds = []
                for _ in range(12):
                    speeds.append(float(np.linalg.norm(host.robot_state(0).eef_lin_vel_mps)))
                    host.step(0, host.hold_action(0))
                rows.append({"hold_at_s": count * host.control_dt, "speeds": speeds})
            (output / "hold_survey.json").write_text(json.dumps(rows, indent=2))
            print(json.dumps({"hold_survey": rows}), flush=True)
        if args.survey:
            host.reset_episode(0, task, args.layout or args.suite + "/demo_0", 0)
            h = host.alloc_hazard(0, host.pool[0].agent)
            mem = h.native
            host.set_hazard_visible(h, True)
            robot = host.robot_state(0)
            rows = []
            for name, centre in zip(robot.collision_body_names, robot.collision_body_pos):
                for axis in range(3):
                    for sign in (-1, 1):
                        normal = np.zeros(3)
                        normal[axis] = sign
                        def contacts(distance):
                            host.set_hazard_pose(h, centre + normal * distance, [1, 0, 0, 0])
                            host._collision(mem, True)
                            s = host._fresh()
                            host.mj.mj_collision(host.m, s)
                            out = []
                            for c in s.contact[:s.ncon]:
                                for g, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                                    if g in mem.geoms:
                                        out.append(host.label_of_body[int(host.m.geom_bodyid[other])])
                            host._collision(mem, False)
                            return sorted(set(out))
                        lo, hi = 0., .4
                        if name not in contacts(lo) or contacts(hi):
                            continue
                        for _ in range(24):
                            mid = (lo + hi) / 2
                            if contacts(mid):
                                lo = mid
                            else:
                                hi = mid
                        first = contacts(lo)
                        if first == [name]:
                            rows.append({"body": name, "direction": (-normal).tolist(),
                                         "radius_m": lo - host.pool[0].agent.half_extents_m[axis],
                                         "body_pos": centre.tolist(), "first": first})
            host.free_hazard(h)
            (output / "strike_survey.json").write_text(json.dumps(rows, indent=2))
            print(json.dumps({"strike_candidates": rows}), flush=True)
        runs = []
        for k in range(args.replay):
            layout = args.suite + "/demo_%d" % k
            host.reset_episode(0, task, layout, 0)
            actions = host._demo(task, args.suite).actions("demo_%d" % k)
            success_at = None
            eef = []
            for j, action in enumerate(actions):
                host.step(0, action)
                eef.append(host.robot_state(0).eef_pose.pos.copy())
                if host.task_success(0) and success_at is None:
                    success_at = j
            row = {"task": task, "layout": layout, "steps": len(actions), "success_at": success_at}
            runs.append(row)
            np.save(output / ("eef_demo_%d.npy" % k), np.asarray(eef))
            print(json.dumps(row), flush=True)
            (output / "replays.json").write_text(json.dumps(runs, indent=2))
    finally:
        host.close()


if __name__ == "__main__":
    main()
