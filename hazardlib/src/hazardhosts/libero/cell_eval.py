"""Evaluate a policy on the generated paradigm cells of the LIBERO batch (L5).

    python -m hazardhosts.libero.cell_eval --gen-root _logs/libero40_gen20 --out <dir> \\
        --tasks all|libero_spatial:0,libero_goal:0 --layouts-per-task 5 --model pi05 \\
        --checkpoint <pi05_libero> --client-src <openpi-client/src> --servers 18010,18011 --workers 8 --seed 0 \\
        [--video none|first-layout|all] [--gpu 0] [--dry-run | --report-only]

Run from the simulator interpreter with ``PYTHONPATH`` / ``LIBERO_CONFIG_PATH`` / ``MUJOCO_GL`` ... set. The OpenPI
servers (``hazardhosts.libero.serve_openpi``, seeded protocol) must already listen on ``--servers``.

What it does:

1. *select* per task the first ``--layouts-per-task`` layouts (coverage order) with an admitted instance
   (``cell_select``); a task whose generation is unfinished before that point is left out and listed;
2. build ONE global queue of episodes: every admitted instance of the selected layouts (all cells) plus the controls
   the core planner requires per layout (hazard-free, same seed protocol, shared exactly as in the run stage);
   episodes that already have a record are not queued (resume at episode granularity);
3. run the queue over ``--workers`` simulator processes (``CUDA_VISIBLE_DEVICES`` / ``MUJOCO_EGL_DEVICE_ID`` = ``--gpu``,
   default 0), worker k bound to ``servers[k % len(servers)]``. A worker runs one episode at a time through the core
   ``run_stage`` (``only_episode_ids``) on a scratch directory and appends the single record to
   ``<out>/runs/<task>/<policy>/episodes.jsonl`` under a file lock, so the core ``load_runs`` / ``summarize`` /
   ``write_report`` read the task directories unchanged. A failed or crashed episode is retried once (the worker is
   restarted); a second failure is written to ``<out>/failed.jsonl`` and skipped;
4. write ``<out>/report/`` (``cell_report.md|json`` per paradigm cell, ``core/`` = the core report over all runs).

Sigma: the calibration (``T_stop``) is per layout, whereas the core run identity holds one ``calibration_hash`` per
run directory, so the run is made without a calibration (records carry ``sigma: null``) and the cell report uses the
instances' achieved Sigma recorded by the generator (computed from the layout's own, uncensored calibration). The
generator never targets Sigma from a censored ``T_stop``, and ``allow_censored_t_stop`` stays false.

Files: ``eval.log`` (one line per finished episode), ``selection.json``, ``config/`` (assembled pilot config),
``workers/w<k>.log`` (simulator / policy stderr), ``failed.jsonl`` (failures of the latest invocation only; the
previous one is archived as ``failed.<timestamp>.jsonl``).

Topple: entities of interest that are not bodies (e.g. ``wooden_cabinet_1_middle_region``) have no pose; they are
skipped, the others are monitored, and every record lists the skipped names (``task_entities_skipped``; also in
``runs/<task>/<policy>/topple_skipped_entities.txt``).
"""
import argparse
import fcntl
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from .cell_select import (Work, build_queue, expected_counts, parse_task_spec, select_all, selection_document,
                          task_name)

PROTOCOL_TAG = "@@EVAL "
VIDEO_MODES = ("none", "first-layout", "all")
RESTART_LIMIT = 30
SKIPPED_ENTITIES_FILE = "topple_skipped_entities.txt"


# ------------------------------------------------------------------------------------------------ orchestrator

def run_dir(out: Path, task: str, policy: str) -> Path:
    return out / "runs" / task / policy


def model_policy_name(model: str, models_dir: Path) -> str:
    return yaml.safe_load((models_dir / (model + ".yaml")).read_text())["name"]


def done_ids(out: Path, selected: Dict[str, Any], policy: str, seed: int) -> Dict[str, Set[str]]:
    """Episode ids with a record per task (a torn last line from a killed worker is repaired here, before any
    worker exists)."""
    from hazardlib.run.runner import read_episode_records
    done = {}
    for task in selected:
        records = read_episode_records(run_dir(out, task, policy) / "episodes.jsonl", repair=True)
        done[task] = {r["episode_id"] for r in records if r["policy_name"] == policy and r["seed"] == seed}
    return done


class Progress(object):
    def __init__(self, total: int, log_path: Path) -> None:
        self.total, self.finished, self.failed = total, 0, 0
        self.start = time.time()
        self.lock = threading.Lock()
        self.log = open(str(log_path), "a")

    def say(self, line: str) -> None:
        with self.lock:
            self.log.write("%s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), line))
            self.log.flush()
        print(line, flush=True)

    def episode(self, work: Work, worker: int, reply: Dict[str, Any]) -> None:
        with self.lock:
            self.finished += 1
            n, elapsed = self.finished, time.time() - self.start
        eta = elapsed / n * (self.total - n)
        self.say("DONE %d/%d %s %s %s %s seconds=%.1f steps=%s success=%s contact=%s worker=%d rss_mb=%s elapsed=%.0f "
                 "eta=%.0f" % (n, self.total, work.task, work.demo, work.kind, work.episode_id.split(".libero")[0],
                               reply["seconds"], reply["steps"], reply["success"], reply["contact"], worker,
                               reply["rss_mb"], elapsed, eta))


class WorkerProcess(object):
    def __init__(self, index: int, command: List[str], env: Dict[str, str], log_path: Path) -> None:
        self.index, self.command, self.env, self.log_path = index, command, env, log_path
        self.process = None  # type: Optional[subprocess.Popen]
        self.restarts = 0

    def start(self, timeout_s: float = 900.0) -> None:
        log = open(str(self.log_path), "a")
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                        env=self.env, universal_newlines=True, bufsize=1)
        deadline = time.time() + timeout_s
        while True:
            reply = self._read(deadline)
            if reply is not None and reply.get("ready"):
                return
            if reply is None:
                raise RuntimeError("worker %d did not become ready (see %s)" % (self.index, self.log_path))

    def _read(self, deadline: Optional[float] = None) -> Optional[Dict[str, Any]]:
        while True:
            line = self.process.stdout.readline()
            if not line:
                return None
            if line.startswith(PROTOCOL_TAG):
                return json.loads(line[len(PROTOCOL_TAG):])

    def request(self, work: Work) -> Optional[Dict[str, Any]]:
        """The worker's reply, or ``None`` when the worker died."""
        try:
            self.process.stdin.write(json.dumps(work.to_dict()) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            return None
        return self._read()

    def stop(self) -> None:
        if self.process is None:
            return
        try:
            self.process.stdin.close()
            self.process.wait(timeout=60)
        except Exception:
            self.process.kill()
        self.process = None

    def kill(self) -> None:
        if self.process is not None:
            self.process.kill()
            self.process.wait()
            self.process = None


def worker_command(args: argparse.Namespace, index: int, port: int) -> List[str]:
    return [sys.executable, "-u", "-m", "hazardhosts.libero.cell_eval", "--worker", "--worker-index", str(index),
            "--gen-root", str(args.gen_root), "--out", str(args.out), "--model", args.model,
            "--checkpoint", args.checkpoint, "--client-src", args.client_src, "--port", str(port),
            "--seed", str(args.seed), "--video", args.video]


def worker_env(gpu: int) -> Dict[str, str]:
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu))


def drive(args: argparse.Namespace, work_items: List[Work], progress: Progress, ports: List[int]) -> None:
    pending = queue.Queue()
    for item in work_items:
        pending.put(item)
    out = Path(args.out)
    (out / "workers").mkdir(exist_ok=True)
    workers = [WorkerProcess(k, worker_command(args, k, ports[k % len(ports)]), worker_env(args.gpu),
                             out / "workers" / ("w%d.log" % k)) for k in range(args.workers)]
    failed_lock = threading.Lock()

    def loop(worker: WorkerProcess) -> None:
        try:
            worker.start()
        except Exception as exc:
            progress.say("WORKER %d failed to start: %s" % (worker.index, exc))
            return
        while True:
            try:
                work = pending.get_nowait()
            except queue.Empty:
                break
            reply = worker.request(work)
            if reply is not None and reply.get("ok"):
                progress.episode(work, worker.index, reply)
                continue
            reason = "worker died" if reply is None else reply.get("error", "error")
            work.attempt += 1
            if reply is None or reply.get("restart"):
                worker.kill()
            if work.attempt <= 1:
                progress.say("RETRY %s %s %s: %s" % (work.task, work.demo, work.episode_id, reason))
                pending.put(work)
            else:
                progress.say("FAIL %s %s %s: %s" % (work.task, work.demo, work.episode_id, reason))
                with failed_lock:
                    with open(str(out / "failed.jsonl"), "a") as handle:
                        handle.write(json.dumps(dict(work.to_dict(), reason=reason)) + "\n")
                progress.failed += 1
            if worker.process is None:
                worker.restarts += 1
                if worker.restarts > RESTART_LIMIT:
                    progress.say("WORKER %d gave up after %d restarts" % (worker.index, RESTART_LIMIT))
                    return
                try:
                    worker.start()
                except Exception as exc:
                    progress.say("WORKER %d failed to restart: %s" % (worker.index, exc))
                    return
        worker.stop()

    threads = [threading.Thread(target=loop, args=(w,), daemon=True) for w in workers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for w in workers:
        w.kill()


def archive_failures(out: Path) -> Optional[Path]:
    """``failed.jsonl`` holds only the failures of the latest invocation: an earlier one is moved to
    ``failed.<timestamp>.jsonl`` before the run (its episodes are queued again unless they have a record by now)."""
    path = out / "failed.jsonl"
    if not path.is_file():
        return None
    target = out / ("failed.%s.jsonl" % time.strftime("%Y%m%d-%H%M%S"))
    path.rename(target)
    return target


def select(args: argparse.Namespace, horizon_extra_s: float):
    tasks = parse_task_spec(args.tasks)
    return select_all(args.gen_root, tasks, args.layouts_per_task, horizon_extra_s, args.seed, args.allow_incomplete)


def write_report(args: argparse.Namespace, selection: Dict[str, Any], policy: str) -> Path:
    from hazardlib.run import report as core
    from hazardlib.run.settings import RunSettings
    from .cell_report import build_cell_report, render_markdown
    out = Path(args.out)
    settings = RunSettings.load(out / "config")
    dirs = [run_dir(out, t, policy) for t in selection["tasks"] if (run_dir(out, t, policy) / "episodes.jsonl").is_file()]
    metas, records = core.load_runs(dirs)
    destination = out / "report"
    destination.mkdir(exist_ok=True)
    cell = build_cell_report(selection, records, settings.report, policy)
    (destination / "cell_report.json").write_text(json.dumps(cell, indent=1, sort_keys=True))
    (destination / "cell_report.md").write_text(render_markdown(cell))
    summary = core.summarize(metas, records, settings.report, settings.config_hash)
    (destination / "core").mkdir(exist_ok=True)
    core.write_report(destination / "core", summary)
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gen-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tasks", default="all", help="all or <suite>:<index>,...")
    parser.add_argument("--layouts-per-task", type=int, default=5)
    parser.add_argument("--model", default="pi05")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--client-src", required=True)
    parser.add_argument("--servers", default="18010", help="OpenPI server ports, comma separated")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0, help="GPU of the simulator processes (default 0; 1 is refused)")
    parser.add_argument("--video", choices=VIDEO_MODES, default="first-layout",
                        help="primary+wrist MP4: none, the first selected layout of each task, or every episode")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="skip layouts the generation has not finished instead of leaving the task out")
    parser.add_argument("--dry-run", action="store_true", help="select and count only")
    parser.add_argument("--report-only", action="store_true", help="only (re)write <out>/report from the records")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.gpu == 1:
        raise SystemExit("GPU 1 is not available to this evaluation")
    if args.worker:
        return worker_main(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from .pilot import assemble_config
    if not (out / "config").exists():
        assemble_config(out / "config")
    from hazardlib.run.settings import RunSettings
    horizon_extra_s = RunSettings.load(out / "config").episode.horizon_extra_s
    from hazardlib.config import default_config_dir
    models_dir = default_config_dir().parent / "models"
    policy = model_policy_name(args.model, models_dir)
    selected, skipped = select(args, horizon_extra_s)
    arguments = {k: getattr(args, k) for k in ("gen_root", "tasks", "layouts_per_task", "model", "seed", "video")}
    document = selection_document(selected, skipped, arguments)
    if not args.report_only:
        (out / "selection.json").write_text(json.dumps(document, indent=1, sort_keys=True))
    if args.report_only:
        document = json.loads((out / "selection.json").read_text())
        print("report:", write_report(args, document, policy))
        return 0
    done = done_ids(out, selected, policy, args.seed)
    work_items = build_queue(selected, done)
    expected = expected_counts(selected)
    print("selected %d tasks (%d layouts); skipped %s" % (len(selected), sum(v["layouts"] for v in expected.values()),
                                                          skipped))
    print("episodes: %d hazard + %d control expected, %d already done, %d queued" % (
        sum(v["hazard"] for v in expected.values()), sum(v["control"] for v in expected.values()),
        sum(len(v) for v in done.values()), len(work_items)))
    if args.dry_run:
        return 0
    ports = [int(p) for p in args.servers.split(",")]
    archive_failures(out)
    progress = Progress(len(work_items), out / "eval.log")
    progress.say("START %d episodes, %d workers on GPU %d, servers %s, seed %d" % (
        len(work_items), args.workers, args.gpu, ports, args.seed))
    drive(args, work_items, progress, ports)
    progress.say("FINISHED %d episodes, %d failed, %.0f s" % (progress.finished, progress.failed,
                                                             time.time() - progress.start))
    print("report:", write_report(args, document, policy))
    return 0 if not progress.failed else 1


# ----------------------------------------------------------------------------------------------------- worker

def _relative_paths(paths: Optional[Dict[str, str]], base: Path) -> Optional[Dict[str, str]]:
    if paths is None:
        return None
    out = {}
    for key, value in paths.items():
        try:
            out[key] = Path(value).resolve().relative_to(base.resolve()).as_posix()
        except ValueError:
            out[key] = str(value)
    return out


def append_locked(directory: Path, record: Dict[str, Any]) -> bool:
    """Append ``record`` to ``episodes.jsonl`` under the directory lock unless the episode already has one."""
    from hazardlib.run.runner import append_record
    directory.mkdir(parents=True, exist_ok=True)
    with open(str(directory / ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = directory / "episodes.jsonl"
        marker = '"episode_id": %s' % json.dumps(record["episode_id"])
        if path.exists() and marker in path.read_text(encoding="utf-8"):
            return False
        append_record(path, record)
        return True


def note_skipped_entities(directory: Path, names: List[str]) -> None:
    """Keep the entities of interest the host reports no pose for (and so are not topple-monitored) in a marker
    file of the task's run directory, one name per line."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / SKIPPED_ENTITIES_FILE
    with open(str(directory / ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        known = set(path.read_text().split()) if path.exists() else set()
        if not set(names) <= known:
            path.write_text("".join(n + "\n" for n in sorted(known.union(names))))


def merge_meta(scratch: Path, directory: Path) -> None:
    """Copy the scratch ``run_meta.yaml`` to the task directory, or check that the identity agrees."""
    from hazardlib.run.runner import IDENTITY_FIELDS, RUN_META_FILE, RunDirectoryError
    mine = scratch / RUN_META_FILE
    theirs = directory / RUN_META_FILE
    directory.mkdir(parents=True, exist_ok=True)
    with open(str(directory / ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not theirs.exists():
            shutil.copyfile(str(mine), str(theirs))
            return
        a, b = yaml.safe_load(mine.read_text()), yaml.safe_load(theirs.read_text())
        diffs = [name for name in IDENTITY_FIELDS if a.get(name) != b.get(name)]
        if diffs:
            raise RunDirectoryError("%s belongs to another run: %s differ" % (directory, diffs))


def build_layout_inputs(folder: Path, root: Path, host_bd: Any, host_adapter_version: str, policy_checkpoint: str):
    """``RunInputs`` of one layout: its instances, its TaskContext with the shared robot geometry restored (the
    stored context only references ``../../robot_geometry.npz``; the instances' ``task_context_hash`` is checked
    against the rebuilt context) and no calibration."""
    import numpy as np
    from hazardlib.pipeline_io import read_records, read_yaml
    from hazardlib.run.inputs import RunInputs, _check_instances, instance_key
    from hazardlib.runtime import Provenance
    from hazardlib.spec.task_context import TaskContext
    from .seeded import PROTOCOL
    from .trajectory_candidates import AnalysisContext
    instances, _ = read_records(folder / "instances.jsonl")
    data = read_yaml(folder / "context_metadata.yaml")
    reference = data["provenance"]["robot_geometry"]
    points = {k: v.tolist() for k, v in np.load(str(root / "robot_geometry.npz")).items()}
    data["provenance"]["robot_geometry"] = {"spacing_m": reference["spacing_m"], "points": points}
    tc = AnalysisContext.from_dict(data)
    contexts = {(tc.task_id, tc.layout_id): tc}
    _check_instances(instances, contexts, host_bd)
    provenance = Provenance(bd_hash=host_bd.hash(), adapter_version=None, manifest_hash=None,
                            benchmark_commits=dict(host_bd.benchmark.commits))
    return RunInputs(pilot=True, manifest_hash=None, instances=instances, contexts=contexts, bd=host_bd,
                     calibration=None, calibration_hash=None, provenance=provenance,
                     sources={"layout_folder": str(folder), "policy_checkpoint": policy_checkpoint,
                              "policy_rng_protocol": PROTOCOL, "adapter_version": host_adapter_version,
                              "calibration": "per layout (see cell_eval)"})


def worker_main(args: argparse.Namespace) -> int:
    import resource
    protocol = os.fdopen(os.dup(1), "w")     # the reply channel; everything the libraries print goes to the log
    os.dup2(2, 1)

    def reply(message: Dict[str, Any]) -> None:
        protocol.write(PROTOCOL_TAG + json.dumps(message) + "\n")
        protocol.flush()

    from hazardlib.config import default_config_dir, load_config
    from hazardlib.driving import EpisodeSession
    from hazardlib.run.runner import RunDirectoryError, read_episode_records, run_stage
    from hazardlib.run.settings import RunSettings
    from hazardlib.runtime import RuntimeConfig
    from .host import LiberoHost
    from .policies import DyniPolicyBridge
    from .video import Mp4FrameSink
    out, gen_root = Path(args.out), Path(args.gen_root)
    config = out / "config"
    settings = RunSettings.load(config)
    pilot = yaml.safe_load((config / "pilot.yaml").read_text())
    host = LiberoHost(config_dir=config)
    session = EpisodeSession(host, RuntimeConfig.load(config))
    # entities of interest without a pose (regions / sites of articulated fixtures) are skipped, not refused
    session.runtime.skip_unreported_entities = True
    models = default_config_dir().parent / "models"
    policy = DyniPolicyBridge(host, models / (args.model + ".yaml"), args.checkpoint, "127.0.0.1", args.port,
                              args.client_src)
    exposure_radius = load_config("semantics", config_dir=config)["exposure"]["low_exposure_radius_m"]
    first_layout = {}   # task -> demo of the first selected layout (video sample)
    selection = json.loads((out / "selection.json").read_text())
    for task, body in selection["tasks"].items():
        first_layout[task] = body["layouts"][0]["demo"]
    state = {"key": None, "inputs": None}
    reply({"ready": True})
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            work = json.loads(line)
            task, demo, episode_id = work["task"], work["demo"], work["episode_id"]
            started = time.time()
            try:
                key = (task, demo)
                if state["key"] != key:
                    state["inputs"] = None      # free the previous layout's geometry first
                    state["inputs"] = build_layout_inputs(gen_root / task / "trajectories" / demo, gen_root / task,
                                                          host.describe(), host.config["adapter_version"],
                                                          args.checkpoint)
                    state["key"] = key
                inputs = state["inputs"]
                directory = run_dir(out, task, policy.name)
                scratch = out / "scratch" / ("w%d" % args.worker_index)
                if scratch.exists():
                    shutil.rmtree(str(scratch))
                video = args.video == "all" or (args.video == "first-layout" and first_layout[task] == demo)
                sink = keys = None
                if video:
                    v = pilot["video"]
                    sink, keys = Mp4FrameSink(directory / "frames", v["fps"], v["crf"], v["preset"]), ["primary", "wrist"]
                result = run_stage(session, inputs, policy, policy.name, settings, scratch, [args.seed], 0,
                                   exposure_radius_m=exposure_radius, labels=None, frame_sink=sink,
                                   frame_keys=keys, monitor_topple=True, allow_censored_t_stop=False,
                                   only_episode_ids={episode_id})
                records = read_episode_records(scratch / "episodes.jsonl", repair=False)
                if result.ran != 1 or len(records) != 1:
                    raise RuntimeError("expected exactly one episode for %r, ran %d" % (episode_id, result.ran))
                record = records[0]
                record["frames"] = _relative_paths(record.get("frames"), directory)
                if record.get("task_entities_skipped"):
                    note_skipped_entities(directory, record["task_entities_skipped"])
                merge_meta(scratch, directory)
                append_locked(directory, record)
                shutil.rmtree(str(scratch))
                reply({"ok": True, "seconds": time.time() - started, "steps": record["n_control_steps"],
                       "success": record["success_t_s"] is not None,
                       "contact": record["log"]["first_official_contact_t_s"] is not None,
                       "rss_mb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)})
            except Exception as exc:    # the episode is retried by the orchestrator after a worker restart
                import traceback
                traceback.print_exc()
                reply({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc), "restart": True})
    finally:
        policy.close()
        host.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
