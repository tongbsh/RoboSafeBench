"""Batch launcher of the LIBERO generation run: many tasks, one global queue of (task, demo) jobs over GPU slots.

    python -m hazardhosts.libero.trajectory_batch --root <batch dir> --source <_logs/libero40> \\
        --tasks libero_spatial:0,libero_10:2 | all --max-demos N --slots 0:10,2:10 [--retries 1] \\
        [--hazard-scale S|M|L]

Run it from an environment with ``PYTHONPATH``, ``LIBERO_CONFIG_PATH``, ``MUJOCO_GL`` ... set (children inherit it);
the launcher sets ``CUDA_VISIBLE_DEVICES`` / ``MUJOCO_EGL_DEVICE_ID`` per job from the slot.

1. *prepare* (per task, in the queue): hard-link ``<source>/<suite>_task<k>`` (collect / derive outputs) into
   ``<root>`` if missing, assemble ``config/`` once (and, for a hazard scale other than the default, give the
   root's ``scene.json`` the level's agent: ``hazard_scale.prepare_root``; the source tree is not modified),
   run ``export`` for the first N usable demonstrations. The root identifies the D21 hazard scale level: use one
   root per level (``_logs/libero40_gen20_M``), all built from the same collect / derive outputs;
2. *validate jobs*: one process per (task, demo) over ``trajectory_analysis validate --demos <demo>``, started
   as soon as a slot is free and the task's export is done, so no slot idles while any task has work left. A
   crashed job is retried (``--retries``, default once; validate resumes from the completed cells);
3. *report* per task when its last demonstration has finished.

Resumable and safe to restart: demos with ``validation.json`` are skipped, ``export`` is skipped when
``generation_inputs`` already holds the N demonstrations. ``<root>/batch.log`` gets one line per event
(``PREPARE`` / ``JOB task demo seconds rc attempt gpu_peak_mb`` / ``REPORT``); job logs are
``<task dir>/jobs/<demo>.log``. ``gpu_peak_mb`` is the peak GPU memory of the job's process tree sampled with
``nvidia-smi`` every ``--sample-s`` seconds (``-`` if the job was shorter than the sampling period).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MODULE = "hazardhosts.libero.trajectory_analysis"


def parse_tasks(text):
    if text == "all":
        return [(s, k) for s in SUITES for k in range(10)]
    out = []
    for item in text.split(","):
        suite, _, index = item.partition(":")
        if suite not in SUITES or not index.isdigit():
            raise SystemExit("bad task %r: use <suite>:<index> (suites %s) or 'all'" % (item, list(SUITES)))
        out.append((suite, int(index)))
    return out


def parse_slots(text):
    slots = []
    for item in text.split(","):
        gpu, _, count = item.partition(":")
        slots += [int(gpu)] * int(count)
    if not slots:
        raise SystemExit("no slots")
    return slots


def usable_demos(task_dir, max_demos):
    rows = json.loads((task_dir / "coverage.json").read_text())["demonstrations"]
    demos = [r["demo"] for r in rows if r["replay_success"]]
    return demos if max_demos is None else demos[:max_demos]


def exported(task_dir, demos):
    index = task_dir / "generation_inputs" / "index.json"
    return index.exists() and json.loads(index.read_text())["demos"][:len(demos)] == demos and \
        len(json.loads(index.read_text())["demos"]) >= len(demos)


def gpu_memory_by_pid():
    """``{pid: MiB}`` of the processes the driver lists. EGL rendering contexts are graphics ("G") processes, which
    ``--query-compute-apps`` does not list: the process table of the plain ``nvidia-smi`` output is parsed."""
    import re
    out = {}
    try:
        text = subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL).decode()
    except (OSError, subprocess.CalledProcessError):
        return out
    for line in text.splitlines():
        m = re.match(r"^\|\s+\d+\s+\S+\s+\S+\s+(\d+)\s+\S+\s+.*?\s(\d+)MiB\s*\|$", line)
        if m:
            out[int(m.group(1))] = out.get(int(m.group(1)), 0) + int(m.group(2))
    return out


def descendants(pid):
    """``pid`` and its child processes (``/proc`` walk)."""
    found, frontier = {pid}, [pid]
    while frontier:
        parent = frontier.pop()
        try:
            children = (Path("/proc/%d/task/%d/children" % (parent, parent))).read_text().split()
        except OSError:
            continue
        for c in children:
            if int(c) not in found:
                found.add(int(c))
                frontier.append(int(c))
    return found


class Job(object):
    def __init__(self, kind, task, demo=None):
        self.kind, self.task, self.demo = kind, task, demo
        self.attempt = 0
        self.peak_mb = 0
        self.started = None
        self.process = None
        self.slot = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="batch directory (one sub-directory per task)")
    parser.add_argument("--source", required=True, help="collect / derive outputs: <source>/<suite>_task<k>")
    parser.add_argument("--tasks", required=True, help="<suite>:<k>,... or all")
    parser.add_argument("--max-demos", type=int, required=True, help="first N usable demonstrations per task")
    parser.add_argument("--slots", required=True, help="gpu:count,... simulator processes per GPU, e.g. 0:10,2:10")
    parser.add_argument("--hazard-scale", default=None, help="D21 hazard scale level S / M / L (default: the "
                        "default level of configs/hazard/pilot/hazard_scale.yaml, the original 4 cm / 0.05 kg box)")
    parser.add_argument("--retries", type=int, default=1, help="retries of a crashed job (default 1)")
    parser.add_argument("--sample-s", type=float, default=5.0, help="GPU memory sampling period")
    args = parser.parse_args(argv)
    root, source = Path(args.root), Path(args.source)
    root.mkdir(parents=True, exist_ok=True)
    slots = parse_slots(args.slots)
    log = (root / "batch.log").open("a")

    def say(line):
        log.write("%s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), line))
        log.flush()
        print(line, flush=True)

    tasks = parse_tasks(args.tasks)
    names = {t: "%s_task%d" % t for t in tasks}
    from .hazard_scale import level_spec, prepare_root
    level = level_spec(args.hazard_scale)[0]
    queue = []
    state = {}      # task -> {"demos": [...], "left": set, "ready": bool}
    for t in tasks:
        d = root / names[t]
        if not d.exists():
            if not (source / names[t] / "coverage.json").exists():
                raise SystemExit("%s has no collect / derive outputs (coverage.json)" % (source / names[t]))
            subprocess.check_call(["cp", "-al", str(source / names[t]), str(d)])
            say("PREPARE %s linked from %s" % (names[t], source))
        prepare_root(d, level)      # config/ and scene.json agent of the level, before any child process
        demos = usable_demos(d, args.max_demos)
        done = [x for x in demos if (d / "trajectories" / x / "validation.json").exists()]
        state[t] = {"demos": demos, "left": set(demos) - set(done), "ready": exported(d, demos), "failed": set()}
        if not state[t]["ready"]:
            queue.append(Job("export", t))
        elif not state[t]["left"]:
            queue.append(Job("report", t))
        (d / "jobs").mkdir(exist_ok=True)
    for t in tasks:        # validate jobs: task-major in the given order (a task finishes, its report runs)
        if state[t]["left"]:
            for demo in state[t]["demos"]:
                if demo in state[t]["left"]:
                    queue.append(Job("validate", t, demo))
    running = []
    free = list(range(len(slots)))
    last_sample = 0.0

    def command(job):
        suite, k = job.task
        d = root / names[job.task]
        base = [sys.executable, "-u", "-m", MODULE]
        common = ["--out", str(d), "--suite", suite, "--task-index", str(k), "--hazard-scale", level]
        if job.kind == "export":
            return base + ["export"] + common + ["--max-demos", str(args.max_demos)]
        if job.kind == "report":
            return base + ["report"] + common
        return base + ["validate"] + common + ["--max-demos", str(args.max_demos), "--demos", job.demo]

    def startable(job):
        return job.kind != "validate" or state[job.task]["ready"]

    def launch(job, slot_index):
        gpu = slots[slot_index]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu))
        d = root / names[job.task]
        out = (d / "jobs" / ("%s.log" % (job.demo or job.kind))).open("a")
        job.attempt += 1
        job.started, job.slot, job.peak_mb = time.time(), slot_index, 0
        job.process = subprocess.Popen(command(job), stdout=out, stderr=subprocess.STDOUT, env=env)
        job.log = out

    def finish(job, rc):
        seconds = time.time() - job.started
        job.log.close()
        s = state[job.task]
        peak = "-" if not job.peak_mb else str(job.peak_mb)
        if job.kind == "validate":
            say("JOB %s %s seconds=%.0f rc=%d attempt=%d gpu=%d gpu_peak_mb=%s" % (
                names[job.task], job.demo, seconds, rc, job.attempt, slots[job.slot], peak))
        else:
            say("%s %s seconds=%.0f rc=%d gpu=%d gpu_peak_mb=%s" % (job.kind.upper(), names[job.task], seconds, rc,
                                                                  slots[job.slot], peak))
        if rc != 0 and job.attempt <= args.retries:
            queue.insert(0, job)
            return
        if job.kind == "export":
            if rc == 0:
                s["ready"] = True
            else:
                s["left"].clear()           # no inputs: the task's validate jobs cannot run
                queue[:] = [j for j in queue if not (j.task == job.task and j.kind == "validate")]
                say("TASK %s skipped: export failed" % names[job.task])
        elif job.kind == "validate":
            s["left"].discard(job.demo)
            if rc != 0:
                s["failed"].add(job.demo)
            if not s["left"]:
                queue.append(Job("report", job.task))
        if job.kind == "export" and rc == 0 and not s["left"]:
            queue.append(Job("report", job.task))

    while queue or running:
        while free and queue:
            pick = next((j for j in queue if startable(j)), None)
            if pick is None:
                break
            queue.remove(pick)
            slot_index = free.pop(0)
            launch(pick, slot_index)
            running.append(pick)
        time.sleep(1.0)
        now = time.time()
        if now - last_sample >= args.sample_s and running:
            last_sample = now
            memory = gpu_memory_by_pid()
            for job in running:
                used = sum(memory.get(p, 0) for p in descendants(job.process.pid))
                job.peak_mb = max(job.peak_mb, used)
        for job in list(running):
            rc = job.process.poll()
            if rc is not None:
                running.remove(job)
                free.append(job.slot)
                finish(job, rc)
    for t in tasks:
        failed = sorted(state[t]["failed"])
        if failed:
            say("TASK %s incomplete: failed demos %s" % (names[t], failed))
    say("BATCH done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
