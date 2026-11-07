"""D14 pilot runs of the hazard pipeline on CALVIN (H1): ``hazardctl`` with the pilot values of
``configs/hazard/pilot/calvin/`` (numbers obtained here are "试点", never formal results).

Every step runs ``python -m hazardlib.cli`` in a subprocess (one host per command) with ``$HAZARDLIB_CONFIG_DIR``
pointing at ``<work>/config``: the repository's config files with the pilot files laid over them (``assemble``).
The host reads ``$CALVIN_DATASET_DIR`` and ``$CALVIN_EGL_DEVICE``; the reference policy is :class:`DemoReplay`.

    python -m hazardhosts.calvin.pilot assemble  --work W
    python -m hazardhosts.calvin.pilot record    --work W --windows windows.json   (probes/windows.py --out)
    python -m hazardhosts.calvin.pilot probe     --work W                          (roles.py --out-dir W/entities first)
    python -m hazardhosts.calvin.pilot calibrate --work W --contexts C [C ...]
    python -m hazardhosts.calvin.pilot gate      --work W --specs S --contexts C [C ...]

``record`` replays every window ``[split, s, e]`` from layout ``split/(s - lead)`` (clamped to the recorded range)
for ``(e - s + lead) / 30 s + margin``, into ``traces/<task>__<split>_<frame>.jsonl``. ``probe`` builds one
TaskContext per successful trace (``contexts/<same name>.yaml``): a TaskContext belongs to one (task, layout), and
the dataset holds one demonstration per (task, layout). ``calibrate`` / ``gate`` take those context names;
``gate`` instantiates the specs on them and runs G0 / G1 (no avoider).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from hazardlib.config import CONFIG_DIR_ENV, default_config_dir, load_config

from .dataset import CalvinDataset
from .host import DATASET_ENV, HOST_CONFIG

HOST = "hazardhosts.calvin.host:CalvinHost"
POLICY = "hazardhosts.calvin.policies:DemoReplay"
PILOT = Path("pilot") / "calvin"
CONTROL_HZ = 30.0


def _hazardctl(work: Path, argv: List[str], log: Any) -> Dict[str, Any]:
    env = dict(os.environ)
    env[CONFIG_DIR_ENV] = str(work / "config")
    proc = subprocess.run([sys.executable, "-m", "hazardlib.cli"] + argv, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True)
    log.write("$ hazardctl %s\n%s%s\n" % (" ".join(argv), proc.stdout, proc.stderr[-4000:] if proc.returncode else ""))
    log.flush()
    if proc.returncode != 0:
        return {"returncode": proc.returncode, "error": proc.stderr.strip().splitlines()[-1:]}
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    return json.loads(lines[-1]) if lines else {}


def _run_name(task: str, layout: str) -> str:
    return "%s__%s" % (task, layout.replace("/", "_"))


def _driver(seed: int = 0) -> List[str]:
    return ["--host", HOST, "--policy", POLICY, "--seed", str(seed), "--env", "0"]


def cmd_assemble(args: argparse.Namespace) -> None:
    repo = default_config_dir()
    out = Path(args.work) / "config"
    if out.exists():
        shutil.rmtree(str(out))
    shutil.copytree(str(repo), str(out), ignore=shutil.ignore_patterns("pilot"))
    for path in sorted((repo / PILOT).glob("*.yaml")):
        shutil.copy(str(path), str(out / path.name))
        print("pilot %s" % path.name)
    print("config dir %s" % out)


def cmd_record(args: argparse.Namespace) -> None:
    work = Path(args.work)
    windows = json.load(open(args.windows))
    dataset = CalvinDataset(os.environ[DATASET_ENV], load_config(HOST_CONFIG)["dataset"]["splits"])
    out_dir = work / "traces"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(str(work / "record.log"), "a") as log:
        for task in sorted(windows):
            for split, start, end in windows[task][:args.per_task]:
                first = max(a for a, b in dataset.split(split).ranges if a <= start <= b)
                start = max(first, start - args.lead)
                horizon = (end - start) / CONTROL_HZ + args.margin_s
                out = out_dir / (_run_name(task, "%s/%d" % (split, start)) + ".jsonl")
                result = _hazardctl(work, ["record"] + _driver() + [
                    "--task", task, "--layout", "%s/%d" % (split, start), "--horizon-s", "%.4f" % horizon,
                    "--out", str(out), "--bd-out", str(work / "bd.yaml")], log)
                rows.append(dict(task=task, layout="%s/%d" % (split, start), window_end=end, horizon_s=horizon,
                                 **{k: result.get(k) for k in ("success", "success_t_s", "steps", "error")}))
                print(json.dumps(rows[-1], sort_keys=True))
    with open(str(work / "record_summary.json"), "w") as fh:
        json.dump(rows, fh, indent=1, sort_keys=True)


def cmd_probe(args: argparse.Namespace) -> None:
    work = Path(args.work)
    rows = json.load(open(str(work / "record_summary.json")))
    (work / "contexts").mkdir(exist_ok=True)
    done = set()
    with open(str(work / "probe.log"), "a") as log:
        for row in rows:
            name = _run_name(row["task"], row["layout"])
            if not row.get("success") or name in done:
                continue
            done.add(name)
            argv = ["probe", "--traces", str(work / "traces" / (name + ".jsonl")),
                    "--out", str(work / "contexts" / (name + ".yaml"))]
            entities = work / "entities" / ("%s.yaml" % row["task"])
            if entities.is_file():
                argv += ["--entities", str(entities)]
            print(json.dumps(dict(context=name, **_hazardctl(work, argv, log)), sort_keys=True))


def cmd_calibrate(args: argparse.Namespace) -> None:
    work = Path(args.work)
    (work / "calibration").mkdir(exist_ok=True)
    with open(str(work / "calibrate.log"), "a") as log:
        for name in args.contexts:
            result = _hazardctl(work, ["calibrate"] + _driver(args.seed) + [
                "--task-context", str(work / "contexts" / ("%s.yaml" % name)),
                "--out", str(work / "calibration" / ("%s.yaml" % name))], log)
            print(json.dumps(dict(context=name, **result), sort_keys=True))


def cmd_gate(args: argparse.Namespace) -> None:
    work = Path(args.work)
    contexts = [str(work / "contexts" / ("%s.yaml" % name)) for name in args.contexts]
    instances = work / ("instances_%s.jsonl" % args.name)
    with open(str(work / "gate.log"), "a") as log:
        print(json.dumps(_hazardctl(work, ["instantiate", "--specs", args.specs, "--task-context"] + contexts + [
            "--bd", str(work / "bd.yaml"), "--seed", str(args.seed), "--count", str(args.count),
            "--out", str(instances)], log), sort_keys=True))
        print(json.dumps(_hazardctl(work, ["gate"] + _driver(args.seed) + [
            "--instances", str(instances), "--task-context"] + contexts + [
            "--out", str(work / ("gated_%s.jsonl" % args.name)),
            "--report", str(work / ("gate_report_%s.yaml" % args.name))], log), sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    sub.required = True
    p = sub.add_parser("assemble")
    p.set_defaults(func=cmd_assemble)
    p = sub.add_parser("record")
    p.add_argument("--windows", required=True)
    p.add_argument("--per-task", type=int, default=10)
    p.add_argument("--margin-s", type=float, default=1.0)
    p.add_argument("--lead", type=int, default=0, help="frames to start earlier than the window (clamped to the "
                                                        "recorded range)")
    p.set_defaults(func=cmd_record)
    p = sub.add_parser("probe")
    p.set_defaults(func=cmd_probe)
    p = sub.add_parser("calibrate")
    p.add_argument("--contexts", nargs="+", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_calibrate)
    p = sub.add_parser("gate")
    p.add_argument("--specs", required=True)
    p.add_argument("--contexts", nargs="+", required=True)
    p.add_argument("--name", default="pilot")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--count", type=int, default=1)
    p.set_defaults(func=cmd_gate)
    for p in sub.choices.values():
        p.add_argument("--work", required=True)
    args = parser.parse_args()
    Path(args.work).mkdir(parents=True, exist_ok=True)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
