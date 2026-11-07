"""``hazardctl``: the instance pipeline as file-to-file commands (T06).

    hazardctl [--config-dir DIR] record      --host M:F --policy M:F --task T --layout L --seed S --horizon-s H
                                             --env E --out trace.jsonl [--bd-out bd.yaml]
    hazardctl [--config-dir DIR] probe       --traces A.jsonl B.jsonl ... --out task_context.yaml
                                             [--entities E.yaml] [--annotations A.yaml] [--workspace W.yaml]
                                             [--under-test NAME ...]
    hazardctl [--config-dir DIR] instantiate --specs specs.yaml --task-context TC.yaml ... --bd bd.yaml
                                             --seed S --count N --out instances.jsonl
    hazardctl [--config-dir DIR] gate        --instances instances.jsonl --task-context TC.yaml ...
                                             --host M:F --policy M:F [--avoider M:F] --seed S --env E
                                             --out gated.jsonl --report gate_report.yaml
    hazardctl [--config-dir DIR] calibrate   --task-context TC.yaml --host M:F --policy M:F --seed S --env E
                                             --out calibration.yaml
    hazardctl [--config-dir DIR] freeze      --instances gated.jsonl --task-context TC.yaml ... --specs specs.yaml
                                             --bd bd.yaml --seed S --adapter-version V [--calibration C.yaml]
                                             --out-dir DIR
    hazardctl [--config-dir DIR] verify      --manifest-dir DIR [--bd bd.yaml]

``M:F`` is ``module:callable``: hosts are built as ``F(**host_args)``, policies and avoiders as
``F(host, **args)``; ``--host-arg`` / ``--policy-arg`` / ``--avoider-arg`` take ``key=value`` (value parsed as
YAML). ``--config-dir`` defaults to ``$HAZARDLIB_CONFIG_DIR`` / the repository's ``configs/hazard``. Every
command prints a one-line JSON summary.
"""
import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from .calibrate.protocol import TStopProtocolSettings, run_t_stop_protocol, write_calibration
from .config import default_config_dir
from .driving import EpisodeSession
from .gates.settings import GateSettings
from .gates.stage import gate_instances
from .hashing import sha256_hex
from .instantiate.builder import InstanceRejected, instantiate
from .instantiate.settings import InstantiateSettings
from .manifest.freeze import freeze, verify_manifest, write_frozen
from .pipeline_io import (load_object, read_bd, read_records, read_specs, read_task_context, read_yaml,
                          write_records, write_yaml)
from .probe.context import ProbeSettings, build_task_context
from .probe.trace import read_probe_trace, record_probe_trace, write_probe_trace
from .runtime import RuntimeConfig


def _kwargs(pairs: Optional[Sequence[str]]) -> Dict[str, Any]:
    out = {}
    for pair in pairs or ():
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise SystemExit("expected key=value, got %r" % (pair,))
        out[key] = yaml.safe_load(value)
    return out


def _session(args: argparse.Namespace) -> Tuple[EpisodeSession, Any]:
    host = load_object(args.host)(**_kwargs(args.host_arg))
    session = EpisodeSession(host, RuntimeConfig.load(args.config_dir))
    policy = load_object(args.policy)(host, **_kwargs(args.policy_arg))
    return session, policy


def _contexts(paths: Sequence[str]) -> Dict[Tuple[str, str], Any]:
    out = {}
    for path in paths:
        tc = read_task_context(path)
        out[(tc.task_id, tc.layout_id)] = tc
    return out


def _emit(command: str, **fields: Any) -> None:
    print(json.dumps(dict(command=command, **fields), sort_keys=True))


def cmd_record(args: argparse.Namespace) -> None:
    session, policy = _session(args)
    trace = record_probe_trace(session, policy, args.task, args.layout, args.seed, args.horizon_s, args.env)
    write_probe_trace(args.out, trace)
    if args.bd_out:
        write_yaml(args.bd_out, session.bd.to_dict())
    _emit("record", out=args.out, source=trace.header.source.to_dict(), success=trace.header.success,
          success_t_s=trace.header.success_t_s, steps=trace.header.n_steps)


def cmd_probe(args: argparse.Namespace) -> None:
    traces = [read_probe_trace(path) for path in args.traces]
    tc = build_task_context(traces, ProbeSettings.load(args.config_dir),
                            static_entities=read_yaml(args.entities) if args.entities else None,
                            annotations=read_yaml(args.annotations) if args.annotations else None,
                            declared_workspace=read_yaml(args.workspace) if args.workspace else None,
                            policies_under_test=args.under_test or ())
    write_yaml(args.out, tc.to_dict())
    _emit("probe", out=args.out, task_context_hash=tc.hash(), probe_set_hash=tc.provenance["probe_set_hash"],
          phases=[p.name for p in tc.phases], entities=sorted(tc.entities), runs_used=tc.provenance["probe_set"]
          ["n_runs_used"])


def cmd_instantiate(args: argparse.Namespace) -> None:
    specs = read_specs(args.specs)
    contexts = _contexts(args.task_context)
    bd = read_bd(args.bd)
    settings = InstantiateSettings.load(args.config_dir)
    records = []  # type: List[Tuple[str, Dict[str, Any]]]
    reasons = {}  # type: Dict[str, int]
    for spec in specs:
        for key in sorted(contexts):
            for seed in range(args.seed, args.seed + args.count):
                try:
                    instance = instantiate(spec, contexts[key], seed, bd, settings)
                except InstanceRejected as exc:
                    reasons[exc.reason] = reasons.get(exc.reason, 0) + 1
                    records.append(("rejection", {"stage": "instantiate", "spec_id": spec.id, "task_id": key[0],
                                                  "layout_id": key[1], "instance_seed": seed,
                                                  "reasons": [exc.reason], "detail": exc.detail}))
                    continue
                records.append(("instance", instance.to_dict()))
    write_records(args.out, records)
    _emit("instantiate", out=args.out, instances=sum(1 for t, _ in records if t == "instance"),
          rejected_by_reason=reasons)


def cmd_gate(args: argparse.Namespace) -> None:
    instances, earlier = read_records(args.instances)
    session, policy = _session(args)
    avoider = load_object(args.avoider)(session.host, **_kwargs(args.avoider_arg)) if args.avoider else None
    output = gate_instances(session, instances, _contexts(args.task_context), policy, avoider,
                            GateSettings.load(args.config_dir), args.seed, args.env)
    records = [("instance", i.to_dict()) for i in output.instances]
    records += [("rejection", r) for r in earlier + output.rejections]
    write_records(args.out, records)
    upstream = {}  # type: Dict[str, int]
    for rejection in earlier:
        for reason in rejection["reasons"]:
            upstream[reason] = upstream.get(reason, 0) + 1
    report = dict(output.summary, upstream_rejections_by_reason=upstream)
    write_yaml(args.report, report)
    _emit("gate", out=args.out, labels=report["labels"], rejections_by_reason=report["rejections_by_reason"],
          n_main_score=report["n_main_score"])


def cmd_calibrate(args: argparse.Namespace) -> None:
    session, policy = _session(args)
    tc = read_task_context(args.task_context)
    result = run_t_stop_protocol(session, policy, tc, TStopProtocolSettings.load(args.config_dir), args.seed,
                                 args.env)
    write_calibration(args.out, result)
    _emit("calibrate", out=args.out, t_stop_median_s=result.summary.median_s,
          n_censored=result.summary.n_censored, t_stop_is_lower_bound=result.is_lower_bound)


def cmd_freeze(args: argparse.Namespace) -> None:
    instances, _ = read_records(args.instances)
    contexts = list(_contexts(args.task_context).values())
    calibration_hash = sha256_hex(read_yaml(args.calibration)) if args.calibration else None
    frozen = freeze(instances, contexts, read_specs(args.specs), read_bd(args.bd), args.config_dir,
                    args.adapter_version, args.seed, calibration_hash)
    write_frozen(args.out_dir, frozen)
    _emit("freeze", out_dir=args.out_dir, manifest_hash=frozen.report["manifest_hash"],
          entries=frozen.report["n_entries"], labels=frozen.report["labels"])


def cmd_verify(args: argparse.Namespace) -> None:
    digest = verify_manifest(args.manifest_dir, config_dir=args.config_dir,
                             bd=read_bd(args.bd) if args.bd else None)
    _emit("verify", manifest_dir=args.manifest_dir, manifest_hash=digest, ok=True)


def _add_driver(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", required=True)
    parser.add_argument("--host-arg", action="append")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--policy-arg", action="append")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--env", type=int, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hazardctl", description="Hazard-as-Spec instance pipeline (T06)")
    parser.add_argument("--config-dir", default=None)
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    p = sub.add_parser("record", help="record one hazard-free probe trace")
    _add_driver(p)
    p.add_argument("--task", required=True)
    p.add_argument("--layout", required=True)
    p.add_argument("--horizon-s", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bd-out")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("probe", help="probe traces -> TaskContext")
    p.add_argument("--traces", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--entities")
    p.add_argument("--annotations")
    p.add_argument("--workspace")
    p.add_argument("--under-test", nargs="*")
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("instantiate", help="specs x TaskContexts x seeds -> instances")
    p.add_argument("--specs", required=True)
    p.add_argument("--task-context", nargs="+", required=True)
    p.add_argument("--bd", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_instantiate)

    p = sub.add_parser("gate", help="G0 / G1 / G2 -> certificates")
    _add_driver(p)
    p.add_argument("--instances", required=True)
    p.add_argument("--task-context", nargs="+", required=True)
    p.add_argument("--avoider")
    p.add_argument("--avoider-arg", action="append")
    p.add_argument("--out", required=True)
    p.add_argument("--report", required=True)
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("calibrate", help="T_stop protocol -> calibration.yaml")
    _add_driver(p)
    p.add_argument("--task-context", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("freeze", help="gated instances -> Manifest")
    p.add_argument("--instances", required=True)
    p.add_argument("--task-context", nargs="+", required=True)
    p.add_argument("--specs", required=True)
    p.add_argument("--bd", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--adapter-version", required=True)
    p.add_argument("--calibration")
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=cmd_freeze)

    p = sub.add_parser("verify", help="check a frozen Manifest directory")
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--bd")
    p.set_defaults(func=cmd_verify)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.config_dir is None:
        args.config_dir = str(default_config_dir())
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
