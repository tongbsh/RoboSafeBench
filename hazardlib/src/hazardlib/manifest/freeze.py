"""Freeze a gated instance set into a Manifest, and verify it (paradigm §2, §9; D11 c).

``freeze`` refuses to run unless every configuration file the pipeline reads (:data:`USED_CONFIGS`) loads with
``strict=True`` (no ``PLACEHOLDER`` / ``unknown``) and the BD is complete. ``config_hashes[name]`` is the
sha256 of the canonical JSON of the parsed file (``hazardlib.hashing``): any change of a value changes it;
comments and formatting do not. ``families.yaml`` (not read by any code) and ``conformance.yaml`` (the T05
suite's) are not part of the instance pipeline and are not hashed.

Entries: every instance with a certificate except ``baseline_unusable`` (not run, only counted in the freeze
report); an instance without a certificate is an error. ``probe_set_hash`` combines the probe-set hashes of
the TaskContexts in use; ``spec_library_hash`` hashes the spec library (sorted by id).

A freeze is bound to a conformance report and to the T_stop calibration (D13 a, b): ``freeze`` reads the report
with :func:`hazardlib.conformance.read_report` and ``calibration.yaml``, refuses a report that
:func:`check_conformance_report` finds problems with, and the Manifest carries ``conformance_report_hash`` and
``calibration_hash``, both ``sha256_hex`` of the parsed content like ``config_hashes``.

Files written by :func:`write_frozen`: ``manifest.yaml`` (the Manifest), ``instances.jsonl`` (one
``HazardInstance`` per line, entry order), ``manifest.sha256`` (``Manifest.hash()``), ``conformance_report.json``,
``calibration.yaml`` and ``freeze_report.yaml`` (counts and excluded instances; its two hashes repeat the
Manifest's fields, which are the ones :func:`verify_manifest` checks).
"""
import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

from ..calibrate.protocol import read_calibration_summary
from ..config import ConfigError, load_config
from ..conformance import read_report
from ..hashing import sha256_hex
from ..runtime import RuntimeConfig
from ..spec.benchmark_descriptor import BenchmarkDescriptor
from ..spec.hazard_spec import HazardSpec
from ..spec.instance import HazardInstance
from ..spec.manifest import Manifest
from ..spec.task_context import TaskContext

MANIFEST_VERSION = "2"
USED_CONFIGS = ("gates", "intensity", "probe", "runtime", "semantics")
NOT_FROZEN_LABELS = ("baseline_unusable",)
CONFORMANCE_REPORT_FILE = "conformance_report.json"
CALIBRATION_FILE = "calibration.yaml"
#: Config files the conformance suite reads besides its own; the report must carry the frozen ones (D13 a).
REPORT_CONFIGS = ("runtime", "semantics", "intensity")
REPORT_VERDICT_PASS = "pass"
#: A setting whose source starts with this came from the suite's ``probe_runtime`` / ``probe_calibration``.
FALLBACK_SOURCE_PREFIX = "conformance.yaml:"
RUNTIME_FIELDS = tuple(field.name for field in dataclasses.fields(RuntimeConfig))


class ManifestError(ValueError):
    """``verify_manifest`` found problems (all of them are listed)."""


class ConformanceReportRejected(ValueError):
    """The conformance report cannot back the freeze (all problems are listed)."""


def _sources(value: Any, path: str) -> Any:
    """``(path, source)`` of every ``{"value": ..., "source": "<file>:<key>"}`` entry nested in ``value``."""
    if isinstance(value, Mapping):
        if "value" in value and isinstance(value.get("source"), str):
            yield path, value["source"]
        for key in sorted(value, key=str):
            yield from _sources(value[key], "%s.%s" % (path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _sources(item, "%s[%d]" % (path, index))


def check_conformance_report(report: Mapping[str, Any], bd: BenchmarkDescriptor, adapter_version: str,
                             config_hashes: Mapping[str, str]) -> List[str]:
    """Every reason why the parsed conformance ``report`` cannot back a freeze of ``bd`` with ``adapter_version``
    and the frozen ``config_hashes`` (D13 a); empty when it can:

    * its ``verdict`` is not ``pass``;
    * its ``bd_hash`` or ``adapter_version`` is not the freeze's;
    * its hash of ``runtime``, ``semantics`` or ``intensity`` is missing or differs from the frozen file's;
    * a setting came from ``conformance.yaml`` (the ``probe_runtime`` / ``probe_calibration`` fallbacks) instead of
      the frozen files: a field of ``runtime_config``, or a ``{"value", "source"}`` entry in a check's ``measured``
      values (C10 records its T_stop settings there). A ``runtime_config`` that does not give the source of every
      ``RuntimeConfig`` field is no evidence either.
    """
    problems = []  # type: List[str]
    if report.get("verdict") != REPORT_VERDICT_PASS:
        problems.append("verdict is %r, not %r (summary %s)"
                        % (report.get("verdict"), REPORT_VERDICT_PASS, report.get("summary")))
    if report.get("bd_hash") != bd.hash():
        problems.append("bd_hash %s is not the hash %s of BD %r" % (report.get("bd_hash"), bd.hash(), bd.name))
    if report.get("adapter_version") != adapter_version:
        problems.append("adapter_version %r is not the freeze's %r" % (report.get("adapter_version"), adapter_version))
    hashes = report.get("config_hashes")
    hashes = hashes if isinstance(hashes, Mapping) else {}
    for name in REPORT_CONFIGS:
        if name not in hashes:
            problems.append("config_hashes has no %r" % name)
        elif hashes[name] != config_hashes.get(name):
            problems.append("config %r hash %s is not the frozen %s.yaml's %s"
                            % (name, hashes[name], name, config_hashes.get(name)))
    runtime = report.get("runtime_config")
    runtime = runtime if isinstance(runtime, Mapping) else {}
    missing = [name for name in RUNTIME_FIELDS if name not in runtime]
    if missing:
        problems.append("runtime_config does not give the source of %s" % missing)
    for name in sorted(runtime, key=str):
        entry = runtime[name]
        source = entry.get("source") if isinstance(entry, Mapping) else None
        if not isinstance(source, str):
            problems.append("runtime_config.%s has no source" % name)
        elif source.startswith(FALLBACK_SOURCE_PREFIX):
            problems.append("runtime_config.%s = %r comes from %s, a fallback, not the frozen runtime / semantics "
                            "config" % (name, entry.get("value"), source))
    for check in report.get("checks") or []:
        if not isinstance(check, Mapping):
            continue
        for path, source in _sources(check.get("measured"), "measured"):
            if source.startswith(FALLBACK_SOURCE_PREFIX):
                problems.append("check %s: %s comes from %s, a fallback, not the frozen config"
                                % (check.get("id"), path, source))
    return problems


def config_hashes(config_dir: Optional[Any], strict: bool = True) -> Dict[str, str]:
    """``{name: sha256}`` of the parsed used config files; ``strict`` refuses unresolved values."""
    problems, out = [], {}
    for name in USED_CONFIGS:
        try:
            out[name] = sha256_hex(load_config(name, strict=strict, config_dir=config_dir))
        except ConfigError as exc:
            problems.append(str(exc))
    if problems:
        raise ConfigError("cannot freeze: %d config file(s) unresolved:\n  %s" % (len(problems), "\n  ".join(problems)))
    return out


@dataclasses.dataclass
class Frozen:
    manifest: Manifest
    instances: List[HazardInstance]
    report: Dict[str, Any]
    conformance_report: Dict[str, Any]
    calibration: Dict[str, Any]


def _load_yaml(path: Any) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_calibration(path: Any) -> Dict[str, Any]:
    """The parsed ``calibration.yaml`` of ``hazardctl calibrate``, after checking that its T_stop summary and its
    lower-bound flag agree."""
    try:
        read_calibration_summary(path)
    except (KeyError, TypeError) as exc:
        raise ValueError("%s is not a calibration file written by hazardctl calibrate (%s: %s)"
                         % (path, type(exc).__name__, exc)) from None
    return _load_yaml(path)


def freeze(instances: Sequence[HazardInstance], task_contexts: Sequence[TaskContext], specs: Sequence[HazardSpec],
           bd: BenchmarkDescriptor, config_dir: Optional[Any], adapter_version: str, seed: int,
           conformance_report: Any, calibration: Any) -> Frozen:
    """``conformance_report``: path of the conformance suite's JSON report (``python -m hazardlib.conformance
    --out``); ``calibration``: path of ``calibration.yaml`` (``hazardctl calibrate``). A report with any problem
    (:func:`check_conformance_report`) raises :class:`ConformanceReportRejected` listing all of them."""
    hashes = config_hashes(config_dir, strict=True)
    bd.require_complete()
    conformance = read_report(conformance_report)
    problems = check_conformance_report(conformance, bd, adapter_version, hashes)
    if problems:
        raise ConformanceReportRejected("cannot freeze with the conformance report %s: %d problem(s):\n  %s"
                                        % (conformance_report, len(problems), "\n  ".join(problems)))
    calibration_data = read_calibration(calibration)
    spec_ids = [s.id for s in specs]
    if len(set(spec_ids)) != len(spec_ids):
        raise ValueError("duplicate spec ids in the library: %s" % spec_ids)
    contexts = {(tc.task_id, tc.layout_id): tc for tc in task_contexts}
    frozen, excluded = [], []  # type: List[HazardInstance], List[Dict[str, Any]]
    for instance in instances:
        if instance.certificate is None:
            raise ValueError("instance %r has no certificate; run the gates before freezing" % (instance.instance_id,))
        prov = instance.provenance
        key = (prov.get("task_id"), prov.get("layout_id"))
        if key not in contexts or prov.get("task_context_hash") != contexts[key].hash():
            raise ValueError("instance %r: its TaskContext %s is not among the given ones" % (instance.instance_id, key))
        if instance.spec_id not in spec_ids:
            raise ValueError("instance %r: spec %r is not in the library" % (instance.instance_id, instance.spec_id))
        if instance.certificate.label in NOT_FROZEN_LABELS:
            excluded.append({"instance_id": instance.instance_id, "label": instance.certificate.label})
            continue
        frozen.append(instance)
    used = sorted({(i.provenance["task_id"], i.provenance["layout_id"]) for i in instances})
    probe_set_hash = sha256_hex(sorted(contexts[k].provenance["probe_set_hash"] for k in used)) if used else \
        sha256_hex([])
    entries = [{"instance_id": i.instance_id, "spec_id": i.spec_id, "task_id": i.provenance["task_id"],
                "layout_id": i.provenance["layout_id"], "instance_seed": i.provenance["instance_seed"],
                "instance_hash": i.hash()} for i in frozen]
    manifest = Manifest.from_dict({
        "manifest_version": MANIFEST_VERSION, "benchmark": bd.name, "bd_hash": bd.hash(),
        "probe_set_hash": probe_set_hash,
        "spec_library_hash": sha256_hex(sorted((s.to_dict() for s in specs), key=lambda d: d["id"])),
        "config_hashes": hashes, "conformance_report_hash": sha256_hex(conformance),
        "calibration_hash": sha256_hex(calibration_data), "adapter_version": adapter_version,
        "benchmark_commits": dict(bd.benchmark.commits), "seed": seed, "entries": entries,
    })
    labels = {}  # type: Dict[str, int]
    for instance in instances:
        labels[instance.certificate.label] = labels.get(instance.certificate.label, 0) + 1
    report = {"manifest_hash": manifest.hash(), "n_entries": len(entries), "labels": dict(sorted(labels.items())),
              "excluded": excluded, "conformance_report_hash": manifest.conformance_report_hash,
              "calibration_hash": manifest.calibration_hash, "task_contexts": ["%s/%s" % k for k in used]}
    return Frozen(manifest=manifest, instances=frozen, report=report, conformance_report=conformance,
                  calibration=calibration_data)


def write_frozen(out_dir: Any, frozen: Frozen) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.yaml").write_text(yaml.safe_dump(frozen.manifest.to_dict(), sort_keys=False), encoding="utf-8")
    with open(out / "instances.jsonl", "w", encoding="utf-8", newline="\n") as handle:
        for instance in frozen.instances:
            handle.write(json.dumps(instance.to_dict(), allow_nan=False, sort_keys=True) + "\n")
    (out / "manifest.sha256").write_text(frozen.manifest.hash() + "\n", encoding="utf-8")
    with open(out / CONFORMANCE_REPORT_FILE, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(frozen.conformance_report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    (out / CALIBRATION_FILE).write_text(yaml.safe_dump(frozen.calibration, sort_keys=False), encoding="utf-8")
    (out / "freeze_report.yaml").write_text(yaml.safe_dump(frozen.report, sort_keys=False), encoding="utf-8")
    return out


def read_manifest(out_dir: Any) -> Manifest:
    with open(Path(out_dir) / "manifest.yaml", "r", encoding="utf-8") as handle:
        return Manifest.from_dict(yaml.safe_load(handle))


def verify_manifest(out_dir: Any, config_dir: Optional[Any] = None, bd: Optional[BenchmarkDescriptor] = None,
                    expected_hash: Optional[str] = None) -> str:
    """Check a frozen directory; return the manifest hash or raise :class:`ManifestError` listing every problem:
    the stored hash, every instance hash, missing / extra instances, the conformance report and the calibration
    file against their Manifest hashes (missing, unreadable or modified), and (when given) the config hashes
    against ``config_dir`` and the BD hash against ``bd``."""
    out = Path(out_dir)
    problems = []  # type: List[str]
    try:
        manifest = read_manifest(out)
    except Exception as exc:  # noqa: BLE001 - any failure to read is a verification failure
        raise ManifestError("%s: cannot read manifest.yaml: %s" % (out, exc)) from None
    digest = manifest.hash()
    stored = (out / "manifest.sha256").read_text(encoding="utf-8").strip() if (out / "manifest.sha256").exists() \
        else None
    for name, value in (("manifest.sha256", stored), ("expected hash", expected_hash)):
        if value is not None and value != digest:
            problems.append("%s %s != manifest hash %s" % (name, value, digest))
    instances = {}  # type: Dict[str, HazardInstance]
    with open(out / "instances.jsonl", "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                instance = HazardInstance.from_dict(json.loads(line))
            except Exception as exc:  # noqa: BLE001
                problems.append("instances.jsonl line %d: %s" % (number, exc))
                continue
            instances[instance.instance_id] = instance
    listed = set()
    for entry in manifest.entries:
        listed.add(entry.instance_id)
        instance = instances.get(entry.instance_id)
        if instance is None:
            problems.append("instance %r is listed but missing" % entry.instance_id)
        elif instance.hash() != entry.instance_hash:
            problems.append("instance %r was modified (hash %s, manifest %s)"
                            % (entry.instance_id, instance.hash(), entry.instance_hash))
    for extra in sorted(set(instances) - listed):
        problems.append("instance %r is not listed in the manifest" % extra)
    for name, recorded, read in ((CONFORMANCE_REPORT_FILE, manifest.conformance_report_hash, read_report),
                                 (CALIBRATION_FILE, manifest.calibration_hash, _load_yaml)):
        if not (out / name).exists():
            problems.append("%s is missing" % name)
            continue
        try:
            found = sha256_hex(read(out / name))
        except Exception as exc:  # noqa: BLE001 - any failure to read is a verification failure
            problems.append("%s cannot be read: %s" % (name, exc))
            continue
        if found != recorded:
            problems.append("%s was modified (hash %s, manifest %s)" % (name, found, recorded))
    if config_dir is not None:
        try:
            current = config_hashes(config_dir, strict=False)
        except ConfigError as exc:
            problems.append(str(exc))
        else:
            for name in sorted(set(current).union(manifest.config_hashes)):
                if current.get(name) != manifest.config_hashes.get(name):
                    problems.append("config %r changed (now %s, manifest %s)"
                                    % (name, current.get(name), manifest.config_hashes.get(name)))
    if bd is not None and bd.hash() != manifest.bd_hash:
        problems.append("BD %r hash %s != manifest bd_hash %s" % (bd.name, bd.hash(), manifest.bd_hash))
    if problems:
        raise ManifestError("%s: %d problem(s):\n  %s" % (out, len(problems), "\n  ".join(problems)))
    return digest
