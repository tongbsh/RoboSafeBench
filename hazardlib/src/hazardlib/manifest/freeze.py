"""Freeze a gated instance set into a Manifest, and verify it (paradigm §2, §9; D11 c).

``freeze`` refuses to run unless every configuration file the pipeline reads (:data:`USED_CONFIGS`) loads with
``strict=True`` (no ``PLACEHOLDER`` / ``unknown``) and the BD is complete. ``config_hashes[name]`` is the
sha256 of the canonical JSON of the parsed file (``hazardlib.hashing``): any change of a value changes it;
comments and formatting do not. ``families.yaml`` (not read by any code) and ``conformance.yaml`` (the T05
suite's) are not part of the instance pipeline and are not hashed.

Entries: every instance with a certificate except ``baseline_unusable`` (not run, only counted in the freeze
report); an instance without a certificate is an error. ``probe_set_hash`` combines the probe-set hashes of
the TaskContexts in use; ``spec_library_hash`` hashes the spec library (sorted by id).

Files written by :func:`write_frozen`: ``manifest.yaml`` (the Manifest), ``instances.jsonl`` (one
``HazardInstance`` per line, entry order), ``manifest.sha256`` (``Manifest.hash()``) and
``freeze_report.yaml`` (counts, excluded instances, optional calibration file hash).
"""
import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

from ..config import ConfigError, load_config
from ..hashing import sha256_hex
from ..spec.benchmark_descriptor import BenchmarkDescriptor
from ..spec.hazard_spec import HazardSpec
from ..spec.instance import HazardInstance
from ..spec.manifest import Manifest
from ..spec.task_context import TaskContext

MANIFEST_VERSION = "1"
USED_CONFIGS = ("gates", "intensity", "probe", "runtime", "semantics")
NOT_FROZEN_LABELS = ("baseline_unusable",)


class ManifestError(ValueError):
    """``verify_manifest`` found problems (all of them are listed)."""


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


def freeze(instances: Sequence[HazardInstance], task_contexts: Sequence[TaskContext], specs: Sequence[HazardSpec],
           bd: BenchmarkDescriptor, config_dir: Optional[Any], adapter_version: str, seed: int,
           calibration_hash: Optional[str] = None) -> Frozen:
    hashes = config_hashes(config_dir, strict=True)
    bd.require_complete()
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
        "config_hashes": hashes, "adapter_version": adapter_version,
        "benchmark_commits": dict(bd.benchmark.commits), "seed": seed, "entries": entries,
    })
    labels = {}  # type: Dict[str, int]
    for instance in instances:
        labels[instance.certificate.label] = labels.get(instance.certificate.label, 0) + 1
    report = {"manifest_hash": manifest.hash(), "n_entries": len(entries), "labels": dict(sorted(labels.items())),
              "excluded": excluded, "calibration_hash": calibration_hash,
              "task_contexts": ["%s/%s" % k for k in used]}
    return Frozen(manifest=manifest, instances=frozen, report=report)


def write_frozen(out_dir: Any, frozen: Frozen) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.yaml").write_text(yaml.safe_dump(frozen.manifest.to_dict(), sort_keys=False), encoding="utf-8")
    with open(out / "instances.jsonl", "w", encoding="utf-8", newline="\n") as handle:
        for instance in frozen.instances:
            handle.write(json.dumps(instance.to_dict(), allow_nan=False, sort_keys=True) + "\n")
    (out / "manifest.sha256").write_text(frozen.manifest.hash() + "\n", encoding="utf-8")
    (out / "freeze_report.yaml").write_text(yaml.safe_dump(frozen.report, sort_keys=False), encoding="utf-8")
    return out


def read_manifest(out_dir: Any) -> Manifest:
    with open(Path(out_dir) / "manifest.yaml", "r", encoding="utf-8") as handle:
        return Manifest.from_dict(yaml.safe_load(handle))


def verify_manifest(out_dir: Any, config_dir: Optional[Any] = None, bd: Optional[BenchmarkDescriptor] = None,
                    expected_hash: Optional[str] = None) -> str:
    """Check a frozen directory; return the manifest hash or raise :class:`ManifestError` listing every problem:
    the stored hash, every instance hash, missing / extra instances, and (when given) the config hashes against
    ``config_dir`` and the BD hash against ``bd``."""
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
