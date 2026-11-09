"""The inputs of ``hazardctl run`` in its two modes (D16 c).

* **Formal** (:func:`load_formal`): a frozen Manifest directory. :func:`verify_manifest` must pass (stored hash,
  every instance hash, the conformance report and the calibration file, the config hashes against the run's
  ``--config-dir`` and the BD hash against ``--bd``); otherwise the run is refused. The instances and the
  calibration are read from the frozen directory. The frozen directory holds neither the TaskContexts nor the BD
  (``write_frozen`` does not copy them), so they are given on the command line and checked against the frozen
  state: the BD by its hash (``verify_manifest``), every TaskContext by the ``task_context_hash`` its instances
  carry, and the set of used TaskContexts by recomputing the Manifest's ``probe_set_hash``. Records carry the
  ``manifest_hash``.
* **Pilot** (:func:`load_pilot`): gated instances (``hazardctl gate`` output), TaskContexts, a BD and optionally a
  calibration, with a pilot configuration directory (D14). Every record and summary is marked ``pilot: true`` and
  carries no ``manifest_hash``; such results are never mixed with formal ones (``hazardctl report`` refuses).

Both modes refuse an instance without a certificate (run the gates first) and duplicate instance ids.
"""
import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..calibrate.protocol import summary_from_dict
from ..calibrate.stop import TStopSummary
from ..hashing import sha256_hex
from ..manifest.freeze import CALIBRATION_FILE, read_calibration, read_manifest, verify_manifest
from ..pipeline_io import read_records, read_task_context
from ..runtime import Provenance
from ..spec.benchmark_descriptor import BenchmarkDescriptor
from ..spec.instance import HazardInstance
from ..spec.task_context import TaskContext

ContextKey = Tuple[str, str]


class RunInputError(ValueError):
    """The run inputs are inconsistent (refused; every problem is listed)."""


@dataclasses.dataclass
class RunInputs:
    """``contexts`` maps ``(task_id, layout_id)`` to its TaskContext; ``calibration`` is the parsed
    ``calibration.yaml`` (``None`` in a pilot run without one) and ``calibration_hash`` its ``sha256_hex``."""
    pilot: bool
    manifest_hash: Optional[str]
    instances: List[HazardInstance]
    contexts: Dict[ContextKey, TaskContext]
    bd: BenchmarkDescriptor
    calibration: Optional[Dict[str, Any]]
    calibration_hash: Optional[str]
    provenance: Provenance
    sources: Dict[str, Any]

    @property
    def t_stop(self) -> Optional[TStopSummary]:
        return None if self.calibration is None else summary_from_dict(self.calibration["t_stop"])

    def context_of(self, instance: HazardInstance) -> TaskContext:
        return self.contexts[instance_key(instance)]


def instance_key(instance: HazardInstance) -> ContextKey:
    prov = instance.provenance
    return prov.get("task_id"), prov.get("layout_id")


def _read_contexts(paths: Sequence[Any]) -> Dict[ContextKey, TaskContext]:
    out = {}  # type: Dict[ContextKey, TaskContext]
    for path in paths:
        tc = read_task_context(path)
        key = (tc.task_id, tc.layout_id)
        if key in out and out[key].hash() != tc.hash():
            raise RunInputError("two different TaskContexts for %s/%s (%s)" % (key[0], key[1], path))
        out[key] = tc
    return out


def _check_instances(instances: Sequence[HazardInstance], contexts: Mapping[ContextKey, TaskContext],
                     bd: BenchmarkDescriptor) -> None:
    problems = []  # type: List[str]
    ids = [i.instance_id for i in instances]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        problems.append("duplicate instance ids %s" % duplicates)
    for instance in instances:
        if instance.certificate is None:
            problems.append("instance %r has no certificate; run the gates first" % instance.instance_id)
        key = instance_key(instance)
        tc = contexts.get(key)
        if tc is None:
            problems.append("instance %r: no TaskContext given for %s/%s" % (instance.instance_id, key[0], key[1]))
        elif instance.provenance.get("task_context_hash") != tc.hash():
            problems.append("instance %r was built on TaskContext %s, not on the given %s/%s (%s)"
                            % (instance.instance_id, instance.provenance.get("task_context_hash"), key[0], key[1],
                               tc.hash()))
        built_on = instance.provenance.get("bd_hash")
        if built_on is not None and built_on != bd.hash():
            problems.append("instance %r was built for BD %s, not for the given BD %r (%s)"
                            % (instance.instance_id, built_on, bd.name, bd.hash()))
    if problems:
        raise RunInputError("%d problem(s) with the run inputs:\n  %s" % (len(problems), "\n  ".join(problems)))


def _read_frozen_instances(manifest_dir: Path, order: Sequence[str]) -> List[HazardInstance]:
    by_id = {}  # type: Dict[str, HazardInstance]
    with open(manifest_dir / "instances.jsonl", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                instance = HazardInstance.from_dict(json.loads(line))
                by_id[instance.instance_id] = instance
    return [by_id[instance_id] for instance_id in order]


def load_formal(manifest_dir: Any, task_context_paths: Sequence[Any], bd: BenchmarkDescriptor,
                config_dir: Any) -> RunInputs:
    """See the module docstring. Raises ``ManifestError`` (from ``verify_manifest``) or :class:`RunInputError`."""
    out = Path(manifest_dir)
    digest = verify_manifest(out, config_dir=config_dir, bd=bd)
    manifest = read_manifest(out)
    instances = _read_frozen_instances(out, [e.instance_id for e in manifest.entries])
    contexts = _read_contexts(task_context_paths)
    _check_instances(instances, contexts, bd)
    used = sorted({instance_key(i) for i in instances})
    probe_set_hash = sha256_hex(sorted(contexts[k].provenance["probe_set_hash"] for k in used)) if used \
        else sha256_hex([])
    if probe_set_hash != manifest.probe_set_hash:
        raise RunInputError("the given TaskContexts' probe sets hash to %s, the Manifest says %s"
                            % (probe_set_hash, manifest.probe_set_hash))
    calibration = read_calibration(out / CALIBRATION_FILE)
    return RunInputs(pilot=False, manifest_hash=digest, instances=instances,
                     contexts={k: contexts[k] for k in used}, bd=bd, calibration=calibration,
                     calibration_hash=manifest.calibration_hash, provenance=Provenance.from_manifest(manifest),
                     sources={"manifest_dir": str(manifest_dir), "task_contexts": [str(p) for p in task_context_paths]})


def load_pilot(instances_path: Any, task_context_paths: Sequence[Any], bd: BenchmarkDescriptor,
               calibration_path: Optional[Any]) -> RunInputs:
    """See the module docstring. Rejection records in ``instances_path`` are ignored (they are not instances)."""
    instances, _ = read_records(instances_path)
    contexts = _read_contexts(task_context_paths)
    _check_instances(instances, contexts, bd)
    calibration = None if calibration_path is None else read_calibration(calibration_path)
    used = sorted({instance_key(i) for i in instances})
    provenance = Provenance(bd_hash=bd.hash(), adapter_version=None, manifest_hash=None,
                            benchmark_commits=dict(bd.benchmark.commits))
    return RunInputs(pilot=True, manifest_hash=None, instances=instances, contexts={k: contexts[k] for k in used},
                     bd=bd, calibration=calibration,
                     calibration_hash=None if calibration is None else sha256_hex(calibration),
                     provenance=provenance,
                     sources={"instances": str(instances_path), "task_contexts": [str(p) for p in task_context_paths],
                              "calibration": None if calibration_path is None else str(calibration_path)})
