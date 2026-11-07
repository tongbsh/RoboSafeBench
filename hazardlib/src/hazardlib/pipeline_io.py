"""File formats passed between ``hazardctl`` stages (possibly in different environments).

* TaskContext, BD, calibration, reports: YAML.
* Spec library: YAML, either a list of HazardSpec mappings or ``{"specs": [...]}``.
* Instances: JSON Lines with one tagged record per line, ``{"instance": <HazardInstance>}`` or
  ``{"rejection": {...}}`` (a builder or G0 rejection with its stage and reason(s)).
"""
import importlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from .spec.benchmark_descriptor import BenchmarkDescriptor
from .spec.hazard_spec import HazardSpec
from .spec.instance import HazardInstance
from .spec.task_context import TaskContext

RECORD_TAGS = ("instance", "rejection")


def write_yaml(path: Any, data: Any) -> None:
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def read_yaml(path: Any) -> Any:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def read_task_context(path: Any) -> TaskContext:
    return TaskContext.from_dict(read_yaml(path), path=str(path))


def read_bd(path: Any) -> BenchmarkDescriptor:
    return BenchmarkDescriptor.from_dict(read_yaml(path), path=str(path))


def read_specs(path: Any) -> List[HazardSpec]:
    data = read_yaml(path)
    if isinstance(data, dict) and set(data) == {"specs"}:
        data = data["specs"]
    if not isinstance(data, list):
        raise ValueError("%s: a spec library is a list of HazardSpec mappings or {specs: [...]}" % (path,))
    return [HazardSpec.from_dict(item, path="%s[%d]" % (path, i)) for i, item in enumerate(data)]


def write_records(path: Any, records: List[Tuple[str, Dict[str, Any]]]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for tag, body in records:
            if tag not in RECORD_TAGS:
                raise ValueError("unknown record tag %r" % (tag,))
            handle.write(json.dumps({tag: body}, allow_nan=False, sort_keys=True) + "\n")


def read_records(path: Any) -> Tuple[List[HazardInstance], List[Dict[str, Any]]]:
    instances, rejections = [], []  # type: List[HazardInstance], List[Dict[str, Any]]
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or len(record) != 1 or next(iter(record)) not in RECORD_TAGS:
                raise ValueError("%s line %d: expected {\"instance\": ...} or {\"rejection\": ...}" % (path, number))
            if "instance" in record:
                instances.append(HazardInstance.from_dict(record["instance"], path="%s:%d" % (path, number)))
            else:
                rejections.append(record["rejection"])
    return instances, rejections


def load_object(reference: str) -> Any:
    """``"package.module:attribute"`` -> the attribute (a host / policy / avoider factory)."""
    module, sep, attribute = reference.partition(":")
    if not sep or not module or not attribute:
        raise ValueError("expected 'module:callable', got %r" % (reference,))
    return getattr(importlib.import_module(module), attribute)
