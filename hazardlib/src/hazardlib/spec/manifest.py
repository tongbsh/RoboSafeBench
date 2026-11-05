"""Frozen instance manifest (paradigm §2, §9)."""
import dataclasses
import re
from typing import Dict, List

from .._records import Record, check_nonempty_str, require

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


@dataclasses.dataclass
class ManifestEntry(Record):
    """Index entry for one frozen instance; ``instance_hash`` is
    ``HazardInstance.hash()`` of the instance stored alongside."""
    instance_id: str
    spec_id: str
    task_id: str
    layout_id: str
    instance_seed: int
    instance_hash: str

    def validate(self) -> None:
        for name in ("instance_id", "spec_id", "task_id", "layout_id"):
            check_nonempty_str(name, getattr(self, name))
        require(bool(_SHA256_HEX.match(self.instance_hash)),
                "instance_hash must be 64 lower-case hex chars, got %r" % (self.instance_hash,))


@dataclasses.dataclass
class Manifest(Record):
    """Everything needed to say *which* instance set a result was measured on.

    The manifest hash is ``Manifest.hash()`` (derived, never stored in itself).
    """
    manifest_version: str
    benchmark: str
    bd_hash: str
    probe_set_hash: str
    spec_library_hash: str
    adapter_version: str
    benchmark_commits: Dict[str, str]
    seed: int
    entries: List[ManifestEntry]

    def validate(self) -> None:
        for name in ("manifest_version", "benchmark", "adapter_version"):
            check_nonempty_str(name, getattr(self, name))
        for name in ("bd_hash", "probe_set_hash", "spec_library_hash"):
            value = getattr(self, name)
            require(bool(_SHA256_HEX.match(value)),
                    "%s must be 64 lower-case hex chars, got %r" % (name, value))
        ids = [e.instance_id for e in self.entries]
        require(len(set(ids)) == len(ids), "duplicate instance_id in entries")
