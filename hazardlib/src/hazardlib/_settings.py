"""Reading the pipeline's numbers from parsed ``configs/hazard`` files (T06).

Every stage collects all the values it needs through one :class:`ValueReader` and fails once, naming every
key that is missing or still ``PLACEHOLDER`` / ``unknown``. Keys are tuples because family names contain dots
(``("mapping", "intrusion.reach", "overshoot_fraction")``). Nothing here supplies a default.
"""
from typing import Any, List, Mapping, Optional

from ._records import MARKERS
from .config import ConfigError, load_config


class ValueReader:
    def __init__(self, owner: str) -> None:
        self.owner = owner
        self.problems = []  # type: List[str]

    def get(self, file_name: str, data: Any, *path: str) -> Any:
        current = data
        for part in path:
            current = current.get(part) if isinstance(current, Mapping) else None
        dotted = "%s.%s" % (file_name, ".".join(path))
        if current is None:
            self.problems.append("%s is missing" % dotted)
        elif isinstance(current, str) and current in MARKERS:
            self.problems.append("%s=%s" % (dotted, current))
            return None
        return current

    def done(self) -> None:
        if self.problems:
            raise ConfigError("%s: %d unresolved value(s): %s"
                              % (self.owner, len(self.problems), "; ".join(self.problems)))


def load_many(names: List[str], config_dir: Optional[Any]) -> Mapping[str, Any]:
    """Parsed ``<config_dir>/<name>.yaml`` for each name (non-strict; the readers check what they use)."""
    return {name: load_config(name, config_dir=config_dir) for name in names}
