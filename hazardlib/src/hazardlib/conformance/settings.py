"""Settings of the conformance suite (T05).

* ``configs/hazard/conformance.yaml`` holds every tolerance and probe magnitude of the suite; the derivations are
  in its comments and in report T05. It is loaded without ``strict``: a key that is still ``PLACEHOLDER`` fails only
  the check that reads it, with the key named in the failure (never a silent default).
* The probes run on a real :class:`~hazardlib.runtime.HazardRuntime`, which needs a full ``RuntimeConfig``. Each of
  its fields is taken from ``runtime.yaml`` / ``semantics.yaml`` when it is resolved there, otherwise from
  ``conformance.yaml: probe_runtime``. The suite's checks do not depend on the probe-only values (see the YAML);
  the source of every field goes into the report.
* T_stop settings (C10) likewise come from ``intensity.yaml: calibration`` when resolved, otherwise from
  ``conformance.yaml: probe_calibration``.
"""
import numbers
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from .._records import MARKERS
from ..config import ConfigError, default_config_dir, load_yaml
from ..hashing import sha256_hex
from ..runtime import RUNTIME_KEYS, SEMANTICS_KEYS, RuntimeConfig

#: Files whose parsed content is hashed into the report (``config_hashes``).
CONFIG_FILES = ("conformance", "runtime", "semantics", "intensity")


def _resolved(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and value in MARKERS)


def _lookup(data: Any, path: Tuple[str, ...]) -> Any:
    current = data
    for part in path:
        current = current.get(part) if isinstance(current, Mapping) else None
    return current


class ConformanceConfig:
    """Parsed config files by name (``conformance``, ``runtime``, ``semantics``, ``intensity``)."""

    def __init__(self, files: Mapping[str, Any], source: str) -> None:
        missing = [name for name in CONFIG_FILES if name not in files]
        if missing:
            raise ConfigError("ConformanceConfig needs the files %s" % missing)
        self.files = dict(files)
        self.source = source

    @classmethod
    def load(cls, config_dir: Optional[Union[str, Path]] = None) -> "ConformanceConfig":
        base = Path(config_dir) if config_dir is not None else default_config_dir()
        return cls({name: load_yaml(base / (name + ".yaml")) for name in CONFIG_FILES}, str(base))

    # ----------------------------------------------------------------------------------- conformance.yaml

    def raw(self, section: str) -> Dict[str, Any]:
        data = self.files["conformance"].get(section) if isinstance(self.files["conformance"], Mapping) else None
        if not isinstance(data, Mapping):
            raise ConfigError("conformance.yaml has no section %r" % section)
        return dict(data)

    def value(self, section: str, key: str) -> Any:
        data = self.raw(section)
        if key not in data:
            raise ConfigError("conformance.yaml: %s.%s is missing" % (section, key))
        value = data[key]
        if not _resolved(value):
            raise ConfigError("conformance.yaml: %s.%s is %s; it must be set (with its derivation) before this "
                              "check can run" % (section, key, value))
        return value

    def is_set(self, section: str, key: str) -> bool:
        """Whether ``section.key`` exists and is not a placeholder."""
        data = self.raw(section)
        return key in data and _resolved(data[key])

    def number(self, section: str, key: str) -> float:
        value = self.value(section, key)
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ConfigError("conformance.yaml: %s.%s must be a number, got %r" % (section, key, value))
        return float(value)

    def integer(self, section: str, key: str) -> int:
        value = self.value(section, key)
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ConfigError("conformance.yaml: %s.%s must be an integer, got %r" % (section, key, value))
        return int(value)

    def config_hashes(self) -> Dict[str, str]:
        """``hashing.sha256_hex`` of each parsed file: formatting and comments do not count, values do."""
        return {name: sha256_hex(self.files[name]) for name in CONFIG_FILES}

    # ---------------------------------------------------------------------------- runtime / calibration

    def runtime_config(self) -> Tuple[RuntimeConfig, Dict[str, Dict[str, Any]]]:
        """The probes' RuntimeConfig and, per field, ``{"value": ..., "source": "<file>.<key>"}``."""
        sources = [("runtime", (key,)) for key in RUNTIME_KEYS]
        sources += [("semantics", path) for path in SEMANTICS_KEYS]
        values = {}  # type: Dict[str, Any]
        used = {}  # type: Dict[str, Dict[str, Any]]
        for file_name, path in sources:
            field = path[-1]
            value = _lookup(self.files[file_name], path)
            where = "%s.yaml:%s" % (file_name, ".".join(path))
            if not _resolved(value):
                value = self.value("probe_runtime", field)
                where = "conformance.yaml:probe_runtime.%s" % field
            values[field] = value
            used[field] = {"value": value, "source": where}
        try:
            return RuntimeConfig(**values), used
        except ValueError as exc:
            raise ConfigError("probe RuntimeConfig: %s" % exc) from None

    def t_stop_settings(self, substeps_per_action: int) -> Tuple[float, int, Dict[str, Dict[str, Any]]]:
        """``(v_threshold_mps, k_samples, sources)`` for ``estimate_t_stop`` on physics-step samples."""
        used = {}  # type: Dict[str, Dict[str, Any]]
        calibration = _lookup(self.files["intensity"], ("calibration",))
        threshold = _lookup(calibration, ("t_stop_speed_threshold_mps",))
        if _resolved(threshold):
            used["v_threshold_mps"] = {"value": threshold, "source": "intensity.yaml:calibration."
                                                                     "t_stop_speed_threshold_mps"}
        else:
            threshold = self.number("probe_calibration", "t_stop_speed_threshold_mps")
            used["v_threshold_mps"] = {"value": threshold, "source": "conformance.yaml:probe_calibration."
                                                                     "t_stop_speed_threshold_mps"}
        k = _lookup(calibration, ("t_stop_min_consecutive_samples",))
        if _resolved(k):
            used["k_samples"] = {"value": k, "source": "intensity.yaml:calibration.t_stop_min_consecutive_samples"}
        else:
            steps = self.integer("probe_calibration", "t_stop_min_consecutive_control_steps")
            k = steps * int(substeps_per_action)
            used["k_samples"] = {"value": k, "source": "conformance.yaml:probe_calibration."
                                                       "t_stop_min_consecutive_control_steps x substeps_per_action"}
        return float(threshold), int(k), used
