"""Numbers of the run stage (``configs/hazard/run.yaml``, R1); no defaults.

``run.yaml`` is always loaded with ``strict=True``: a ``PLACEHOLDER`` anywhere in it refuses the command (the
values that ``run`` and ``report`` use are all in this one small file). ``run_config_hash`` is the sha256 of the
parsed file, computed like ``Manifest.config_hashes`` (``hazardlib.hashing``) and recorded in every run.
"""
import dataclasses
from typing import Any, Dict, List, Mapping, Optional

from .._numeric import finite_float, non_negative_float
from .._settings import ValueReader
from ..config import ConfigError, load_config
from ..hashing import sha256_hex
from .stats import INTERVAL_METHODS

RUN_CONFIG = "run"


def _edges(name: str, value: Any) -> List[float]:
    if not isinstance(value, list) or not value:
        raise ValueError("%s must be a non-empty list of increasing numbers, got %r" % (name, value))
    edges = [finite_float("%s[%d]" % (name, k), v) for k, v in enumerate(value)]
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("%s must be strictly increasing, got %r" % (name, edges))
    return edges


@dataclasses.dataclass(frozen=True)
class EpisodeSettings:
    horizon_extra_s: float
    stop_on_success: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "horizon_extra_s", non_negative_float("horizon_extra_s", self.horizon_extra_s))
        if not isinstance(self.stop_on_success, bool):
            raise ValueError("stop_on_success must be true or false, got %r" % (self.stop_on_success,))


@dataclasses.dataclass(frozen=True)
class ReportSettings:
    tau_bin_edges_s: List[float]
    sigma_bin_edges: List[float]
    interval_method: str
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "tau_bin_edges_s", _edges("tau_bin_edges_s", self.tau_bin_edges_s))
        object.__setattr__(self, "sigma_bin_edges", _edges("sigma_bin_edges", self.sigma_bin_edges))
        if self.interval_method not in INTERVAL_METHODS:
            raise ValueError("interval.method %r is not one of %s" % (self.interval_method, list(INTERVAL_METHODS)))
        confidence = finite_float("confidence", self.confidence)
        if not 0.0 < confidence < 1.0:
            raise ValueError("interval.confidence must lie in (0, 1), got %r" % (confidence,))
        object.__setattr__(self, "confidence", confidence)


@dataclasses.dataclass(frozen=True)
class RunSettings:
    episode: EpisodeSettings
    report: ReportSettings
    config_hash: str

    @classmethod
    def from_config(cls, run: Mapping[str, Any]) -> "RunSettings":
        reader = ValueReader("RunSettings")
        episode = {key: reader.get("run", run, "episode", key) for key in ("horizon_extra_s", "stop_on_success")}
        report = {key: reader.get("run", run, "report", key) for key in ("tau_bin_edges_s", "sigma_bin_edges")}
        report["interval_method"] = reader.get("run", run, "report", "interval", "method")
        report["confidence"] = reader.get("run", run, "report", "interval", "confidence")
        reader.done()
        try:
            return cls(episode=EpisodeSettings(**episode), report=ReportSettings(**report),
                       config_hash=sha256_hex(run))
        except ValueError as exc:
            raise ConfigError("RunSettings: %s" % exc) from None

    @classmethod
    def load(cls, config_dir: Optional[Any] = None) -> "RunSettings":
        return cls.from_config(load_config(RUN_CONFIG, strict=True, config_dir=config_dir))

    def to_dict(self) -> Dict[str, Any]:
        return {"episode": dataclasses.asdict(self.episode), "report": dataclasses.asdict(self.report),
                "config_hash": self.config_hash}
