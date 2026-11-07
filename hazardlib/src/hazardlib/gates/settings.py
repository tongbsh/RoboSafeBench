"""Numbers the gates read (``gates.yaml``, ``runtime.yaml``); no defaults."""
import dataclasses
from typing import Any, Dict, Mapping, Optional

from .._numeric import non_negative_float, positive_float, positive_int
from .._settings import ValueReader, load_many
from ..config import ConfigError


@dataclasses.dataclass(frozen=True)
class G0Settings:
    max_penetration_m: float
    workspace_margin_m: float
    early_contact_tol_m: float
    max_step_ratio: float

    def __post_init__(self) -> None:
        for name in ("max_penetration_m", "workspace_margin_m"):
            object.__setattr__(self, name, non_negative_float(name, getattr(self, name)))
        for name in ("early_contact_tol_m", "max_step_ratio"):
            object.__setattr__(self, name, positive_float(name, getattr(self, name)))


@dataclasses.dataclass(frozen=True)
class EpisodeGateSettings:
    """G1 / G2 pass criterion: at most ``max_official_contacts`` official contact events and, if
    ``require_task_success``, a success, in every one of ``repeats`` episodes of ``nominal success time +
    horizon_extra_s`` seconds."""
    max_official_contacts: int
    require_task_success: bool
    repeats: int
    horizon_extra_s: float

    def __post_init__(self) -> None:
        count = self.max_official_contacts
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("max_official_contacts must be an int >= 0, got %r" % (count,))
        if not isinstance(self.require_task_success, bool):
            raise ValueError("require_task_success must be a bool, got %r" % (self.require_task_success,))
        object.__setattr__(self, "repeats", positive_int("repeats", self.repeats))
        object.__setattr__(self, "horizon_extra_s", positive_float("horizon_extra_s", self.horizon_extra_s))


@dataclasses.dataclass(frozen=True)
class GateSettings:
    """``baseline_success_required``: minimum success fraction of the reference policy over ``g1.repeats``
    hazard-free episodes. The Oracle resumes once the released hazard has been slower than
    ``resume_hazard_speed_mps`` (or outside the workspace) for ``resume_settle_s``."""
    g0: G0Settings
    g1: EpisodeGateSettings
    g2: EpisodeGateSettings
    baseline_success_required: float
    resume_hazard_speed_mps: float
    resume_settle_s: float

    def __post_init__(self) -> None:
        fraction = positive_float("baseline_success_required", self.baseline_success_required)
        if fraction > 1.0:
            raise ValueError("baseline_success_required is a fraction <= 1, got %r" % (fraction,))
        object.__setattr__(self, "baseline_success_required", fraction)
        object.__setattr__(self, "resume_hazard_speed_mps",
                           positive_float("resume_hazard_speed_mps", self.resume_hazard_speed_mps))
        object.__setattr__(self, "resume_settle_s", non_negative_float("resume_settle_s", self.resume_settle_s))

    @classmethod
    def from_configs(cls, gates: Mapping[str, Any], runtime: Mapping[str, Any]) -> "GateSettings":
        reader = ValueReader("GateSettings")
        g0 = {key: reader.get("gates", gates, "g0", key)
              for key in ("max_penetration_m", "workspace_margin_m", "early_contact_tol_m")}
        g0["max_step_ratio"] = reader.get("runtime", runtime, "max_step_ratio")
        episode = {}
        for gate in ("g1", "g2"):
            episode[gate] = {key: reader.get("gates", gates, gate, key)
                             for key in ("max_official_contacts", "require_task_success", "repeats",
                                         "horizon_extra_s")}
        rest = {key: reader.get("gates", gates, "g1", key)
                for key in ("baseline_success_required", "resume_hazard_speed_mps", "resume_settle_s")}
        reader.done()
        try:
            return cls(g0=G0Settings(**g0), g1=EpisodeGateSettings(**episode["g1"]),
                       g2=EpisodeGateSettings(**episode["g2"]), **rest)
        except ValueError as exc:
            raise ConfigError("GateSettings: %s" % exc) from None

    @classmethod
    def load(cls, config_dir: Optional[Any] = None) -> "GateSettings":
        files = load_many(["gates", "runtime"], config_dir)
        return cls.from_configs(files["gates"], files["runtime"])

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)
