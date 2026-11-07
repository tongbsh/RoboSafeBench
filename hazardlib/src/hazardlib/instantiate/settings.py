"""Numbers the instance builder reads (``runtime.yaml``, ``intensity.yaml``); no defaults."""
import dataclasses
from typing import Any, Dict, Mapping, Optional, Tuple

from .._numeric import finite_float, non_negative_float, positive_float
from .._settings import ValueReader, load_many
from ..config import ConfigError
from ..kinematics.hit import FAMILY_FALL, FAMILY_REACH, FAMILY_SLIDE

#: Intensity axes per family: (required, optional).
FAMILY_AXES = {
    FAMILY_FALL: (("tau_s",), ("mass_kg",)),
    FAMILY_SLIDE: (("tau_s", "approach_speed_mps"), ()),
    FAMILY_REACH: (("tau_s", "approach_speed_mps"), ("limb_size_m",)),
}  # type: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]]


@dataclasses.dataclass(frozen=True)
class InstantiateSettings:
    """``intensity_ranges[family][axis] = (lo, hi)``: a spec value outside is a spec error."""
    release_velocity_window_s: float
    keyframe_dt_s: float
    keyframe_interp_tol_m: float
    reach_overshoot_fraction: float
    intensity_ranges: Mapping[str, Mapping[str, Tuple[float, float]]]

    def __post_init__(self) -> None:
        for name in ("release_velocity_window_s", "keyframe_dt_s", "keyframe_interp_tol_m"):
            object.__setattr__(self, name, positive_float(name, getattr(self, name)))
        # 0 is allowed: the builder then rejects every reach instance (D6).
        object.__setattr__(self, "reach_overshoot_fraction",
                           non_negative_float("reach_overshoot_fraction", self.reach_overshoot_fraction))
        ranges = {}
        for family, axes in self.intensity_ranges.items():
            ranges[family] = {}
            for axis, (lo, hi) in axes.items():
                lo = finite_float("%s.%s.lo" % (family, axis), lo)
                hi = finite_float("%s.%s.hi" % (family, axis), hi)
                if lo > hi:
                    raise ValueError("%s.%s: lo=%r exceeds hi=%r" % (family, axis, lo, hi))
                ranges[family][axis] = (lo, hi)
        object.__setattr__(self, "intensity_ranges", ranges)

    @classmethod
    def from_configs(cls, runtime: Mapping[str, Any], intensity: Mapping[str, Any]) -> "InstantiateSettings":
        reader = ValueReader("InstantiateSettings")
        values = {key: reader.get("runtime", runtime, key)
                  for key in ("release_velocity_window_s", "keyframe_dt_s", "keyframe_interp_tol_m")}
        values["reach_overshoot_fraction"] = reader.get("intensity", intensity, "mapping", FAMILY_REACH,
                                                        "overshoot_fraction")
        ranges = {}  # type: Dict[str, Dict[str, Tuple[Any, Any]]]
        for family, (required, optional) in FAMILY_AXES.items():
            ranges[family] = {}
            for axis in required + optional:
                lo = reader.get("intensity", intensity, "intensity", family, axis, "lo")
                hi = reader.get("intensity", intensity, "intensity", family, axis, "hi")
                ranges[family][axis] = (lo, hi)
        values["intensity_ranges"] = ranges
        reader.done()
        try:
            return cls(**values)
        except ValueError as exc:
            raise ConfigError("InstantiateSettings: %s" % exc) from None

    @classmethod
    def load(cls, config_dir: Optional[Any] = None) -> "InstantiateSettings":
        files = load_many(["runtime", "intensity"], config_dir)
        return cls.from_configs(files["runtime"], files["intensity"])

    def to_dict(self) -> Dict[str, Any]:
        return {"release_velocity_window_s": self.release_velocity_window_s, "keyframe_dt_s": self.keyframe_dt_s,
                "keyframe_interp_tol_m": self.keyframe_interp_tol_m,
                "reach_overshoot_fraction": self.reach_overshoot_fraction,
                "intensity_ranges": {f: {a: list(r) for a, r in sorted(axes.items())}
                                     for f, axes in sorted(self.intensity_ranges.items())}}
