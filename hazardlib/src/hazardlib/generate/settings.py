"""Numbers the generator reads (a mapping parsed from the benchmark's pilot YAML); no defaults in code.

Read through :class:`hazardlib._settings.ValueReader` like ``InstantiateSettings`` / ``GateSettings``: every
missing or ``PLACEHOLDER`` / ``unknown`` key is collected and reported once.
"""
import dataclasses
import math
from typing import Any, Dict, Mapping, Optional, Tuple

from .._numeric import finite_float, non_negative_float, positive_float
from .._settings import ValueReader
from ..config import ConfigError, load_config
from ..kinematics.hit import FAMILY_FALL, FAMILY_REACH, FAMILY_SLIDE

FAMILIES = (FAMILY_REACH, FAMILY_FALL, FAMILY_SLIDE)
DIRECTION_FRAMES = ("segment_entity", "demo_first_entity")
PREFILTER_MODES = ("reject", "record", "off")
OWNER = "GenerateSettings"


@dataclasses.dataclass(frozen=True)
class EntrySettings:
    """Entry directions of one family: ``azimuth_count`` equally spaced azimuths in the segment entity's initial
    frame x each elevation (radians below the horizontal the hazard moves down along; 0 = from the side)."""
    azimuth_count: int
    elevations_rad: Tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class GenerateSettings:
    families: Tuple[str, ...]
    sigma_targets: Tuple[float, ...]
    minimum_lead_control_steps: int
    step_tolerance_fraction: float
    anchor_quantiles: Tuple[float, ...]
    speed_quantiles: Tuple[float, ...]
    max_candidates_per_family_layout: int
    follow_through: Mapping[str, str]
    entry_directions: Mapping[str, EntrySettings]
    direction_frame: str
    coverage_radius_m: float
    coverage_ordering: bool
    thinned_point_spacing_m: float
    gripper_body_prefixes: Tuple[str, ...]
    risk_search_m: float
    scene_clearance_tol_m: float
    scene_clearance_spacing_half_extent_fraction: float
    layout_prefilter: str
    layout_prefilter_slack_m: float
    workspace_object_margin_multiple: float

    def __post_init__(self) -> None:
        for family in self.families:
            if family not in FAMILIES:
                raise ValueError("unknown family %r" % family)
        if self.direction_frame not in DIRECTION_FRAMES:
            raise ValueError("direction_frame %r is not one of %s" % (self.direction_frame, list(DIRECTION_FRAMES)))
        if self.layout_prefilter not in PREFILTER_MODES:
            raise ValueError("layout_prefilter %r is not one of %s" % (self.layout_prefilter, list(PREFILTER_MODES)))
        for name in ("step_tolerance_fraction", "coverage_radius_m", "thinned_point_spacing_m", "risk_search_m",
                     "scene_clearance_tol_m", "scene_clearance_spacing_half_extent_fraction"):
            positive_float(name, getattr(self, name))
        non_negative_float("layout_prefilter_slack_m", self.layout_prefilter_slack_m)
        for family in self.families:
            if family == FAMILY_FALL:
                continue
            if family not in self.entry_directions:
                raise ValueError("entry_directions has no entry for %s" % family)
            entry = self.entry_directions[family]
            if entry.azimuth_count < 1 or not entry.elevations_rad:
                raise ValueError("entry_directions.%s needs azimuth_count >= 1 and at least one elevation" % family)
            if family == FAMILY_SLIDE and any(e != 0.0 for e in entry.elevations_rad):
                raise ValueError("a slide moves along a support surface: entry_directions.%s.elevations_rad must "
                                 "be [0.0]" % family)
            if any(not 0.0 <= e < 0.5 * math.pi for e in entry.elevations_rad):
                raise ValueError("entry_directions.%s.elevations_rad must lie in [0, pi/2)" % family)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], file_name: str = "trajectory_analysis") -> "GenerateSettings":
        reader = ValueReader(OWNER)
        keys = ("families", "sigma_targets", "minimum_lead_control_steps", "step_tolerance_fraction",
                "anchor_quantiles", "speed_quantiles", "max_candidates_per_family_layout", "follow_through",
                "direction_frame", "coverage_radius_m", "coverage_ordering", "thinned_point_spacing_m",
                "gripper_body_prefixes", "risk_search_m", "scene_clearance_tol_m",
                "scene_clearance_spacing_half_extent_fraction", "layout_prefilter", "layout_prefilter_slack_m",
                "workspace_object_margin_multiple")
        values = {key: reader.get(file_name, data, key) for key in keys}
        entry_raw = reader.get(file_name, data, "entry_directions")
        entry = {}  # type: Dict[str, EntrySettings]
        if isinstance(entry_raw, Mapping):
            for family, e in entry_raw.items():
                count = reader.get(file_name, e, "azimuth_count")
                elevations = reader.get(file_name, e, "elevations_rad")
                if count is not None and elevations is not None:
                    entry[family] = EntrySettings(int(count), tuple(float(x) for x in elevations))
        reader.done()
        for key in ("families", "sigma_targets", "anchor_quantiles", "speed_quantiles", "gripper_body_prefixes"):
            values[key] = tuple(values[key])
        for key in ("sigma_targets", "anchor_quantiles", "speed_quantiles"):
            values[key] = tuple(finite_float(key, v) for v in values[key])
        values["entry_directions"] = entry
        values["follow_through"] = dict(values["follow_through"])
        try:
            return cls(**values)
        except ValueError as exc:
            raise ConfigError("%s: %s" % (OWNER, exc)) from None

    @classmethod
    def load(cls, config_dir: Optional[Any] = None, name: str = "trajectory_analysis") -> "GenerateSettings":
        return cls.from_mapping(load_config(name, config_dir=config_dir), name)

    def elevations(self, family: str) -> Tuple[Optional[float], ...]:
        """The elevation classes of a family: ``(None,)`` for the vertical fall."""
        if family == FAMILY_FALL:
            return (None,)
        return self.entry_directions[family].elevations_rad


def elevation_tag(elevation_rad: Optional[float]) -> str:
    """``vertical`` (fall) or ``el<degrees>`` (``el00`` from the side, ``el30``, ``el60`` from above)."""
    if elevation_rad is None:
        return "vertical"
    return "el%02d" % int(round(elevation_rad * 180.0 / math.pi))


def cell_id(family: str, elevation_rad: Optional[float]) -> str:
    """``<family>@<tag>``: one (family, entry-direction class); a result cell is this x a Sigma target."""
    return "%s@%s" % (family, elevation_tag(elevation_rad))
