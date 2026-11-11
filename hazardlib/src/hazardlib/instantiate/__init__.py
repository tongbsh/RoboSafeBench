"""Instance builder: HazardSpec + TaskContext -> HazardInstance (T06, paradigm §5.2)."""
from .builder import (IDENTITY_QUAT, PRIMITIVES, REJECT_REASONS, InstanceRejected, half_height_m, instantiate,
                      slide_support_rule, support_radius_m)
from .mapping import (ReachGeometry, ZeroOvershootError, fall_release_height_m, reach_geometry,
                      slide_start_distance_m)
from .settings import FAMILY_AXES, InstantiateSettings

__all__ = ["FAMILY_AXES", "IDENTITY_QUAT", "InstanceRejected", "InstantiateSettings", "PRIMITIVES",
           "REJECT_REASONS", "ReachGeometry", "ZeroOvershootError", "fall_release_height_m", "half_height_m",
           "instantiate", "reach_geometry", "slide_start_distance_m", "slide_support_rule", "support_radius_m"]
