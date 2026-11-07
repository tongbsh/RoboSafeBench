"""hazardlib: simulator-agnostic core of the Hazard-as-Spec paradigm.

T01 scope: data structures and interfaces only (no kinematics, semantics,
runtime, probes or conformance tests). Depends on numpy and pyyaml only.
"""
__version__ = "0.1.0"

from . import hashing, spec  # noqa: F401  (import order: host first via spec)
from .config import (ConfigError, default_config_dir, load_bd, load_config, load_yaml)
from .host import (Action, CapabilityError, ContactObs, HazardHandle, HazardHost, HazardState,
                   OptionalHazardHost, PartnerClass, Pose, RobotState, Surface, TimeState, Tier,
                   missing_methods, require_methods)
from ._records import PLACEHOLDER, UNKNOWN, SpecError

__all__ = [
    "Action", "CapabilityError", "ConfigError", "ContactObs", "HazardHandle", "HazardHost",
    "HazardState", "OptionalHazardHost", "PLACEHOLDER", "PartnerClass", "Pose", "RobotState",
    "SpecError", "Surface", "Tier", "TimeState", "UNKNOWN", "default_config_dir", "hashing",
    "load_bd", "load_config", "load_yaml", "missing_methods", "require_methods", "spec",
]
