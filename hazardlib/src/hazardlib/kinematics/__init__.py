"""Seconds-based motion primitives, nominal-collision construction and sub-stepping (T02)."""
from .hit import (FAMILIES, FAMILY_FALL, FAMILY_REACH, FAMILY_SLIDE, HitPlan, build_hit_path)
from .path import Path, SampledPath
from .primitives import (PROFILES, BallisticPath, LinePath, arc_path, fall_height_for_tau,
                         fall_time_for_height, free_fall_path, min_jerk, min_jerk_derivative,
                         min_jerk_inverse, profile_inverse, reach_path, straight_path)
from .subdivide import hook_times, needs_substeps, step_displacements, subdivide

__all__ = [
    "BallisticPath", "FAMILIES", "FAMILY_FALL", "FAMILY_REACH", "FAMILY_SLIDE", "HitPlan",
    "LinePath", "PROFILES", "Path", "SampledPath", "arc_path", "build_hit_path",
    "fall_height_for_tau", "fall_time_for_height", "free_fall_path", "hook_times", "min_jerk",
    "min_jerk_derivative", "min_jerk_inverse", "needs_substeps", "profile_inverse", "reach_path",
    "step_displacements", "straight_path", "subdivide",
]
