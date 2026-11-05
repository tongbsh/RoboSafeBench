"""Simulator-independent spec objects (paradigm §2). Data + validation only."""
from .benchmark_descriptor import (CONTACT_TIERS, DISTANCE_KINDS, LOOP_OWNERS, REPRO_LEVELS,
                                   SPAWN_MODES, TIER_A_REQUIREMENTS, BenchmarkDescriptor,
                                   TierUndeterminedError)
from .certificate import CERT_LABELS, Certificate
from .hazard_spec import (TIMING_MODES, AgentSpec, ExposureSpec, HazardSpec, MotionSpec,
                          PhaseRef, ReleaseSpec, TimingSpec)
from .instance import HazardInstance, ReleaseVelocity
from .manifest import Manifest, ManifestEntry
from .pose_sequence import PoseSequence
from .task_context import (ENTITY_SOURCES, AABB, EntityRef, Occupancy, Phase, TaskContext)

__all__ = [
    "AABB", "AgentSpec", "BenchmarkDescriptor", "CERT_LABELS", "CONTACT_TIERS", "Certificate",
    "DISTANCE_KINDS", "ENTITY_SOURCES", "EntityRef", "ExposureSpec", "HazardInstance",
    "HazardSpec", "LOOP_OWNERS", "Manifest", "ManifestEntry", "MotionSpec", "Occupancy",
    "Phase", "PhaseRef", "PoseSequence", "REPRO_LEVELS", "ReleaseSpec", "ReleaseVelocity",
    "SPAWN_MODES", "TIER_A_REQUIREMENTS", "TIMING_MODES", "TaskContext", "TierUndeterminedError",
    "TimingSpec",
]
