"""Unified safety semantics (00 §6, D9): engine-independent pure functions and data classes (T03)."""
from .contact_graph import (ConstraintState, ContactSegment, ContactStep, ContactSummary,
                            continuous_runs, summarize_contacts)
from .event_log import (SCHEMA_VERSION, HazardEventLog, append_jsonl, read_jsonl, write_jsonl)
from .events import HELD_OBJECT_TOKEN, ContactEvent, EventKind, classify_partner
from .exposure import is_exposure_low, min_distance_to_hit_point
from .invalid import (DenominatorSummary, InvalidAttribution, InvalidClass, classify_invalid,
                      metric_denominator)
from .proximity import ProximitySummary, SeparationKind, compute_proximity
from .severity import analytic_impulse_ns, closing_normal_speed
from .topple import is_toppled, tilt_deg_up_axis

__all__ = [
    "ConstraintState", "ContactEvent", "ContactSegment", "ContactStep", "ContactSummary",
    "DenominatorSummary", "EventKind", "HELD_OBJECT_TOKEN", "HazardEventLog", "InvalidAttribution",
    "InvalidClass", "ProximitySummary", "SCHEMA_VERSION", "SeparationKind", "analytic_impulse_ns",
    "append_jsonl", "classify_invalid", "classify_partner", "closing_normal_speed", "compute_proximity",
    "continuous_runs", "is_exposure_low", "is_toppled", "metric_denominator", "min_distance_to_hit_point",
    "read_jsonl", "summarize_contacts", "tilt_deg_up_axis", "write_jsonl",
]
