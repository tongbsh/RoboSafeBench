"""Behavior diagnostics: per-step trace capture and react/stop/avoid/recover metrics.

``core/`` owns outcomes (success x contact x force). This package owns process:
what the policy did between disturbance onset and episode end.

Rollout wiring uses ``DiagnosticsSession`` (the four primitives stay available)::

    session = DiagnosticsSession.create(cfg, episode_index=i, bound=bound, controller=ctrl)
    session.record_step(t=task_step, eef_pos=..., action=..., in_contact=..., ...)
    session.finalize(success=..., success_step=..., metrics=metrics)
    record = build_episode_record(..., diagnostics=session.episode_diagnostics())

Thresholds live in ``configs/base.yaml`` under ``diagnostics.*`` and reach this
package as ``DiagnosticsConfig`` (defined in ``dynisafety.config.schema``, next
to the other config dataclasses).

See docs/user/behavior_diagnostics.md for the operational definitions.
"""

from __future__ import annotations

from dynisafety.diagnostics.analyze import analyze_trace
from dynisafety.diagnostics.probes import make_probe
from dynisafety.diagnostics.reader import (
    TraceSchemaVersionError,
    TruncatedTraceError,
    iter_traces,
    read_index,
    read_trace,
)
from dynisafety.diagnostics.recorder import StepTraceRecorder
from dynisafety.diagnostics.schema import (
    TRACE_SCHEMA_VERSION,
    BehaviorClass,
    BehaviorDiagnostics,
    DisturbancePhase,
    DisturbanceProbe,
    DisturbanceState,
    StepRecord,
    Trace,
    TraceHeader,
)
from dynisafety.diagnostics.wiring import DiagnosticsSession, copy_vec
from dynisafety.diagnostics.writer import TraceWriter

__all__ = [
    "TRACE_SCHEMA_VERSION",
    "BehaviorClass",
    "BehaviorDiagnostics",
    "DiagnosticsSession",
    "DisturbancePhase",
    "DisturbanceProbe",
    "DisturbanceState",
    "StepRecord",
    "StepTraceRecorder",
    "Trace",
    "TraceHeader",
    "TraceSchemaVersionError",
    "TraceWriter",
    "TruncatedTraceError",
    "analyze_trace",
    "copy_vec",
    "iter_traces",
    "make_probe",
    "read_index",
    "read_trace",
]
