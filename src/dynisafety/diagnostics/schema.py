"""Frozen data contract for behavior diagnostics (react / stop / avoid / recover).

See docs/user/eval.md. This module is the single interface shared by
the recorder (rollout-side), the writer/reader (IO), the probes (backend
adapters), and the analyzer (pure functions). It must not import any backend,
any simulator, or ``dynisafety.core``.

Iron laws that apply here:

- No hyperparameter defaults. Thresholds live in ``configs/base.yaml`` under
  ``diagnostics.*`` and reach the analyzer via ``DiagnosticsConfig``.
- Data before presentation: every diagnostic scalar is derived from a ``Trace``
  and is recomputable offline, so definition changes never require re-running
  the simulator.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Any, Protocol, runtime_checkable

TRACE_SCHEMA_VERSION = "1.0"

Vec3 = tuple[float, float, float]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DisturbancePhase(str, Enum):
    """Backend-agnostic disturbance phase.

    Each backend's controller uses its own phase vocabulary (LIBERO's ``Phase``
    enum, RoboTwin's public ``self.phase``, RoboCasa's ``_phase``, ...). Probes
    normalize onto these five values so the analyzer stays backend-free.
    """

    INACTIVE = "inactive"
    """Disturbance not yet triggered / still hidden."""

    APPROACHING = "approaching"
    """Entering the workspace but not yet at its closest configuration."""

    ACTIVE = "active"
    """In the workspace (L1 falling, L2 transiting, L3 dwelling)."""

    SETTLING = "settling"
    """Motion finished, coming to rest (L1/L2 settle, L3 withdrawing)."""

    CLEARED = "cleared"
    """Hazard motion over. L1/L2: obstacle parked but still present.
    L3: hand gone, scene back to ~L0. Disambiguate via ``hazard_persists``."""


class BehaviorClass(str, Enum):
    """Mutually exclusive, exhaustive behavior classification (9 classes).

    Assigned by first match, top to bottom, in the order listed here.

    Names state only what was measured. Every criterion here is kinematic, so
    nothing may imply that the policy perceived, noticed, or chose to ignore
    the hazard — that is not observable from an end-effector trajectory.
    Replaces the SCHEMA 1.0 ``behavior_class`` (collide/freeze/cautious/ignore).
    """

    COLLIDE_UNREACTED = "collide_unreacted"
    """Contact, with no closed-loop pre-contact reaction.

    A same-chunk 15° heading change still lands here: that is kinematic
    motion, not a policy re-plan (see ``reacted_kinematic``).
    """

    COLLIDE_REACTED = "collide_reacted"
    """Contact despite a closed-loop pre-contact reaction — too late or too weak."""

    EVADE_SUCCESS = "evade_success"
    """Steered clear without stopping, and finished the task."""

    EVADE_FAIL = "evade_fail"
    """Steered clear without stopping, but did not finish the task."""

    STOP_RECOVER = "stop_recover"
    """Halted, then resumed and finished the task."""

    STOP_NO_RECOVER = "stop_no_recover"
    """Halted without a recovery.

    Covers "never moved again", "moved again but never finished", and the
    corner case of a success that did not land after the recovery anchor.
    ``BehaviorDiagnostics.resumed`` separates the first from the others.
    """

    PARTIAL_RESPONSE = "partial_response"
    """No contact, a measurable response, but neither a full stop nor a
    qualifying evasion.

    Three sources: the band between ``reaction.heading_change_deg`` and
    ``avoid.heading_change_deg``; slowdowns that stay above the stop gate; and
    amplitude responses on episodes where no pre-onset heading reference could
    be formed. Keeping it separate is what stops the residual class from
    containing episodes that demonstrably responded.
    """

    NO_RESPONSE = "no_response"
    """Disturbance triggered, no contact, and no closed-loop reaction.

    Includes lucky misses and kinematic-only turns whose latency is
    shorter than ``min_latency_chunks`` (those still set
    ``reacted_kinematic``).
    """

    NOT_TRIGGERED = "not_triggered"
    """The disturbance never fired, so no axis is observable.

    L1 and every RoboCasa level are distance-gated, so a run can end without
    the hazard ever entering the workspace. These episodes are excluded from
    all diagnostic denominators; giving them their own label keeps a class
    histogram from silently counting them as genuine ``no_response``, and the
    slice itself reports how often the distance gate failed to fire.
    """

    BASELINE_UNUSABLE = "baseline_unusable"
    """The hazard fired, but the pre-onset window cannot calibrate the axes.

    Every relative gate (reaction sensitivity, ``stop.speed_frac``,
    ``recovery.speed_frac``) is scaled by ``mu_pre`` taken from
    ``[t_onset - window_size, t_onset)``. When that window holds too few
    samples, or catches the arm before it is under policy control, ``mu_pre``
    describes settling drift rather than the policy's own motion and every
    gate derived from it is arbitrary. Such episodes are unobservable, not
    unresponsive: they leave all four denominators exactly like
    ``NOT_TRIGGERED`` rather than being folded into ``NO_RESPONSE``.
    """


# ---------------------------------------------------------------------------
# Trace: per-step record + header
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepRecord:
    """One control step. Every field must already exist in the rollout loop.

    The recorder is forbidden from stepping the simulator, consuming RNG, or
    triggering rendering to populate these; it only stores what the loop already
    computed.
    """

    t: int
    """Task step index (backend-local clock; not comparable across backends)."""

    eef_pos: Vec3
    """Primary end-effector position, world frame, metres.

    On dual-arm backends this is the arm the analyzer treats as the subject.
    RoboTwin uses the left arm, matching ``_eef_context``.
    """

    eef_quat: tuple[float, ...] | None
    """End-effector orientation if cheaply available, else None."""

    base_xy_yaw: Vec3 | None
    """Mobile-base (x, y, yaw). RoboCasa only; None on tabletop backends."""

    action: tuple[float, ...]
    """The action actually executed this step (post action-chunk dequeue)."""

    gripper_cmd: float | None
    """Gripper component of ``action``, sliced per action space."""

    obstacle_pos: Vec3 | None
    """Disturbance body origin, world frame. None while hidden."""

    d_eef_obstacle: float | None
    """Surrogate clearance: ``||eef_pos - obstacle_pos||``.

    NOT a true geometric minimum distance (no geom-level query is wrapped in
    this repo). Never rank absolute metres across backends.
    """

    in_contact: bool
    force_n: float
    impulse_ns: float

    phase: DisturbancePhase | None
    """Normalized phase from the backend probe; None when no probe is active."""

    eef_pos_secondary: Vec3 | None = None
    """Second end-effector on dual-arm backends (RoboTwin), else None.

    The analyzer does not consume this: all four axes are defined on a single
    kinematic subject, and ``eef_pos`` is the primary arm (left, matching
    ``_eef_context``). It is captured anyway because a dual-arm policy can
    react with either arm, and re-running the simulator to recover the other
    arm later would defeat the point of storing traces at all.
    """

    def to_json_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["phase"] = self.phase.value if self.phase is not None else None
        for key in (
            "eef_pos",
            "eef_quat",
            "eef_pos_secondary",
            "base_xy_yaw",
            "action",
            "obstacle_pos",
        ):
            value = out[key]
            if value is not None:
                out[key] = [float(v) for v in value]
        return out

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> StepRecord:
        phase = data.get("phase")
        return cls(
            t=int(data["t"]),
            eef_pos=_as_vec3(data["eef_pos"]),
            eef_quat=_as_tuple_or_none(data.get("eef_quat")),
            eef_pos_secondary=_as_vec3_or_none(data.get("eef_pos_secondary")),
            base_xy_yaw=_as_vec3_or_none(data.get("base_xy_yaw")),
            action=_as_tuple(data["action"]),
            gripper_cmd=_as_float_or_none(data.get("gripper_cmd")),
            obstacle_pos=_as_vec3_or_none(data.get("obstacle_pos")),
            d_eef_obstacle=_as_float_or_none(data.get("d_eef_obstacle")),
            in_contact=bool(data["in_contact"]),
            force_n=float(data["force_n"]),
            impulse_ns=float(data["impulse_ns"]),
            phase=DisturbancePhase(phase) if phase is not None else None,
        )


@dataclass(frozen=True)
class TraceHeader:
    """Episode-level trace metadata. Written as the first line of a trace file."""

    trace_schema_version: str
    run_id: str
    episode_index: int
    backend: str
    level: int
    seed: int

    disturbance_triggered: bool
    """False when a distance-triggered disturbance (L1, all RoboCasa levels)
    never fired. Such episodes are excluded from every diagnostic denominator;
    they must NOT be counted as 'no reaction'."""

    t_onset: int | None
    t_hazard_end: int | None
    """Step at which hazard motion stops. L1: obstacle at rest. L2: settle end.
    L3: hand withdrawn. None when the disturbance never triggered or never
    finished within the step budget."""

    hazard_persists: bool | None
    """True for L1/L2 (obstacle stays in the scene as a static hazard), False
    for L3 (scene cleared, back to ~L0), None for L0. Recovery on persisting vs
    cleared hazards is NOT the same quantity and must not be pooled."""

    action_chunk_len: int
    """Action-chunk length. A chunked policy cannot re-decide inside a chunk, so
    reaction latency has an irreducible floor of up to ``action_chunk_len - 1``
    steps. Reported alongside every latency."""

    action_space: str
    control_freq_hz: float | None
    """Recorded for provenance only. Step clocks are backend-specific and never
    form a shared time axis."""

    stride: int
    """Recording stride. ``stride=n`` keeps one record every n task steps.

    The analyzer needs this to keep persistence windows physical: a threshold
    of "sustained for N steps" must become "sustained for N/stride records",
    otherwise downsampling silently widens every window.
    """

    n_steps: int
    """Number of records stored, which equals the task-step count only when
    ``stride == 1``."""

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> TraceHeader:
        known = {f.name for f in fields(cls)}
        missing = known - set(data)
        if missing:
            raise KeyError(f"TraceHeader missing keys: {sorted(missing)}")
        unknown = set(data) - known
        if unknown:
            raise KeyError(f"TraceHeader unknown keys: {sorted(unknown)}")
        return cls(
            trace_schema_version=str(data["trace_schema_version"]),
            run_id=str(data["run_id"]),
            episode_index=int(data["episode_index"]),
            backend=str(data["backend"]),
            level=int(data["level"]),
            seed=int(data["seed"]),
            disturbance_triggered=bool(data["disturbance_triggered"]),
            t_onset=_as_int_or_none(data["t_onset"]),
            t_hazard_end=_as_int_or_none(data["t_hazard_end"]),
            hazard_persists=_as_bool_or_none(data["hazard_persists"]),
            action_chunk_len=int(data["action_chunk_len"]),
            action_space=str(data["action_space"]),
            control_freq_hz=_as_float_or_none(data["control_freq_hz"]),
            stride=int(data["stride"]),
            n_steps=int(data["n_steps"]),
        )


@dataclass(frozen=True)
class Trace:
    """A full episode trace: header + ordered steps."""

    header: TraceHeader
    steps: tuple[StepRecord, ...]

    def __post_init__(self) -> None:
        if self.header.n_steps != len(self.steps):
            raise ValueError(
                f"header.n_steps={self.header.n_steps} != len(steps)={len(self.steps)}"
            )


# ---------------------------------------------------------------------------
# Probe contract (backend adapters)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DisturbanceState:
    """Read-only snapshot of a disturbance controller, normalized across backends."""

    phase: DisturbancePhase
    obstacle_pos: Vec3 | None
    onset_step: int | None
    hazard_end_step: int | None
    hazard_persists: bool | None


@runtime_checkable
class DisturbanceProbe(Protocol):
    """Read-only adapter over a backend disturbance controller.

    Implementations live in ``dynisafety.diagnostics.probes`` — NEVER inside
    ``backends/*/disturbance/``, which is frozen (docs/user/eval.md
    §0.1). A probe may only read attributes; it must not mutate controller
    state, step the simulator, or influence control flow.
    """

    def snapshot(self, task_step: int) -> DisturbanceState:
        """Return the controller state at ``task_step`` without side effects."""
        ...


# ---------------------------------------------------------------------------
# Analyzer output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BehaviorDiagnostics:
    """Per-episode process metrics. Serialized into ``EpisodeRecord.diagnostics``.

    Every field is derived from a ``Trace`` plus ``DiagnosticsConfig``; nothing
    here requires re-running the simulator.
    """

    # --- axis 1: reaction ---
    reacted: bool
    """Closed-loop pre-contact reaction (default).

    True only when a pre-contact kinematic deviation was found **and**
    ``reaction_latency_chunks >= diagnostics.reaction.min_latency_chunks``.
    A 15° task-space turn one step after onset is not a closed-loop
    reaction: the policy has not yet been allowed to re-plan.
    """

    reacted_kinematic: bool
    """Pre-contact 15° / amplitude deviation, ignoring the chunk gate.

    This is the P5-frozen kinematic bit (L0 FPR ≈ 1.0). Kept so tables can
    still report the unsaturated heading gate without calling it ``reacted``.
    """

    reaction_latency_steps: int | None
    reaction_latency_chunks: float | None
    """``reaction_latency_steps / action_chunk_len``; below 1.0 means the delay
    is execution inertia, not slow policy."""

    reaction_post_contact: bool
    """A deviation was found but only at/after first contact — that is the
    collision perturbing the arm, not perception."""

    # --- axis 2: stop ---
    stopped: bool
    stop_preemptive: bool
    """Halted before first contact (braked) rather than after (was stopped)."""

    stop_latency_steps: int | None
    stop_duration_steps: int | None

    # --- axis 3: avoid ---
    evaded: bool
    min_clearance_m: float | None
    """Surrogate clearance; not comparable across backends."""

    max_heading_change_deg: float | None

    # --- axis 4: recovery ---
    resumed: bool
    recovered: bool
    recovery_applicable: bool
    """False when the episode is outside the recovery denominator: disturbance
    never triggered, hazard never ended, fewer than
    ``recovery.min_post_hazard_steps`` remained, or the robot never stopped."""

    recovery_latency_steps: int | None

    # --- summary ---
    behavior_class: BehaviorClass

    # --- provenance / denominators ---
    baseline_usable: bool
    """The pre-onset window supported a ``mu_pre`` the relative gates can use.

    False means the four axes were not evaluated (see
    ``BehaviorClass.BASELINE_UNUSABLE``); every axis field is then blank and
    must not be read as a negative observation.
    """

    disturbance_triggered: bool
    t_onset: int | None
    t_hazard_end: int | None
    hazard_persists: bool | None
    trace_path: str | None
    """Trace file path relative to the run directory; None when not saved."""

    def to_json_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["behavior_class"] = self.behavior_class.value
        return out

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> BehaviorDiagnostics:
        known = {f.name for f in fields(cls)}
        payload = dict(data)
        # Pre-2026-08-29 JSONL has no reacted_kinematic: ``reacted`` *was*
        # the kinematic bit. Do not invent a closed-loop True from that.
        if "reacted_kinematic" not in payload and "reacted" in payload:
            payload["reacted_kinematic"] = bool(payload["reacted"])
        # Pre-2026-08-30 JSONL predates the baseline-validity guard: every
        # triggered episode was scored, so the flag is unknown rather than
        # False. Treat it as True so old rows stay readable, and re-run
        # ``dynisafety diagnostics recompute`` to get the real value.
        if "baseline_usable" not in payload:
            payload["baseline_usable"] = True
        missing = known - set(payload)
        if missing:
            raise KeyError(f"BehaviorDiagnostics missing keys: {sorted(missing)}")
        unknown = set(payload) - known
        if unknown:
            raise KeyError(f"BehaviorDiagnostics unknown keys: {sorted(unknown)}")
        kwargs = dict(payload)
        kwargs["behavior_class"] = BehaviorClass(payload["behavior_class"])
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _as_tuple(value: Sequence[Any]) -> tuple[float, ...]:
    return tuple(float(v) for v in value)


def _as_tuple_or_none(value: Sequence[Any] | None) -> tuple[float, ...] | None:
    return None if value is None else _as_tuple(value)


def _as_vec3(value: Sequence[Any]) -> Vec3:
    out = _as_tuple(value)
    if len(out) != 3:
        raise ValueError(f"expected 3 components, got {len(out)}")
    return (out[0], out[1], out[2])


def _as_vec3_or_none(value: Sequence[Any] | None) -> Vec3 | None:
    return None if value is None else _as_vec3(value)


def _as_float_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


def _as_int_or_none(value: Any) -> int | None:
    return None if value is None else int(value)


def _as_bool_or_none(value: Any) -> bool | None:
    return None if value is None else bool(value)


__all__ = [
    "TRACE_SCHEMA_VERSION",
    "BehaviorClass",
    "BehaviorDiagnostics",
    "DisturbancePhase",
    "DisturbanceProbe",
    "DisturbanceState",
    "StepRecord",
    "Trace",
    "TraceHeader",
    "Vec3",
]
