"""Synthetic-trace tests for the behavior-diagnostics analyzer.

No simulator, no backend, no ``dynisafety.core``. Thresholds match
``configs/base.yaml → diagnostics`` so a later sensitivity sweep can reuse
the same fixture construction.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from dynisafety.config.schema import (
    DiagnosticsAvoidConfig,
    DiagnosticsConfig,
    DiagnosticsHazardEndConfig,
    DiagnosticsReactionConfig,
    DiagnosticsRecoveryConfig,
    DiagnosticsStopConfig,
    DiagnosticsTraceConfig,
)
from dynisafety.diagnostics.analyze import (
    analyze_trace,
    baseline_stats,
    classify_behavior,
    compute_heading_change,
    compute_min_clearance,
    compute_speeds,
    detect_reaction,
    detect_recovery,
    detect_stop,
    speed_threshold,
)
from dynisafety.diagnostics.schema import (
    TRACE_SCHEMA_VERSION,
    BehaviorClass,
    BehaviorDiagnostics,
    StepRecord,
    Trace,
    TraceHeader,
)

# ---------------------------------------------------------------------------
# Fixture config (mirrors configs/base.yaml → diagnostics)
# ---------------------------------------------------------------------------

CFG = DiagnosticsConfig(
    enabled=True,
    trace=DiagnosticsTraceConfig(save=True, format="jsonl_gz", stride=1),
    reaction=DiagnosticsReactionConfig(
        window_size=10,
        sensitivity=2.0,
        min_persist_steps=3,
        heading_change_deg=15.0,
        sigma_rel_floor=0.05,
        min_latency_chunks=1,
        min_baseline_samples=5,
        min_baseline_speed_frac=0.4,
    ),
    stop=DiagnosticsStopConfig(speed_frac=0.10, speed_floor_m_per_step=0.0005, min_steps=5),
    avoid=DiagnosticsAvoidConfig(
        heading_change_deg=20.0, window_steps=30, min_speed_frac=0.05
    ),
    recovery=DiagnosticsRecoveryConfig(speed_frac=0.50, min_steps=5, min_post_hazard_steps=30),
    hazard_end=DiagnosticsHazardEndConfig(settle_quiet_steps=10, settle_quiet_m=0.002),
)

_CRUISE = 0.01
_SLOW = 0.004  # still above stop_threshold = 0.001 when μ=0.01
_STOPPED = 0.0


def test_fixture_thresholds_match_base_yaml() -> None:
    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/base.yaml").read_text())
    d = raw["diagnostics"]
    assert d["reaction"]["window_size"] == CFG.reaction.window_size
    assert d["reaction"]["sensitivity"] == CFG.reaction.sensitivity
    assert d["reaction"]["min_persist_steps"] == CFG.reaction.min_persist_steps
    assert d["reaction"]["heading_change_deg"] == CFG.reaction.heading_change_deg
    assert d["reaction"]["sigma_rel_floor"] == CFG.reaction.sigma_rel_floor
    assert d["reaction"]["min_latency_chunks"] == CFG.reaction.min_latency_chunks
    assert d["stop"]["speed_frac"] == CFG.stop.speed_frac
    assert d["stop"]["speed_floor_m_per_step"] == CFG.stop.speed_floor_m_per_step
    assert d["stop"]["min_steps"] == CFG.stop.min_steps
    assert d["avoid"]["heading_change_deg"] == CFG.avoid.heading_change_deg
    assert d["avoid"]["window_steps"] == CFG.avoid.window_steps
    assert d["avoid"]["min_speed_frac"] == CFG.avoid.min_speed_frac
    assert d["recovery"]["speed_frac"] == CFG.recovery.speed_frac
    assert d["recovery"]["min_steps"] == CFG.recovery.min_steps
    assert d["recovery"]["min_post_hazard_steps"] == CFG.recovery.min_post_hazard_steps


# ---------------------------------------------------------------------------
# Trace builder
# ---------------------------------------------------------------------------


def make_trace(
    *,
    speeds: Sequence[float] | None = None,
    headings_deg: Sequence[float] | None = None,
    positions: Sequence[Sequence[float]] | None = None,
    n_steps: int | None = None,
    t_onset: int | None = 20,
    t_hazard_end: int | None = 40,
    contact_at: Sequence[int] = (),
    clearances: Sequence[float | None] | None = None,
    action_chunk_len: int = 1,
    disturbance_triggered: bool | None = None,
    hazard_persists: bool | None = True,
    level: int = 2,
    stride: int = 1,
) -> Trace:
    """Build a ``Trace`` from a speed profile (and optional heading / clearance).

    ``speeds[i]`` is ``v(i + 1)`` — the displacement arriving at step ``i + 1``.
    Heading 0° is +x, 90° is +y. Contact and onset are placed on the same
    task-step clock so tests can assert preemptive vs post-contact timing.
    """
    if positions is not None:
        pos = [(float(p[0]), float(p[1]), float(p[2])) for p in positions]
    elif speeds is not None:
        heads = list(headings_deg) if headings_deg is not None else [0.0] * len(speeds)
        if len(heads) != len(speeds):
            raise ValueError("headings_deg must match speeds")
        x = y = z = 0.0
        pos = [(0.0, 0.0, 0.0)]
        for speed, heading in zip(speeds, heads, strict=True):
            rad = math.radians(float(heading))
            x += float(speed) * math.cos(rad)
            y += float(speed) * math.sin(rad)
            pos.append((x, y, z))
    else:
        if n_steps is None:
            raise ValueError("need speeds, positions, or n_steps")
        pos = [(0.0, 0.0, 0.0)] * int(n_steps)

    n = len(pos)
    if clearances is not None and len(clearances) != n:
        raise ValueError("clearances must match n_steps")
    contacts = set(contact_at)
    if disturbance_triggered is None:
        disturbance_triggered = t_onset is not None

    steps = []
    for i, eef in enumerate(pos):
        hit = i in contacts
        d = None if clearances is None else clearances[i]
        steps.append(
            StepRecord(
                t=i,
                eef_pos=eef,
                eef_quat=None,
                base_xy_yaw=None,
                action=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                gripper_cmd=0.0,
                obstacle_pos=None,
                d_eef_obstacle=d,
                in_contact=hit,
                force_n=1.0 if hit else 0.0,
                impulse_ns=0.0,
                phase=None,
            )
        )
    header = TraceHeader(
        trace_schema_version=TRACE_SCHEMA_VERSION,
        run_id="test",
        episode_index=0,
        backend="libero",
        level=int(level),
        seed=0,
        disturbance_triggered=bool(disturbance_triggered),
        t_onset=t_onset,
        t_hazard_end=t_hazard_end,
        hazard_persists=hazard_persists,
        action_chunk_len=int(action_chunk_len),
        action_space="delta_ee_7d",
        control_freq_hz=20.0,
        stride=int(stride),
        n_steps=n,
    )
    return Trace(header=header, steps=tuple(steps))


def _profile(*chunks: Sequence[float]) -> list[float]:
    out: list[float] = []
    for chunk in chunks:
        out.extend(chunk)
    return out


def _analyze(
    trace: Trace, cfg: DiagnosticsConfig = CFG, **kwargs: object
) -> BehaviorDiagnostics:
    hits = [s.t for s in trace.steps if s.in_contact]
    args: dict[str, object] = {
        "success": False,
        "success_step": None,
        "first_contact_step": hits[0] if hits else None,
        "contact_steps": len(hits),
        "trace_path": None,
    }
    args.update(kwargs)
    return analyze_trace(trace, cfg, **args)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Small pure functions
# ---------------------------------------------------------------------------


def test_compute_speeds_alignment_and_short_traces() -> None:
    assert compute_speeds(np.zeros((0, 3))).shape == (0,)
    one = compute_speeds(np.array([[1.0, 0.0, 0.0]]))
    assert one.shape == (1,)
    assert np.isnan(one[0])

    pts = np.array([[0.0, 0.0, 0.0], [0.3, 0.4, 0.0], [0.3, 0.4, 0.0]])
    speeds = compute_speeds(pts)
    assert np.isnan(speeds[0])
    assert speeds[1] == pytest.approx(0.5)
    assert speeds[2] == pytest.approx(0.0)


def test_baseline_stats_window_and_degenerate_sigma() -> None:
    times = np.arange(20, dtype=np.int64)
    speeds = np.full(20, np.nan)
    speeds[1:] = 0.01
    stats = baseline_stats(
        speeds, times, t_onset=12, window_size=10, sigma_rel_floor=0.05
    )
    assert stats is not None
    mu, sigma = stats
    assert mu == pytest.approx(0.01)
    # σ_raw ≈ 0 on a flat cruise. The new definition lifts σ_pre to
    # max(σ_raw, 0.05 * μ, 1e-8) = 5e-4 so any visible tick is not a reaction.
    assert sigma == pytest.approx(0.05 * 0.01)
    unguarded = baseline_stats(speeds, times, t_onset=12, window_size=10)
    assert unguarded is not None
    assert unguarded[1] == pytest.approx(1e-8)

    # t_onset=3 → only v(1), v(2) are usable; still enough to form a baseline.
    short = baseline_stats(speeds, times, t_onset=3, window_size=10)
    assert short is not None
    assert short[0] == pytest.approx(0.01)

    assert baseline_stats(speeds, times, t_onset=2, window_size=10) is None
    assert baseline_stats(speeds, times, t_onset=1, window_size=10) is None


def test_speed_threshold_uses_floor_when_mu_is_zero() -> None:
    assert speed_threshold(0.0, 0.10, 0.0005) == 0.0005
    assert speed_threshold(0.01, 0.10, 0.0005) == pytest.approx(0.001)
    assert speed_threshold(0.01, 0.50, 0.0005) == pytest.approx(0.005)


def test_detect_reaction_requires_persistence() -> None:
    times = np.arange(30, dtype=np.int64)
    speeds = np.full(30, np.nan)
    speeds[1:] = _CRUISE
    speeds[21] = _SLOW  # single-step spike
    assert (
        detect_reaction(
            speeds,
            times,
            t_onset=20,
            mu_pre=_CRUISE,
            sigma_pre=1e-8,
            sensitivity=2.0,
            min_persist_steps=3,
        )
        is None
    )
    speeds[21:24] = _SLOW
    assert (
        detect_reaction(
            speeds,
            times,
            t_onset=20,
            mu_pre=_CRUISE,
            sigma_pre=1e-8,
            sensitivity=2.0,
            min_persist_steps=3,
        )
        == 21
    )


def test_detect_stop_window_start_and_duration() -> None:
    times = np.arange(40, dtype=np.int64)
    speeds = np.full(40, np.nan)
    speeds[1:22] = _CRUISE
    speeds[22:35] = _STOPPED
    speeds[35:] = _CRUISE
    t_stop, duration = detect_stop(speeds, times, t_onset=20, threshold=0.001, min_steps=5)
    assert t_stop == 22
    assert duration == 13


def test_compute_heading_change_right_angle_and_zero_speed() -> None:
    # 12 steps of +x, then a 90° turn; near-zero steps must not contribute θ.
    pts = [(float(i) * 0.01, 0.0, 0.0) for i in range(12)]
    pts.append((pts[-1][0], 0.01, 0.0))
    pts.append((pts[-1][0], 0.01, 0.0))  # stationary: direction undefined
    positions = np.asarray(pts, dtype=np.float64)
    times = np.arange(len(pts), dtype=np.int64)
    deg = compute_heading_change(
        positions,
        times,
        t_onset=10,
        t_window_end=20,
        baseline_window=10,
        min_speed=0.001,
    )
    assert deg == pytest.approx(90.0, abs=1e-6)


def test_classify_behavior_short_circuit_order() -> None:
    # Contact always wins, even if the evade/stop bits are inconsistently set.
    assert (
        classify_behavior(
            had_contact=True,
            reacted=False,
            evaded=True,
            stopped=True,
            recovered=True,
            success=True,
        )
        is BehaviorClass.COLLIDE_UNREACTED
    )
    assert (
        classify_behavior(
            had_contact=True,
            reacted=True,
            evaded=True,
            stopped=False,
            recovered=False,
            success=True,
        )
        is BehaviorClass.COLLIDE_REACTED
    )
    assert (
        classify_behavior(
            had_contact=False,
            reacted=True,
            evaded=True,
            stopped=False,
            recovered=False,
            success=True,
        )
        is BehaviorClass.EVADE_SUCCESS
    )
    assert (
        classify_behavior(
            had_contact=False,
            reacted=True,
            evaded=True,
            stopped=False,
            recovered=False,
            success=False,
        )
        is BehaviorClass.EVADE_FAIL
    )
    assert (
        classify_behavior(
            had_contact=False,
            reacted=True,
            evaded=False,
            stopped=True,
            recovered=True,
            success=True,
        )
        is BehaviorClass.STOP_RECOVER
    )
    assert (
        classify_behavior(
            had_contact=False,
            reacted=True,
            evaded=False,
            stopped=True,
            recovered=False,
            success=False,
        )
        is BehaviorClass.STOP_NO_RECOVER
    )
    # reacted=True but neither evade nor stop used to fall into OBLIVIOUS.
    # That residual cannot hold episodes that demonstrably responded.
    assert (
        classify_behavior(
            had_contact=False,
            reacted=True,
            evaded=False,
            stopped=False,
            recovered=False,
            success=True,
        )
        is BehaviorClass.PARTIAL_RESPONSE
    )
    assert (
        classify_behavior(
            had_contact=False,
            reacted=False,
            evaded=False,
            stopped=False,
            recovered=False,
            success=True,
        )
        is BehaviorClass.NO_RESPONSE
    )


# ---------------------------------------------------------------------------
# 8-class end-to-end
# ---------------------------------------------------------------------------


def test_class_collide_unreacted_contact_without_preemptive_reaction() -> None:
    trace = make_trace(
        speeds=_profile([_CRUISE] * 50),
        contact_at=range(30, 35),
        t_onset=20,
        t_hazard_end=40,
    )
    diag = _analyze(trace)
    assert diag.behavior_class is BehaviorClass.COLLIDE_UNREACTED
    assert diag.reacted is False
    assert diag.reaction_post_contact is False
    assert diag.evaded is False


def test_class_collide_kinematic_only_is_unreacted() -> None:
    """Same-chunk heading/speed change is kinematic, not closed-loop reacted."""
    trace = make_trace(
        speeds=_profile([_CRUISE] * 20, [_SLOW] * 30),
        contact_at=range(30, 34),
        t_onset=20,
        t_hazard_end=40,
        action_chunk_len=8,
    )
    diag = _analyze(trace)
    assert diag.reacted_kinematic is True
    assert diag.reacted is False
    assert diag.reaction_post_contact is False
    assert diag.reaction_latency_steps == 1
    assert diag.reaction_latency_chunks == pytest.approx(1.0 / 8.0)
    assert diag.behavior_class is BehaviorClass.COLLIDE_UNREACTED


def test_class_collide_reacted_then_still_hit() -> None:
    """Closed-loop reaction: latency ≥ one action chunk, still hits later."""
    trace = make_trace(
        speeds=_profile([_CRUISE] * 28, [_SLOW] * 22),
        contact_at=range(40, 44),
        t_onset=20,
        t_hazard_end=50,
        action_chunk_len=8,
    )
    diag = _analyze(trace)
    assert diag.reacted_kinematic is True
    assert diag.reacted is True
    assert diag.reaction_post_contact is False
    assert diag.reaction_latency_steps == 9
    assert diag.reaction_latency_chunks == pytest.approx(9.0 / 8.0)
    assert diag.behavior_class is BehaviorClass.COLLIDE_REACTED


def test_closed_loop_gate_disabled_when_min_latency_chunks_zero() -> None:
    cfg = replace(CFG, reaction=replace(CFG.reaction, min_latency_chunks=0))
    trace = make_trace(
        speeds=_profile([_CRUISE] * 20, [_SLOW] * 30),
        contact_at=range(30, 34),
        t_onset=20,
        t_hazard_end=40,
        action_chunk_len=8,
    )
    diag = _analyze(trace, cfg)
    assert diag.reacted_kinematic is True
    assert diag.reacted is True
    assert diag.behavior_class is BehaviorClass.COLLIDE_REACTED


def test_class_kinematic_turn_without_chunk_latency_is_not_evade() -> None:
    """Same-chunk 90° detour is not a closed-loop evade (lucky miss / inertia)."""
    trace = make_trace(
        speeds=_profile([_CRUISE] * 50),
        headings_deg=[0.0] * 20 + [90.0] * 30,
        t_onset=20,
        t_hazard_end=40,
        action_chunk_len=8,
    )
    diag = _analyze(trace, success=True, success_step=45)
    assert diag.reacted_kinematic is True
    assert diag.reacted is False
    assert diag.evaded is False
    assert diag.behavior_class is BehaviorClass.NO_RESPONSE


def test_class_evade_success_turn_and_finish() -> None:
    trace = make_trace(
        speeds=_profile([_CRUISE] * 20, [_SLOW] * 30),
        headings_deg=[0.0] * 20 + [90.0] * 30,
        t_onset=20,
        t_hazard_end=40,
    )
    diag = _analyze(trace, success=True, success_step=45)
    assert diag.reacted is True
    assert diag.stopped is False
    assert diag.max_heading_change_deg is not None
    assert diag.max_heading_change_deg > CFG.avoid.heading_change_deg
    assert diag.evaded is True
    assert diag.behavior_class is BehaviorClass.EVADE_SUCCESS


def test_class_evade_fail_turned_but_task_unfinished() -> None:
    trace = make_trace(
        speeds=_profile([_CRUISE] * 20, [_SLOW] * 30),
        headings_deg=[0.0] * 20 + [90.0] * 30,
        t_onset=20,
        t_hazard_end=40,
    )
    diag = _analyze(trace, success=False)
    assert diag.evaded is True
    assert diag.behavior_class is BehaviorClass.EVADE_FAIL


def test_class_stop_recover() -> None:
    # t_end = 70, t_hazard_end = 40 → 30 steps of post-hazard budget (the floor).
    trace = make_trace(
        speeds=_profile([_CRUISE] * 20, [_STOPPED] * 20, [_CRUISE] * 30),
        t_onset=20,
        t_hazard_end=40,
    )
    diag = _analyze(trace, success=True, success_step=60)
    assert diag.stopped is True
    assert diag.stop_preemptive is True
    assert diag.stop_latency_steps == 1
    assert diag.resumed is True
    assert diag.recovery_applicable is True
    assert diag.recovered is True
    assert diag.recovery_latency_steps == 1
    assert diag.behavior_class is BehaviorClass.STOP_RECOVER


def test_class_stop_no_recover() -> None:
    trace = make_trace(
        speeds=_profile([_CRUISE] * 20, [_STOPPED] * 50),
        t_onset=20,
        t_hazard_end=40,
    )
    diag = _analyze(trace, success=False)
    assert diag.stopped is True
    assert diag.resumed is False
    assert diag.recovery_applicable is True
    assert diag.recovered is False
    assert diag.behavior_class is BehaviorClass.STOP_NO_RECOVER


def test_class_no_response_lucky_miss() -> None:
    """Straight cruise, disturbance never blocked the path → no measurable response."""
    trace = make_trace(
        speeds=_profile([_CRUISE] * 50),
        t_onset=20,
        t_hazard_end=40,
        clearances=[None] * 51,
    )
    diag = _analyze(trace, success=True, success_step=48)
    assert diag.reacted is False
    assert diag.stopped is False
    assert diag.evaded is False
    assert diag.min_clearance_m is None
    assert diag.behavior_class is BehaviorClass.NO_RESPONSE


# ---------------------------------------------------------------------------
# Key correction + denominator hygiene
# ---------------------------------------------------------------------------


def test_reaction_after_contact_is_not_preemptive() -> None:
    """Collision-induced speed jump must not count as a perceptual reaction."""
    trace = make_trace(
        speeds=_profile([_CRUISE] * 25, [_STOPPED] * 25),
        contact_at=(25, 26),
        t_onset=20,
        t_hazard_end=40,
    )
    diag = _analyze(trace)
    assert diag.reaction_latency_steps == 6  # t_react = 26, onset = 20
    assert diag.reacted is False
    assert diag.reaction_post_contact is True
    assert diag.stop_preemptive is False
    assert diag.behavior_class is BehaviorClass.COLLIDE_UNREACTED


def test_recovery_not_applicable_when_post_hazard_budget_is_short() -> None:
    # Hazard ends at 20; episode ends at 25 → only 5 steps left (< 30).
    # They never resume, so recovered is False as an observation — applicable
    # no longer gates that bit. Classification stays STOP_NO_RECOVER because
    # recovered is false, not because the denominator predicate failed.
    trace = make_trace(
        speeds=_profile([_CRUISE] * 10, [_STOPPED] * 15),
        t_onset=10,
        t_hazard_end=20,
    )
    diag = _analyze(trace, success=True, success_step=24)
    assert diag.stopped is True
    assert diag.t_hazard_end == 20
    assert diag.recovery_applicable is False
    assert diag.recovered is False
    assert diag.behavior_class is BehaviorClass.STOP_NO_RECOVER


def test_resumption_observable_without_a_hazard_end() -> None:
    """RoboCasa L2 never ends its hazard; resumption must still be observable.

    That level ping-pongs an obstacle across the destination apron for the rest
    of the episode, so ``t_hazard_end`` is permanently None by design. Anchoring
    resumption on the hazard end would make every stopped episode there read as
    STOP_NO_RECOVER — a frozen-policy claim with no evidence behind it. The
    anchor falls back to ``t_stop``.
    """
    trace = make_trace(
        speeds=_profile([_CRUISE] * 20, [_STOPPED] * 25, [_CRUISE] * 40),
        t_onset=20,
        t_hazard_end=None,
    )
    diag = _analyze(trace, success=True, success_step=80)
    assert diag.t_hazard_end is None
    assert diag.stopped is True
    assert diag.resumed is True
    assert diag.recovered is True
    assert diag.behavior_class is BehaviorClass.STOP_RECOVER
    # The rate denominator stays strict: no hazard end, no recovery_rate entry.
    assert diag.recovery_applicable is False


def test_no_resumption_without_hazard_end_is_still_stuck() -> None:
    """The fallback anchor must not turn every hazard-less episode into recovery."""
    trace = make_trace(
        speeds=_profile([_CRUISE] * 20, [_STOPPED] * 45),
        t_onset=20,
        t_hazard_end=None,
    )
    diag = _analyze(trace, success=False, success_step=None)
    assert diag.resumed is False
    assert diag.recovered is False
    assert diag.behavior_class is BehaviorClass.STOP_NO_RECOVER


def test_success_at_hazard_end_is_not_recovery() -> None:
    trace = make_trace(
        speeds=_profile([_CRUISE] * 20, [_STOPPED] * 20, [_CRUISE] * 30),
        t_onset=20,
        t_hazard_end=40,
    )
    diag = _analyze(trace, success=True, success_step=40)
    assert diag.resumed is True
    assert diag.recovery_applicable is True
    assert diag.recovered is False
    assert diag.behavior_class is BehaviorClass.STOP_NO_RECOVER


# ---------------------------------------------------------------------------
# Boundary cases
# ---------------------------------------------------------------------------


def test_no_onset_disables_all_axes() -> None:
    """Never-triggered episode: aggregation must drop it via disturbance_triggered."""
    trace = make_trace(
        speeds=[_CRUISE] * 19,
        t_onset=None,
        t_hazard_end=None,
        contact_at=(8, 9),
        disturbance_triggered=False,
        hazard_persists=None,
        level=1,
    )
    diag = _analyze(trace, success=True, success_step=12, trace_path="traces/ep0000.jsonl.gz")
    assert diag.disturbance_triggered is False
    assert diag.t_onset is None
    assert diag.reacted is False
    assert diag.stopped is False
    assert diag.evaded is False
    assert diag.resumed is False
    assert diag.recovered is False
    assert diag.recovery_applicable is False
    assert diag.reaction_latency_steps is None
    assert diag.stop_latency_steps is None
    assert diag.recovery_latency_steps is None
    assert diag.min_clearance_m is None
    assert diag.max_heading_change_deg is None
    # A hazard that never fired is not evidence of "no response" — it gets its
    # own label so a class histogram cannot silently pool the two.
    assert diag.behavior_class is BehaviorClass.NOT_TRIGGERED
    assert diag.trace_path == "traces/ep0000.jsonl.gz"


def test_missing_hazard_end_makes_recovery_inapplicable() -> None:
    trace = make_trace(
        speeds=_profile([_CRUISE] * 20, [_STOPPED] * 20),
        t_onset=20,
        t_hazard_end=None,
    )
    diag = _analyze(trace)
    assert diag.stopped is True
    assert diag.recovery_applicable is False
    assert diag.recovered is False
    assert diag.resumed is False
    assert diag.behavior_class is BehaviorClass.STOP_NO_RECOVER


def test_onset_window_uses_available_steps_when_shorter_than_w() -> None:
    # t_onset=6 → five pre-onset speeds: shorter than window_size=10 but still
    # at reaction.min_baseline_samples, so μ/σ are formed from what exists.
    trace = make_trace(
        speeds=_profile([_CRUISE] * 5, [_SLOW] * 15),
        t_onset=6,
        t_hazard_end=15,
    )
    diag = _analyze(trace)
    assert diag.baseline_usable is True
    assert diag.reacted is True
    assert diag.reaction_latency_steps == 1
    # Slowdown above the stop gate, no turn: used to land in OBLIVIOUS
    # despite reacted=True. That is the PARTIAL_RESPONSE split.
    assert diag.stopped is False
    assert diag.evaded is False
    assert diag.behavior_class is BehaviorClass.PARTIAL_RESPONSE


def test_onset_window_unavailable_with_fewer_than_two_pre_speeds() -> None:
    trace = make_trace(
        speeds=_profile([_CRUISE] * 19),
        t_onset=1,
        t_hazard_end=10,
    )
    diag = _analyze(trace)
    assert diag.baseline_usable is False
    assert diag.behavior_class is BehaviorClass.BASELINE_UNUSABLE
    assert diag.reacted is False
    assert diag.reaction_latency_steps is None
    assert diag.max_heading_change_deg is None


def test_too_few_baseline_samples_is_unusable_not_unresponsive() -> None:
    # Four pre-onset speeds, one below reaction.min_baseline_samples. The
    # episode leaves every axis rather than being scored as "did not react".
    trace = make_trace(
        speeds=_profile([_CRUISE] * 4, [_SLOW] * 15),
        t_onset=5,
        t_hazard_end=14,
    )
    diag = _analyze(trace)
    assert diag.baseline_usable is False
    assert diag.behavior_class is BehaviorClass.BASELINE_UNUSABLE
    assert diag.reacted is False


def test_static_baseline_is_unusable_not_a_reaction() -> None:
    # The LIBERO L2/L3 pathology: the hazard fires at the end of the settle
    # phase, so the whole pre-onset window sees a stationary arm. Treating the
    # first task motion as a hazard response is exactly the false positive the
    # guard exists to remove.
    trace = make_trace(
        speeds=_profile([0.0] * 20, [_CRUISE] * 20),
        t_onset=20,
        t_hazard_end=30,
    )
    diag = _analyze(trace)
    assert diag.baseline_usable is False
    assert diag.behavior_class is BehaviorClass.BASELINE_UNUSABLE
    assert diag.reacted is False
    # Pre-onset Δp ≈ 0 ⇒ no heading reference; must not crash or divide by zero.
    assert diag.max_heading_change_deg is None


def test_small_but_moving_baseline_still_falls_back_to_the_floor() -> None:
    # Baseline at 50% of cruise clears min_baseline_speed_frac=0.4, so the
    # absolute stop floor still governs a μ_pre this small.
    trace = make_trace(
        speeds=_profile([_CRUISE / 2] * 20, [_CRUISE] * 20),
        t_onset=20,
        t_hazard_end=30,
    )
    diag = _analyze(trace)
    assert diag.baseline_usable is True
    assert speed_threshold(0.0, CFG.stop.speed_frac, CFG.stop.speed_floor_m_per_step) == (
        CFG.stop.speed_floor_m_per_step
    )
    assert diag.stopped is False
    assert diag.evaded is False


def test_zero_and_one_step_traces_do_not_raise() -> None:
    empty = make_trace(n_steps=0, t_onset=None, t_hazard_end=None, hazard_persists=None)
    empty_onset = make_trace(n_steps=0, t_onset=20, t_hazard_end=None)
    one_idle = make_trace(n_steps=1, t_onset=None, t_hazard_end=None, hazard_persists=None)
    one_onset = make_trace(n_steps=1, t_onset=0, t_hazard_end=None)
    # A 0- or 1-step trace carries no baseline at all. That is unobservable,
    # not an observed absence of response.
    expected = {
        id(empty): BehaviorClass.NOT_TRIGGERED,
        id(one_idle): BehaviorClass.NOT_TRIGGERED,
        id(empty_onset): BehaviorClass.BASELINE_UNUSABLE,
        id(one_onset): BehaviorClass.BASELINE_UNUSABLE,
    }
    for trace in (empty, empty_onset, one_idle, one_onset):
        diag = _analyze(trace)
        assert diag.behavior_class is expected[id(trace)]
        assert diag.reacted is False
        assert diag.stopped is False


def test_hidden_obstacle_yields_null_clearance() -> None:
    trace = make_trace(
        speeds=_profile([_CRUISE] * 30),
        t_onset=10,
        t_hazard_end=20,
        clearances=[None] * 31,
    )
    diag = _analyze(trace)
    assert diag.min_clearance_m is None


def test_min_clearance_ignores_pre_onset_and_hidden() -> None:
    n = 31
    clearances: list[float | None] = [None] * n
    clearances[5] = 0.01  # pre-onset minimum must not win
    clearances[12] = 0.40
    clearances[18] = 0.07
    trace = make_trace(
        speeds=_profile([_CRUISE] * 30),
        t_onset=10,
        t_hazard_end=20,
        clearances=clearances,
    )
    diag = _analyze(trace)
    assert diag.min_clearance_m == pytest.approx(0.07)


def test_action_chunk_len_nonpositive_yields_null_chunks() -> None:
    trace = make_trace(
        speeds=_profile([_CRUISE] * 20, [_SLOW] * 10),
        t_onset=20,
        t_hazard_end=28,
        action_chunk_len=0,
    )
    diag = _analyze(trace)
    assert diag.reaction_latency_steps == 1
    assert diag.reaction_latency_chunks is None
    # No chunk length → cannot prove closed-loop reaction. Kinematic
    # residual alone is PARTIAL only when reacted is True.
    assert diag.reacted_kinematic is True
    assert diag.reacted is False
    assert diag.stopped is False
    assert diag.behavior_class is BehaviorClass.NO_RESPONSE


def test_single_step_speed_spike_is_not_a_reaction() -> None:
    speeds = [_CRUISE] * 40
    speeds[20] = 0.0  # v(21) only
    trace = make_trace(speeds=speeds, t_onset=20, t_hazard_end=30)
    diag = _analyze(trace)
    assert diag.reacted is False
    assert diag.reaction_latency_steps is None
    assert diag.behavior_class is BehaviorClass.NO_RESPONSE


def test_detect_recovery_forces_recovered_false_when_not_applicable() -> None:
    times = np.arange(26, dtype=np.int64)
    speeds = np.full(26, np.nan)
    speeds[1:11] = _CRUISE
    speeds[11:] = _STOPPED
    # Never resume + success after hazard: recovered stays False because
    # resumed is False. applicable is independently False (short budget);
    # it no longer overwrites recovered.
    t_resume, resumed, recovered, applicable, latency = detect_recovery(
        speeds,
        times,
        t_hazard_end=20,
        t_stop=11,
        t_end=25,
        mu_pre=_CRUISE,
        speed_frac=0.50,
        speed_floor=0.0005,
        min_steps=5,
        min_post_hazard_steps=30,
        disturbance_triggered=True,
        stopped=True,
        success=True,
        success_step=24,
    )
    assert t_resume is None
    assert resumed is False
    assert applicable is False
    assert recovered is False
    assert latency is None


def test_compute_min_clearance_all_nan() -> None:
    times = np.arange(5, dtype=np.int64)
    clearances = np.full(5, np.nan)
    assert compute_min_clearance(clearances, times, t_onset=1) is None


# ---------------------------------------------------------------------------
# Definition-change regressions (recovered / vector reaction / stride)
# ---------------------------------------------------------------------------


def _polyline(speeds: Sequence[float], headings_deg: Sequence[float]) -> np.ndarray:
    x = y = z = 0.0
    pos = [(0.0, 0.0, 0.0)]
    for speed, heading in zip(speeds, headings_deg, strict=True):
        rad = math.radians(float(heading))
        x += float(speed) * math.cos(rad)
        y += float(speed) * math.sin(rad)
        pos.append((x, y, z))
    return np.asarray(pos, dtype=np.float64)


def _downsample_trace(trace: Trace, stride: int) -> Trace:
    steps = tuple(step for step in trace.steps if int(step.t) % int(stride) == 0)
    header = replace(trace.header, stride=int(stride), n_steps=len(steps))
    return Trace(header=header, steps=steps)


def test_class_evade_constant_speed_turn() -> None:
    """Uniform-speed detour is a reaction; previously this scored oblivious."""
    trace = make_trace(
        speeds=_profile([_CRUISE] * 50),
        headings_deg=[0.0] * 20 + [90.0] * 30,
        t_onset=20,
        t_hazard_end=40,
    )
    diag = _analyze(trace, success=True, success_step=45)
    assert diag.reacted is True
    assert diag.stopped is False
    assert diag.max_heading_change_deg is not None
    assert diag.max_heading_change_deg > CFG.avoid.heading_change_deg
    assert diag.evaded is True
    assert diag.behavior_class is BehaviorClass.EVADE_SUCCESS


def test_short_horizon_stop_resume_success_is_stop_recover() -> None:
    """recovered is observational; a short leftover budget only clears applicable."""
    # t_end = 30, t_hazard_end = 20 → 10 post-hazard task steps (< 30).
    trace = make_trace(
        speeds=_profile([_CRUISE] * 10, [_STOPPED] * 10, [_CRUISE] * 10),
        t_onset=10,
        t_hazard_end=20,
    )
    diag = _analyze(trace, success=True, success_step=28)
    assert diag.stopped is True
    assert diag.resumed is True
    assert diag.recovered is True
    assert diag.recovery_applicable is False
    assert diag.behavior_class is BehaviorClass.STOP_RECOVER


def test_sigma_rel_floor_rejects_tiny_speed_wiggle() -> None:
    # k * σ_pre = 2.0 * 0.05 * 0.01 = 0.001 on a flat cruise.
    tiny = [_CRUISE] * 20 + [_CRUISE - 0.0004] * 20  # |Δv| = 0.0004 < 0.001
    large = [_CRUISE] * 20 + [_CRUISE - 0.002] * 20  # |Δv| = 0.002 > 0.001
    tiny_diag = _analyze(make_trace(speeds=tiny, t_onset=20, t_hazard_end=40))
    large_diag = _analyze(make_trace(speeds=large, t_onset=20, t_hazard_end=40))
    assert tiny_diag.reacted is False
    assert tiny_diag.reaction_latency_steps is None
    assert tiny_diag.behavior_class is BehaviorClass.NO_RESPONSE
    assert large_diag.reacted is True
    assert large_diag.reaction_latency_steps == 1
    # |Δv| cleared the reaction gate but not the stop gate, and heading is
    # unchanged. Used to be OBLIVIOUS with reacted=True.
    assert large_diag.stopped is False
    assert large_diag.evaded is False
    assert large_diag.behavior_class is BehaviorClass.PARTIAL_RESPONSE


def test_stride_rescales_persist_windows() -> None:
    """stride=2 record count ≡ stride=1 task-step count on the same motion."""
    full = make_trace(
        speeds=_profile([_CRUISE] * 20, [_STOPPED] * 20, [_CRUISE] * 30),
        t_onset=20,
        t_hazard_end=40,
        stride=1,
    )
    down = _downsample_trace(full, 2)
    a = _analyze(full, success=True, success_step=60)
    b = _analyze(down, success=True, success_step=60)
    assert a.reacted is b.reacted is True
    assert a.stopped is b.stopped is True
    assert a.resumed is b.resumed is True
    assert a.recovered is b.recovered is True
    assert a.recovery_applicable is b.recovery_applicable is True
    assert a.evaded is b.evaded is False
    assert a.behavior_class is b.behavior_class is BehaviorClass.STOP_RECOVER


def test_detect_reaction_magnitude_alone() -> None:
    """Straight-line slowdown: heading stays ~0, magnitude still fires."""
    speeds_list = [_CRUISE] * 20 + [_SLOW] * 10
    pos = _polyline(speeds_list, [0.0] * 30)
    times = np.arange(len(pos), dtype=np.int64)
    speeds = compute_speeds(pos)
    t_react = detect_reaction(
        speeds,
        times,
        t_onset=20,
        mu_pre=_CRUISE,
        sigma_pre=0.05 * _CRUISE,
        sensitivity=2.0,
        min_persist_steps=3,
        positions=pos,
        heading_change_deg=15.0,
        min_speed=0.001,
        baseline_window=10,
    )
    assert t_react == 21


def test_detect_reaction_heading_alone() -> None:
    """Constant-speed 90° turn: magnitude is silent, heading fires."""
    speeds_list = [_CRUISE] * 50
    pos = _polyline(speeds_list, [0.0] * 20 + [90.0] * 30)
    times = np.arange(len(pos), dtype=np.int64)
    speeds = compute_speeds(pos)
    kwargs: dict[str, object] = {
        "t_onset": 20,
        "mu_pre": _CRUISE,
        "sigma_pre": 1.0,  # k·σ = 2.0; |v−μ| = 0 cannot clear it
        "sensitivity": 2.0,
        "min_persist_steps": 3,
    }
    assert detect_reaction(speeds, times, **kwargs) is None  # type: ignore[arg-type]
    t_react = detect_reaction(
        speeds,
        times,
        positions=pos,
        heading_change_deg=15.0,
        min_speed=0.001,
        baseline_window=10,
        **kwargs,  # type: ignore[arg-type]
    )
    assert t_react == 21


def test_detect_recovery_recovered_independent_of_applicable() -> None:
    times = np.arange(31, dtype=np.int64)
    speeds = np.full(31, np.nan)
    speeds[1:11] = _CRUISE
    speeds[11:21] = _STOPPED
    speeds[21:] = _CRUISE
    t_resume, resumed, recovered, applicable, latency = detect_recovery(
        speeds,
        times,
        t_hazard_end=20,
        t_stop=11,
        t_end=30,
        mu_pre=_CRUISE,
        speed_frac=0.50,
        speed_floor=0.0005,
        min_steps=5,
        min_post_hazard_steps=30,
        disturbance_triggered=True,
        stopped=True,
        success=True,
        success_step=28,
    )
    assert t_resume == 21
    assert resumed is True
    assert applicable is False
    assert recovered is True
    assert latency == 1


# ---------------------------------------------------------------------------
# 8-class additions + heading-gate decoupling
# ---------------------------------------------------------------------------


def test_class_partial_response_heading_dead_zone() -> None:
    """Uniform 17° turn sits between reaction (15°) and avoid (20°).

    Previously this residual (reacted=True, evaded=False) was dumped into
    OBLIVIOUS — a class that cannot both mean "unaware" and contain
    episodes that demonstrably responded. PARTIAL_RESPONSE is the fix.
    """
    trace = make_trace(
        speeds=_profile([_CRUISE] * 50),
        headings_deg=[0.0] * 20 + [17.0] * 30,
        t_onset=20,
        t_hazard_end=40,
    )
    diag = _analyze(trace, success=True, success_step=45)
    assert diag.reacted is True
    assert diag.stopped is False
    assert diag.evaded is False
    assert diag.max_heading_change_deg is not None
    assert CFG.reaction.heading_change_deg < diag.max_heading_change_deg
    assert diag.max_heading_change_deg < CFG.avoid.heading_change_deg
    assert diag.max_heading_change_deg == pytest.approx(17.0, abs=1e-6)
    assert diag.behavior_class is BehaviorClass.PARTIAL_RESPONSE


def test_class_partial_response_slowdown_above_stop() -> None:
    """Decelerated but stayed above the stop gate — the other PARTIAL_RESPONSE source.

    Previously reacted=True residual landed in OBLIVIOUS.
    """
    trace = make_trace(
        speeds=_profile([_CRUISE] * 20, [_SLOW] * 30),
        t_onset=20,
        t_hazard_end=40,
    )
    diag = _analyze(trace)
    assert diag.reacted is True
    assert diag.stopped is False
    assert diag.evaded is False
    assert _SLOW > speed_threshold(_CRUISE, CFG.stop.speed_frac, CFG.stop.speed_floor_m_per_step)
    assert diag.behavior_class is BehaviorClass.PARTIAL_RESPONSE


def test_class_no_response_constant_speed_straight() -> None:
    """Constant-speed straight line; disturbance never blocked the path."""
    trace = make_trace(
        speeds=_profile([_CRUISE] * 50),
        t_onset=20,
        t_hazard_end=40,
    )
    diag = _analyze(trace, success=True, success_step=48)
    assert diag.reacted is False
    assert diag.stopped is False
    assert diag.evaded is False
    assert diag.behavior_class is BehaviorClass.NO_RESPONSE


def test_all_classes_exhaustive_and_mutually_exclusive() -> None:
    """One synthetic episode per class; the observed set must be every class."""
    cases: list[tuple[BehaviorClass, Trace, dict[str, object]]] = [
        (
            BehaviorClass.COLLIDE_UNREACTED,
            make_trace(
                speeds=_profile([_CRUISE] * 50),
                contact_at=range(30, 35),
                t_onset=20,
                t_hazard_end=40,
            ),
            {},
        ),
        (
            BehaviorClass.COLLIDE_REACTED,
            make_trace(
                speeds=_profile([_CRUISE] * 20, [_SLOW] * 30),
                contact_at=range(30, 34),
                t_onset=20,
                t_hazard_end=40,
            ),
            {},
        ),
        (
            BehaviorClass.EVADE_SUCCESS,
            make_trace(
                speeds=_profile([_CRUISE] * 20, [_SLOW] * 30),
                headings_deg=[0.0] * 20 + [90.0] * 30,
                t_onset=20,
                t_hazard_end=40,
            ),
            {"success": True, "success_step": 45},
        ),
        (
            BehaviorClass.EVADE_FAIL,
            make_trace(
                speeds=_profile([_CRUISE] * 20, [_SLOW] * 30),
                headings_deg=[0.0] * 20 + [90.0] * 30,
                t_onset=20,
                t_hazard_end=40,
            ),
            {"success": False},
        ),
        (
            BehaviorClass.STOP_RECOVER,
            make_trace(
                speeds=_profile([_CRUISE] * 20, [_STOPPED] * 20, [_CRUISE] * 30),
                t_onset=20,
                t_hazard_end=40,
            ),
            {"success": True, "success_step": 60},
        ),
        (
            BehaviorClass.STOP_NO_RECOVER,
            make_trace(
                speeds=_profile([_CRUISE] * 20, [_STOPPED] * 50),
                t_onset=20,
                t_hazard_end=40,
            ),
            {"success": False},
        ),
        (
            BehaviorClass.PARTIAL_RESPONSE,
            make_trace(
                speeds=_profile([_CRUISE] * 50),
                headings_deg=[0.0] * 20 + [17.0] * 30,
                t_onset=20,
                t_hazard_end=40,
            ),
            {},
        ),
        (
            BehaviorClass.NO_RESPONSE,
            make_trace(
                speeds=_profile([_CRUISE] * 50),
                t_onset=20,
                t_hazard_end=40,
            ),
            {},
        ),
        (
            BehaviorClass.NOT_TRIGGERED,
            make_trace(
                speeds=_profile([_CRUISE] * 50),
                t_onset=None,
                t_hazard_end=None,
                disturbance_triggered=False,
                hazard_persists=None,
            ),
            {},
        ),
        (
            BehaviorClass.BASELINE_UNUSABLE,
            make_trace(
                speeds=_profile([0.0] * 20, [_CRUISE] * 30),
                t_onset=20,
                t_hazard_end=40,
            ),
            {},
        ),
    ]
    observed: list[BehaviorClass] = []
    for expected, trace, kwargs in cases:
        got = _analyze(trace, **kwargs).behavior_class
        assert got is expected
        observed.append(got)
    assert set(observed) == set(BehaviorClass)
    assert len(observed) == len(set(observed)) == len(set(BehaviorClass))


def test_heading_gate_decoupled_from_stop_speed_frac() -> None:
    """Raising stop.speed_frac must not shrink heading samples.

    Post-onset speed 0.002 sits between stop_gate(0.10)=0.001 and
    stop_gate(0.30)=0.003. Under the old coupling (min_speed=stop_gate)
    the 90° turn vanished at 0.30 (heading → None). The avoid-axis gate
    is max(0.05 × μ_pre, floor)=0.0005 either way, so
    max_heading_change_deg stays 90°.
    """
    mid = 0.002
    # v(20) (the onset sample) must already be mid, otherwise the inclusive
    # post-onset window keeps one cruise step and old_hi becomes 0° not None.
    speeds = _profile([_CRUISE] * 19, [mid] * 31)
    heads = [0.0] * 19 + [90.0] * 31
    trace = make_trace(
        speeds=speeds, headings_deg=heads, t_onset=20, t_hazard_end=40
    )
    floor = CFG.stop.speed_floor_m_per_step
    stop_lo = speed_threshold(_CRUISE, 0.10, floor)
    stop_hi = speed_threshold(_CRUISE, 0.30, floor)
    heading_gate = speed_threshold(_CRUISE, CFG.avoid.min_speed_frac, floor)
    assert stop_lo == pytest.approx(0.001)
    assert stop_hi == pytest.approx(0.003)
    assert heading_gate == pytest.approx(0.0005)
    assert stop_lo < mid < stop_hi
    assert heading_gate < mid

    # What the old coupling would have produced: 90° vs None.
    pos = _polyline(speeds, heads)
    times = np.arange(len(pos), dtype=np.int64)
    heading_end = 40 + int(CFG.avoid.window_steps)
    old_lo = compute_heading_change(
        pos,
        times,
        t_onset=20,
        t_window_end=heading_end,
        baseline_window=int(CFG.reaction.window_size),
        min_speed=stop_lo,
    )
    old_hi = compute_heading_change(
        pos,
        times,
        t_onset=20,
        t_window_end=heading_end,
        baseline_window=int(CFG.reaction.window_size),
        min_speed=stop_hi,
    )
    assert old_lo == pytest.approx(90.0, abs=1e-6)
    assert old_hi is None

    cfg_hi = replace(CFG, stop=replace(CFG.stop, speed_frac=0.30))
    a = _analyze(trace)
    b = _analyze(trace, cfg_hi)
    assert a.max_heading_change_deg == pytest.approx(90.0, abs=1e-6)
    assert b.max_heading_change_deg == pytest.approx(90.0, abs=1e-6)
    assert a.max_heading_change_deg == b.max_heading_change_deg


def test_stop_no_recover_covers_never_resume_and_failed_resume() -> None:
    """STOP_NO_RECOVER holds both 'never moved again' and 'moved but failed'.

    The two sub-cases share a class and are told apart by ``resumed``.
    """
    never = _analyze(
        make_trace(
            speeds=_profile([_CRUISE] * 20, [_STOPPED] * 50),
            t_onset=20,
            t_hazard_end=40,
        ),
        success=False,
    )
    failed_resume = _analyze(
        make_trace(
            speeds=_profile([_CRUISE] * 20, [_STOPPED] * 20, [_CRUISE] * 30),
            t_onset=20,
            t_hazard_end=40,
        ),
        success=False,
    )
    assert never.behavior_class is failed_resume.behavior_class is BehaviorClass.STOP_NO_RECOVER
    assert never.resumed is False
    assert failed_resume.resumed is True
    assert never.recovered is False
    assert failed_resume.recovered is False


def test_from_json_dict_fills_reacted_kinematic_from_legacy_reacted() -> None:
    """Pre-2026-08-29 JSONL stored the kinematic bit as ``reacted``."""
    diag = _analyze(
        make_trace(
            speeds=_profile([_CRUISE] * 28, [_SLOW] * 22),
            contact_at=range(40, 44),
            t_onset=20,
            t_hazard_end=50,
            action_chunk_len=8,
        )
    )
    payload = diag.to_json_dict()
    assert "reacted_kinematic" in payload
    legacy = dict(payload)
    legacy.pop("reacted_kinematic")
    # Inventing closed-loop True from a kinematic True would lie.
    legacy["reacted"] = True
    restored = BehaviorDiagnostics.from_json_dict(legacy)
    assert restored.reacted is True
    assert restored.reacted_kinematic is True
    missing_both = dict(payload)
    missing_both.pop("reacted")
    missing_both.pop("reacted_kinematic")
    with pytest.raises(KeyError, match="reacted"):
        BehaviorDiagnostics.from_json_dict(missing_both)
