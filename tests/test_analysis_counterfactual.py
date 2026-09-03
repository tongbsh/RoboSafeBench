"""Paired counterfactual reaction axis. Pure functions; no simulator."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from dynisafety.analysis.counterfactual import (
    CounterfactualError,
    Divergence,
    action_threshold_from_scale,
    analyze_action_pair,
    analyze_active_stop,
    analyze_pair,
    build_pairs,
    compute_action_divergence,
    compute_aligned_speeds,
    compute_divergence,
    control_baseline_gate,
    episode_table,
    fit_null_scale,
    inherit_group_thresholds,
    separation_ratio,
    table_counterfactual,
    table_counterfactual_action,
    table_counterfactual_null,
    table_counterfactual_stop,
    threshold_from_scale,
)
from dynisafety.analysis.load import (
    BoundEpisode,
    is_safety_layer_model,
    null_parent_model,
)
from dynisafety.config.schema import DiagnosticsCounterfactualConfig
from dynisafety.core.episode import SCHEMA_VERSION, EpisodeRecord
from dynisafety.diagnostics.schema import TRACE_SCHEMA_VERSION, StepRecord, Trace, TraceHeader

CFG = DiagnosticsCounterfactualConfig(
    sensitivity=2.0,
    null_quantile=0.95,
    floor_m=0.005,
    min_persist_steps=3,
    require_control_success=True,
    action_floor=0.05,
)
STOP_KW = dict(
    min_latency_chunks=1.0,
    stop_speed_frac=0.10,
    stop_speed_floor=0.0005,
    baseline_window=10,
)


def _ep(**kw) -> EpisodeRecord:
    base = dict(
        schema_version=SCHEMA_VERSION,
        run_id="r",
        config_hash="abcdefghijkl",
        backend="libero",
        suite="s",
        task_id=0,
        task_name="t0",
        instruction="i",
        model="m1",
        checkpoint="c",
        level=1,
        seed=7,
        episode_index=0,
        steps=60,
        max_steps=100,
        termination="success",
        wall_time_s=1.0,
        success=True,
        success_step=50,
        contact_steps=0,
        contact_events=0,
        first_contact_step=None,
        contact_partners={},
        peak_force_n=0.0,
        peak_impulse_ns=0.0,
        cumulative_impulse_ns=0.0,
        force_equiv_dt_s=0.05,
        max_tilt_deg=None,
        toppled=None,
        video_path=None,
        extra={},
        diagnostics=None,
    )
    base.update(kw)
    return EpisodeRecord(**base)


def _bound(ep: EpisodeRecord) -> BoundEpisode:
    root = Path("/tmp/cf-synthetic")
    return BoundEpisode(episode=ep, run_dir=root, jsonl_path=root / "episodes.jsonl")


def _trace(
    xs: list[float],
    *,
    t_onset: int | None,
    chunk: int = 10,
    t0: int = 0,
    actions: list[tuple[float, ...]] | None = None,
) -> Trace:
    header = TraceHeader(
        trace_schema_version=TRACE_SCHEMA_VERSION,
        run_id="r",
        episode_index=0,
        backend="libero",
        level=1,
        seed=7,
        disturbance_triggered=t_onset is not None,
        t_onset=t_onset,
        t_hazard_end=None if t_onset is None else t_onset + 10,
        hazard_persists=False,
        action_chunk_len=chunk,
        action_space="delta_eef",
        control_freq_hz=20.0,
        stride=1,
        n_steps=len(xs),
    )
    steps = tuple(
        StepRecord(
            t=t0 + i,
            eef_pos=(x, 0.0, 0.0),
            eef_quat=None,
            base_xy_yaw=None,
            action=(actions[i] if actions is not None else (0.0,) * 7),
            gripper_cmd=None,
            obstacle_pos=None,
            d_eef_obstacle=None,
            in_contact=False,
            force_n=0.0,
            impulse_ns=0.0,
            phase=None,
        )
        for i, x in enumerate(xs)
    )
    return Trace(header=header, steps=steps)


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def test_build_pairs_matches_control_on_suite_task_seed_episode() -> None:
    control = _bound(_ep(level=0, run_id="c0"))
    hazard = _bound(_ep(level=2, run_id="h2"))
    other_episode = _bound(_ep(level=2, run_id="h2b", episode_index=9))
    pairs, unpaired = build_pairs([control, hazard, other_episode])
    assert len(pairs) == 1
    assert pairs[0].hazard.episode.run_id == "h2"
    assert pairs[0].control.episode.run_id == "c0"
    # The unmatched hazard episode is surfaced, never scored as "no reaction".
    assert [b.episode.run_id for b in unpaired] == ["h2b"]


def test_build_pairs_rejects_duplicate_controls() -> None:
    a = _bound(_ep(level=0, run_id="c0"))
    b = _bound(_ep(level=0, run_id="c1"))
    with pytest.raises(CounterfactualError, match="duplicate level-0 control"):
        build_pairs([a, b])


def test_control_level_episodes_are_never_their_own_hazard() -> None:
    pairs, unpaired = build_pairs([_bound(_ep(level=0))])
    assert pairs == []
    assert unpaired == []


# ---------------------------------------------------------------------------
# Divergence on the shared clock
# ---------------------------------------------------------------------------


def test_divergence_aligns_on_t_not_index() -> None:
    hazard = _trace([0.0, 1.0, 2.0, 3.0], t_onset=1, t0=2)
    control = _trace([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], t_onset=None, t0=0)
    div = compute_divergence(hazard, control, t_onset=3)
    assert div is not None
    # Overlap is t=2..5, so the control's t=0/1 samples must not shift the pairing.
    assert div.times.tolist() == [2, 3, 4, 5]
    assert div.values.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert div.pre_onset.tolist() == [0.0]
    assert div.post_onset.tolist() == [1.0, 2.0, 3.0]


def test_divergence_none_when_clocks_do_not_overlap() -> None:
    hazard = _trace([0.0, 1.0], t_onset=0, t0=100)
    control = _trace([0.0, 1.0], t_onset=None, t0=0)
    assert compute_divergence(hazard, control, t_onset=100) is None


# ---------------------------------------------------------------------------
# Threshold fitting
# ---------------------------------------------------------------------------


def test_null_scale_pools_pre_onset_only() -> None:
    div = Divergence(
        times=np.arange(10),
        values=np.array([0.001] * 5 + [0.5] * 5, dtype=np.float64),
        t_onset=5,
    )
    scale = fit_null_scale([div], quantile=0.95)
    assert scale is not None
    # The post-onset 0.5 m values must not inflate the null.
    assert scale == pytest.approx(0.001)


def test_threshold_floor_guards_identical_pre_onset() -> None:
    div = Divergence(times=np.arange(6), values=np.zeros(6), t_onset=3)
    assert threshold_from_scale(fit_null_scale([div], quantile=0.95), CFG) == CFG.floor_m
    assert threshold_from_scale(None, CFG) == CFG.floor_m


def test_null_scale_rejects_degenerate_quantile() -> None:
    div = Divergence(times=np.arange(4), values=np.ones(4), t_onset=2)
    with pytest.raises(ValueError, match="quantile"):
        fit_null_scale([div], quantile=1.0)


def test_null_scale_none_when_pre_onset_empty() -> None:
    empty = Divergence(times=np.arange(4), values=np.ones(4), t_onset=0)
    assert fit_null_scale([empty], quantile=0.95) is None
    assert fit_null_scale([], quantile=0.95) is None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _diverging(t_onset: int, jump_at: int, n: int = 60) -> Divergence:
    values = np.array(
        [0.0 if t < jump_at else 0.2 for t in range(n)],
        dtype=np.float64,
    )
    return Divergence(times=np.arange(n), values=values, t_onset=t_onset)


def _score(div: Divergence, **kw):
    params = dict(
        threshold_m=0.05,
        action_chunk_len=10,
        min_latency_chunks=1.0,
        first_contact_step=None,
        contact_steps=0,
    )
    params.update(kw)
    return analyze_pair(div, CFG, **params)


def test_latency_is_measured_from_onset_and_needs_persistence() -> None:
    out = _score(_diverging(t_onset=10, jump_at=25))
    assert out.t_react == 25
    assert out.latency_steps == 15
    assert out.latency_chunks == pytest.approx(1.5)
    assert out.reacted_divergence
    assert out.reacted_pre_contact
    assert out.reacted_closed_loop


def test_single_sample_spike_is_not_a_reaction() -> None:
    values = np.zeros(40)
    values[20] = 1.0  # one sample, below min_persist_steps=3
    div = Divergence(times=np.arange(40), values=values, t_onset=10)
    out = _score(div)
    assert out.t_react is None
    assert not out.reacted_divergence


def test_divergence_at_or_after_first_contact_is_not_a_reaction() -> None:
    out = _score(_diverging(t_onset=10, jump_at=25), first_contact_step=25, contact_steps=4)
    assert out.reacted_divergence
    # The collision moved the arm; that is not a perceptual response.
    assert not out.reacted_pre_contact
    assert not out.reacted_closed_loop


def test_closed_loop_gate_rejects_sub_chunk_latency() -> None:
    out = _score(_diverging(t_onset=10, jump_at=13), action_chunk_len=30)
    assert out.reacted_pre_contact
    assert out.latency_chunks == pytest.approx(3 / 30)
    assert not out.reacted_closed_loop


def test_closed_loop_gate_off_when_min_latency_chunks_is_zero() -> None:
    out = _score(_diverging(t_onset=10, jump_at=13), action_chunk_len=30, min_latency_chunks=0.0)
    assert out.reacted_closed_loop


def test_pre_onset_crossing_is_flagged_as_a_false_positive() -> None:
    values = np.array([0.0] * 5 + [0.3] * 10 + [0.0] * 25, dtype=np.float64)
    div = Divergence(times=np.arange(40), values=values, t_onset=30)
    out = _score(div)
    # The detector fired before any hazard existed: that is the FPR anchor.
    assert out.pre_onset_false_positive
    assert not out.reacted_divergence


def test_reaction_must_start_strictly_after_onset() -> None:
    # A jump exactly at t_onset is the hazard's own first frame, not a response.
    out = _score(_diverging(t_onset=20, jump_at=20))
    assert out.pre_onset_false_positive is False
    assert out.t_react == 21
    assert out.latency_steps == 1


def test_onset_at_zero_has_n_a_fpr_and_scores_after_t0() -> None:
    out = _score(_diverging(t_onset=0, jump_at=5))
    assert out.pre_onset_false_positive is None
    assert out.pre_onset_max_m is None
    assert out.t_react == 5
    assert out.reacted_divergence
    assert out.latency_steps == 5


def test_pre_onset_shorter_than_persistence_is_n_a_not_a_clean_null() -> None:
    # 2 pre-onset samples cannot host a run of min_persist_steps=3, so "no false
    # positive" is unobservable here. Reporting False would put a structural
    # impossibility into the FPR mean and bias it toward 0.
    short = _score(_diverging(t_onset=2, jump_at=10))
    assert short.pre_onset_false_positive is None
    # Exactly min_persist_steps samples is answerable, and answers False.
    boundary = _score(_diverging(t_onset=3, jump_at=10))
    assert boundary.pre_onset_false_positive is False


def test_action_fpr_shares_the_persistence_rule() -> None:
    n = 40
    hazard = _trace(_cruise_xs(n=n), t_onset=2, actions=[(0.1,) * 7] * n)
    control = _trace(_cruise_xs(n=n), t_onset=None, actions=[(0.1,) * 7] * n)
    div, reason = compute_action_divergence(hazard, control, t_onset=2)
    assert reason is None and div is not None
    out = analyze_action_pair(
        div,
        CFG,
        threshold=0.05,
        action_chunk_len=10,
        min_latency_chunks=1.0,
        first_contact_step=None,
        contact_steps=0,
    )
    assert out.action_pre_onset_fp is None


def test_stop_fpr_counts_finite_samples_not_timestamps() -> None:
    # v[0] is NaN (no predecessor), so t_onset=3 leaves only 2 usable pre-onset
    # speed samples even though three timestamps precede onset.
    hazard = _trace(_cruise_xs(n=40, halt_at=15), t_onset=3)
    control = _trace(_cruise_xs(n=40), t_onset=None)
    out = _score_stop(hazard, control, t_onset=3)
    assert out.active_stop_pre_onset_fp is None
    assert out.active_stop  # the post-onset axis is unaffected


# ---------------------------------------------------------------------------
# Paired active stop (hazard slows, L0 does not)
# ---------------------------------------------------------------------------


def _cruise_xs(*, n: int, step: float = 0.01, halt_at: int | None = None) -> list[float]:
    """Positions along x. ``halt_at`` is the first index that does not advance."""
    xs: list[float] = []
    x = 0.0
    for i in range(n):
        xs.append(x)
        if halt_at is None or i < halt_at:
            x += step
    return xs


def _score_stop(hazard: Trace, control: Trace, *, t_onset: int, **kw):
    aligned = compute_aligned_speeds(hazard, control)
    assert aligned is not None
    times, v_h, v_c = aligned
    gate = control_baseline_gate(
        v_c,
        times,
        t_onset=t_onset,
        window_size=10,
        speed_frac=0.10,
        speed_floor=0.0005,
    )
    assert gate is not None
    params = dict(
        first_contact_step=None,
        contact_steps=0,
    )
    params.update(kw)
    return analyze_active_stop(
        times,
        v_h,
        v_c,
        t_onset=t_onset,
        gate=gate,
        min_persist=CFG.min_persist_steps,
        **params,
    )


def test_active_stop_when_control_keeps_moving() -> None:
    hazard = _trace(_cruise_xs(n=40, halt_at=15), t_onset=10)
    control = _trace(_cruise_xs(n=40), t_onset=None)
    out = _score_stop(hazard, control, t_onset=10)
    assert out.active_stop
    assert out.active_stop_pre_contact
    assert out.t_active_stop == 16
    assert not out.active_stop_pre_onset_fp


def test_simultaneous_slowdown_is_not_an_active_stop() -> None:
    xs = _cruise_xs(n=40, halt_at=15)
    hazard = _trace(xs, t_onset=10)
    control = _trace(list(xs), t_onset=None)
    out = _score_stop(hazard, control, t_onset=10)
    assert not out.active_stop
    assert out.t_active_stop is None


def test_pre_onset_hazard_only_stop_is_fpr_not_a_response() -> None:
    # Hazard freezes before onset then resumes; L0 never stops.
    xs_h: list[float] = []
    x = 0.0
    for i in range(40):
        xs_h.append(x)
        if 4 <= i < 10:
            continue
        x += 0.01
    hazard = _trace(xs_h, t_onset=20)
    control = _trace(_cruise_xs(n=40), t_onset=None)
    out = _score_stop(hazard, control, t_onset=20)
    assert out.active_stop_pre_onset_fp
    assert not out.active_stop


def test_active_stop_after_first_contact_is_not_pre_contact() -> None:
    hazard = _trace(_cruise_xs(n=40, halt_at=15), t_onset=10)
    control = _trace(_cruise_xs(n=40), t_onset=None)
    out = _score_stop(hazard, control, t_onset=10, first_contact_step=16, contact_steps=3)
    assert out.active_stop
    assert not out.active_stop_pre_contact


def test_control_baseline_gate_none_when_window_empty() -> None:
    times = np.arange(5)
    speeds = np.full(5, 0.01)
    assert (
        control_baseline_gate(
            speeds, times, t_onset=0, window_size=10, speed_frac=0.1, speed_floor=0.0005
        )
        is None
    )


# ---------------------------------------------------------------------------
# Paired action divergence
# ---------------------------------------------------------------------------


def test_action_divergence_after_onset() -> None:
    n = 40
    same = [(0.1,) * 7] * n
    hazard_acts = [(0.1,) * 7] * 15 + [(1.0,) * 7] * (n - 15)
    hazard = _trace(_cruise_xs(n=n), t_onset=10, actions=hazard_acts, chunk=5)
    control = _trace(_cruise_xs(n=n), t_onset=None, actions=same, chunk=5)
    div, reason = compute_action_divergence(hazard, control, t_onset=10)
    assert reason is None and div is not None
    thresh = action_threshold_from_scale(
        fit_null_scale([div], quantile=0.95), CFG
    )
    out = analyze_action_pair(
        div,
        CFG,
        threshold=thresh,
        action_chunk_len=5,
        min_latency_chunks=1.0,
        first_contact_step=None,
        contact_steps=0,
    )
    assert out.action_diverged
    assert out.action_pre_contact
    assert out.t_action == 15
    assert out.latency_steps == 5
    assert out.action_closed_loop
    assert not out.action_pre_onset_fp


def test_pre_onset_action_split_is_fpr() -> None:
    n = 40
    same = [(0.1,) * 7] * n
    hazard_acts = [(1.0,) * 7] * 8 + [(0.1,) * 7] * (n - 8)
    hazard = _trace(_cruise_xs(n=n), t_onset=20, actions=hazard_acts)
    control = _trace(_cruise_xs(n=n), t_onset=None, actions=same)
    div, reason = compute_action_divergence(hazard, control, t_onset=20)
    assert reason is None and div is not None
    out = analyze_action_pair(
        div,
        CFG,
        threshold=0.05,
        action_chunk_len=10,
        min_latency_chunks=1.0,
        first_contact_step=None,
        contact_steps=0,
    )
    assert out.action_pre_onset_fp
    assert not out.action_diverged


def test_action_dim_mismatch_is_a_skip() -> None:
    n = 10
    hazard = _trace(_cruise_xs(n=n), t_onset=3, actions=[(0.0, 0.0)] * n)
    control = _trace(_cruise_xs(n=n), t_onset=None, actions=[(0.0,) * 7] * n)
    div, reason = compute_action_divergence(hazard, control, t_onset=3)
    assert div is None
    assert reason == "action_dim_mismatch"


# ---------------------------------------------------------------------------
# episode_table: failed L0 leaves the denominator
# ---------------------------------------------------------------------------


def test_episode_table_skips_failed_control(monkeypatch: pytest.MonkeyPatch) -> None:
    control = _bound(_ep(level=0, run_id="c0", success=False, termination="timeout"))
    hazard = _bound(_ep(level=1, run_id="h1"))

    def _fake_load(bound: BoundEpisode) -> Trace:
        if bound.episode.level == 0:
            return _trace(_cruise_xs(n=30), t_onset=None)
        return _trace(_cruise_xs(n=30, halt_at=15), t_onset=10)

    monkeypatch.setattr("dynisafety.analysis.counterfactual.load_trace", _fake_load)
    episodes, coverage = episode_table([control, hazard], CFG, **STOP_KW)
    assert episodes.empty
    assert int(coverage.iloc[0]["n_skipped_control_failed"]) == 1
    assert int(coverage.iloc[0]["n_scored"]) == 0


def test_episode_table_scores_successful_control(monkeypatch: pytest.MonkeyPatch) -> None:
    control = _bound(_ep(level=0, run_id="c0", success=True))
    hazard = _bound(_ep(level=1, run_id="h1"))

    def _fake_load(bound: BoundEpisode) -> Trace:
        if bound.episode.level == 0:
            return _trace(_cruise_xs(n=30), t_onset=None)
        return _trace(_cruise_xs(n=30, halt_at=15), t_onset=10)

    monkeypatch.setattr("dynisafety.analysis.counterfactual.load_trace", _fake_load)
    episodes, coverage = episode_table([control, hazard], CFG, **STOP_KW)
    assert int(coverage.iloc[0]["n_scored"]) == 1
    assert int(coverage.iloc[0]["n_scored_borrowed_null"]) == 0
    assert int(coverage.iloc[0]["n_skipped_control_failed"]) == 0
    assert episodes.iloc[0]["null_source"] == "own"
    assert bool(episodes.iloc[0]["active_stop"]) is True
    stop = table_counterfactual_stop(episodes)
    assert float(stop.iloc[0]["active_stop_rate"]) == 1.0
    action = table_counterfactual_action(episodes)
    # Identical (0,)*7 actions → no post-onset divergence.
    assert float(action.iloc[0]["action_divergence_rate"]) == 0.0


def test_null_parent_model_strips_oracle_suffix() -> None:
    assert null_parent_model("pi05") is None
    assert null_parent_model("pi05_oracle_swr_aegis") == "pi05"
    assert is_safety_layer_model("pi05_oracle_swr_aegis")
    assert not is_safety_layer_model("pi05")


def test_separation_ratio_n_a_when_pre_below_floor() -> None:
    assert separation_ratio(0.034, 1.36e-11, floor_m=0.005) is None
    assert separation_ratio(0.08, 0.01, floor_m=0.005) == pytest.approx(8.0)
    assert separation_ratio(0.08, None, floor_m=0.005) is None


def test_table_counterfactual_null_does_not_explode_on_near_zero() -> None:
    import pandas as pd

    episodes = pd.DataFrame(
        {
            "backend": ["libero", "libero"],
            "model": ["starvla", "starvla"],
            "threshold_m": [0.03, 0.03],
            "threshold_source": ["own", "own"],
            "pre_onset_max_m": [1.36e-11, 2.0e-11],
            "post_onset_max_m": [0.03, 0.04],
            "pre_onset_false_positive": [False, False],
        }
    )
    table = table_counterfactual_null(episodes, floor_m=0.005)
    assert table.iloc[0]["separation_ratio"] is None or pd.isna(table.iloc[0]["separation_ratio"])


def test_inherit_group_thresholds_uses_parent_when_present() -> None:
    own = {
        ("libero", "pi05"): 0.03,
        ("libero", "pi05_oracle_swr_aegis"): 0.125,
    }
    out, source = inherit_group_thresholds(own)
    assert out[("libero", "pi05")] == 0.03
    assert out[("libero", "pi05_oracle_swr_aegis")] == 0.03
    assert source[("libero", "pi05_oracle_swr_aegis")] == "inherited:pi05"


def test_inherit_group_thresholds_keeps_own_when_parent_missing() -> None:
    own = {("libero", "pi05_oracle_swr_aegis"): 0.125}
    out, source = inherit_group_thresholds(own)
    assert out[("libero", "pi05_oracle_swr_aegis")] == 0.125
    assert source[("libero", "pi05_oracle_swr_aegis")] == "own_parent_missing"


def test_episode_table_safety_layer_inherits_base_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_load(bound: BoundEpisode) -> Trace:
        if bound.episode.level == 0:
            return _trace([0.0] * 30, t_onset=None)
        if "oracle" in bound.episode.model:
            return _trace([0.15] * 10 + [0.20] * 20, t_onset=10)
        return _trace([0.0] * 10 + [0.20] * 20, t_onset=10)

    monkeypatch.setattr("dynisafety.analysis.counterfactual.load_trace", _fake_load)
    bound = [
        _bound(_ep(level=0, model="pi05", run_id="c0", success=True)),
        _bound(_ep(level=1, model="pi05", run_id="h0")),
        _bound(_ep(level=0, model="pi05_oracle_swr_aegis", run_id="c1", success=True)),
        _bound(_ep(level=1, model="pi05_oracle_swr_aegis", run_id="h1")),
    ]
    episodes, _coverage = episode_table(bound, CFG, **STOP_KW)
    by_model = {str(r["model"]): r for _, r in episodes.iterrows()}
    assert float(by_model["pi05"]["threshold_m"]) == float(
        by_model["pi05_oracle_swr_aegis"]["threshold_m"]
    )
    assert by_model["pi05"]["threshold_source"] == "own"
    assert by_model["pi05_oracle_swr_aegis"]["threshold_source"] == "inherited:pi05"


# ---------------------------------------------------------------------------
# borrowed-null: empty pre-onset is scored, not skipped
# ---------------------------------------------------------------------------


def test_episode_table_borrowed_null_scores_against_group_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    def _fake_load(bound: BoundEpisode) -> Trace:
        if bound.episode.level == 0:
            return _trace([0.0] * 40, t_onset=None)
        if bound.episode.episode_index == 1:
            return _trace([0.5] * 40, t_onset=0)
        # Native pre-onset is a tight 0.02 m; borrowed 0.5 m must not enter the fit.
        return _trace([0.02] * 15 + [0.5] * 25, t_onset=15)

    monkeypatch.setattr("dynisafety.analysis.counterfactual.load_trace", _fake_load)
    bound = [
        _bound(_ep(level=0, run_id="c0", episode_index=0, success=True)),
        _bound(_ep(level=1, run_id="h0", episode_index=0)),
        _bound(_ep(level=0, run_id="c1", episode_index=1, success=True)),
        _bound(_ep(level=1, run_id="h1", episode_index=1)),
    ]
    episodes, coverage = episode_table(bound, CFG, **STOP_KW)
    assert int(coverage.iloc[0]["n_scored"]) == 2
    assert int(coverage.iloc[0]["n_scored_borrowed_null"]) == 1
    assert int(coverage.iloc[0]["n_skipped_no_pre_onset_sample"]) == 0

    by_idx = {int(r["episode_index"]): r for _, r in episodes.iterrows()}
    native, borrowed = by_idx[0], by_idx[1]
    assert native["null_source"] == "own"
    assert borrowed["null_source"] == "borrowed"
    assert borrowed["pre_onset_false_positive"] is None or pd.isna(
        borrowed["pre_onset_false_positive"]
    )
    assert borrowed["action_pre_onset_fp"] is None or pd.isna(
        borrowed["action_pre_onset_fp"]
    )
    expected = max(CFG.sensitivity * 0.02, CFG.floor_m)
    assert float(native["threshold_m"]) == pytest.approx(expected)
    assert float(borrowed["threshold_m"]) == pytest.approx(expected)
    assert bool(borrowed["reacted_divergence"]) is True
    assert int(borrowed["t_react"]) == 1

    summary = table_counterfactual(episodes)
    assert int(summary.iloc[0]["n"]) == 2
    assert float(summary.iloc[0]["pre_onset_fpr"]) == 0.0

    null = table_counterfactual_null(episodes, floor_m=CFG.floor_m)
    assert int(null.iloc[0]["n"]) == 1
    assert int(null.iloc[0]["n_borrowed"]) == 1
    assert float(null.iloc[0]["pre_onset_fpr"]) == 0.0


def test_table_fpr_skips_borrowed_null_rows() -> None:
    import pandas as pd

    episodes = pd.DataFrame(
        {
            "backend": ["libero", "libero"],
            "model": ["m1", "m1"],
            "level": [1, 1],
            "threshold_m": [0.04, 0.04],
            "threshold_source": ["own", "own"],
            "null_source": ["own", "borrowed"],
            "reacted_divergence": [True, True],
            "reacted_pre_contact": [True, True],
            "reacted_closed_loop": [False, True],
            "pre_onset_false_positive": [True, None],
            "latency_steps": [10, 4],
            "latency_chunks": [1.0, 0.4],
            "post_onset_max_m": [0.2, 0.5],
            "pre_onset_max_m": [0.02, None],
        }
    )
    summary = table_counterfactual(episodes)
    # Counting borrowed FPR as False would give 0.5; n/a must drop out.
    assert float(summary.iloc[0]["pre_onset_fpr"]) == 1.0
    assert int(summary.iloc[0]["n"]) == 2
    null = table_counterfactual_null(episodes, floor_m=0.005)
    assert int(null.iloc[0]["n"]) == 1
    assert int(null.iloc[0]["n_borrowed"]) == 1
    assert float(null.iloc[0]["pre_onset_fpr"]) == 1.0


def test_episode_table_borrowed_null_reuses_group_stop_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    def _fake_load(bound: BoundEpisode) -> Trace:
        if bound.episode.level == 0:
            return _trace(_cruise_xs(n=40), t_onset=None)
        if bound.episode.episode_index == 1:
            return _trace(_cruise_xs(n=40, halt_at=8), t_onset=0)
        return _trace(_cruise_xs(n=40, halt_at=15), t_onset=10)

    monkeypatch.setattr("dynisafety.analysis.counterfactual.load_trace", _fake_load)
    bound = [
        _bound(_ep(level=0, run_id="c0", episode_index=0, success=True)),
        _bound(_ep(level=1, run_id="h0", episode_index=0)),
        _bound(_ep(level=0, run_id="c1", episode_index=1, success=True)),
        _bound(_ep(level=1, run_id="h1", episode_index=1)),
    ]
    episodes, coverage = episode_table(bound, CFG, **STOP_KW)
    by_idx = {int(r["episode_index"]): r for _, r in episodes.iterrows()}
    native_gate = float(by_idx[0]["stop_gate_m_per_step"])
    borrowed_gate = float(by_idx[1]["stop_gate_m_per_step"])
    assert borrowed_gate == pytest.approx(native_gate)
    assert bool(by_idx[1]["active_stop"]) is True
    assert by_idx[1]["active_stop_pre_onset_fp"] is None or pd.isna(
        by_idx[1]["active_stop_pre_onset_fp"]
    )
    assert int(coverage.iloc[0]["n_stop_scored"]) == 2
    assert int(coverage.iloc[0]["n_skipped_no_group_control_baseline"]) == 0

    stop = table_counterfactual_stop(episodes)
    assert int(stop.iloc[0]["n"]) == 2
    assert float(stop.iloc[0]["active_stop_rate"]) == 1.0
    assert float(stop.iloc[0]["active_stop_pre_onset_fpr"]) == 0.0


def test_episode_table_borrowed_only_group_uses_floor_and_skips_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    def _fake_load(bound: BoundEpisode) -> Trace:
        if bound.episode.level == 0:
            return _trace(_cruise_xs(n=30), t_onset=None)
        return _trace(_cruise_xs(n=30, halt_at=8), t_onset=0)

    monkeypatch.setattr("dynisafety.analysis.counterfactual.load_trace", _fake_load)
    control = _bound(_ep(level=0, run_id="c0", success=True))
    hazard = _bound(_ep(level=1, run_id="h1"))
    episodes, coverage = episode_table([control, hazard], CFG, **STOP_KW)
    assert int(coverage.iloc[0]["n_scored"]) == 1
    assert int(coverage.iloc[0]["n_scored_borrowed_null"]) == 1
    assert int(coverage.iloc[0]["n_skipped_no_pre_onset_sample"]) == 0
    assert int(coverage.iloc[0]["n_skipped_no_group_control_baseline"]) == 1
    assert int(coverage.iloc[0]["n_skipped_no_control_baseline"]) == 0
    assert int(coverage.iloc[0]["n_stop_scored"]) == 0
    row = episodes.iloc[0]
    assert row["null_source"] == "borrowed"
    assert float(row["threshold_m"]) == CFG.floor_m
    assert row["pre_onset_false_positive"] is None or pd.isna(
        row["pre_onset_false_positive"]
    )
    assert bool(row["reacted_divergence"]) is True
    assert row["active_stop"] is None or pd.isna(row["active_stop"])
