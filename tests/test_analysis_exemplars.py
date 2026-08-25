"""Offline exemplar selection and trajectory plots. No simulator, no backend."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from dynisafety.analysis.cli import main as analysis_main
from dynisafety.analysis.exemplars import (
    plot_exemplar_trajectory,
    select_exemplars,
    write_exemplars,
)
from dynisafety.analysis.load import BoundEpisode, fingerprint_results
from dynisafety.config.schema import DiagnosticsTraceConfig
from dynisafety.core.episode import SCHEMA_VERSION, EpisodeRecord
from dynisafety.diagnostics.schema import (
    TRACE_SCHEMA_VERSION,
    BehaviorClass,
    Trace,
)
from dynisafety.diagnostics.writer import TraceWriter
from test_diagnostics_analyze import make_trace


def _diag(**kw) -> dict:
    base = {
        "reacted": False,
        "reaction_latency_steps": None,
        "reaction_latency_chunks": None,
        "reaction_post_contact": False,
        "stopped": False,
        "stop_preemptive": False,
        "stop_latency_steps": None,
        "stop_duration_steps": None,
        "evaded": False,
        "min_clearance_m": None,
        "max_heading_change_deg": None,
        "resumed": False,
        "recovered": False,
        "recovery_applicable": False,
        "behavior_class": BehaviorClass.NO_RESPONSE.value,
        "disturbance_triggered": True,
        "t_onset": 10,
        "t_hazard_end": 20,
        "hazard_persists": True,
        "trace_path": None,
    }
    base.update(kw)
    return base


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
        level=2,
        seed=7,
        episode_index=0,
        steps=30,
        max_steps=50,
        termination="max_steps",
        wall_time_s=1.0,
        success=False,
        success_step=None,
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
        diagnostics=_diag(),
    )
    base.update(kw)
    return EpisodeRecord(**base)


def _bound(ep: EpisodeRecord, run_dir: Path) -> BoundEpisode:
    return BoundEpisode(episode=ep, run_dir=run_dir, jsonl_path=run_dir / "episodes.jsonl")


def test_select_skips_episodes_without_behavior_class(tmp_path: Path) -> None:
    missing = _bound(_ep(diagnostics=None, episode_index=0), tmp_path)
    kept = _bound(_ep(episode_index=1), tmp_path)
    chosen = select_exemplars([missing, kept])
    assert [c.episode.episode_index for c in chosen] == [1]


def test_select_prefers_trace_then_timeout(tmp_path: Path) -> None:
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    run_a.mkdir()
    run_b.mkdir()
    no_trace = _bound(
        _ep(
            run_id="aaa",
            episode_index=0,
            success=False,
            termination="max_steps",
            steps=50,
            max_steps=50,
            diagnostics=_diag(behavior_class=BehaviorClass.NO_RESPONSE.value),
        ),
        run_a,
    )
    with_trace = _bound(
        _ep(
            run_id="zzz",
            episode_index=3,
            success=False,
            termination="success",
            steps=12,
            max_steps=50,
            diagnostics=_diag(
                behavior_class=BehaviorClass.NO_RESPONSE.value,
                trace_path="traces/ep0003.jsonl",
            ),
        ),
        run_b,
    )
    writer = TraceWriter(run_b, DiagnosticsTraceConfig(save=True, format="jsonl", stride=1))
    trace = make_trace(n_steps=16, t_onset=6, t_hazard_end=12, level=2)
    stored = Trace(
        header=replace(
            trace.header,
            run_id="zzz",
            episode_index=3,
            backend="libero",
            level=2,
            seed=7,
            n_steps=len(trace.steps),
        ),
        steps=trace.steps,
    )
    writer.write(3, stored)
    chosen = select_exemplars([no_trace, with_trace], per_class=1)
    assert len(chosen) == 1
    assert chosen[0].episode.episode_index == 3


def test_skip_not_triggered() -> None:
    root = Path("/tmp/unused")
    idle = _bound(
        _ep(
            diagnostics=_diag(
                behavior_class=BehaviorClass.NOT_TRIGGERED.value,
                disturbance_triggered=False,
            )
        ),
        root,
    )
    assert select_exemplars([idle], include_not_triggered=False) == []
    assert len(select_exemplars([idle], include_not_triggered=True)) == 1


def test_plot_and_write_do_not_mutate_results(tmp_path: Path) -> None:
    results = tmp_path / "results"
    run_dir = results / "l2" / "libero-m1-s-t0-l2-s7-synth"
    run_dir.mkdir(parents=True)
    trace = make_trace(n_steps=20, t_onset=6, t_hazard_end=14, level=2)
    stored = Trace(
        header=replace(
            trace.header,
            run_id="libero-m1-s-t0-l2-s7-synth",
            episode_index=0,
            backend="libero",
            level=2,
            seed=7,
            n_steps=len(trace.steps),
            trace_schema_version=TRACE_SCHEMA_VERSION,
        ),
        steps=trace.steps,
    )
    writer = TraceWriter(run_dir, DiagnosticsTraceConfig(save=True, format="jsonl", stride=1))
    rel = writer.write(0, stored)
    ep = _ep(
        run_id="libero-m1-s-t0-l2-s7-synth",
        diagnostics=_diag(trace_path=rel, t_onset=6, t_hazard_end=14),
    )
    (run_dir / "episodes.jsonl").write_text(
        __import__("json").dumps(ep.to_json_dict(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    before = fingerprint_results(results)
    out = tmp_path / "out"
    rc = analysis_main(["exemplars", "--results", str(results), "--out", str(out)])
    assert rc == 0
    assert (out / "exemplars.csv").is_file()
    plots = list((out / "plots").glob("*.png"))
    assert len(plots) == 1
    assert fingerprint_results(results) == before


def test_plot_exemplar_writes_png(tmp_path: Path) -> None:
    trace = make_trace(n_steps=12, t_onset=4, t_hazard_end=8, contact_at=(6,), level=2)
    dest = tmp_path / "traj.png"
    plot_exemplar_trajectory(trace, dest, first_contact_step=6, title="unit")
    assert dest.is_file()
    assert dest.stat().st_size > 0


def test_write_exemplars_keeps_row_without_trace(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    bound = [_bound(_ep(), run_dir)]
    rows = write_exemplars(bound, tmp_path / "out")
    assert len(rows) == 1
    assert rows[0].plot_path is None
    assert rows[0].behavior_class == BehaviorClass.NO_RESPONSE.value
