"""Offline diagnostics rewrite from traces. No simulator."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from dynisafety.core.episode import SCHEMA_VERSION, EpisodeRecord
from dynisafety.diagnostics.calibrate import fingerprint_results, load_baseline_diagnostics
from dynisafety.diagnostics.recompute import recompute_results, recompute_run
from dynisafety.diagnostics.schema import BehaviorClass
from dynisafety.diagnostics.writer import TraceWriter
from dynisafety.config.schema import DiagnosticsTraceConfig
from test_diagnostics_analyze import CFG, _CRUISE, _SLOW, _analyze, _profile, make_trace

REPO = Path(__file__).resolve().parents[1]


def _ep(**kw) -> EpisodeRecord:
    base = dict(
        schema_version=SCHEMA_VERSION,
        run_id="libero-xvla-safelibero_spatial-t0-l1-s7-synth",
        config_hash="abcdefghijkl",
        backend="libero",
        suite="safelibero_spatial",
        task_id=0,
        task_name="t0",
        instruction="i",
        model="xvla",
        checkpoint="c",
        level=1,
        seed=7,
        episode_index=0,
        steps=50,
        max_steps=50,
        termination="max_steps",
        wall_time_s=1.0,
        success=False,
        success_step=None,
        contact_steps=4,
        contact_events=1,
        first_contact_step=30,
        contact_partners={},
        peak_force_n=1.0,
        peak_impulse_ns=0.0,
        cumulative_impulse_ns=0.0,
        force_equiv_dt_s=0.05,
        max_tilt_deg=None,
        toppled=None,
        video_path=None,
        extra={},
        diagnostics={"reacted": True, "behavior_class": "collide_reacted"},
    )
    base.update(kw)
    return EpisodeRecord(**base)


def _write_completed(run_dir: Path) -> None:
    (run_dir / "summary.json").write_text(
        json.dumps({"status": "completed", "n_episodes": 1}),
        encoding="utf-8",
    )


def _write_run(
    results: Path,
    trace,
    record: EpisodeRecord,
    *,
    status: str = "completed",
) -> Path:
    run_dir = results / "libero" / record.suite / f"l{record.level}" / record.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    header = replace(
        trace.header,
        run_id=record.run_id,
        episode_index=int(record.episode_index),
        level=int(record.level),
        backend=record.backend,
        seed=int(record.seed),
    )
    stored = replace(trace, header=header)
    TraceWriter(
        run_dir,
        DiagnosticsTraceConfig(save=True, format="jsonl", stride=1),
    ).write(int(record.episode_index), stored)
    jsonl = run_dir / "episodes.jsonl"
    jsonl.write_text(
        json.dumps(record.to_json_dict(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"status": status, "n_episodes": 1}),
        encoding="utf-8",
    )
    return run_dir


def test_recompute_applies_closed_loop_gate_and_leaves_traces(tmp_path: Path) -> None:
    trace = make_trace(
        speeds=_profile([_CRUISE] * 20, [_SLOW] * 30),
        contact_at=range(30, 34),
        t_onset=20,
        t_hazard_end=40,
        action_chunk_len=8,
    )
    live = _analyze(trace)
    assert live.reacted_kinematic is True
    assert live.reacted is False
    assert live.behavior_class is BehaviorClass.COLLIDE_UNREACTED

    stale = {
        "reacted": True,
        "reaction_latency_steps": 1,
        "reaction_latency_chunks": 0.125,
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
        "behavior_class": "collide_reacted",
        "disturbance_triggered": True,
        "t_onset": 20,
        "t_hazard_end": 40,
        "hazard_persists": True,
        "trace_path": "traces/ep0000.jsonl",
    }
    run_dir = _write_run(tmp_path, trace, _ep(diagnostics=stale))
    before = fingerprint_results(tmp_path)
    report = recompute_results(tmp_path, CFG)
    after = fingerprint_results(tmp_path)
    assert before == after
    assert report.rewritten == 1
    assert report.skipped_no_trace == 0

    row = json.loads((run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["success"] is False
    assert row["contact_steps"] == 4
    assert row["first_contact_step"] == 30
    diag = row["diagnostics"]
    assert diag["reacted"] is False
    assert diag["reacted_kinematic"] is True
    assert diag["behavior_class"] == "collide_unreacted"
    assert diag["reaction_latency_chunks"] == pytest.approx(1.0 / 8.0)


def test_recompute_dry_run_does_not_write(tmp_path: Path) -> None:
    trace = make_trace(
        speeds=_profile([_CRUISE] * 20, [_SLOW] * 30),
        contact_at=range(30, 34),
        t_onset=20,
        t_hazard_end=40,
        action_chunk_len=8,
    )
    run_dir = _write_run(
        tmp_path,
        trace,
        _ep(diagnostics={"reacted": True, "behavior_class": "collide_reacted"}),
    )
    before = (run_dir / "episodes.jsonl").read_bytes()
    report = recompute_results(tmp_path, CFG, dry_run=True)
    assert report.dry_run is True
    assert report.rewritten == 1
    assert (run_dir / "episodes.jsonl").read_bytes() == before


def test_recompute_skips_incomplete_runs(tmp_path: Path) -> None:
    trace = make_trace(
        speeds=_profile([_CRUISE] * 20, [_SLOW] * 10),
        t_onset=20,
        t_hazard_end=28,
        action_chunk_len=1,
    )
    _write_run(tmp_path, trace, _ep(diagnostics=None), status="incomplete")
    report = recompute_results(tmp_path, CFG)
    assert report.runs == 0
    assert report.episodes == 0


def test_recompute_skips_episode_without_trace(tmp_path: Path) -> None:
    run_dir = tmp_path / "libero" / "s" / "l1" / "run"
    run_dir.mkdir(parents=True)
    rec = _ep(diagnostics=None, contact_steps=0, first_contact_step=None)
    (run_dir / "episodes.jsonl").write_text(
        json.dumps(rec.to_json_dict(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_completed(run_dir)
    report = recompute_run(run_dir, CFG)
    assert report["skipped_no_trace"] == 1
    assert report["rewritten"] == 0
    row = json.loads((run_dir / "episodes.jsonl").read_text(encoding="utf-8"))
    assert row["diagnostics"] is None


def test_recompute_loads_repo_base_yaml() -> None:
    cfg = load_baseline_diagnostics(REPO / "configs" / "base.yaml")
    assert cfg.reaction.min_latency_chunks == pytest.approx(1.0)
