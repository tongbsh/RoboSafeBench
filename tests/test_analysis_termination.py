"""Offline termination / CFF-timeout tables. No simulator, no backend."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import pytest

from dynisafety.analysis.cli import main as analysis_main
from dynisafety.analysis.load import BoundEpisode, fingerprint_results, load_bound_episodes
from dynisafety.analysis.termination import (
    cff_bucket,
    classify_termination,
    plot_cff_termination,
    table_termination,
    write_termination_tables,
)
from dynisafety.core.episode import SCHEMA_VERSION, EpisodeRecord
from dynisafety.diagnostics.schema import BehaviorClass


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
        "t_hazard_end": 40,
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
        level=1,
        seed=7,
        episode_index=0,
        steps=10,
        max_steps=50,
        termination="success",
        wall_time_s=1.0,
        success=True,
        success_step=5,
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


def _bound(ep: EpisodeRecord, run_dir: Path | None = None) -> BoundEpisode:
    root = run_dir or Path("/tmp/analysis-synthetic")
    return BoundEpisode(episode=ep, run_dir=root, jsonl_path=root / "episodes.jsonl")


def test_classify_success_wins_over_max_steps() -> None:
    ep = _ep(success=True, termination="max_steps", steps=50, max_steps=50)
    assert classify_termination(ep) == "success"


def test_classify_error_wins_over_max_steps() -> None:
    ep = _ep(success=False, termination="error", steps=50, max_steps=50)
    assert classify_termination(ep) == "error"


def test_classify_timeout_from_max_steps_and_from_budget() -> None:
    named = _ep(success=False, termination="max_steps", steps=40, max_steps=50)
    assert classify_termination(named) == "timeout"
    budget = _ep(success=False, termination="other", steps=50, max_steps=50)
    assert classify_termination(budget) == "timeout"


def test_cff_bucket_splits_timeout_not_triggered_remainder() -> None:
    timeout = _ep(
        success=False,
        contact_steps=0,
        termination="max_steps",
        steps=50,
        max_steps=50,
        diagnostics=_diag(behavior_class=BehaviorClass.NO_RESPONSE.value),
    )
    assert cff_bucket(timeout) == "timeout"
    idle = _ep(
        success=False,
        contact_steps=0,
        termination="success",
        steps=8,
        max_steps=50,
        diagnostics=_diag(
            behavior_class=BehaviorClass.NOT_TRIGGERED.value,
            disturbance_triggered=False,
            t_onset=None,
        ),
    )
    assert cff_bucket(idle) == "not_triggered"
    early = _ep(
        success=False,
        contact_steps=0,
        termination="success",
        steps=12,
        max_steps=50,
        diagnostics=_diag(behavior_class=BehaviorClass.STOP_NO_RECOVER.value, stopped=True),
    )
    assert cff_bucket(early) == "remainder"
    unlabeled = _ep(success=False, contact_steps=0, termination="success", steps=12)
    assert cff_bucket(unlabeled) == "unlabeled"
    not_cff = _ep(success=False, contact_steps=3, termination="max_steps", steps=50)
    assert cff_bucket(not_cff) is None


def test_table_missing_diagnostics_not_triggered_is_na() -> None:
    bound = [
        _bound(
            _ep(
                success=False,
                contact_steps=0,
                termination="max_steps",
                steps=50,
                max_steps=50,
                diagnostics=None,
            )
        )
    ]
    table = table_termination(bound)
    assert len(table) == 1
    row = table.iloc[0]
    assert int(row["n_cff"]) == 1
    assert int(row["n_cff_timeout"]) == 1
    assert pd.isna(row["n_cff_not_triggered"]) or row["n_cff_not_triggered"] is None


def test_table_not_triggered_is_counted_when_diagnostics_present() -> None:
    bound = [
        _bound(
            _ep(
                episode_index=0,
                success=False,
                contact_steps=0,
                termination="success",
                steps=8,
                diagnostics=_diag(
                    behavior_class=BehaviorClass.NOT_TRIGGERED.value,
                    disturbance_triggered=False,
                ),
            )
        ),
        _bound(
            _ep(
                episode_index=1,
                success=False,
                contact_steps=0,
                termination="max_steps",
                steps=50,
                max_steps=50,
                diagnostics=_diag(behavior_class=BehaviorClass.NO_RESPONSE.value),
            )
        ),
    ]
    table = table_termination(bound)
    row = table.iloc[0]
    assert int(row["n_cff"]) == 2
    assert int(row["n_cff_timeout"]) == 1
    assert int(row["n_cff_not_triggered"]) == 1
    assert pytest.approx(float(row["cff_timeout_share"])) == 0.5


def test_empty_input_has_headers_no_rows() -> None:
    table = table_termination([])
    assert table.empty
    assert "timeout_rate" in table.columns
    assert "n_cff_not_triggered" in table.columns


def _write_jsonl_run(root: Path, episodes: list[EpisodeRecord]) -> Path:
    run_dir = root / "l1" / "libero-m1-s-t0-l1-s7-synth"
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "episodes.jsonl"
    path.write_text(
        "".join(
            __import__("json").dumps(ep.to_json_dict(), ensure_ascii=False) + "\n"
            for ep in episodes
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        __import__("json").dumps(
            {"status": "completed", "n_episodes_written": len(episodes)}
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def test_cli_writes_out_and_does_not_touch_results(tmp_path: Path) -> None:
    results = tmp_path / "results"
    run_dir = _write_jsonl_run(
        results,
        [
            _ep(
                success=False,
                contact_steps=0,
                termination="max_steps",
                steps=50,
                max_steps=50,
                diagnostics=_diag(),
            )
        ],
    )
    before = fingerprint_results(results)
    original = (run_dir / "episodes.jsonl").read_bytes()
    out = tmp_path / "out"
    rc = analysis_main(
        ["termination", "--results", str(results), "--out", str(out)]
    )
    assert rc == 0
    assert (out / "table_termination.csv").is_file()
    assert (out / "table_termination.md").is_file()
    assert (out / "plots" / "cff_termination.png").is_file()
    assert (run_dir / "episodes.jsonl").read_bytes() == original
    assert fingerprint_results(results) == before


def test_cli_refuses_out_inside_results(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _write_jsonl_run(results, [_ep()])
    rc = analysis_main(
        [
            "termination",
            "--results",
            str(results),
            "--out",
            str(results / "nested"),
        ]
    )
    assert rc == 1


def test_skips_sweep_bookkeeping_tree(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _write_jsonl_run(results, [_ep()])
    sweep = results / "_sweep" / "job" 
    sweep.mkdir(parents=True)
    junk = _ep(model="sweep-only", run_id="sweep")
    (sweep / "episodes.jsonl").write_text(
        __import__("json").dumps(junk.to_json_dict()) + "\n", encoding="utf-8"
    )
    bound = load_bound_episodes(results)
    assert {b.episode.model for b in bound} == {"m1"}


def test_plot_empty_still_writes_file(tmp_path: Path) -> None:
    dest = tmp_path / "empty.png"
    plot_cff_termination([], dest)
    assert dest.is_file()
    assert dest.stat().st_size > 0


def test_write_termination_tables_roundtrip(tmp_path: Path) -> None:
    bound = [_bound(_ep(success=True))]
    table = write_termination_tables(bound, tmp_path)
    loaded = pd.read_csv(tmp_path / "table_termination.csv")
    assert int(loaded.iloc[0]["n_success"]) == 1
    assert int(table.iloc[0]["n_success"]) == 1
