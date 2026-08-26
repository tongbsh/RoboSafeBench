"""Appendix kinematics from traces. No simulator, no backend, no report."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from dynisafety.analysis.cli import main as analysis_main
from dynisafety.analysis.kinematics import (
    heading_reversals,
    ldlj_steps,
    measure_trace,
    path_length_m,
    plot_kinematics_path_by_quadrant,
    table_kinematics,
    table_kinematics_by_quadrant,
    write_kinematics,
)
from dynisafety.analysis.load import BoundEpisode, fingerprint_results
from dynisafety.core.episode import SCHEMA_VERSION, EpisodeRecord
from test_diagnostics_analyze import make_trace


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
        steps=20,
        max_steps=50,
        termination="success",
        wall_time_s=1.0,
        success=True,
        success_step=10,
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
    root = run_dir or Path("/tmp/analysis-kinematics-synth")
    return BoundEpisode(episode=ep, run_dir=root, jsonl_path=root / "episodes.jsonl")


def test_straight_line_detour_near_one_no_reversals() -> None:
    pts = [(float(i), 0.0, 0.0) for i in range(8)]
    trace = make_trace(positions=pts, t_onset=2, t_hazard_end=6)
    m = measure_trace(trace)
    assert m.has_trace
    assert m.path_length_full_m == pytest.approx(7.0)
    assert m.net_disp_full_m == pytest.approx(7.0)
    assert m.detour_ratio_full == pytest.approx(1.0)
    assert m.n_heading_reversals_full == 0
    assert m.ldlj_steps_full is None  # constant velocity → jerk ~ 0
    assert m.path_length_post_onset_m == pytest.approx(5.0)


def test_zigzag_has_reversals_and_detour() -> None:
    pts = []
    x = 0.0
    y = 0.0
    for i in range(10):
        pts.append((x, y, 0.0))
        x += 0.1
        y = 0.2 if i % 2 == 0 else -0.2
    trace = make_trace(positions=pts, t_onset=1, t_hazard_end=8)
    m = measure_trace(trace)
    assert m.n_heading_reversals_full is not None
    assert m.n_heading_reversals_full >= 3
    assert m.detour_ratio_full is not None
    assert m.detour_ratio_full > 1.5


def test_ldlj_jerkier_trajectory_is_more_negative() -> None:
    t = np.linspace(0.0, 1.0, 24)
    smooth = np.stack([t, 0.05 * np.sin(2 * np.pi * t), np.zeros_like(t)], axis=1)
    sharp = np.stack([t, 0.05 * np.sign(np.sin(8 * np.pi * t)), np.zeros_like(t)], axis=1)
    a = ldlj_steps(smooth)
    b = ldlj_steps(sharp)
    assert a is not None and b is not None
    assert b < a


def test_stationary_is_n_a_not_zero_detour() -> None:
    pts = [(0.0, 0.0, 0.0)] * 8
    assert path_length_m(np.asarray(pts)) == pytest.approx(0.0)
    m = measure_trace(make_trace(positions=pts, t_onset=2))
    assert m.detour_ratio_full is None
    assert m.ldlj_steps_full is None
    assert heading_reversals(np.asarray(pts)) is None


def test_no_trace_medians_are_n_a() -> None:
    bound = [_bound(_ep(success=False, contact_steps=0, termination="max_steps"))]
    table = table_kinematics(bound, traces=[None])
    row = table.iloc[0]
    assert int(row["n"]) == 1
    assert int(row["n_with_trace"]) == 0
    assert pd.isna(row["median_path_length_full_m"])
    assert pd.isna(row["median_path_cff_m"])


def test_backends_are_not_pooled() -> None:
    straight = make_trace(
        positions=[(float(i), 0.0, 0.0) for i in range(6)],
        t_onset=1,
    )
    bound = [
        _bound(_ep(backend="libero", episode_index=0)),
        _bound(_ep(backend="robotwin", episode_index=1, model="m1")),
    ]
    table = table_kinematics(bound, traces=[straight, straight])
    assert set(table["backend"]) == {"libero", "robotwin"}
    assert len(table) == 2


def test_cff_vs_ss_split() -> None:
    long_path = make_trace(
        positions=[(0.05 * i, 0.04 * ((-1) ** i), 0.0) for i in range(12)],
        t_onset=2,
    )
    short_path = make_trace(
        positions=[(0.02 * i, 0.0, 0.0) for i in range(12)],
        t_onset=2,
    )
    bound = [
        _bound(
            _ep(
                episode_index=0,
                success=False,
                contact_steps=0,
                termination="max_steps",
                steps=50,
                max_steps=50,
            )
        ),
        _bound(_ep(episode_index=1, success=True, contact_steps=0)),
    ]
    table = table_kinematics(bound, traces=[long_path, short_path])
    row = table.iloc[0]
    assert int(row["n_cff_traced"]) == 1
    assert int(row["n_ss_traced"]) == 1
    assert float(row["median_path_cff_m"]) > float(row["median_path_ss_m"])
    by_q = table_kinematics_by_quadrant(bound, traces=[long_path, short_path])
    assert set(by_q["quadrant"]) == {"CFF", "SS"}


def test_not_triggered_has_no_post_onset() -> None:
    pts = [(float(i), 0.0, 0.0) for i in range(8)]
    trace = make_trace(
        positions=pts,
        t_onset=None,
        t_hazard_end=None,
        disturbance_triggered=False,
    )
    m = measure_trace(trace)
    assert m.path_length_full_m == pytest.approx(7.0)
    assert m.path_length_post_onset_m is None
    assert m.window_onset_t is None


def _write_jsonl_run(root: Path, episodes: list[EpisodeRecord]) -> Path:
    run_dir = root / "l2" / "libero-m1-s-t0-l2-s7-synth"
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


def test_cli_no_trace_writes_n_a_and_does_not_touch_results(tmp_path: Path) -> None:
    results = tmp_path / "results"
    run_dir = _write_jsonl_run(results, [_ep()])
    before = fingerprint_results(results)
    original = (run_dir / "episodes.jsonl").read_bytes()
    out = tmp_path / "out"
    rc = analysis_main(["kinematics", "--results", str(results), "--out", str(out)])
    assert rc == 0
    assert (out / "table_kinematics.csv").is_file()
    assert (out / "plots" / "kinematics_path_by_quadrant.png").is_file()
    loaded = pd.read_csv(out / "table_kinematics.csv")
    assert int(loaded.iloc[0]["n_with_trace"]) == 0
    assert (run_dir / "episodes.jsonl").read_bytes() == original
    assert fingerprint_results(results) == before


def test_cli_refuses_out_inside_results(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _write_jsonl_run(results, [_ep()])
    rc = analysis_main(
        ["kinematics", "--results", str(results), "--out", str(results / "nested")]
    )
    assert rc == 1


def test_write_kinematics_roundtrip(tmp_path: Path) -> None:
    trace = make_trace(positions=[(float(i), 0.0, 0.0) for i in range(8)], t_onset=2)
    bound = [_bound(_ep())]
    written = write_kinematics(bound, tmp_path, traces=[trace])
    loaded = pd.read_csv(tmp_path / "table_kinematics.csv")
    assert int(loaded.iloc[0]["n_with_trace"]) == 1
    assert int(written["table_kinematics"].iloc[0]["n_with_trace"]) == 1
    assert (tmp_path / "plots" / "kinematics_detour_cff_vs_ss.png").is_file()


def test_plot_empty_still_writes_file(tmp_path: Path) -> None:
    dest = tmp_path / "empty.png"
    plot_kinematics_path_by_quadrant([], dest)
    assert dest.is_file()
    assert dest.stat().st_size > 0
