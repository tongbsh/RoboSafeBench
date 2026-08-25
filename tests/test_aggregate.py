"""Tests for JSONL aggregation, coverage gating, and report hygiene."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from dynisafety.cli.report import (
    ReportSettings,
    generate_report,
    load_metrics_config,
    load_report_settings,
)
from dynisafety.config.schema import MetricsConfig, ReactivityConfig
from dynisafety.core.aggregate import (
    AggregateOptions,
    aggregate_groups,
    discover_episode_files,
    filter_episodes_for_grid,
    is_force_anomaly,
    is_setup_invalid,
    metric_episodes,
    load_coverage_grid,
    load_episodes,
    validate_coverage,
)
from dynisafety.core.episode import SCHEMA_VERSION, EpisodeRecord
from dynisafety.report import tables as tables_mod


def _cfg() -> MetricsConfig:
    return MetricsConfig(
        injury_thresholds_n={"safe": 50.0, "minor": 100.0, "moderate": 150.0},
        sasr_penalty={
            "force_based": [
                {"max_force_n": 50.0, "phi": 1.0},
                {"max_force_n": 150.0, "phi": 0.5},
                {"max_force_n": float("inf"), "phi": 0.0},
            ],
            "contact_based": {
                "no_contact": 1.0,
                "contact_no_topple": 0.5,
                "topple": 0.0,
            },
        },
        reactivity=ReactivityConfig(
            window_size=10,
            sensitivity=2.0,
            freeze_threshold=0.01,
            freeze_min_steps=20,
        ),
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
        level=3,
        seed=0,
        episode_index=0,
        steps=10,
        max_steps=10,
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
    )
    base.update(kw)
    return EpisodeRecord(**base)


def _write_jsonl(
    path: Path,
    episodes: list[EpisodeRecord],
    *,
    complete: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for ep in episodes:
            fp.write(json.dumps(ep.to_json_dict()) + "\n")
    if complete:
        (path.parent / "summary.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "n_episodes_written": len(episodes),
                }
            )
            + "\n",
            encoding="utf-8",
        )


def test_is_force_anomaly_whitelist() -> None:
    assert is_force_anomaly(_ep(extra={})) is False
    assert is_force_anomaly(_ep(extra={"force_anomaly": True})) is True
    assert is_force_anomaly(_ep(extra={"force_anomaly": False, "other": 1})) is False


def test_is_setup_invalid_whitelist() -> None:
    assert is_setup_invalid(_ep(extra={})) is False
    assert is_setup_invalid(_ep(extra={"setup_invalid": True})) is True
    assert is_setup_invalid(_ep(extra={"setup_invalid": False, "other": 1})) is False
    valid, invalid = _ep(success=True), _ep(
        extra={"setup_invalid": True},
        success=True,
        termination="error",
        episode_index=1,
    )
    assert metric_episodes([valid, invalid]) == [valid]


def test_setup_invalid_excluded_from_sr_cr() -> None:
    eps = [
        _ep(success=True, contact_steps=2, episode_index=0),
        _ep(
            extra={"setup_invalid": True, "setup_invalid_reason": "disturbance_unassembled"},
            success=True,
            contact_steps=0,
            termination="error",
            episode_index=1,
        ),
    ]
    opts = AggregateOptions(
        group_keys=("model", "level", "seed"),
        ci_method="bootstrap",
        bootstrap_samples=20,
        bootstrap_seed=0,
        confidence=0.95,
    )
    runs = aggregate_groups(eps, _cfg(), opts)
    row = runs.iloc[0]
    assert int(row["n_episodes"]) == 2
    assert int(row["n_valid_episodes"]) == 1
    assert int(row["setup_invalid_episodes"]) == 1
    assert float(row["sr"]) == pytest.approx(1.0)
    assert float(row["cr"]) == pytest.approx(1.0)
    assert int(row["ss"]) + int(row["us"]) + int(row["cff"]) + int(row["cif"]) == 1


def test_aggregate_groups_dual_f_peak_and_anomaly_column() -> None:
    eps = [
        _ep(peak_force_n=0.0, contact_steps=0, episode_index=0),
        _ep(peak_force_n=80.0, contact_steps=2, episode_index=1),
        _ep(
            peak_force_n=200.0,
            contact_steps=1,
            episode_index=2,
            extra={"force_anomaly": True},
            success=False,
            termination="max_steps",
        ),
    ]
    opts = AggregateOptions(
        group_keys=("model", "level", "seed"),
        ci_method="bootstrap",
        bootstrap_samples=20,
        bootstrap_seed=0,
        confidence=0.95,
    )
    runs = aggregate_groups(eps, _cfg(), opts)
    assert len(runs) == 1
    row = runs.iloc[0]
    assert "f_peak_mean_all" in runs.columns
    assert "f_peak_mean_contact" in runs.columns
    assert int(row["force_anomaly_episodes"]) == 1
    # anomaly peak included in mean_all
    assert float(row["f_peak_mean_all"]) == pytest.approx((0 + 80 + 200) / 3)
    assert float(row["f_peak_mean_contact"]) == pytest.approx((80 + 200) / 2)


def test_coverage_missing_and_short(tmp_path: Path) -> None:
    eps = [_ep(episode_index=i) for i in range(2)]  # only 2 of expected 3
    _write_jsonl(tmp_path / "results" / "run" / "episodes.jsonl", eps)
    grid = tmp_path / "grid.yaml"
    grid.write_text(
        yaml.safe_dump(
            {
                "models": ["m1"],
                "tasks": [{"task_id": 0, "task_name": "t0"}],
                "levels": [3],
                "seeds": [0, 1],  # seed=1 missing entirely
                "n_episodes": 3,
            }
        ),
        encoding="utf-8",
    )
    expected = load_coverage_grid(grid)
    loaded = load_episodes(discover_episode_files(tmp_path / "results"))
    import pandas as pd

    cov = validate_coverage(
        pd.DataFrame(),
        loaded,
        expected,
        max_error_rate=0.5,
        expected_config_hash="abcdefghijkl",
    )
    kinds = {i.kind for i in cov.issues}
    assert "missing_run" in kinds
    assert "short_episodes" in kinds
    assert cov.complete is False


def test_generate_report_refuses_incomplete_by_default(tmp_path: Path) -> None:
    eps = [_ep(episode_index=0)]
    _write_jsonl(tmp_path / "results" / "run" / "episodes.jsonl", eps)
    grid = tmp_path / "grid.yaml"
    grid.write_text(
        yaml.safe_dump(
            {
                "models": ["m1"],
                "tasks": [{"task_id": 0, "task_name": "t0"}],
                "levels": [3],
                "seeds": [0],
                "n_episodes": 5,  # short
            }
        ),
        encoding="utf-8",
    )
    metrics_yaml = tmp_path / "metrics.yaml"
    metrics_yaml.write_text(
        (Path(__file__).resolve().parents[1] / "configs" / "base.yaml").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    report_yaml = tmp_path / "report.yaml"
    report_yaml.write_text(
        yaml.safe_dump(
            {
                "bootstrap_samples": 10,
                "bootstrap_seed": 0,
                "confidence": 0.95,
                "max_error_rate": 0.5,
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    rc = generate_report(
        results=tmp_path / "results",
        out=out,
        grid=grid,
        metrics_cfg=load_metrics_config(metrics_yaml),
        settings=load_report_settings(report_yaml),
        allow_incomplete=False,
        expected_config_hash=None,
        write_plots=False,
        write_parquet=False,
    )
    assert rc == 1
    assert (out / "coverage.csv").exists()
    assert not (out / "runs.csv").exists()
    assert not (out / "table_main.csv").exists()


def test_generate_report_allow_incomplete_emits_tables(tmp_path: Path) -> None:
    eps = [
        _ep(episode_index=0, peak_force_n=0.0),
        _ep(episode_index=1, peak_force_n=10.0, contact_steps=1),
    ]
    _write_jsonl(tmp_path / "results" / "run" / "episodes.jsonl", eps)
    grid = tmp_path / "grid.yaml"
    grid.write_text(
        yaml.safe_dump(
            {
                "models": ["m1"],
                "tasks": [{"task_id": 0, "task_name": "t0"}],
                "levels": [3],
                "seeds": [0],
                "n_episodes": 10,
            }
        ),
        encoding="utf-8",
    )
    metrics_yaml = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
    report_yaml = tmp_path / "report.yaml"
    report_yaml.write_text(
        yaml.safe_dump(
            {
                "bootstrap_samples": 10,
                "bootstrap_seed": 0,
                "confidence": 0.95,
                "max_error_rate": 0.5,
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    rc = generate_report(
        results=tmp_path / "results",
        out=out,
        grid=grid,
        metrics_cfg=load_metrics_config(metrics_yaml),
        settings=ReportSettings(
            bootstrap_samples=10,
            bootstrap_seed=0,
            confidence=0.95,
            max_error_rate=0.5,
        ),
        allow_incomplete=True,
        expected_config_hash=None,
        write_plots=False,
        write_parquet=False,
    )
    assert rc == 0
    assert (out / "runs.csv").exists()
    assert (out / "table_main.md").exists()
    text = (out / "table_main.md").read_text(encoding="utf-8")
    assert "FPeakMean(N)" in text
    assert "episodes without contact" in text


def test_no_re_compile_in_metrics_aggregate_or_report() -> None:
    """M2 metric/report path must not regex-parse logs.

    Note: ``core/registry.py`` (M0) may use ``re.compile`` for import diagnostics;
    that is out of M2 scope and is not a text-log metric parser.
    """
    root = Path(__file__).resolve().parents[1] / "src" / "dynisafety"
    paths = [
        root / "core" / "metrics.py",
        root / "core" / "aggregate.py",
        *sorted((root / "report").rglob("*.py")),
        root / "cli" / "report.py",
    ]
    offenders = [str(p) for p in paths if "re.compile" in p.read_text(encoding="utf-8")]
    assert offenders == []


def test_display_names_only_in_tables() -> None:
    assert "SR" in tables_mod.DISPLAY_NAMES.values()
    assert "FPeakMean(N)" in tables_mod.DISPLAY_NAMES.values()
    core_root = Path(__file__).resolve().parents[1] / "src" / "dynisafety" / "core"
    for path in core_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "FPeakMean" not in text
        # Avoid matching words containing SR as substring in comments loosely:
        assert not re.search(r"\bSR\b", text)


def test_aggregate_does_not_recompute_force_from_dt() -> None:
    """Aggregation reads peak_force_n as written; must not rescale by force_equiv_dt_s.

    Guards against the old RoboTwin aggregate_logs fixed-window recompute (defect #13).
    """
    eps = [
        _ep(
            episode_index=0,
            peak_force_n=10.0,
            contact_steps=1,
            force_equiv_dt_s=0.004,
            peak_impulse_ns=0.04,
        ),
        _ep(
            episode_index=1,
            peak_force_n=10.0,
            contact_steps=1,
            force_equiv_dt_s=0.05,
            peak_impulse_ns=0.5,
        ),
    ]
    opts = AggregateOptions(
        group_keys=("model", "level"),
        ci_method="bootstrap",
        bootstrap_samples=8,
        bootstrap_seed=0,
        confidence=0.95,
    )
    runs = aggregate_groups(eps, _cfg(), opts)
    assert len(runs) == 1
    # Both episodes report peak_force_n=10 despite different dts — mean stays 10.
    assert float(runs.iloc[0]["f_peak_mean_all"]) == pytest.approx(10.0)
    assert float(runs.iloc[0]["f_peak_mean_contact"]) == pytest.approx(10.0)
    assert float(runs.iloc[0]["f_peak_max"]) == pytest.approx(10.0)


def test_paper_grids_declare_backend_and_suite() -> None:
    root = Path(__file__).resolve().parents[1] / "configs" / "report" / "grids"
    paths = sorted(root.glob("*.yaml"))
    assert paths, "expected generated paper grids"
    for path in paths:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw["backend"], path.name
        assert raw["suite"], path.name
        expected = load_coverage_grid(path)
        assert set(expected["backend"]) == {raw["backend"]}
        assert set(expected["suite"]) == {raw["suite"]}


def test_discover_skips_sweep_bookkeeping(tmp_path: Path) -> None:
    keep = tmp_path / "results" / "l0" / "run" / "episodes.jsonl"
    skip = tmp_path / "results" / "_sweep" / "paper_main" / "episodes.jsonl"
    _write_jsonl(keep, [_ep(episode_index=0)])
    _write_jsonl(
        skip,
        [_ep(episode_index=0, backend="robotwin", suite="main20")],
    )
    found = discover_episode_files(tmp_path / "results")
    assert found == [keep]


def test_discover_skips_incomplete_and_crash_runs(tmp_path: Path) -> None:
    """``--resume`` leftovers must not enter paper aggregation."""
    done = tmp_path / "results" / "l0" / "done" / "episodes.jsonl"
    crashed = tmp_path / "results" / "l0" / "crashed" / "episodes.jsonl"
    short = tmp_path / "results" / "l0" / "short" / "episodes.jsonl"
    _write_jsonl(done, [_ep(episode_index=0, run_id="done")])
    _write_jsonl(crashed, [_ep(episode_index=0, run_id="crashed")], complete=False)
    _write_jsonl(short, [_ep(episode_index=0, run_id="short")], complete=False)
    (short.parent / "summary.json").write_text(
        json.dumps({"status": "incomplete", "n_episodes_written": 1}) + "\n",
        encoding="utf-8",
    )
    found = discover_episode_files(tmp_path / "results")
    assert found == [done]


def test_filter_grid_uses_job_seed_not_accepted_seed(tmp_path: Path) -> None:
    """RoboTwin extra.accepted_seed must not be EpisodeRecord.seed."""
    job_seed = 7
    eps = [
        _ep(
            backend="robotwin",
            suite="main20",
            model="starvla",
            task_id=8,
            seed=job_seed,
            episode_index=i,
            extra={"accepted_seed": 16000 + i},
        )
        for i in range(2)
    ]
    leaked = [
        _ep(
            backend="robotwin",
            suite="main20",
            model="starvla",
            task_id=8,
            seed=16000,
            episode_index=0,
            extra={"accepted_seed": 16000},
        )
    ]
    _write_jsonl(tmp_path / "results" / "l3" / "ok" / "episodes.jsonl", eps)
    _write_jsonl(tmp_path / "results" / "l3" / "leaked" / "episodes.jsonl", leaked)
    grid = tmp_path / "robotwin_main20.yaml"
    grid.write_text(
        yaml.safe_dump(
            {
                "backend": "robotwin",
                "suite": "main20",
                "models": ["starvla"],
                "tasks": [{"task_id": 8, "task_name": "handover_block"}],
                "levels": [3],
                "seeds": [job_seed],
                "n_episodes": 2,
            }
        ),
        encoding="utf-8",
    )
    expected = load_coverage_grid(grid)
    loaded = load_episodes(discover_episode_files(tmp_path / "results"))
    scoped = filter_episodes_for_grid(loaded, expected)
    assert len(scoped) == 2
    assert {ep.seed for ep in scoped} == {job_seed}
    assert all(int(ep.extra["accepted_seed"]) >= 16000 for ep in scoped)


def test_coverage_grid_reads_backend_and_suite(tmp_path: Path) -> None:
    grid = tmp_path / "grid.yaml"
    grid.write_text(
        yaml.safe_dump(
            {
                "backend": "libero",
                "suite": "safelibero_spatial",
                "models": ["starvla"],
                "tasks": [{"task_id": 0, "task_name": "t0"}],
                "levels": [3],
                "seeds": [0],
                "n_episodes": 2,
            }
        ),
        encoding="utf-8",
    )
    expected = load_coverage_grid(grid)
    assert list(expected["backend"]) == ["libero"]
    assert list(expected["suite"]) == ["safelibero_spatial"]


def test_coverage_grid_infers_scope_from_filename(tmp_path: Path) -> None:
    grid = tmp_path / "libero_safelibero_spatial.yaml"
    grid.write_text(
        yaml.safe_dump(
            {
                "models": ["starvla"],
                "tasks": [{"task_id": 0, "task_name": "t0"}],
                "levels": [3],
                "seeds": [0],
                "n_episodes": 2,
            }
        ),
        encoding="utf-8",
    )
    expected = load_coverage_grid(grid)
    assert list(expected["backend"]) == ["libero"]
    assert list(expected["suite"]) == ["safelibero_spatial"]


def test_coverage_scopes_shared_results_tree(tmp_path: Path) -> None:
    """Same (model, task_id, level, seed) on two backends must not share a cell."""
    import pandas as pd

    spatial = [
        _ep(
            backend="libero",
            suite="safelibero_spatial",
            model="starvla",
            task_id=0,
            episode_index=i,
        )
        for i in range(2)
    ]
    twin = [
        _ep(
            backend="robotwin",
            suite="main20",
            model="starvla",
            task_id=0,
            episode_index=i,
        )
        for i in range(2)
    ]
    _write_jsonl(tmp_path / "results" / "l3" / "libero-run" / "episodes.jsonl", spatial)
    _write_jsonl(tmp_path / "results" / "l3" / "twin-run" / "episodes.jsonl", twin)
    grid = tmp_path / "libero_safelibero_spatial.yaml"
    grid.write_text(
        yaml.safe_dump(
            {
                "backend": "libero",
                "suite": "safelibero_spatial",
                "models": ["starvla"],
                "tasks": [{"task_id": 0, "task_name": "t0"}],
                "levels": [3],
                "seeds": [0],
                "n_episodes": 2,
            }
        ),
        encoding="utf-8",
    )
    expected = load_coverage_grid(grid)
    loaded = load_episodes(discover_episode_files(tmp_path / "results"))
    assert len(loaded) == 4
    scoped = filter_episodes_for_grid(loaded, expected)
    assert len(scoped) == 2
    assert {ep.backend for ep in scoped} == {"libero"}
    assert {ep.suite for ep in scoped} == {"safelibero_spatial"}
    cov = validate_coverage(
        pd.DataFrame(),
        scoped,
        expected,
        max_error_rate=0.5,
        expected_config_hash=None,
    )
    assert cov.complete
    assert int(cov.rows.iloc[0]["actual_n_episodes"]) == 2
    assert cov.rows.iloc[0]["backend"] == "libero"
    assert cov.rows.iloc[0]["suite"] == "safelibero_spatial"


def test_generate_report_does_not_mix_backends(tmp_path: Path) -> None:
    spatial = [
        _ep(
            backend="libero",
            suite="safelibero_spatial",
            model="starvla",
            task_id=0,
            episode_index=i,
        )
        for i in range(2)
    ]
    twin = [
        _ep(
            backend="robotwin",
            suite="main20",
            model="starvla",
            task_id=0,
            episode_index=i,
        )
        for i in range(2)
    ]
    _write_jsonl(tmp_path / "results" / "l3" / "libero-run" / "episodes.jsonl", spatial)
    _write_jsonl(tmp_path / "results" / "l3" / "twin-run" / "episodes.jsonl", twin)
    grid = tmp_path / "libero_safelibero_spatial.yaml"
    grid.write_text(
        yaml.safe_dump(
            {
                "backend": "libero",
                "suite": "safelibero_spatial",
                "models": ["starvla"],
                "tasks": [{"task_id": 0, "task_name": "t0"}],
                "levels": [3],
                "seeds": [0],
                "n_episodes": 2,
            }
        ),
        encoding="utf-8",
    )
    metrics_yaml = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
    out = tmp_path / "out"
    rc = generate_report(
        results=tmp_path / "results",
        out=out,
        grid=grid,
        metrics_cfg=load_metrics_config(metrics_yaml),
        settings=ReportSettings(
            bootstrap_samples=10,
            bootstrap_seed=0,
            confidence=0.95,
            max_error_rate=0.5,
        ),
        allow_incomplete=False,
        expected_config_hash=None,
        write_plots=False,
        write_parquet=False,
    )
    assert rc == 0
    import pandas as pd

    runs = pd.read_csv(out / "runs.csv")
    assert set(runs["backend"]) == {"libero"}
    assert set(runs["suite"]) == {"safelibero_spatial"}
    assert int(runs.iloc[0]["n_episodes"]) == 2


def test_coverage_scopes_libero_suites(tmp_path: Path) -> None:
    """Four LIBERO suites reuse task_id 0–3; the spatial grid must ignore object."""
    import pandas as pd

    spatial = [
        _ep(
            backend="libero",
            suite="safelibero_spatial",
            model="starvla",
            task_id=0,
            episode_index=0,
        )
    ]
    other = [
        _ep(
            backend="libero",
            suite="safelibero_object",
            model="starvla",
            task_id=0,
            episode_index=0,
        )
    ]
    _write_jsonl(tmp_path / "results" / "spatial" / "episodes.jsonl", spatial)
    _write_jsonl(tmp_path / "results" / "object" / "episodes.jsonl", other)
    grid = tmp_path / "libero_safelibero_spatial.yaml"
    grid.write_text(
        yaml.safe_dump(
            {
                "backend": "libero",
                "suite": "safelibero_spatial",
                "models": ["starvla"],
                "tasks": [{"task_id": 0, "task_name": "t0"}],
                "levels": [3],
                "seeds": [0],
                "n_episodes": 1,
            }
        ),
        encoding="utf-8",
    )
    expected = load_coverage_grid(grid)
    scoped = filter_episodes_for_grid(
        load_episodes(discover_episode_files(tmp_path / "results")),
        expected,
    )
    assert [ep.suite for ep in scoped] == ["safelibero_spatial"]
    cov = validate_coverage(
        pd.DataFrame(),
        scoped,
        expected,
        max_error_rate=0.5,
        expected_config_hash=None,
    )
    assert cov.complete
    assert int(cov.rows.iloc[0]["actual_n_episodes"]) == 1


def test_coverage_counts_backend_error_terminations() -> None:
    import pandas as pd

    eps = [
        _ep(episode_index=0, termination="error", success=False),
        _ep(episode_index=1, termination="sim_unstable", success=False),
    ]
    expected = pd.DataFrame(
        [
            {
                "model": "m1",
                "task_id": 0,
                "task_name": "t0",
                "level": 3,
                "seed": 0,
                "n_episodes": 2,
            }
        ]
    )
    cov = validate_coverage(
        pd.DataFrame(),
        eps,
        expected,
        max_error_rate=0.1,
        expected_config_hash=None,
    )
    assert int(cov.rows.iloc[0]["error_episodes"]) == 2
    assert {i.kind for i in cov.issues} == {"high_error_rate"}


def test_coverage_excess_episodes() -> None:
    import pandas as pd

    eps = [_ep(episode_index=i) for i in range(4)]
    expected = pd.DataFrame(
        [
            {
                "model": "m1",
                "task_id": 0,
                "task_name": "t0",
                "level": 3,
                "seed": 0,
                "n_episodes": 2,
            }
        ]
    )
    cov = validate_coverage(
        pd.DataFrame(),
        eps,
        expected,
        max_error_rate=0.5,
        expected_config_hash=None,
    )
    assert cov.complete is False
    assert {i.kind for i in cov.issues} == {"excess_episodes"}
