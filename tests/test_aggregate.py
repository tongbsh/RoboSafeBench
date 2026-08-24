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
    is_force_anomaly,
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


def _write_jsonl(path: Path, episodes: list[EpisodeRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for ep in episodes:
            fp.write(json.dumps(ep.to_json_dict()) + "\n")


def test_is_force_anomaly_whitelist() -> None:
    assert is_force_anomaly(_ep(extra={})) is False
    assert is_force_anomaly(_ep(extra={"force_anomaly": True})) is True
    assert is_force_anomaly(_ep(extra={"force_anomaly": False, "other": 1})) is False


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
