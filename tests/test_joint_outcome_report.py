"""Tests for joint-outcome report tables, plots, and CLI regeneration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dynisafety.cli.report import (
    load_joint_report_settings,
    run_report_joint,
)
from dynisafety.report.plots import (
    plot_joint_outcome_stacked,
    plot_pareto_sr_vs,
    render_joint_plots,
)
from dynisafety.report.tables import (
    DISPLAY_NAMES,
    UNAVAILABLE,
    check_joint_table_consistency,
    render_joint_tables,
    table_base_vs_oracle,
    table_joint_outcomes,
    table_sr_ssr_cr,
)


def _four_quadrant_runs() -> pd.DataFrame:
    """Deterministic fixture covering SS/US/CFF/CIF for two models × two levels."""
    rows = [
        # model A L0: mostly SS + CFF (no contact)
        dict(
            model="model_a",
            level=0,
            task_id=0,
            n_episodes=100,
            sr=0.70,
            cr=0.0,
            ss=70,
            us=0,
            cff=30,
            cif=0,
            ssr=0.70,
            us_rate=0.0,
            cff_rate=0.30,
            cif_rate=0.0,
            sasr=0.70,
        ),
        # model A L3: all four quadrants
        dict(
            model="model_a",
            level=3,
            task_id=0,
            n_episodes=100,
            sr=0.55,
            cr=0.40,
            ss=40,
            us=15,
            cff=20,
            cif=25,
            ssr=0.40,
            us_rate=0.15,
            cff_rate=0.20,
            cif_rate=0.25,
            sasr=0.48,
        ),
        # model B L0
        dict(
            model="model_b",
            level=0,
            task_id=1,
            n_episodes=50,
            sr=0.80,
            cr=0.0,
            ss=40,
            us=0,
            cff=10,
            cif=0,
            ssr=0.80,
            us_rate=0.0,
            cff_rate=0.20,
            cif_rate=0.0,
            sasr=0.80,
        ),
        # model B L3: contact-heavy
        dict(
            model="model_b",
            level=3,
            task_id=1,
            n_episodes=50,
            sr=0.30,
            cr=0.70,
            ss=10,
            us=5,
            cff=5,
            cif=30,
            ssr=0.20,
            us_rate=0.10,
            cff_rate=0.10,
            cif_rate=0.60,
            sasr=0.22,
        ),
    ]
    return pd.DataFrame(rows)


def test_display_names_ssr_not_overwriting_sasr() -> None:
    assert DISPLAY_NAMES["sasr"] == "SASR"
    assert DISPLAY_NAMES["ssr"] == "SSR"
    assert DISPLAY_NAMES["ss"] == "SS"
    assert DISPLAY_NAMES["cff"] == "CFF"
    assert DISPLAY_NAMES["cif"] == "CIF"
    # Locked: never present forbidden aliases as display names
    assert "Safe Stop" not in DISPLAY_NAMES.values()
    assert "Collision-induced Failure" not in DISPLAY_NAMES.values()


def test_table_joint_outcomes_four_quadrants() -> None:
    df = _four_quadrant_runs()
    tab = table_joint_outcomes(df, group_cols=("model", "level"))
    assert "SS" in tab.columns
    assert "US" in tab.columns
    assert "CFF" in tab.columns
    assert "CIF" in tab.columns
    # Counts sum to N for each row
    for _, row in tab.iterrows():
        total = int(row["SS"]) + int(row["US"]) + int(row["CFF"]) + int(row["CIF"])
        assert total == int(row["N"])


def test_table_sr_ssr_cr_keeps_sasr_separate() -> None:
    df = _four_quadrant_runs()
    tab = table_sr_ssr_cr(df, group_cols=("model", "level"))
    assert list(tab.columns) == ["model", "level", "N", "SR", "SSR", "CR", "SASR"]
    # L0: SSR == SR when no contact
    row = tab[(tab["model"] == "model_a") & (tab["level"] == 0)].iloc[0]
    assert float(row["SSR"]) == pytest.approx(float(row["SR"]))
    # L3: SSR < SR when US > 0
    row3 = tab[(tab["model"] == "model_a") & (tab["level"] == 3)].iloc[0]
    assert float(row3["SSR"]) < float(row3["SR"])
    # SASR present and not equal to SSR for L3 (φ-penalized)
    assert float(row3["SASR"]) != float(row3["SSR"])


def test_table_base_vs_oracle_unavailable_without_oracle() -> None:
    df = _four_quadrant_runs()
    tab = table_base_vs_oracle(df, group_cols=("model", "level"))
    assert "Base_SR" in tab.columns
    assert "Oracle_SR" in tab.columns
    assert "Oracle_SSR" in tab.columns
    assert "Oracle_CR" in tab.columns
    # Raw None before render; after render_joint_tables → unavailable
    assert tab["Oracle_SR"].isna().all()


def test_missing_joint_cols_show_unavailable(tmp_path: Path) -> None:
    legacy = pd.DataFrame(
        [
            {"model": "old", "level": 0, "n_episodes": 10, "sr": 0.5, "cr": 0.0, "sasr": 0.5},
        ]
    )
    out = tmp_path / "joint"
    render_joint_tables(legacy, out, group_cols=("model", "level"))
    md = (out / "table_joint_outcomes.md").read_text(encoding="utf-8")
    assert UNAVAILABLE in md
    joint_csv = pd.read_csv(out / "table_joint_outcomes.csv")
    assert str(joint_csv.loc[0, "SS"]) == UNAVAILABLE
    assert str(joint_csv.loc[0, "US"]) == UNAVAILABLE
    base_ora = pd.read_csv(out / "table_base_vs_oracle.csv")
    assert str(base_ora.loc[0, "Oracle_SR"]) == UNAVAILABLE


def test_check_joint_table_consistency_ok() -> None:
    df = _four_quadrant_runs()
    assert check_joint_table_consistency(df) == []


def test_check_joint_table_consistency_detects_bad_sum() -> None:
    df = _four_quadrant_runs()
    df = df.copy()
    df.loc[0, "ss"] = 99
    issues = check_joint_table_consistency(df)
    assert any("ss+us+cff+cif" in m for m in issues)


def test_stacked_plot_and_pareto(tmp_path: Path) -> None:
    df = _four_quadrant_runs()
    stack = tmp_path / "stack.png"
    plot_joint_outcome_stacked(df, stack, group_col="model")
    assert stack.exists() and stack.stat().st_size > 0
    pareto = tmp_path / "pareto.png"
    plot_pareto_sr_vs(df, pareto, y_metric="ssr")
    assert pareto.exists() and pareto.stat().st_size > 0


def test_render_joint_plots_writes_artifacts(tmp_path: Path) -> None:
    df = _four_quadrant_runs()
    render_joint_plots(df, tmp_path)
    plots = tmp_path / "plots"
    assert (plots / "joint_stack_by_model.png").exists()
    assert (plots / "pareto_sr_ssr.png").exists()
    assert (plots / "heatmap_ssr_task_level.png").exists()


def test_cli_report_joint_from_runs(tmp_path: Path) -> None:
    runs = tmp_path / "runs.csv"
    _four_quadrant_runs().to_csv(runs, index=False)
    joint_yaml = Path(__file__).resolve().parents[1] / "configs" / "report" / "joint_outcomes.yaml"
    out = tmp_path / "out"
    ns = type(
        "Args",
        (),
        {
            "runs": runs,
            "results": None,
            "out": out,
            "joint_yaml": joint_yaml,
            "report_yaml": None,
            "metrics_yaml": None,
            "oracle_runs": None,
            "no_plots": False,
        },
    )()
    rc = run_report_joint(ns)
    assert rc == 0
    assert (out / "table_joint_outcomes.md").exists()
    assert (out / "table_sr_ssr_cr.md").exists()
    assert (out / "table_base_vs_oracle.csv").exists()
    text = (out / "table_sr_ssr_cr.md").read_text(encoding="utf-8")
    assert "SSR" in text and "SASR" in text
    header = text.splitlines()[0]
    assert "Safe Stop" not in header
    assert "Collision-induced Failure" not in header
    ora = (out / "table_base_vs_oracle.md").read_text(encoding="utf-8")
    assert UNAVAILABLE in ora
    assert (out / "plots" / "joint_stack_by_model.png").exists()


def test_load_joint_report_settings() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "report" / "joint_outcomes.yaml"
    cfg = load_joint_report_settings(path)
    assert cfg.unavailable_token == UNAVAILABLE
    assert cfg.group_by == ("model", "level")


def test_stacked_plot_unavailable_without_cols(tmp_path: Path) -> None:
    legacy = pd.DataFrame([{"model": "m", "level": 0, "n_episodes": 5, "sr": 1.0}])
    out = tmp_path / "missing.png"
    plot_joint_outcome_stacked(legacy, out)
    assert out.exists()
