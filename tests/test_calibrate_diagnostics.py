"""Offline P5 calibration: synthetic traces only, no simulator.

The original trace files on disk must stay byte-identical after a run.
Thresholds must come from YAML (or an explicit DiagnosticsConfig), never
from literals inside the calibrator.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from dynisafety.config.schema import DiagnosticsTraceConfig
from dynisafety.diagnostics.calibrate import (
    CalibrationError,
    EpisodeBundle,
    EpisodeOutcome,
    baseline_health_row,
    compute_fpr_curve,
    compute_sensitivity_table,
    discover_bundles,
    fingerprint_results,
    inject_virtual_onset,
    load_baseline_diagnostics,
    run_calibration,
    set_knob,
    summarize_diagnostics,
)
from dynisafety.diagnostics.reader import read_trace
from dynisafety.diagnostics.schema import TRACE_SCHEMA_VERSION, Trace
from dynisafety.diagnostics.writer import TraceWriter
from test_diagnostics_analyze import _CRUISE, CFG, make_trace

REPO = Path(__file__).resolve().parents[1]


def _diag_yaml_dict(*, sensitivity: float = 2.0, sigma_rel_floor: float = 0.05) -> dict:
    """Complete diagnostics block. Values are arguments, not calibrator defaults."""
    return {
        "enabled": True,
        "trace": {"save": True, "format": "jsonl_gz", "stride": 1},
        "reaction": {
            "window_size": 10,
            "sensitivity": float(sensitivity),
            "min_persist_steps": 3,
            "heading_change_deg": 15.0,
            "sigma_rel_floor": float(sigma_rel_floor),
            "min_latency_chunks": 1,
            "min_baseline_samples": 5,
            "min_baseline_speed_frac": 0.4,
        },
        "stop": {
            "speed_frac": 0.10,
            "speed_floor_m_per_step": 0.0005,
            "min_steps": 5,
        },
        "avoid": {
            "heading_change_deg": 20.0,
            "window_steps": 30,
            "min_speed_frac": 0.05,
        },
        "recovery": {
            "speed_frac": 0.50,
            "min_steps": 5,
            "min_post_hazard_steps": 30,
        },
        "hazard_end": {"settle_quiet_steps": 10, "settle_quiet_m": 0.002},
        "counterfactual": {
            "sensitivity": 2.0,
            "null_quantile": 0.95,
            "floor_m": 0.005,
            "min_persist_steps": 3,
            "require_control_success": True,
            "action_floor": 0.05,
        },
    }


def _write_base_yaml(path: Path, *, sensitivity: float = 2.0) -> Path:
    path.write_text(
        yaml.safe_dump({"diagnostics": _diag_yaml_dict(sensitivity=sensitivity)}),
        encoding="utf-8",
    )
    return path


def _write_run(
    results_root: Path,
    trace: Trace,
    *,
    level: int,
    task_id: int = 0,
    episode_index: int = 0,
    success: bool = False,
    contact_steps: int | None = None,
    first_contact_step: int | None = None,
) -> Path:
    run_id = f"libero-pi05-safelibero_spatial-t{task_id}-l{level}-s7-synth"
    run_dir = results_root / f"l{level}" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    header = replace(
        trace.header,
        run_id=run_id,
        episode_index=int(episode_index),
        level=int(level),
        backend="libero",
        seed=7,
    )
    stored = Trace(header=header, steps=trace.steps)
    writer = TraceWriter(
        run_dir,
        DiagnosticsTraceConfig(save=True, format="jsonl", stride=1),
    )
    writer.write(int(episode_index), stored)
    hits = [int(step.t) for step in stored.steps if step.in_contact]
    row = {
        "episode_index": int(episode_index),
        "success": bool(success),
        "success_step": 30 if success else None,
        "first_contact_step": first_contact_step if first_contact_step is not None else (hits[0] if hits else None),
        "contact_steps": contact_steps if contact_steps is not None else len(hits),
        "backend": "libero",
        "model": "pi05",
        "suite": "safelibero_spatial",
        "task_id": int(task_id),
        "level": int(level),
        "seed": 7,
        "run_id": run_id,
    }
    with (run_dir / "episodes.jsonl").open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    return run_dir


def _l0_with_post_onset_bump(t_bump: int = 20) -> Trace:
    """Smooth cruise, then an 8% speed bump that a tight k catches and a loose k misses."""
    pre = [_CRUISE] * t_bump
    bump = [0.0108] * 8
    post = [_CRUISE] * 12
    return make_trace(
        speeds=pre + bump + post,
        t_onset=None,
        t_hazard_end=None,
        disturbance_triggered=False,
        hazard_persists=None,
        level=0,
    )


def _l1_triggered(*, t_onset: int = 20, truncated: bool = False) -> Trace:
    onset = 6 if truncated else t_onset
    speeds = [_CRUISE] * 40
    return make_trace(
        speeds=speeds,
        t_onset=onset,
        t_hazard_end=onset + 10,
        disturbance_triggered=True,
        hazard_persists=True,
        level=1,
    )


def test_load_baseline_diagnostics_reads_yaml_not_literals(tmp_path: Path) -> None:
    path = _write_base_yaml(tmp_path / "base.yaml", sensitivity=9.25)
    cfg = load_baseline_diagnostics(path)
    assert cfg.reaction.sensitivity == pytest.approx(9.25)
    assert cfg.reaction.sigma_rel_floor == pytest.approx(0.05)


def test_load_baseline_diagnostics_matches_repo_base_yaml() -> None:
    cfg = load_baseline_diagnostics(REPO / "configs" / "base.yaml")
    raw = yaml.safe_load((REPO / "configs" / "base.yaml").read_text(encoding="utf-8"))
    assert cfg.reaction.sensitivity == raw["diagnostics"]["reaction"]["sensitivity"]
    assert cfg.reaction.sigma_rel_floor == raw["diagnostics"]["reaction"]["sigma_rel_floor"]
    assert cfg.reaction.min_latency_chunks == raw["diagnostics"]["reaction"]["min_latency_chunks"]
    assert cfg.stop.speed_frac == raw["diagnostics"]["stop"]["speed_frac"]
    assert cfg.avoid.heading_change_deg == raw["diagnostics"]["avoid"]["heading_change_deg"]
    assert cfg.recovery.min_post_hazard_steps == raw["diagnostics"]["recovery"]["min_post_hazard_steps"]


def test_incomplete_yaml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "base.yaml"
    path.write_text("diagnostics:\n  enabled: true\n", encoding="utf-8")
    with pytest.raises(CalibrationError, match="missing diagnostics fields"):
        load_baseline_diagnostics(path)


def test_inject_virtual_onset_does_not_mutate_original() -> None:
    original = _l0_with_post_onset_bump()
    assert original.header.t_onset is None
    assert original.header.disturbance_triggered is False
    virtual = inject_virtual_onset(original, 20)
    assert virtual.header.t_onset == 20
    assert virtual.header.disturbance_triggered is True
    assert original.header.t_onset is None
    assert original.header.disturbance_triggered is False
    assert original.header.trace_schema_version == TRACE_SCHEMA_VERSION
    assert virtual.steps is original.steps


def test_set_knob_does_not_mutate_config() -> None:
    before = float(CFG.reaction.sensitivity)
    updated = set_knob(CFG, "reaction.sensitivity", 4.0)
    assert CFG.reaction.sensitivity == pytest.approx(before)
    assert updated.reaction.sensitivity == pytest.approx(4.0)


def _memory_bundle(trace: Trace, *, level: int) -> EpisodeBundle:
    header = replace(trace.header, level=int(level), episode_index=0, run_id="mem")
    stored = Trace(header=header, steps=trace.steps)
    hits = [int(step.t) for step in stored.steps if step.in_contact]
    outcome = EpisodeOutcome(
        success=False,
        success_step=None,
        first_contact_step=hits[0] if hits else None,
        contact_steps=len(hits),
        backend="libero",
        model="pi05",
        suite="safelibero_spatial",
        task_id=0,
        level=int(level),
        seed=7,
        run_id="mem",
        episode_index=0,
    )
    return EpisodeBundle(
        run_dir=Path("."),
        trace_path=Path("traces/ep0000.jsonl"),
        trace=stored,
        outcome=outcome,
    )


def test_fpr_falls_as_sensitivity_rises() -> None:
    l0 = _memory_bundle(_l0_with_post_onset_bump(), level=0)
    curve = compute_fpr_curve(
        [l0],
        onset_pool=[20],
        cfg=CFG,
        n_virtual_onsets=3,
        seed=7,
        sensitivity_values=[1.0, 2.0, 8.0],
        sigma_rel_floor_values=[0.05],
    )
    by_k = {float(row.sensitivity): float(row.fpr) for row in curve.itertuples()}
    assert by_k[1.0] == pytest.approx(1.0)
    assert by_k[2.0] == pytest.approx(0.0)
    assert by_k[8.0] == pytest.approx(0.0)
    assert by_k[1.0] > by_k[8.0]


def test_empirical_onset_pool_drops_degenerate_t0() -> None:
    from dynisafety.diagnostics.calibrate import empirical_onset_pool

    zero = _memory_bundle(_l1_triggered(t_onset=0), level=1)
    late = _memory_bundle(_l1_triggered(t_onset=20), level=1)
    assert empirical_onset_pool([zero, late], min_onset=0) == [0, 20]
    assert empirical_onset_pool([zero, late]) == [20]


def test_missing_onset_pool_raises(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _write_run(results, _l0_with_post_onset_bump(), level=0)
    with pytest.raises(CalibrationError, match="onset"):
        run_calibration(
            results=results,
            out=tmp_path / "out",
            cfg=CFG,
            n_virtual_onsets=2,
            fpr_scales=(1.0,),
            factors=(1.0,),
        )


def test_baseline_health_flags_truncated_l1_window(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _write_run(results, _l1_triggered(truncated=True), level=1)
    bundle = discover_bundles(results)[0]
    row = baseline_health_row(
        bundle,
        CFG,
        t_onset=bundle.trace.header.t_onset,
        onset_source="real",
    )
    assert row["t_onset"] == 6
    assert row["window_size"] == 10
    assert row["window_truncated"] is True
    assert row["n_baseline_steps"] < 10
    assert row["mu_pre"] is not None


def test_summarize_diagnostics_uses_none_not_zero_for_empty() -> None:
    row = summarize_diagnostics([])
    assert row["n_triggered"] == 0
    assert row["reaction_rate"] is None
    assert row["stop_rate"] is None
    assert row["evade_rate"] is None
    assert row["recovery_rate"] is None
    assert row["n_recovery"] == 0


def test_run_calibration_does_not_modify_trace_files(tmp_path: Path) -> None:
    results = tmp_path / "results"
    l0_dir = _write_run(results, _l0_with_post_onset_bump(), level=0)
    l1_dir = _write_run(results, _l1_triggered(t_onset=20), level=1)
    before = fingerprint_results(results)
    assert before

    l0_on_disk = next((l0_dir / "traces").glob("ep*.jsonl*"))
    header_before = read_trace(l0_on_disk).header
    assert header_before.t_onset is None
    assert header_before.disturbance_triggered is False
    raw_before = l0_on_disk.read_bytes()

    out = tmp_path / "calib"
    result = run_calibration(
        results=results,
        out=out,
        cfg=CFG,
        target_fpr=0.05,
        n_virtual_onsets=3,
        seed=7,
        fpr_scales=(0.5, 1.0, 1.5),
        factors=(0.5, 1.0, 1.5),
    )

    assert result.traces_unmodified is True
    assert fingerprint_results(results) == before
    assert l0_on_disk.read_bytes() == raw_before
    header_after = read_trace(l0_on_disk).header
    assert header_after.t_onset is None
    assert header_after.disturbance_triggered is False
    assert header_after == header_before

    l1_file = next((l1_dir / "traces").glob("ep*.jsonl*"))
    assert read_trace(l1_file).header.t_onset == 20

    assert (out / "fpr_curve.csv").is_file()
    assert (out / "fpr_heading.csv").is_file()
    assert (out / "fpr_curve_magnitude.csv").is_file()
    assert (out / "sensitivity.csv").is_file()
    assert (out / "baseline_health.csv").is_file()
    assert (out / "recommendation.md").is_file()
    assert (out / "plots" / "fpr_curve.png").is_file()
    assert (out / "plots" / "sensitivity.png").is_file()
    assert (out / "plots" / "fpr_heading.png").is_file()
    text = (out / "recommendation.md").read_text(encoding="utf-8")
    assert "heading" in text.lower()
    assert "sensitivity" in text
    assert "window_truncated" in text or "truncated" in text.lower()


def test_run_calibration_grid_centers_on_yaml_sensitivity(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _write_run(results, _l0_with_post_onset_bump(), level=0)
    _write_run(results, _l1_triggered(t_onset=20), level=1)
    yaml_path = _write_base_yaml(tmp_path / "base.yaml", sensitivity=6.0)
    result = run_calibration(
        results=results,
        out=tmp_path / "out",
        base_yaml=yaml_path,
        n_virtual_onsets=2,
        fpr_scales=(1.0, 1.5),
        factors=(1.0,),
    )
    ks = sorted(result.fpr_curve["sensitivity"].unique())
    assert 6.0 in set(float(x) for x in ks)
    assert 9.0 in set(float(x) for x in ks)
    assert 2.0 not in set(float(x) for x in ks)


def test_sensitivity_table_has_knob_rows() -> None:
    bundle = _memory_bundle(_l1_triggered(t_onset=20), level=1)
    table = compute_sensitivity_table([bundle], CFG, knobs=("stop.speed_frac",), factors=(0.5, 1.0, 1.5))
    assert list(table["factor"]) == [0.5, 1.0, 1.5]
    assert table["value"].tolist() == pytest.approx([0.05, 0.10, 0.15])
    assert "reaction_rate" in table.columns
    assert "bc_not_triggered" in table.columns


def test_cli_main_writes_outputs_and_leaves_traces(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _write_run(results, _l0_with_post_onset_bump(), level=0)
    _write_run(results, _l1_triggered(t_onset=20), level=1)
    yaml_path = _write_base_yaml(tmp_path / "base.yaml", sensitivity=2.0)
    out = tmp_path / "out"
    before = fingerprint_results(results)

    import importlib.util

    cli_path = REPO / "tools" / "calibrate_diagnostics.py"
    spec = importlib.util.spec_from_file_location("calibrate_diagnostics_cli", cli_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    code = cli.main(
        [
            "--results",
            str(results),
            "--out",
            str(out),
            "--base-yaml",
            str(yaml_path),
            "--n-virtual-onsets",
            "2",
            "--seed",
            "7",
        ]
    )
    assert code == 0
    assert fingerprint_results(results) == before
    assert (out / "fpr_curve.csv").is_file()


def test_cli_help_exits_zero() -> None:
    import importlib.util

    cli_path = REPO / "tools" / "calibrate_diagnostics.py"
    spec = importlib.util.spec_from_file_location("calibrate_diagnostics_cli_help", cli_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(["--help"])
    assert exc.value.code == 0
