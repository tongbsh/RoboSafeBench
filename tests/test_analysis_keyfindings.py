"""Cross-backend key findings. Synthetic EpisodeRecord only; no simulator."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dynisafety.analysis.cli import main as analysis_main
from dynisafety.analysis.keyfindings import (
    is_preschedule_l2l3,
    scan_results_tree,
    table_cff_process,
    table_coverage,
    table_joint,
    table_outcomes,
    table_warnings,
    write_key_findings,
)
from dynisafety.analysis.load import BoundEpisode, fingerprint_results
from dynisafety.core.episode import SCHEMA_VERSION, EpisodeRecord
from dynisafety.diagnostics.calibrate import load_baseline_diagnostics
from dynisafety.diagnostics.schema import BehaviorClass

REPO = Path(__file__).resolve().parents[1]
BASE_YAML = REPO / "configs" / "base.yaml"


def _diag(**kw) -> dict:
    base = {
        "reacted": False,
        "reacted_kinematic": False,
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
        "baseline_usable": True,
        "disturbance_triggered": True,
        "t_onset": 20,
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
    root = run_dir or Path("/tmp/keyfindings-synthetic")
    return BoundEpisode(episode=ep, run_dir=root, jsonl_path=root / "episodes.jsonl")


def _cfg():
    return load_baseline_diagnostics(BASE_YAML)


def test_empty_input_headers_and_na(tmp_path: Path) -> None:
    written = write_key_findings([], tmp_path, _cfg(), results_root=tmp_path)
    assert written["table_outcomes"].empty
    assert "sr" in written["table_outcomes"].columns
    assert "n_cff_stop_no_recover" in written["table_cff_process"].columns
    md = (tmp_path / "key_findings.md").read_text(encoding="utf-8")
    assert "n/a" in md
    assert "0.0000 (0/0)" not in md


def test_outcome_denominators_exclude_setup_invalid() -> None:
    bound = [
        _bound(_ep(episode_index=0, success=True, contact_steps=0)),
        _bound(
            _ep(
                episode_index=1,
                success=False,
                contact_steps=0,
                extra={"setup_invalid": True},
            )
        ),
    ]
    row = table_outcomes(bound).iloc[0]
    assert int(row["n"]) == 2
    assert int(row["n_valid"]) == 1
    assert pytest.approx(float(row["sr"])) == 1.0
    assert int(row["n_success"]) == 1
    assert row["sr_lo"] is not None


def test_timeout_uses_all_completed_not_n_valid() -> None:
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
        _bound(
            _ep(
                episode_index=1,
                success=False,
                extra={"setup_invalid": True},
                termination="error",
            )
        ),
    ]
    row = table_outcomes(bound).iloc[0]
    assert int(row["n"]) == 2
    assert int(row["n_timeout"]) == 1
    assert pytest.approx(float(row["timeout_rate"])) == 0.5


def test_empty_valid_rates_are_na_not_zero() -> None:
    bound = [
        _bound(_ep(success=True, extra={"setup_invalid": True})),
    ]
    row = table_outcomes(bound).iloc[0]
    assert int(row["n_valid"]) == 0
    assert row["sr"] is None or pd.isna(row["sr"])
    assert row["ssr"] is None or pd.isna(row["ssr"])
    joint = table_joint(bound).iloc[0]
    assert joint["ss_share"] is None or pd.isna(joint["ss_share"])
    cff = table_cff_process(bound).iloc[0]
    assert int(cff["n_cff"]) == 0
    assert cff["cff_stop_no_recover_share"] is None or pd.isna(cff["cff_stop_no_recover_share"])


def test_cff_process_denominator_is_n_cff() -> None:
    bound = [
        _bound(
            _ep(
                episode_index=0,
                success=False,
                contact_steps=0,
                diagnostics=_diag(behavior_class=BehaviorClass.STOP_NO_RECOVER.value, stopped=True),
            )
        ),
        _bound(
            _ep(
                episode_index=1,
                success=False,
                contact_steps=0,
                diagnostics=_diag(behavior_class=BehaviorClass.NO_RESPONSE.value),
            )
        ),
        _bound(
            _ep(
                episode_index=2,
                success=True,
                contact_steps=0,
                diagnostics=_diag(behavior_class=BehaviorClass.EVADE_SUCCESS.value, evaded=True),
            )
        ),
    ]
    row = table_cff_process(bound).iloc[0]
    assert int(row["n_cff"]) == 2
    assert int(row["n_cff_stop_no_recover"]) == 1
    assert int(row["n_cff_no_response"]) == 1
    assert int(row["n_cff_evade_fail"]) == 0
    assert pytest.approx(float(row["cff_stop_no_recover_share"])) == 0.5


def test_coverage_counts_safety_layer() -> None:
    bound = [
        _bound(_ep(model="pi05")),
        _bound(_ep(model="pi05_oracle_swr_aegis", episode_index=1)),
    ]
    table = table_coverage(bound)
    assert set(table["model"]) == {"pi05", "pi05_oracle_swr_aegis"}
    oracle = table.loc[table["model"] == "pi05_oracle_swr_aegis"].iloc[0]
    assert int(oracle["n_safety_layer"]) == 1


def test_warnings_fire_on_synthetic(tmp_path: Path) -> None:
    cfg = _cfg()
    min_base = int(cfg.reaction.min_baseline_samples)
    wait = 10
    results = tmp_path / "eval" / "libero"
    run = results / "s" / "l2" / "run-pre"
    run.mkdir(parents=True)
    pre = _ep(
        backend="libero",
        level=2,
        extra={"num_steps_wait": wait},
        diagnostics=_diag(
            t_onset=wait,
            behavior_class=BehaviorClass.BASELINE_UNUSABLE.value,
            baseline_usable=False,
        ),
    )
    short = _ep(
        episode_index=1,
        backend="libero",
        level=1,
        diagnostics=_diag(
            t_onset=min_base,
            behavior_class=BehaviorClass.NOT_TRIGGERED.value,
            disturbance_triggered=False,
        ),
    )
    missing_key = _ep(
        episode_index=2,
        backend="simplerenv",
        level=1,
        diagnostics=_diag(t_onset=20),
    )
    # drop the key after construction — EpisodeRecord copies the dict
    missing_diag = dict(missing_key.diagnostics or {})
    missing_diag.pop("baseline_usable")
    missing_key = _ep(
        episode_index=2,
        backend="simplerenv",
        level=1,
        diagnostics=missing_diag,
    )
    (run / "episodes.jsonl").write_text(
        json.dumps(pre.to_json_dict(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run / "summary.json").write_text(
        json.dumps({"status": "completed", "n_episodes_written": 1}) + "\n",
        encoding="utf-8",
    )
    incomplete = results / "s" / "l3" / "run-incomplete"
    incomplete.mkdir(parents=True)
    pre_incomplete = _ep(
        episode_index=9,
        backend="libero",
        level=3,
        extra={"num_steps_wait": wait},
        diagnostics=_diag(t_onset=wait),
    )
    (incomplete / "episodes.jsonl").write_text(
        json.dumps(pre_incomplete.to_json_dict(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    bound = [
        _bound(pre, run),
        _bound(short),
        _bound(missing_key),
    ]
    tree = scan_results_tree(results)
    paired = pd.DataFrame(
        [{"n_scored_own_unusable": 3, "n_scored": 10, "source": "/tmp/libero/paired_axes/x.csv"}]
    )
    warnings = table_warnings(
        bound,
        tree,
        min_baseline_samples=min_base,
        paired_coverage=paired,
        results_root=results,
    )
    def _row(warning_id: str, backend: str | None = None) -> pd.Series:
        mask = warnings["warning_id"] == warning_id
        if backend is not None:
            mask = mask & (warnings["backend"] == backend)
        return warnings.loc[mask].iloc[0]

    assert int(_row("not_triggered", "libero")["n"]) == 1
    assert int(_row("baseline_unusable", "libero")["n"]) == 1
    assert int(_row("short_onset_structural", "libero")["n"]) == 1
    assert int(_row("libero_preschedule")["n"]) == 2
    assert int(_row("baseline_usable_key_missing", "simplerenv")["n"]) == 1
    recov = _row("paired_recovered_unusable")
    assert int(recov["n"]) == 3
    assert int(recov["n_ref"]) == 10


def test_preschedule_uses_episode_num_steps_wait() -> None:
    match = _ep(
        backend="libero",
        level=2,
        extra={"num_steps_wait": 10},
        diagnostics=_diag(t_onset=10),
    )
    randomized = _ep(
        backend="libero",
        level=2,
        extra={"num_steps_wait": 10, "onset_plan": {"bin": "approach"}},
        diagnostics=_diag(t_onset=24),
    )
    other_wait = _ep(
        backend="libero",
        level=2,
        extra={"num_steps_wait": 12},
        diagnostics=_diag(t_onset=10),
    )
    assert is_preschedule_l2l3(match) is True
    assert is_preschedule_l2l3(randomized) is False
    assert is_preschedule_l2l3(other_wait) is False


def _write_jsonl_run(root: Path, episodes: list[EpisodeRecord]) -> Path:
    run_dir = root / "l1" / "libero-m1-s-t0-l1-s7-synth"
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "episodes.jsonl"
    path.write_text(
        "".join(json.dumps(ep.to_json_dict(), ensure_ascii=False) + "\n" for ep in episodes),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"status": "completed", "n_episodes_written": len(episodes)}) + "\n",
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
                extra={"num_steps_wait": 10},
                diagnostics=_diag(),
            )
        ],
    )
    before = fingerprint_results(results)
    original = (run_dir / "episodes.jsonl").read_bytes()
    out = tmp_path / "out"
    rc = analysis_main(
        [
            "key-findings",
            "--results",
            str(results),
            "--out",
            str(out),
            "--base-yaml",
            str(BASE_YAML),
        ]
    )
    assert rc == 0
    stage = out / "key_findings"
    assert (stage / "key_findings.md").is_file()
    assert (stage / "table_outcomes.csv").is_file()
    assert (stage / "table_warnings.csv").is_file()
    assert not (out / "key_findings.md").exists()
    assert (run_dir / "episodes.jsonl").read_bytes() == original
    assert fingerprint_results(results) == before


def test_cli_refuses_out_inside_results(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _write_jsonl_run(results, [_ep()])
    rc = analysis_main(
        [
            "key-findings",
            "--results",
            str(results),
            "--out",
            str(results / "nested"),
            "--base-yaml",
            str(BASE_YAML),
        ]
    )
    assert rc == 1
