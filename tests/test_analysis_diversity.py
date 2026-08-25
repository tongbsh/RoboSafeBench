"""Offline spawn/reach diversity tables. No simulator, no backend."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import pytest

from dynisafety.analysis.cli import main as analysis_main
from dynisafety.analysis.diversity import (
    FAMILY_REACH,
    FAMILY_SPAWN,
    extract_plan,
    plot_diversity_slots,
    table_diversity_acceptance,
    table_diversity_by_mode,
    table_diversity_coverage,
    table_diversity_episodes,
    table_diversity_slots,
    write_diversity,
)
from dynisafety.analysis.load import BoundEpisode, fingerprint_results
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
        level=2,
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
    root = run_dir or Path("/tmp/analysis-diversity-synth")
    return BoundEpisode(episode=ep, run_dir=root, jsonl_path=root / "episodes.jsonl")


def _spawn(
    *,
    requested: str,
    accepted: str,
    slot: str | None,
    path_style: str | None,
    fallback_reason: str | None = None,
) -> dict:
    return {
        "backend": "libero",
        "enabled": True,
        "requested_mode": requested,
        "accepted_mode": accepted,
        "slot": slot,
        "path_style": path_style,
        "fallback_reason": fallback_reason,
        "fallback_chain": [requested] if accepted == requested else [requested, accepted],
        "start": [0.0, 0.0, 0.1],
        "lateral_axis": [0.0, 1.0],
        "lateral_axis_degenerate": False,
        "min_clearance_m": 0.05,
        "linear_solve_mode": "na",
        "validated_path_sha1": "a" * 40,
        "rollback_steps": 0,
    }


def _reach(
    *,
    requested: str,
    accepted: str,
    slot: str | None,
    depth: str | None,
    fallback_reason: str | None = None,
) -> dict:
    return {
        "reach_diversity_enabled": True,
        "requested_mode": requested,
        "accepted_mode": accepted,
        "slot": slot,
        "depth": depth,
        "path_style": None if accepted == "legacy_opposite" else "straight",
        "fallback_reason": fallback_reason,
        "fallback_chain": [requested] if accepted == requested else [requested, accepted],
        "start": [0.0, 0.1, 0.2],
        "dwell": [0.0, 0.0, 0.2],
    }


def test_missing_plan_key_is_none_not_default_mode() -> None:
    ep = _ep(extra={"obstacle_id": "moka"})
    assert extract_plan(ep, "spawn_plan", FAMILY_SPAWN) is None
    assert extract_plan(ep, "l3_plan", FAMILY_REACH) is None


def test_empty_plan_object_is_ignored() -> None:
    ep = _ep(extra={"spawn_plan": {}})
    assert extract_plan(ep, "spawn_plan", FAMILY_SPAWN) is None


def test_libero_geometric_l3_plan_is_not_diversity() -> None:
    """paper_main writes this snapshot even when reach_diversity is off."""
    ep = _ep(
        level=3,
        extra={
            "l3_plan": {
                "start_mode": "opposite",
                "target_mode": "path_intercept",
                "eef_pos": [0.1, 0.2, 0.3],
                "target_pos": [0.0, 0.0, 0.2],
                "enter_height": 0.15,
            }
        },
    )
    assert extract_plan(ep, "l3_plan", FAMILY_REACH) is None
    coverage = table_diversity_coverage([_bound(ep)])
    assert int(coverage.iloc[0]["n_l3_plan"]) == 0


def test_coverage_zero_plans_are_counts_acceptance_is_empty() -> None:
    bound = [_bound(_ep(extra={}))]
    coverage = table_diversity_coverage(bound)
    assert int(coverage.iloc[0]["n"]) == 1
    assert int(coverage.iloc[0]["n_spawn_plan"]) == 0
    assert int(coverage.iloc[0]["n_l3_plan"]) == 0
    assert pytest.approx(float(coverage.iloc[0]["spawn_plan_share"])) == 0.0
    acceptance = table_diversity_acceptance(bound)
    assert acceptance.empty
    assert "legacy_share" in acceptance.columns
    assert table_diversity_by_mode(bound).empty
    assert table_diversity_episodes(bound).empty


def test_accept_fallback_legacy_and_slot_histogram() -> None:
    bound = [
        _bound(
            _ep(
                episode_index=0,
                extra={"spawn_plan": _spawn(
                    requested="left_arc",
                    accepted="left_arc",
                    slot="left",
                    path_style="arc",
                )},
            )
        ),
        _bound(
            _ep(
                episode_index=1,
                success=False,
                contact_steps=3,
                termination="max_steps",
                steps=50,
                extra={"spawn_plan": _spawn(
                    requested="right_linear",
                    accepted="center_arc",
                    slot="center",
                    path_style="arc",
                    fallback_reason="candidate_rejected",
                )},
            )
        ),
        _bound(
            _ep(
                episode_index=2,
                success=False,
                contact_steps=0,
                extra={"spawn_plan": _spawn(
                    requested="left_linear",
                    accepted="legacy_jitter",
                    slot=None,
                    path_style=None,
                    fallback_reason="all_candidates_rejected",
                )},
            )
        ),
    ]
    acc = table_diversity_acceptance(bound)
    assert len(acc) == 1
    row = acc.iloc[0]
    assert row["family"] == FAMILY_SPAWN
    assert int(row["n_planned"]) == 3
    assert int(row["n_accepted_as_requested"]) == 1
    assert int(row["n_fallback"]) == 2
    assert int(row["n_legacy"]) == 1
    assert pytest.approx(float(row["legacy_share"])) == pytest.approx(1 / 3)

    slots = table_diversity_slots(bound)
    by_slot = {str(r["slot"]): int(r["n"]) for _, r in slots.iterrows()}
    assert by_slot == {"left": 1, "center": 1, "_none": 1}


def test_mode_cross_sr_cr_ssr_and_n_a_classes() -> None:
    bound = [
        _bound(
            _ep(
                episode_index=0,
                success=True,
                contact_steps=0,
                extra={"spawn_plan": _spawn(
                    requested="left_arc",
                    accepted="left_arc",
                    slot="left",
                    path_style="arc",
                )},
            )
        ),
        _bound(
            _ep(
                episode_index=1,
                success=True,
                contact_steps=2,
                extra={"spawn_plan": _spawn(
                    requested="left_arc",
                    accepted="left_arc",
                    slot="left",
                    path_style="arc",
                )},
            )
        ),
    ]
    table = table_diversity_by_mode(bound)
    assert len(table) == 1
    row = table.iloc[0]
    assert row["accepted_mode"] == "left_arc"
    assert pytest.approx(float(row["sr"])) == 1.0
    assert pytest.approx(float(row["cr"])) == 0.5
    assert pytest.approx(float(row["ssr"])) == 0.5
    assert int(row["n_ss"]) == 1
    assert int(row["n_us"]) == 1
    assert pd.isna(row["n_class_no_response"])


def test_l3_plan_depth_and_legacy_opposite() -> None:
    bound = [
        _bound(
            _ep(
                level=3,
                extra={"l3_plan": _reach(
                    requested="left_near",
                    accepted="left_near",
                    slot="left",
                    depth="near",
                )},
            )
        ),
        _bound(
            _ep(
                level=3,
                episode_index=1,
                extra={"l3_plan": _reach(
                    requested="right_mid",
                    accepted="legacy_opposite",
                    slot=None,
                    depth=None,
                    fallback_reason="all_candidates_rejected",
                )},
            )
        ),
    ]
    acc = table_diversity_acceptance(bound)
    assert int(acc.iloc[0]["n_legacy"]) == 1
    modes = set(table_diversity_by_mode(bound)["accepted_mode"])
    assert modes == {"left_near", "legacy_opposite"}
    left = table_diversity_by_mode(bound)
    near = left.loc[left["accepted_mode"] == "left_near"].iloc[0]
    assert near["depth"] == "near"
    assert near["slot"] == "left"


def test_class_counts_when_diagnostics_present() -> None:
    bound = [
        _bound(
            _ep(
                extra={"spawn_plan": _spawn(
                    requested="left_arc",
                    accepted="left_arc",
                    slot="left",
                    path_style="arc",
                )},
                diagnostics=_diag(behavior_class=BehaviorClass.NO_RESPONSE.value),
            )
        )
    ]
    row = table_diversity_by_mode(bound).iloc[0]
    assert int(row["n_class_no_response"]) == 1
    assert int(row["n_class_evade_fail"]) == 0


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
    return run_dir


def test_cli_no_plan_still_writes_and_does_not_touch_results(tmp_path: Path) -> None:
    results = tmp_path / "results"
    run_dir = _write_jsonl_run(results, [_ep(extra={})])
    before = fingerprint_results(results)
    original = (run_dir / "episodes.jsonl").read_bytes()
    out = tmp_path / "out"
    rc = analysis_main(["diversity", "--results", str(results), "--out", str(out)])
    assert rc == 0
    assert (out / "table_diversity_coverage.csv").is_file()
    assert (out / "table_diversity_acceptance.csv").is_file()
    assert (out / "plots" / "diversity_slots.png").is_file()
    assert (out / "plots" / "diversity_legacy.png").is_file()
    coverage = pd.read_csv(out / "table_diversity_coverage.csv")
    assert int(coverage.iloc[0]["n_spawn_plan"]) == 0
    assert (run_dir / "episodes.jsonl").read_bytes() == original
    assert fingerprint_results(results) == before


def test_cli_refuses_out_inside_results(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _write_jsonl_run(results, [_ep()])
    rc = analysis_main(
        ["diversity", "--results", str(results), "--out", str(results / "nested")]
    )
    assert rc == 1


def test_write_diversity_roundtrip(tmp_path: Path) -> None:
    bound = [
        _bound(
            _ep(
                extra={"spawn_plan": _spawn(
                    requested="center_arc",
                    accepted="center_arc",
                    slot="center",
                    path_style="arc",
                )}
            )
        )
    ]
    written = write_diversity(bound, tmp_path)
    loaded = pd.read_csv(tmp_path / "table_diversity_acceptance.csv")
    assert int(loaded.iloc[0]["n_planned"]) == 1
    assert int(written["table_diversity_acceptance"].iloc[0]["n_planned"]) == 1


def test_plot_empty_still_writes_file(tmp_path: Path) -> None:
    dest = tmp_path / "empty.png"
    plot_diversity_slots([], dest)
    assert dest.is_file()
    assert dest.stat().st_size > 0
