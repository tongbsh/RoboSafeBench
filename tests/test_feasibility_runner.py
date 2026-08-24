"""CPU tests for feasibility runner (Prompt 14) — fixtures only, no paper numbers."""

from __future__ import annotations

from pathlib import Path

import pytest

from dynisafety.feasibility.aggregate import (
    aggregate_episodes_to_cells,
    build_base_vs_oracle_table,
    build_feasibility_matrix,
    load_episode_dicts_from_jsonl,
)
from dynisafety.feasibility.classify import (
    FeasibilityClass,
    FeasibilityThresholds,
    classify_cell,
    classify_task_level,
)
from dynisafety.feasibility.estimate import (
    estimate_matrix_resources,
    expand_matrix_jobs,
    job_to_eval_command,
    load_feasibility_matrix,
    thresholds_from_matrix,
)
from dynisafety.feasibility.report import narrative_notes, write_feasibility_artifacts

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests/fixtures/feasibility/episodes_fixture.jsonl"
MATRIX = REPO / "configs/feasibility/libero_openvla_oft_small.yaml"


@pytest.fixture
def thresholds() -> FeasibilityThresholds:
    return thresholds_from_matrix(load_feasibility_matrix(MATRIX))


def test_matrix_expands_base_and_oracle_jobs() -> None:
    data = load_feasibility_matrix(MATRIX)
    jobs = expand_matrix_jobs(data)
    # 2 tasks × 3 levels × 2 conditions × 1 seed = 12
    assert len(jobs) == 12
    assert {j.condition for j in jobs} == {"base", "oracle_swr"}
    ora = [j for j in jobs if j.condition == "oracle_swr"][0]
    cmd = job_to_eval_command(ora)
    assert "--config" in cmd
    assert "oracle.enabled=true" in cmd
    base_cmd = job_to_eval_command([j for j in jobs if j.condition == "base"][0])
    assert "oracle.enabled=false" in base_cmd


def test_resource_estimate_positive() -> None:
    data = load_feasibility_matrix(MATRIX)
    jobs = expand_matrix_jobs(data)
    est = estimate_matrix_resources(jobs, seconds_per_episode_low=45, seconds_per_episode_high=120)
    assert est.n_episodes_total == 12 * 5
    assert est.gpu_hours_high >= est.gpu_hours_low > 0


def test_oracle_failure_not_invalid_geometry(thresholds: FeasibilityThresholds) -> None:
    cls, reason = classify_cell(
        n=5,
        ssr=0.0,
        cr=0.8,
        ss=0,
        oracle_pause_steps_mean=0.0,
        stop_rate=0.0,
        resume_rate=0.0,
        geometry_invalid=False,
        oracle_failure_reason="hold_factory_crash",
        thresholds=thresholds,
        condition="oracle_swr",
    )
    assert cls is FeasibilityClass.UNRESOLVED
    assert "NOT" in reason or "not" in reason.lower()
    assert "invalid-geometry" in reason.lower() or "NOT" in reason


def test_explicit_geometry_invalid(thresholds: FeasibilityThresholds) -> None:
    cls, _ = classify_cell(
        n=5,
        ssr=0.0,
        cr=0.0,
        ss=0,
        oracle_pause_steps_mean=0.0,
        stop_rate=0.0,
        resume_rate=None,
        geometry_invalid=True,
        oracle_failure_reason="hold_factory_crash",  # must not override geometry
        thresholds=thresholds,
        condition="oracle_swr",
    )
    assert cls is FeasibilityClass.INVALID_GEOMETRY


def test_fixture_pipeline_and_classes(thresholds: FeasibilityThresholds, tmp_path: Path) -> None:
    eps = load_episode_dicts_from_jsonl([FIXTURE])
    cells = aggregate_episodes_to_cells(eps, thresholds=thresholds)
    assert cells
    compare = build_base_vs_oracle_table(cells)
    matrix = build_feasibility_matrix(cells, thresholds=thresholds)

    by = {(r["task_id"], r["level"]): r for r in matrix}
    assert by[(0, 1)]["feasibility_class"] == "safe-stop-only"
    assert by[(0, 2)]["feasibility_class"] == "safe-stop-only"
    assert by[(0, 3)]["feasibility_class"] == "safe-success-feasible"
    assert by[(1, 1)]["feasibility_class"] == "invalid-geometry"
    assert by[(1, 2)]["feasibility_class"] == "unresolved"
    assert "invalid-geometry" not in by[(1, 2)]["feasibility_reason"] or "NOT" in by[(1, 2)][
        "feasibility_reason"
    ]

    # CR should drop on fixture oracle for task0 L1
    row01 = next(r for r in compare if r["task_id"] == 0 and r["level"] == 1)
    assert row01["delta_CR"] is not None and row01["delta_CR"] < 0
    assert row01["Oracle_pause_steps_mean"] > 0
    assert row01["Oracle_policy_infer_steps_mean"] > 0

    out = write_feasibility_artifacts(
        out_dir=tmp_path,
        cells=cells,
        compare_rows=compare,
        matrix_rows=matrix,
        estimate=None,
        status="fixture",
    )
    assert out["compare"].is_file()
    assert out["matrix"].is_file()
    assert out["report"].is_file()
    assert "heatmap" in out
    text = out["report"].read_text(encoding="utf-8")
    assert "SafeVLA" in text or "static obstacle" in text.lower() or "SSR" in text


def test_classify_task_level_prefers_oracle_ss(thresholds: FeasibilityThresholds) -> None:
    base = {
        "condition": "base",
        "feasibility_class": "unresolved",
        "feasibility_reason": "x",
        "geometry_invalid": False,
        "level": 3,
    }
    ora = {
        "condition": "oracle_swr",
        "feasibility_class": "safe-success-feasible",
        "feasibility_reason": "ssr high",
        "geometry_invalid": False,
        "level": 3,
    }
    cls, reason = classify_task_level(base=base, oracle=ora, thresholds=thresholds)
    assert cls is FeasibilityClass.SAFE_SUCCESS_FEASIBLE
    assert "oracle_swr" in reason


def test_narrative_mentions_expected_story() -> None:
    notes = " ".join(narrative_notes()).lower()
    assert "cr" in notes
    assert "l1" in notes and "l2" in notes
    assert "l3" in notes
    assert "invalid-geometry" in notes
    assert "safevla" in notes or "avoidance" in notes
