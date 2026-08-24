"""A2 feasibility matrix tests — dry expand/analyze only (no GPU / no paper numbers)."""

from __future__ import annotations

from pathlib import Path

from dynisafety.feasibility.aggregate import (
    aggregate_episodes_to_cells,
    build_base_vs_oracle_table,
    build_feasibility_matrix,
)
from dynisafety.feasibility.classify import FeasibilityClass, classify_task_level
from dynisafety.feasibility.estimate import (
    expand_matrix_jobs,
    job_to_eval_command,
    load_feasibility_matrix,
    thresholds_from_matrix,
)
from dynisafety.feasibility.report import narrative_notes

REPO = Path(__file__).resolve().parents[1]
A2_MATRIX = REPO / "configs/feasibility/libero_openvla_oft_aegis_small.yaml"
PI05_MATRIX = REPO / "configs/feasibility/libero_pi05_aegis_small.yaml"
LEGACY_MATRIX = REPO / "configs/feasibility/libero_openvla_oft_small.yaml"


def test_a2_matrix_expands_three_conditions_and_l3_proof() -> None:
    data = load_feasibility_matrix(A2_MATRIX)
    jobs = expand_matrix_jobs(data)
    # L1/L2 × 3 conditions × 1 seed = 6; L3 oracle_swr × 1 = 1 → 7 jobs
    assert len(jobs) == 7
    assert {j.condition for j in jobs} == {"base", "oracle_swr", "oracle_swr_aegis"}
    l12 = [j for j in jobs if j.level in (1, 2)]
    assert len(l12) == 6
    assert all(j.episodes == 3 for j in l12)
    assert all(j.task_id == 0 for j in jobs)
    l3 = [j for j in jobs if j.level == 3]
    assert len(l3) == 1
    assert l3[0].condition == "oracle_swr"
    assert l3[0].episodes == 1
    assert l3[0].aegis_config is None


def test_commands_keep_aegis_off_for_base_and_oracle() -> None:
    data = load_feasibility_matrix(A2_MATRIX)
    jobs = expand_matrix_jobs(data)
    by_cond = {j.condition: j for j in jobs if j.level == 1}
    base_cmd = job_to_eval_command(by_cond["base"])
    ora_cmd = job_to_eval_command(by_cond["oracle_swr"])
    aeg_cmd = job_to_eval_command(by_cond["oracle_swr_aegis"])

    assert "aegis.enabled=false" in base_cmd
    assert "oracle.enabled=false" in base_cmd
    assert "aegis.enabled=false" in ora_cmd
    assert "oracle.enabled=true" in ora_cmd
    assert "aegis/enable.yaml" not in ora_cmd

    assert "aegis.enabled=true" in aeg_cmd
    assert "oracle.enabled=true" in aeg_cmd
    assert "aegis/enable.yaml" in aeg_cmd
    assert "feasibility_condition=oracle_swr_aegis" in aeg_cmd


def test_legacy_prompt14_matrix_still_expands() -> None:
    data = load_feasibility_matrix(LEGACY_MATRIX)
    jobs = expand_matrix_jobs(data)
    assert len(jobs) == 12
    assert {j.condition for j in jobs} == {"base", "oracle_swr"}
    ora = next(j for j in jobs if j.condition == "oracle_swr")
    cmd = job_to_eval_command(ora)
    assert "aegis.enabled=false" in cmd


def test_condition_inference_from_aegis_flags() -> None:
    thr = thresholds_from_matrix(load_feasibility_matrix(A2_MATRIX))
    eps = [
        {
            "backend": "libero",
            "model": "openvla_oft",
            "suite": "safelibero_spatial",
            "task_id": 0,
            "task_name": "t0",
            "level": 1,
            "success": False,
            "contact_steps": 0,
            "extra": {
                "oracle_enabled": True,
                "aegis_enabled": True,
                "oracle_pause_steps": 10,
                "aegis_intervened_steps": 4,
                "policy_infer_steps": 20,
                "oracle_resume_occurred": True,
            },
        },
        {
            "backend": "libero",
            "model": "openvla_oft",
            "suite": "safelibero_spatial",
            "task_id": 0,
            "task_name": "t0",
            "level": 1,
            "success": False,
            "contact_steps": 0,
            "extra": {
                "oracle_enabled": True,
                "aegis_enabled": True,
                "oracle_pause_steps": 12,
                "aegis_intervened_steps": 6,
                "policy_infer_steps": 18,
                "oracle_resume_occurred": True,
            },
        },
        {
            "backend": "libero",
            "model": "openvla_oft",
            "suite": "safelibero_spatial",
            "task_id": 0,
            "task_name": "t0",
            "level": 1,
            "success": False,
            "contact_steps": 0,
            "extra": {
                "oracle_enabled": True,
                "aegis_enabled": True,
                "oracle_pause_steps": 11,
                "aegis_intervened_steps": 5,
                "policy_infer_steps": 19,
                "oracle_resume_occurred": True,
            },
        },
    ]
    cells = aggregate_episodes_to_cells(eps, thresholds=thr)
    assert len(cells) == 1
    assert cells[0].condition == "oracle_swr_aegis"
    assert cells[0].aegis_intervened_steps_mean == 5.0
    assert cells[0].oracle_pause_steps_mean > 0


def test_three_way_compare_and_classify_prefers_aegis() -> None:
    thr = thresholds_from_matrix(load_feasibility_matrix(A2_MATRIX))
    cells_raw = []
    for cond, contact, pause, aeg_iv, success in (
        ("base", 3, 0, 0, False),
        ("oracle_swr", 0, 10, 0, False),
        ("oracle_swr_aegis", 0, 10, 5, True),
    ):
        for _ in range(3):
            cells_raw.append(
                {
                    "backend": "libero",
                    "model": "openvla_oft",
                    "suite": "safelibero_spatial",
                    "task_id": 0,
                    "task_name": "t0",
                    "level": 2,
                    "success": success,
                    "contact_steps": contact,
                    "extra": {
                        "feasibility_condition": cond,
                        "oracle_enabled": cond != "base",
                        "aegis_enabled": cond == "oracle_swr_aegis",
                        "oracle_pause_steps": pause,
                        "aegis_intervened_steps": aeg_iv,
                        "policy_infer_steps": 15,
                        "oracle_resume_occurred": cond != "base",
                    },
                }
            )
    cells = aggregate_episodes_to_cells(cells_raw, thresholds=thr)
    compare = build_base_vs_oracle_table(cells)
    assert len(compare) == 1
    row = compare[0]
    assert row["N_aegis"] == 3
    assert row["Aegis_intervened_steps_mean"] == 5.0
    assert row["Oracle_pause_steps_mean"] == 10.0
    assert row["delta_CR"] is not None and row["delta_CR"] < 0

    matrix = build_feasibility_matrix(cells, thresholds=thr)
    assert matrix[0]["feasibility_class"] == "safe-success-feasible"
    assert "oracle_swr_aegis" in matrix[0]["feasibility_reason"]

    cls, reason = classify_task_level(
        base=next(c.as_dict() for c in cells if c.condition == "base"),
        oracle=next(c.as_dict() for c in cells if c.condition == "oracle_swr"),
        aegis=next(c.as_dict() for c in cells if c.condition == "oracle_swr_aegis"),
        thresholds=thr,
    )
    assert cls is FeasibilityClass.SAFE_SUCCESS_FEASIBLE
    assert "oracle_swr_aegis" in reason


def test_narrative_mentions_aegis_and_pause_metrics() -> None:
    text = " ".join(narrative_notes()).lower()
    assert "aegis" in text
    assert "oracle_pause_steps" in text or "pause" in text
    assert "aegis_intervened" in text or "intervened" in text
    assert "aegis.enabled=false" in text or "closing aegis" in text


def test_pi05_matrix_same_aegis_overlays_as_oft() -> None:
    """π₀.₅ uses the same Oracle/AEGIS overlays; only model stem differs."""
    data = load_feasibility_matrix(PI05_MATRIX)
    jobs = expand_matrix_jobs(data)
    assert len(jobs) == 7
    assert {j.model for j in jobs} == {"pi05"}
    assert {j.condition for j in jobs} == {"base", "oracle_swr", "oracle_swr_aegis"}
    aeg = next(j for j in jobs if j.condition == "oracle_swr_aegis")
    cmd = job_to_eval_command(aeg)
    assert "--model pi05" in cmd
    assert "aegis/enable.yaml" in cmd
    assert "oracle.enabled=true" in cmd
    assert "aegis.enabled=true" in cmd
    ora = next(j for j in jobs if j.condition == "oracle_swr" and j.level == 1)
    ora_cmd = job_to_eval_command(ora)
    assert "aegis.enabled=false" in ora_cmd
    assert "--model pi05" in ora_cmd
