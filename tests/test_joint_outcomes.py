"""Unit tests for joint success×contact outcomes and SSR."""

from __future__ import annotations

import pytest

from dynisafety.config.schema import MetricsConfig, ReactivityConfig
from dynisafety.core.aggregate import AggregateOptions, aggregate_groups, summarize_run
from dynisafety.core.episode import SCHEMA_VERSION, EpisodeRecord
from dynisafety.core.joint_outcomes import (
    check_joint_outcome_consistency,
    classify_joint_outcome,
    count_joint_outcomes,
    joint_outcome_metrics,
    reconstruct_sr_cr_from_counts,
    ssr,
)
from dynisafety.core.metrics import cr, sasr, sr


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


def _quadrant_fixture() -> list[EpisodeRecord]:
    """One episode per quadrant (N=4): SS, US, CFF, CIF."""
    return [
        _ep(episode_index=0, success=True, contact_steps=0, termination="success"),
        _ep(episode_index=1, success=True, contact_steps=3, termination="success"),
        _ep(
            episode_index=2,
            success=False,
            contact_steps=0,
            termination="max_steps",
            success_step=None,
        ),
        _ep(
            episode_index=3,
            success=False,
            contact_steps=2,
            termination="max_steps",
            success_step=None,
            peak_force_n=10.0,
        ),
    ]


def test_classify_four_quadrants() -> None:
    assert classify_joint_outcome(success=True, contact_steps=0) == "SS"
    assert classify_joint_outcome(success=True, contact_steps=1) == "US"
    assert classify_joint_outcome(success=False, contact_steps=0) == "CFF"
    assert classify_joint_outcome(success=False, contact_steps=5) == "CIF"


def test_quadrant_sum_equals_n() -> None:
    eps = _quadrant_fixture()
    counts = count_joint_outcomes(eps)
    assert counts.n == 4
    assert counts.ss + counts.us + counts.cff + counts.cif == counts.n
    assert (counts.ss, counts.us, counts.cff, counts.cif) == (1, 1, 1, 1)


def test_ssr_le_sr_and_us_le_cr() -> None:
    eps = _quadrant_fixture()
    sr_v = sr(eps)
    cr_v = cr(eps)
    ssr_v = ssr(eps)
    metrics = joint_outcome_metrics(eps)
    assert ssr_v == pytest.approx(0.25)
    assert ssr_v <= sr_v
    assert float(metrics["us_rate"]) <= cr_v
    assert sr_v == pytest.approx(0.5)
    assert cr_v == pytest.approx(0.5)


def test_reconstruct_sr_cr_from_quadrants() -> None:
    eps = _quadrant_fixture()
    counts = count_joint_outcomes(eps)
    sr_hat, cr_hat = reconstruct_sr_cr_from_counts(counts)
    assert sr_hat == pytest.approx(sr(eps))
    assert cr_hat == pytest.approx(cr(eps))
    # Explicit reconstruction identities
    assert counts.ss + counts.us == sum(1 for ep in eps if ep.success)
    assert counts.us + counts.cif == sum(1 for ep in eps if ep.contact_steps > 0)


def test_check_joint_outcome_consistency_ok() -> None:
    eps = _quadrant_fixture()
    check_joint_outcome_consistency(eps, sr_value=sr(eps), cr_value=cr(eps))


def test_empty_episodes_joint_metrics() -> None:
    metrics = joint_outcome_metrics([])
    assert metrics["ss"] == 0
    assert metrics["us"] == 0
    assert metrics["cff"] == 0
    assert metrics["cif"] == 0
    assert metrics["ssr"] == 0.0
    assert ssr([]) == 0.0
    assert reconstruct_sr_cr_from_counts(count_joint_outcomes([])) == (0.0, 0.0)


def test_all_safe_success() -> None:
    eps = [_ep(episode_index=i, success=True, contact_steps=0) for i in range(5)]
    metrics = joint_outcome_metrics(eps)
    assert metrics["ss"] == 5
    assert metrics["ssr"] == pytest.approx(1.0)
    assert metrics["ssr"] == pytest.approx(sr(eps))
    assert metrics["us"] == 0
    assert cr(eps) == 0.0


def test_sasr_preserved_and_distinct_from_ssr() -> None:
    """sasr (success×φ) must remain; ssr is a separate ID."""
    eps = [
        _ep(episode_index=0, success=True, contact_steps=0, peak_force_n=0.0),
        _ep(episode_index=1, success=True, contact_steps=2, peak_force_n=80.0),
        _ep(
            episode_index=2,
            success=False,
            contact_steps=1,
            peak_force_n=200.0,
            termination="max_steps",
            success_step=None,
        ),
    ]
    # sasr force-based: (1*1 + 1*0.5 + 0*0) / 3 = 0.5
    sasr_v = sasr(eps, _cfg().sasr_penalty, use_force_based=True)
    ssr_v = ssr(eps)  # only first episode is SS → 1/3
    assert sasr_v == pytest.approx(0.5)
    assert ssr_v == pytest.approx(1.0 / 3.0)
    assert sasr_v != pytest.approx(ssr_v)

    summary = summarize_run(eps, _cfg(), use_force_based_sasr=True)
    assert "sasr" in summary and "ssr" in summary
    assert float(summary["sasr"]) == pytest.approx(0.5)
    assert float(summary["ssr"]) == pytest.approx(1.0 / 3.0)
    assert int(summary["ss"]) == 1
    assert int(summary["us"]) == 1
    assert int(summary["cff"]) == 0
    assert int(summary["cif"]) == 1


def test_aggregate_groups_emits_joint_columns() -> None:
    eps = _quadrant_fixture()
    opts = AggregateOptions(
        group_keys=("model", "level", "seed"),
        ci_method="bootstrap",
        bootstrap_samples=8,
        bootstrap_seed=0,
        confidence=0.95,
    )
    runs = aggregate_groups(eps, _cfg(), opts)
    assert len(runs) == 1
    row = runs.iloc[0]
    for col in ("ss", "us", "cff", "cif", "ssr", "us_rate", "cff_rate", "cif_rate", "sasr"):
        assert col in runs.columns
    assert int(row["ss"]) + int(row["us"]) + int(row["cff"]) + int(row["cif"]) == int(
        row["n_episodes"]
    )
    assert float(row["ssr"]) <= float(row["sr"])
    assert float(row["us_rate"]) <= float(row["cr"])
    assert float(row["ssr"]) == pytest.approx(float(row["ss"]) / float(row["n_episodes"]))
    # Marginal rates rebuild from quadrants
    n = float(row["n_episodes"])
    assert float(row["sr"]) == pytest.approx((int(row["ss"]) + int(row["us"])) / n)
    assert float(row["cr"]) == pytest.approx((int(row["us"]) + int(row["cif"])) / n)
    # sasr retained as a separate column from ssr
    assert float(row["sasr"]) == pytest.approx(0.5)
    assert float(row["ssr"]) == pytest.approx(0.25)


def test_legacy_jsonl_fields_suffice() -> None:
    """Only success + contact_steps are required to recompute joint outcomes."""
    eps = [
        _ep(success=True, contact_steps=0),
        _ep(success=False, contact_steps=0, episode_index=1, termination="max_steps"),
    ]
    # Simulate "old" row view
    derived = [
        classify_joint_outcome(success=bool(ep.success), contact_steps=int(ep.contact_steps))
        for ep in eps
    ]
    assert derived == ["SS", "CFF"]
