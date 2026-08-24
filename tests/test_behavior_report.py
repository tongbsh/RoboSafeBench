"""Behavior-diagnostics aggregation, table_behavior, and §7.2 plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import pytest

from dynisafety.config.schema import MetricsConfig, ReactivityConfig
from dynisafety.core.aggregate import (
    AggregateOptions,
    aggregate_groups,
    behavior_group_metrics,
    episodes_to_frame,
)
from dynisafety.core.episode import SCHEMA_VERSION, EpisodeRecord
from dynisafety.diagnostics.schema import BehaviorClass
from dynisafety.report.plots import plot_behavior_classes, plot_failure_decomposition
from dynisafety.report.tables import (
    DISPLAY_NAMES,
    render_all_tables,
    table_behavior,
    table_main,
)

_DIAG_SCALAR_PREFIX_KEYS = (
    "diag_reacted",
    "diag_reaction_latency_steps",
    "diag_reaction_latency_chunks",
    "diag_reaction_post_contact",
    "diag_stopped",
    "diag_stop_preemptive",
    "diag_stop_latency_steps",
    "diag_stop_duration_steps",
    "diag_evaded",
    "diag_min_clearance_m",
    "diag_max_heading_change_deg",
    "diag_resumed",
    "diag_recovered",
    "diag_recovery_latency_steps",
    "diag_recovery_applicable",
    "diag_behavior_class",
    "diag_disturbance_triggered",
    "diag_t_onset",
    "diag_t_hazard_end",
    "diag_hazard_persists",
    "diag_trace_path",
)
_PROCESS_NA_KEYS = (
    "n_triggered",
    "reaction_rate",
    "n_reaction_latency",
    "reaction_latency_p50",
    "reaction_latency_chunks_p50",
    "stop_rate",
    "evade_rate",
    "n_recovery",
    "recovery_rate",
    "n_min_clearance",
    "min_clearance_p50",
    "hazard_persists",
)


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


def _opts() -> AggregateOptions:
    return AggregateOptions(
        group_keys=("model", "level"),
        ci_method="bootstrap",
        bootstrap_samples=8,
        bootstrap_seed=0,
        confidence=0.95,
    )


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
        "recovery_latency_steps": None,
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
        level=1,
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
        diagnostics=None,
    )
    base.update(kw)
    return EpisodeRecord(**base)


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def test_flatten_diagnostics_none_is_not_zero_or_false() -> None:
    """The most important n/a hygiene: missing diagnostics stay missing."""
    df = episodes_to_frame([_ep(diagnostics=None), _ep(episode_index=1, diagnostics=None)])
    for col in _DIAG_SCALAR_PREFIX_KEYS:
        assert col in df.columns, col
        series = df[col]
        assert series.isna().all(), col
        # Explicitly not the disguised "no reaction" sentinels
        assert not (series == 0).any(), col
        assert not (series == 0.0).any(), col
        assert not (series == False).any(), col  # noqa: E712
        for value in series:
            assert value is not False, col
            assert value != 0, col


def test_flatten_diagnostics_scalars() -> None:
    df = episodes_to_frame(
        [
            _ep(
                diagnostics=_diag(
                    reacted=True,
                    reaction_latency_steps=4,
                    reaction_latency_chunks=0.5,
                    behavior_class="evade_success",
                )
            )
        ]
    )
    assert bool(df.loc[0, "diag_reacted"]) is True
    assert int(df.loc[0, "diag_reaction_latency_steps"]) == 4
    assert df.loc[0, "diag_behavior_class"] == "evade_success"


def test_reaction_rate_denominator_is_n_triggered_not_n() -> None:
    eps = [
        _ep(
            episode_index=0,
            diagnostics=_diag(
                reacted=True,
                reaction_latency_steps=3,
                behavior_class="partial_response",
            ),
        ),
        _ep(
            episode_index=1,
            diagnostics=_diag(
                reacted=False,
                behavior_class="no_response",
            ),
        ),
        _ep(
            episode_index=2,
            success=True,
            diagnostics=_diag(
                disturbance_triggered=False,
                t_onset=None,
                t_hazard_end=None,
                hazard_persists=None,
                reacted=False,
                behavior_class="not_triggered",
            ),
        ),
    ]
    metrics = behavior_group_metrics(eps)
    assert metrics["n_triggered"] == 2
    assert metrics["reaction_rate"] == pytest.approx(0.5)
    assert metrics["reaction_rate"] != pytest.approx(1.0 / 3.0)

    runs = aggregate_groups(eps, _cfg(), _opts())
    row = runs.iloc[0]
    assert int(row["n_triggered"]) == 2
    assert float(row["reaction_rate"]) == pytest.approx(0.5)
    assert int(row["n_episodes"]) == 3


def test_recovery_rate_uses_applicable_subset() -> None:
    eps = [
        _ep(
            episode_index=0,
            diagnostics=_diag(
                stopped=True,
                stop_preemptive=True,
                recovered=True,
                recovery_applicable=True,
                behavior_class="stop_recover",
            ),
        ),
        _ep(
            episode_index=1,
            success=False,
            termination="max_steps",
            success_step=None,
            diagnostics=_diag(
                stopped=True,
                stop_preemptive=True,
                recovered=False,
                recovery_applicable=True,
                behavior_class="stop_no_recover",
            ),
        ),
        _ep(
            episode_index=2,
            # Finished after a short horizon: class counts it, rate does not.
            diagnostics=_diag(
                stopped=True,
                stop_preemptive=True,
                recovered=True,
                recovery_applicable=False,
                behavior_class="stop_recover",
            ),
        ),
        _ep(
            episode_index=3,
            diagnostics=_diag(
                evaded=True,
                reacted=True,
                recovery_applicable=False,
                behavior_class="evade_success",
            ),
        ),
    ]
    metrics = behavior_group_metrics(eps)
    assert metrics["n_recovery"] == 2
    assert metrics["recovery_rate"] == pytest.approx(0.5)
    assert int(metrics["bc_stop_recover"]) == 2  # ungated class count


def test_stop_rate_excludes_crash_stop() -> None:
    crash = _ep(
        episode_index=0,
        success=False,
        contact_steps=3,
        termination="max_steps",
        success_step=None,
        diagnostics=_diag(
            reacted=False,
            stopped=True,
            stop_preemptive=False,  # knocked to a halt, not a brake
            recovery_applicable=True,
            recovered=False,
            behavior_class="collide_unreacted",
        ),
    )
    brake = _ep(
        episode_index=1,
        diagnostics=_diag(
            reacted=True,
            stopped=True,
            stop_preemptive=True,
            recovery_applicable=True,
            recovered=True,
            behavior_class="stop_recover",
        ),
    )
    metrics = behavior_group_metrics([crash, brake])
    assert metrics["n_triggered"] == 2
    assert metrics["stop_rate"] == pytest.approx(0.5)
    crash_only = behavior_group_metrics([crash])
    assert crash_only["stop_rate"] == pytest.approx(0.0)
    assert crash_only["stop_rate"] != pytest.approx(1.0)


def test_nine_class_counts_sum_to_n() -> None:
    classes = [c.value for c in BehaviorClass]
    eps = [
        _ep(
            episode_index=i,
            success=(name in {"evade_success", "stop_recover"}),
            contact_steps=1 if name.startswith("collide") else 0,
            diagnostics=_diag(
                disturbance_triggered=name != "not_triggered",
                behavior_class=name,
                reacted=name
                in {
                    "collide_reacted",
                    "evade_success",
                    "evade_fail",
                    "stop_recover",
                    "stop_no_recover",
                    "partial_response",
                },
            ),
        )
        for i, name in enumerate(classes)
    ]
    metrics = behavior_group_metrics(eps)
    total = sum(int(metrics[f"bc_{name}"]) for name in classes)
    assert total == len(eps)
    rate_sum = sum(float(metrics[f"bc_{name}_rate"]) for name in classes)
    assert rate_sum == pytest.approx(1.0)


def test_all_none_diagnostics_group_is_na_not_zero() -> None:
    eps = [_ep(episode_index=i, diagnostics=None) for i in range(3)]
    metrics = behavior_group_metrics(eps)
    for key in _PROCESS_NA_KEYS:
        assert _is_missing(metrics[key]), key
        assert metrics[key] is not False
        assert metrics[key] != 0
        assert metrics[key] != 0.0
    for name in (c.value for c in BehaviorClass):
        assert _is_missing(metrics[f"bc_{name}"]), name
        assert _is_missing(metrics[f"bc_{name}_rate"]), name

    runs = aggregate_groups(eps, _cfg(), _opts())
    row = runs.iloc[0]
    assert _is_missing(row["reaction_rate"])
    assert _is_missing(row["n_triggered"])
    assert _is_missing(row["stop_rate"])


def test_empty_episodes_behavior_metrics_are_na() -> None:
    metrics = behavior_group_metrics([])
    for key in _PROCESS_NA_KEYS:
        assert _is_missing(metrics[key]), key


def test_empty_aggregate_groups_has_behavior_columns() -> None:
    runs = aggregate_groups([], _cfg(), _opts())
    assert runs.empty
    for key in ("n_triggered", "reaction_rate", "bc_no_response", "hazard_persists"):
        assert key in runs.columns


def test_hazard_persists_mixed_is_none() -> None:
    eps = [
        _ep(level=1, diagnostics=_diag(hazard_persists=True)),
        _ep(
            episode_index=1,
            level=3,
            diagnostics=_diag(hazard_persists=False, t_hazard_end=80),
        ),
    ]
    # Same aggregate group (model, level would split; force one group)
    metrics = behavior_group_metrics(eps)
    assert metrics["hazard_persists"] is None


def test_hazard_persists_survives_not_triggered_episodes() -> None:
    """A never-fired disturbance must not blank the flag for its group.

    ``not_triggered`` episodes carry ``hazard_persists=None``. Treating that as
    a third distinct value made an otherwise-uniform L1 group look mixed, which
    silently removed the very column that keeps L1/L2 Recovery from being pooled
    with L3 (docs/behavior_diagnostics.md §1.1).
    """
    eps = [
        _ep(episode_index=0, level=1, diagnostics=_diag(hazard_persists=True)),
        _ep(episode_index=1, level=1, diagnostics=_diag(hazard_persists=True)),
        _ep(
            episode_index=2,
            level=1,
            diagnostics=_diag(
                disturbance_triggered=False,
                t_onset=None,
                t_hazard_end=None,
                hazard_persists=None,
                behavior_class=BehaviorClass.NOT_TRIGGERED.value,
            ),
        ),
    ]
    metrics = behavior_group_metrics(eps)
    assert metrics["hazard_persists"] is True
    assert metrics["n_triggered"] == 2


def test_table_behavior_columns_and_na_render(tmp_path: Path) -> None:
    wired = [
        _ep(
            episode_index=0,
            diagnostics=_diag(
                reacted=True,
                reaction_latency_steps=5,
                reaction_latency_chunks=0.625,
                stop_preemptive=True,
                stopped=True,
                recovered=True,
                recovery_applicable=True,
                min_clearance_m=0.12,
                behavior_class="stop_recover",
            ),
        )
    ]
    runs = aggregate_groups(wired, _cfg(), _opts())
    tab = table_behavior(runs, group_cols=("model", "level"))
    for header in (
        "N_trig",
        "ReactionRate",
        "StopRate",
        "EvadeRate",
        "RecoveryRate",
        "N_recov",
        "ReactLat_p50",
        "MinClear_p50",
        "HazardPersists",
        "Collide (no reaction)",
        "Stop→recover",
        "Not triggered",
    ):
        assert header in tab.columns, header

    blank = [_ep(episode_index=i, diagnostics=None) for i in range(2)]
    blank_runs = aggregate_groups(blank, _cfg(), _opts())
    out = tmp_path / "tables"
    render_all_tables(blank_runs, out, backends_present=["libero"])
    csv_text = (out / "table_behavior.csv").read_text(encoding="utf-8")
    md_text = (out / "table_behavior.md").read_text(encoding="utf-8")
    assert "ReactionRate" in csv_text
    assert "N_trig" in csv_text
    assert "n/a" in csv_text
    assert "n/a" in md_text
    csv_df = pd.read_csv(out / "table_behavior.csv", keep_default_na=False)
    reaction = str(csv_df.loc[0, "ReactionRate"])
    assert reaction == "n/a"
    assert reaction not in {"0", "0.0", "nan"}


def test_table_main_unchanged_columns() -> None:
    eps = [_ep(diagnostics=_diag(reacted=True, behavior_class="partial_response"))]
    runs = aggregate_groups(eps, _cfg(), _opts())
    main = table_main(runs, backends_present=["libero"])
    assert "SR" in main.columns
    assert "CR" in main.columns
    assert "SASR" in main.columns
    assert "ReactionRate" not in main.columns
    assert "StopRate" not in main.columns


def test_rendered_tables_round_but_csv_keeps_precision(tmp_path: Path) -> None:
    """md / tex are read by humans; csv feeds analysis.

    A float mean like 0.30000000000000004 is noise in a paper table but real
    precision downstream, so rounding happens only at render time.
    """
    eps = [
        _ep(episode_index=i, level=1, diagnostics=_diag(evaded=i < 3, reacted=True))
        for i in range(7)
    ]
    runs = aggregate_groups(eps, _cfg(), _opts())
    out = tmp_path / "tables"
    render_all_tables(runs, out, backends_present=["libero"])

    raw = float(pd.read_csv(out / "table_behavior.csv")["EvadeRate"].iloc[0])
    assert raw == pytest.approx(3 / 7, abs=1e-12)
    assert len(repr(raw)) > 6  # full precision survives in csv

    md = (out / "table_behavior.md").read_text(encoding="utf-8")
    assert "0.4286" in md
    assert "0.42857142857142855" not in md


def test_behavior_class_display_names() -> None:
    assert DISPLAY_NAMES["bc_collide_unreacted"] == "Collide (no reaction)"
    assert DISPLAY_NAMES["bc_collide_reacted"] == "Collide (late)"
    assert DISPLAY_NAMES["bc_evade_success"] == "Evade"
    assert DISPLAY_NAMES["bc_evade_fail"] == "Evade→fail"
    assert DISPLAY_NAMES["bc_stop_recover"] == "Stop→recover"
    assert DISPLAY_NAMES["bc_stop_no_recover"] == "Stop→no recover"
    assert DISPLAY_NAMES["bc_partial_response"] == "Partial"
    assert DISPLAY_NAMES["bc_no_response"] == "No response"
    assert DISPLAY_NAMES["bc_not_triggered"] == "Not triggered"


def test_behavior_plots_write_png(tmp_path: Path) -> None:
    eps = [
        _ep(
            episode_index=0,
            level=1,
            success=True,
            contact_steps=0,
            diagnostics=_diag(behavior_class="evade_success", evaded=True, reacted=True),
        ),
        _ep(
            episode_index=1,
            level=1,
            success=False,
            contact_steps=0,
            termination="max_steps",
            success_step=None,
            diagnostics=_diag(
                behavior_class="stop_no_recover",
                stopped=True,
                stop_preemptive=True,
            ),
        ),
        _ep(
            episode_index=2,
            level=2,
            success=False,
            contact_steps=0,
            termination="max_steps",
            success_step=None,
            diagnostics=_diag(behavior_class="evade_fail", evaded=True, reacted=True),
        ),
        _ep(
            episode_index=3,
            level=2,
            success=False,
            contact_steps=0,
            termination="max_steps",
            success_step=None,
            diagnostics=_diag(behavior_class="no_response"),
        ),
        _ep(
            episode_index=4,
            level=0,
            diagnostics=_diag(
                disturbance_triggered=False,
                behavior_class="not_triggered",
                t_onset=None,
                hazard_persists=None,
            ),
        ),
        _ep(
            episode_index=5,
            level=3,
            success=False,
            contact_steps=2,
            termination="max_steps",
            success_step=None,
            diagnostics=_diag(
                hazard_persists=False,
                behavior_class="collide_unreacted",
            ),
        ),
    ]
    frame = episodes_to_frame(eps)
    classes_png = tmp_path / "behavior_classes.png"
    decomp_png = tmp_path / "failure_decomposition.png"
    plot_behavior_classes(frame, classes_png)
    plot_failure_decomposition(frame, decomp_png)
    assert classes_png.exists() and classes_png.stat().st_size > 0
    assert decomp_png.exists() and decomp_png.stat().st_size > 0
