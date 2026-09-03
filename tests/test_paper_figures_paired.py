"""Paired four-axis paper figures (docs/user/eval.md §2.5). No simulator, no report generate."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import pytest

from dynisafety.analysis.paired_axes import PAIRED_CLASS_ORDER
from dynisafety.config.resolve import find_workspace_root
from dynisafety.paper_figures.data import (
    base_paired_episodes,
    load_bundle,
    load_display,
    load_peers,
    load_settings,
)
from dynisafety.paper_figures.paired import (
    paired_axis_rates,
    paired_baseline_states,
    paired_class_shares,
)
from dynisafety.paper_figures.render import render_figures

_ROOT = find_workspace_root()
_SUITE = _ROOT / "configs" / "suites" / "paper_main.yaml"
_SETTINGS = _ROOT / "configs" / "paper_figures" / "settings.yaml"
_PEERS = _ROOT / "configs" / "paper_figures" / "peers.yaml"
_DISPLAY = _ROOT / "configs" / "paper_figures" / "display.yaml"
_PAIRED_FIGURES = ("fig09", "figA5")
_UNOBSERVABLE = ("not_triggered", "baseline_unusable")


def _row(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = dict(
        run_dir="out/eval/libero/safelibero_goal/l1/run",
        run_id="run",
        backend="libero",
        model="pi05",
        suite="safelibero_goal",
        task_id=0,
        task_name="put_the_bowl_on_the_plate",
        level=1,
        seed=7,
        episode_index=0,
        control_run_id="control",
        control_success=True,
        t_onset=27,
        t_hazard_end=28.0,
        first_contact_step=None,
        contact_steps=0,
        success=True,
        action_chunk_len=5,
        n_shared_steps=81,
        own_baseline="usable",
        own_class="no_response",
        paired_class="no_response",
        threshold_m=0.0299,
        threshold_source="own",
        mu_ref=0.0115,
        reference_window="both",
        n_reference_pre=10,
        n_reference_post=10,
        stop_gate_m_per_step=0.00115,
        heading_gate_m_per_step=0.00057,
        reacted=False,
        reaction_latency_steps=None,
        reaction_latency_chunks=None,
        reaction_pre_onset_fp=False,
        stopped=False,
        stop_latency_steps=None,
        stop_pre_onset_fp=False,
        evaded=False,
        max_heading_deg=30.0,
        heading_pre_onset_max_deg=5.0,
        evade_pre_onset_fp=False,
        resumed=False,
        recovered=False,
        recovery_applicable=False,
        recovery_latency_steps=None,
    )
    base.update(kw)
    return base


def _write_paired_csv(root: Path, backend: str, rows: Iterable[Mapping[str, Any]]) -> Path:
    path = root / backend / "paired_axes" / "paired_axes_episodes.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(r) for r in rows]).to_csv(path, index=False)
    return path


def _bundle(analysis_root: Path | None):
    return load_bundle(
        results=None,
        suite_yaml=_SUITE,
        settings=load_settings(_SETTINGS),
        display=load_display(_DISPLAY),
        peers=load_peers(_PEERS),
        workspace=_ROOT,
        analysis_root=analysis_root,
    )


def _one_group(analysis_root: Path, rows: Iterable[Mapping[str, Any]]) -> pd.Series:
    _write_paired_csv(analysis_root, "libero", rows)
    rates = paired_axis_rates(base_paired_episodes(_bundle(analysis_root)))
    assert len(rates) == 1
    return rates.iloc[0]


def test_missing_paired_csv_renders_na_placeholder(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "no_such_analysis")
    assert not bundle.paired.available
    message = bundle.paired.unavailable_message()
    assert "libero/paired_axes/paired_axes_episodes.csv" in message
    assert "file not found" in message
    assert "03_analysis.sh" in message
    # No frame at all beats a frame of zeros: a 0 here would read as "no policy reacted".
    assert paired_axis_rates(base_paired_episodes(bundle)).empty
    assert paired_class_shares(base_paired_episodes(bundle)).empty
    written = render_figures(bundle, tmp_path / "figs", figures=_PAIRED_FIGURES)
    for stem in ("fig09_process", "figA5_behavior_paired"):
        assert written[stem]
        assert all(Path(p).is_file() and Path(p).stat().st_size > 0 for p in written[stem])


def test_empty_paired_csv_is_unavailable(tmp_path: Path) -> None:
    path = tmp_path / "libero" / "paired_axes" / "paired_axes_episodes.csv"
    path.parent.mkdir(parents=True)
    path.write_text("backend,model,level\n", encoding="utf-8")
    bundle = _bundle(tmp_path)
    assert not bundle.paired.available
    assert "no rows" in bundle.paired.unavailable_message()
    render_figures(bundle, tmp_path / "figs", figures=_PAIRED_FIGURES)


def test_null_axis_bit_refuses_the_file(tmp_path: Path) -> None:
    _write_paired_csv(tmp_path, "libero", [_row(reacted=None)])
    bundle = _bundle(tmp_path)
    assert not bundle.paired.available
    assert "null axis bits" in bundle.paired.unavailable_message()


def test_recovery_rate_denominator_is_recovery_applicable(tmp_path: Path) -> None:
    row = _one_group(
        tmp_path,
        [
            _row(episode_index=0, recovery_applicable=True, recovered=True),
            _row(episode_index=1, recovery_applicable=True, recovered=False),
            _row(episode_index=2, recovery_applicable=False, recovered=True),
            _row(episode_index=3, recovery_applicable=False, recovered=True),
        ],
    )
    assert int(row["n"]) == 4
    assert int(row["n_recovery_applicable"]) == 2
    assert float(row["recovery_rate"]) == pytest.approx(0.5)


def test_recovery_rate_is_na_when_nothing_is_applicable(tmp_path: Path) -> None:
    row = _one_group(tmp_path, [_row(recovery_applicable=False, recovered=False)])
    assert int(row["n_recovery_applicable"]) == 0
    assert row["recovery_rate"] is None or pd.isna(row["recovery_rate"])


def test_pre_onset_fpr_drops_null_rows_instead_of_reading_them_as_false(tmp_path: Path) -> None:
    row = _one_group(
        tmp_path,
        [
            _row(episode_index=0, reaction_pre_onset_fp=True),
            _row(episode_index=1, reaction_pre_onset_fp=False),
            _row(episode_index=2, reaction_pre_onset_fp=None, reference_window="post_only", t_onset=0),
            _row(episode_index=3, reaction_pre_onset_fp=None, reference_window="post_only", t_onset=1),
        ],
    )
    assert float(row["reaction_pre_onset_fpr"]) == pytest.approx(0.5)


def test_pre_onset_fpr_is_na_when_no_row_has_a_pre_onset_segment(tmp_path: Path) -> None:
    row = _one_group(
        tmp_path,
        [
            _row(
                episode_index=i,
                t_onset=0,
                reference_window="post_only",
                reaction_pre_onset_fp=None,
                stop_pre_onset_fp=None,
                evade_pre_onset_fp=None,
            )
            for i in range(2)
        ],
    )
    for col in ("reaction_pre_onset_fpr", "stop_pre_onset_fpr", "evade_pre_onset_fpr"):
        assert row[col] is None or pd.isna(row[col])


def test_false_strings_are_not_read_as_true(tmp_path: Path) -> None:
    row = _one_group(
        tmp_path,
        [
            _row(episode_index=0, reacted="true"),
            _row(episode_index=1, reacted="false"),
        ],
    )
    assert float(row["reaction_rate"]) == pytest.approx(0.5)


def test_paired_classes_split_by_own_baseline_and_stay_eight(tmp_path: Path) -> None:
    _write_paired_csv(
        tmp_path,
        "libero",
        [
            _row(episode_index=0, own_baseline="usable", paired_class="collide_reacted"),
            _row(episode_index=1, own_baseline="usable", paired_class="no_response"),
            _row(episode_index=2, own_baseline="unusable", own_class=None, paired_class="stop_recover"),
            _row(episode_index=3, own_baseline="unknown", own_class=None, paired_class="evade_success"),
        ],
    )
    shares = paired_class_shares(base_paired_episodes(_bundle(tmp_path)))
    assert paired_baseline_states(shares) == ("usable", "unusable", "unknown")
    assert set(shares.columns) >= {f"{c}_rate" for c in PAIRED_CLASS_ORDER}
    assert not any(c in shares.columns for c in _UNOBSERVABLE)
    assert not any(f"{c}_rate" in shares.columns for c in _UNOBSERVABLE)
    by_state = shares.set_index("own_baseline")
    assert int(by_state.loc["usable", "n"]) == 2
    assert int(by_state.loc["unusable", "n"]) == 1
    assert float(by_state.loc["unusable", "stop_recover_rate"]) == pytest.approx(1.0)


def test_paired_figures_render_from_two_backends(tmp_path: Path) -> None:
    _write_paired_csv(
        tmp_path / "analysis",
        "libero",
        [
            _row(episode_index=0, level=1, reacted=True, paired_class="collide_reacted"),
            _row(
                episode_index=1,
                level=2,
                own_baseline="unusable",
                own_class=None,
                stopped=True,
                recovered=True,
                recovery_applicable=True,
                paired_class="stop_recover",
            ),
        ],
    )
    _write_paired_csv(
        tmp_path / "analysis",
        "simplerenv",
        [
            _row(
                episode_index=0,
                backend="simplerenv",
                model="xvla_official",
                suite="bridge",
                level=3,
                own_baseline="unknown",
                own_class=None,
                t_onset=0,
                reference_window="post_only",
                reaction_pre_onset_fp=None,
                stop_pre_onset_fp=None,
                evade_pre_onset_fp=None,
                paired_class="evade_success",
                evaded=True,
            )
        ],
    )
    bundle = _bundle(tmp_path / "analysis")
    assert bundle.paired.available
    assert set(bundle.paired.episodes["backend"]) == {"libero", "simplerenv"}
    written = render_figures(bundle, tmp_path / "figs", figures=_PAIRED_FIGURES)
    for stem in ("fig09_process", "figA5_behavior_paired"):
        assert all(Path(p).stat().st_size > 0 for p in written[stem])


def test_oracle_arm_is_not_ranked_in_paired_figures(tmp_path: Path) -> None:
    _write_paired_csv(
        tmp_path,
        "libero",
        [
            _row(episode_index=0, model="pi05"),
            _row(episode_index=1, model="pi05_oracle_swr_aegis"),
        ],
    )
    bundle = _bundle(tmp_path)
    assert len(bundle.paired.episodes) == 2
    assert set(base_paired_episodes(bundle)["model"]) == {"pi05"}
