"""ICLR paper-figure package. No simulator, no report generate."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import pytest

from dynisafety.config.resolve import find_workspace_root
from dynisafety.core.episode import SCHEMA_VERSION, EpisodeRecord
from dynisafety.paper_figures.cli import main as paper_main
from dynisafety.paper_figures.data import (
    base_runs,
    collapse_episodes,
    expand_paper_main,
    load_bundle,
    load_display,
    load_peers,
    load_settings,
)
from dynisafety.paper_figures.render import render_figures
from dynisafety.paper_figures.style import DOUBLE, SINGLE, TEXTWIDTH, apply_iclr_style, save_figure

_ROOT = find_workspace_root()
_SUITE = _ROOT / "configs" / "suites" / "paper_main.yaml"
_SETTINGS = _ROOT / "configs" / "paper_figures" / "settings.yaml"
_PEERS = _ROOT / "configs" / "paper_figures" / "peers.yaml"
_DISPLAY = _ROOT / "configs" / "paper_figures" / "display.yaml"


def _ep(**kw) -> EpisodeRecord:
    base = dict(
        schema_version=SCHEMA_VERSION,
        run_id="r",
        config_hash="abcdefghijkl",
        backend="libero",
        suite="safelibero_spatial",
        task_id=0,
        task_name="t0",
        instruction="i",
        model="pi05",
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
        diagnostics={
            "behavior_class": "evade_success",
            "disturbance_triggered": True,
            "reacted": True,
            "stop_preemptive": False,
            "evaded": True,
            "recovered": True,
            "recovery_applicable": True,
        },
    )
    base.update(kw)
    return EpisodeRecord(**base)


def _write_run(root: Path, ep: EpisodeRecord) -> None:
    run = (
        root
        / ep.backend
        / ep.suite
        / f"l{ep.level}"
        / f"{ep.backend}-{ep.model}-{ep.suite}-t{ep.task_id}-l{ep.level}-s{ep.seed}"
    )
    run.mkdir(parents=True, exist_ok=True)
    (run / "summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "backend": ep.backend,
                "model": ep.model,
                "suite": ep.suite,
                "level": ep.level,
                "task_id": ep.task_id,
                "seed": ep.seed,
                "n_episodes_written": 1,
            }
        ),
        encoding="utf-8",
    )
    (run / "episodes.jsonl").write_text(
        json.dumps(ep.to_json_dict()) + "\n", encoding="utf-8"
    )


def test_expand_paper_main_matches_yaml_comment() -> None:
    scale = expand_paper_main(_SUITE, workspace=_ROOT)
    by = {b.backend: b for b in scale.blocks}
    assert by["libero"].n_jobs == 896
    assert by["libero"].n_episodes == 44800
    assert by["robotwin"].n_jobs == 400
    assert by["robotwin"].n_episodes == 8000
    assert by["simplerenv"].n_jobs == 36
    assert by["simplerenv"].n_episodes == 1800
    assert scale.n_jobs == 1332
    assert scale.n_episodes == 54600
    assert scale.n_tasks == 16 + 20 + 3
    assert scale.n_backends == 3
    assert len(scale.planned) == 1332
    assert "robocasa" not in by
    assert set(by["libero"].models) == {
        "openvla_oft",
        "xvla",
        "starvla",
        "pi0",
        "pi05",
        "univla",
        "openvla_oft_rlinf",
    }


def test_collapse_never_invents_missing_level() -> None:
    rows = [
        _ep(success=True, contact_steps=0, episode_index=0),
        _ep(success=True, contact_steps=3, episode_index=1),
    ]
    from dynisafety.core.aggregate import episodes_to_frame

    df = episodes_to_frame(rows)
    out = collapse_episodes(df, ("model", "level"), confidence=0.95)
    assert len(out) == 1
    assert float(out.iloc[0]["sr"]) == pytest.approx(1.0)
    assert float(out.iloc[0]["ssr"]) == pytest.approx(0.5)
    assert int(out.iloc[0]["ss"]) == 1
    assert int(out.iloc[0]["us"]) == 1
    assert "2" not in set(out["level"].astype(str))


def test_robocasa_episodes_are_dropped_from_paper_bundle(tmp_path: Path) -> None:
    results = tmp_path / "eval"
    _write_run(results, _ep(model="pi05", backend="libero", success=True, contact_steps=0))
    _write_run(
        results,
        _ep(
            model="random_mobile",
            backend="robocasa",
            suite="robocasa_nav",
            success=True,
            contact_steps=0,
            run_id="rc",
        ),
    )
    bundle = load_bundle(
        results=results,
        suite_yaml=_SUITE,
        settings=load_settings(_SETTINGS),
        display=load_display(_DISPLAY),
        peers=load_peers(_PEERS),
        workspace=_ROOT,
        analysis_root=tmp_path / "no_analysis",
    )
    assert set(bundle.runs["backend"].astype(str)) == {"libero"}
    assert bundle.n_completed_jobs == 1


def test_oracle_arm_excluded_from_base_runs(tmp_path: Path) -> None:
    results = tmp_path / "eval"
    _write_run(results, _ep(model="pi05", success=True, contact_steps=0))
    _write_run(
        results,
        _ep(model="pi05_oracle_swr_aegis", success=True, contact_steps=0, run_id="r2"),
    )
    bundle = load_bundle(
        results=results,
        suite_yaml=_SUITE,
        settings=load_settings(_SETTINGS),
        display=load_display(_DISPLAY),
        peers=load_peers(_PEERS),
        workspace=_ROOT,
        analysis_root=tmp_path / "no_analysis",
    )
    assert set(bundle.runs["model"].astype(str)) == {"pi05", "pi05_oracle_swr_aegis"}
    assert set(base_runs(bundle)["model"].astype(str)) == {"pi05"}
    assert bundle.n_completed_jobs == 1  # oracle is not a paper_main cell


def test_iclr_figure_width_is_single_column() -> None:
    assert TEXTWIDTH == 5.50
    assert SINGLE == TEXTWIDTH
    assert DOUBLE == TEXTWIDTH


def test_save_figure_writes_pdf_and_png(tmp_path: Path) -> None:
    apply_iclr_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    written = save_figure(fig, tmp_path, "demo", formats=("pdf", "png"), dpi=120)
    stems = {p.suffix for p in written}
    assert stems == {".pdf", ".png"}
    assert all(p.is_file() and p.stat().st_size > 0 for p in written)


def test_overview_only_without_results(tmp_path: Path) -> None:
    bundle = load_bundle(
        results=None,
        suite_yaml=_SUITE,
        settings=load_settings(_SETTINGS),
        display=load_display(_DISPLAY),
        peers=load_peers(_PEERS),
        workspace=_ROOT,
        analysis_root=tmp_path / "no_analysis",
    )
    written = render_figures(bundle, tmp_path, overview_only=True)
    assert "fig01_scale_overview" in written
    assert "fig02_framework" in written
    assert "fig03_main_metrics" not in written
    assert (tmp_path / "manifest.json").is_file()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_planned_jobs"] == 1332
    assert manifest["benchmark"] == "RoboSafeBench"
    catalog = (tmp_path / "catalog.md").read_text(encoding="utf-8")
    assert "fig03_main_metrics" in catalog
    assert len(manifest["figures"]) == 17
    assert {f["fig_id"] for f in manifest["figures"]} >= {"fig09", "figA4", "figA5"}


def test_cli_refuses_out_inside_results(tmp_path: Path) -> None:
    results = tmp_path / "eval"
    results.mkdir()
    code = paper_main(
        [
            "--results",
            str(results),
            "--out",
            str(results / "nested"),
            "--suite-yaml",
            str(_SUITE),
            "--settings-yaml",
            str(_SETTINGS),
            "--peers-yaml",
            str(_PEERS),
            "--display-yaml",
            str(_DISPLAY),
            "--overview-only",
        ]
    )
    assert code != 0


def test_render_result_figures_from_tiny_eval(tmp_path: Path) -> None:
    results = tmp_path / "eval"
    for i, (success, contact, cls) in enumerate(
        (
            (True, 0, "evade_success"),
            (True, 2, "collide_reacted"),
            (False, 0, "no_response"),
            (False, 4, "collide_unreacted"),
        )
    ):
        _write_run(
            results,
            _ep(
                episode_index=i,
                run_id=f"r{i}",
                success=success,
                contact_steps=contact,
                peak_force_n=float(contact) * 3.0,
                diagnostics={
                    "behavior_class": cls,
                    "disturbance_triggered": True,
                    "reacted": cls != "no_response",
                    "stop_preemptive": False,
                    "evaded": cls == "evade_success",
                    "recovered": cls == "evade_success",
                    "recovery_applicable": True,
                },
            ),
        )
    # Same cell: discover_episode_files takes completed runs. Multiple run dirs
    # for the same cell are allowed; coverage counts the cell once.
    bundle = load_bundle(
        results=results,
        suite_yaml=_SUITE,
        settings=load_settings(_SETTINGS),
        display=load_display(_DISPLAY),
        peers=load_peers(_PEERS),
        workspace=_ROOT,
        analysis_root=tmp_path / "no_analysis",
    )
    written = render_figures(
        bundle,
        tmp_path / "figs",
        figures=(
            "fig03",
            "fig04",
            "fig05",
            "fig06",
            "fig07",
            "fig08",
            "fig09",
            "fig10",
            "figA1",
            "figA2",
            "figA3",
            "figA4",
            "figA5",
        ),
    )
    assert "fig03_main_metrics" in written
    assert "fig04_joint_stack" in written
    assert "figA4_process_single_trace" in written
    assert written["fig09_process"]
    assert written["figA5_behavior_paired"]
    pdfs = list((tmp_path / "figs").glob("*.pdf"))
    assert pdfs
    for path in pdfs:
        assert path.stat().st_size > 0


def test_weighted_collapse_recomputes_ssr_from_counts() -> None:
    from dynisafety.paper_figures.data import weighted_collapse

    runs = pd.DataFrame(
        [
            {
                "backend": "libero",
                "model": "pi05",
                "level": 1,
                "n_episodes": 10,
                "ss": 2,
                "us": 5,
                "cff": 1,
                "cif": 2,
                "sr": 0.7,
                "cr": 0.7,
                "ssr": 0.2,
            },
            {
                "backend": "libero",
                "model": "pi05",
                "level": 1,
                "n_episodes": 10,
                "ss": 4,
                "us": 1,
                "cff": 0,
                "cif": 5,
                "sr": 0.5,
                "cr": 0.6,
                "ssr": 0.4,
            },
        ]
    )
    out = weighted_collapse(runs, ("backend", "model", "level"), confidence=0.95)
    assert len(out) == 1
    assert int(out.iloc[0]["ss"]) == 6
    assert float(out.iloc[0]["ssr"]) == pytest.approx(0.3)
    assert float(out.iloc[0]["sr"]) == pytest.approx(0.6)
