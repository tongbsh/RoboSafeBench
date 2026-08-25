"""Rank-stability appendix. No simulator, no backend, no report."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from dynisafety.analysis.cli import main as analysis_main
from dynisafety.analysis.load import BoundEpisode, fingerprint_results
from dynisafety.analysis.rank import (
    average_ranks,
    cells_from_runs_csv,
    kendall_tau,
    rank_models,
    score_cells,
    spearman_rho,
    table_rank_backend_pairs,
    table_rank_coverage,
    table_rank_holdout,
    table_rank_seed_pairs,
    write_rank,
)
from dynisafety.core.episode import SCHEMA_VERSION, EpisodeRecord


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
        model="m_good",
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


def _bound(ep: EpisodeRecord) -> BoundEpisode:
    root = Path("/tmp/analysis-rank-synth")
    return BoundEpisode(episode=ep, run_dir=root, jsonl_path=root / "episodes.jsonl")


def test_spearman_and_kendall_identity_and_reverse() -> None:
    a = [0.9, 0.5, 0.1]
    b = [0.8, 0.4, 0.0]
    assert spearman_rho(a, b, higher_is_better=True) == pytest.approx(1.0)
    assert kendall_tau(a, b, higher_is_better=True) == pytest.approx(1.0)
    assert spearman_rho(a, list(reversed(a)), higher_is_better=True) == pytest.approx(-1.0)
    assert kendall_tau(a, list(reversed(a)), higher_is_better=True) == pytest.approx(-1.0)


def test_average_ranks_ties_and_cr_orientation() -> None:
    assert average_ranks([0.5, 0.5, 0.1], higher_is_better=True) == [1.5, 1.5, 3.0]
    # lower CR is better → 0.1 ranks first
    assert average_ranks([0.4, 0.1, 0.9], higher_is_better=False) == [2.0, 1.0, 3.0]


def _grid(models_ssr: dict[str, dict[str, float]], *, backend: str, seed: int = 7) -> list[BoundEpisode]:
    """Each model gets 10 episodes; SSR ≈ given rate via success/no-contact."""
    out: list[BoundEpisode] = []
    idx = 0
    for model, by_backend in models_ssr.items():
        rate = by_backend[backend]
        n_ok = int(round(10 * rate))
        for i in range(10):
            out.append(
                _bound(
                    _ep(
                        backend=backend,
                        model=model,
                        seed=seed,
                        episode_index=idx,
                        success=i < n_ok,
                        contact_steps=0,
                        termination="success" if i < n_ok else "max_steps",
                    )
                )
            )
            idx += 1
    return out


def test_same_order_across_backends_spearman_one() -> None:
    spec = {
        "m_good": {"libero": 0.9, "robotwin": 0.8},
        "m_mid": {"libero": 0.5, "robotwin": 0.4},
        "m_bad": {"libero": 0.1, "robotwin": 0.0},
    }
    bound = _grid(spec, backend="libero") + _grid(spec, backend="robotwin")
    cells = score_cells(
        bound,
        group_cols=("backend", "model", "level", "seed", "suite"),
        metrics=("ssr",),
    )
    pairs = table_rank_backend_pairs(cells)
    row = pairs.iloc[0]
    assert {row["backend_a"], row["backend_b"]} == {"libero", "robotwin"}
    assert int(row["n_shared_models"]) == 3
    assert float(row["spearman"]) == pytest.approx(1.0)


def test_reversed_backends_spearman_minus_one() -> None:
    spec = {
        "m_good": {"libero": 0.9, "robotwin": 0.1},
        "m_mid": {"libero": 0.5, "robotwin": 0.5},
        "m_bad": {"libero": 0.1, "robotwin": 0.9},
    }
    bound = _grid(spec, backend="libero") + _grid(spec, backend="robotwin")
    cells = score_cells(
        bound,
        group_cols=("backend", "model", "level", "seed", "suite"),
        metrics=("ssr",),
    )
    row = table_rank_backend_pairs(cells).iloc[0]
    assert float(row["spearman"]) == pytest.approx(-1.0)


def test_single_model_correlation_is_n_a() -> None:
    bound = _grid({"only": {"libero": 0.5, "robotwin": 0.6}}, backend="libero")
    bound += _grid({"only": {"libero": 0.5, "robotwin": 0.6}}, backend="robotwin")
    cells = score_cells(
        bound,
        group_cols=("backend", "model", "level", "seed", "suite"),
        metrics=("ssr",),
    )
    row = table_rank_backend_pairs(cells).iloc[0]
    assert int(row["n_shared_models"]) == 1
    assert pd.isna(row["spearman"])
    cov = table_rank_coverage(cells).iloc[0]
    assert int(cov["n_models"]) == 1
    assert "fewer than 2 models" in str(cov["note"])


def test_single_seed_pair_table_empty() -> None:
    spec = {
        "m_good": {"libero": 0.9},
        "m_bad": {"libero": 0.1},
    }
    bound = _grid(spec, backend="libero", seed=7)
    cells = score_cells(
        bound,
        group_cols=("backend", "model", "level", "seed", "suite"),
        metrics=("ssr",),
    )
    seeds = table_rank_seed_pairs(cells)
    assert seeds.empty
    assert "spearman" in seeds.columns
    cov = table_rank_coverage(cells).iloc[0]
    assert int(cov["n_seeds"]) == 1


def test_two_seeds_are_scored() -> None:
    spec = {"m_good": {"libero": 0.9}, "m_bad": {"libero": 0.2}}
    bound = _grid(spec, backend="libero", seed=7) + _grid(spec, backend="libero", seed=11)
    cells = score_cells(
        bound,
        group_cols=("backend", "model", "level", "seed", "suite"),
        metrics=("ssr",),
    )
    seeds = table_rank_seed_pairs(cells)
    assert len(seeds) == 1
    assert float(seeds.iloc[0]["spearman"]) == pytest.approx(1.0)


def test_holdout_vs_core() -> None:
    core = [
        _bound(_ep(model="m_good", suite="main", success=True, episode_index=0)),
        _bound(_ep(model="m_bad", suite="main", success=False, episode_index=1)),
    ]
    hold = [
        _bound(_ep(model="m_good", suite="task_holdout", success=True, episode_index=2)),
        _bound(_ep(model="m_bad", suite="task_holdout", success=False, episode_index=3)),
    ]
    cells = score_cells(
        core + hold,
        group_cols=("backend", "model", "level", "seed", "suite"),
        metrics=("ssr",),
    )
    table = table_rank_holdout(cells)
    assert len(table) == 1
    assert float(table.iloc[0]["spearman"]) == pytest.approx(1.0)
    empty = table_rank_holdout(
        score_cells(
            core,
            group_cols=("backend", "model", "level", "seed", "suite"),
            metrics=("ssr",),
        )
    )
    assert empty.empty


def test_cr_ranks_lower_better() -> None:
    bound = [
        _bound(_ep(model="safe", success=True, contact_steps=0, episode_index=0)),
        _bound(_ep(model="risky", success=True, contact_steps=4, episode_index=1)),
    ]
    cells = score_cells(
        bound,
        group_cols=("backend", "model", "level"),
        metrics=("cr",),
    )
    ranks = rank_models(cells, slice_cols=("backend", "level"))
    by_model = {str(r["model"]): float(r["rank"]) for _, r in ranks.iterrows()}
    assert by_model["safe"] == 1.0
    assert by_model["risky"] == 2.0


def test_runs_csv_weighted_collapse(tmp_path: Path) -> None:
    csv = tmp_path / "runs.csv"
    pd.DataFrame(
        [
            {
                "backend": "libero",
                "model": "m",
                "level": 1,
                "seed": 7,
                "suite": "s",
                "n_episodes": 1,
                "ssr": 1.0,
                "sr": 1.0,
                "cr": 0.0,
            },
            {
                "backend": "libero",
                "model": "m",
                "level": 1,
                "seed": 7,
                "suite": "s",
                "n_episodes": 3,
                "ssr": 0.0,
                "sr": 0.0,
                "cr": 1.0,
            },
        ]
    ).to_csv(csv, index=False)
    cells = cells_from_runs_csv(csv, metrics=("ssr",))
    assert float(cells.iloc[0]["value"]) == pytest.approx(0.25)
    assert int(cells.iloc[0]["n"]) == 4


def _write_jsonl_run(root: Path, episodes: list[EpisodeRecord]) -> Path:
    run_dir = root / "l2" / "libero-m-s-t0-l2-s7-synth"
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


def test_cli_writes_n_a_and_does_not_touch_results(tmp_path: Path) -> None:
    results = tmp_path / "results"
    run_dir = _write_jsonl_run(results, [_ep()])
    before = fingerprint_results(results)
    original = (run_dir / "episodes.jsonl").read_bytes()
    out = tmp_path / "out"
    rc = analysis_main(["rank", "--results", str(results), "--out", str(out)])
    assert rc == 0
    assert (out / "table_rank_coverage.csv").is_file()
    assert (out / "table_rank_seed_pairs.csv").is_file()
    assert (out / "plots" / "rank_spearman_ssr.png").is_file()
    cov = pd.read_csv(out / "table_rank_coverage.csv")
    assert int(cov.iloc[0]["n_seeds"]) == 1
    assert (run_dir / "episodes.jsonl").read_bytes() == original
    assert fingerprint_results(results) == before


def test_write_rank_roundtrip(tmp_path: Path) -> None:
    spec = {
        "m_good": {"libero": 0.9, "robotwin": 0.8},
        "m_bad": {"libero": 0.2, "robotwin": 0.1},
    }
    bound = _grid(spec, backend="libero") + _grid(spec, backend="robotwin")
    written = write_rank(bound, tmp_path, metrics=("ssr",))
    loaded = pd.read_csv(tmp_path / "table_rank_backend_pairs.csv")
    assert float(loaded.iloc[0]["spearman"]) == pytest.approx(1.0)
    assert float(written["table_rank_backend_pairs"].iloc[0]["spearman"]) == pytest.approx(1.0)
