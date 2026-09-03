"""CPU tests for ``dynisafety doctor`` (no simulator)."""

from __future__ import annotations

import argparse
from pathlib import Path

from dynisafety.cli.doctor import (
    BACKEND_ENV_NAME,
    check_assets,
    mujoco_matches_robocasa_pin,
    numpy_matches_robocasa_pin,
    register_doctor_parser,
)
from dynisafety.config.schema import KNOWN_BACKENDS
from dynisafety.runner.scheduler import expand_jobs, load_sweep_suite, resolve_suite_path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_backend_env_name_includes_robocasa() -> None:
    assert BACKEND_ENV_NAME["robocasa"] == "dynisafety-robocasa"
    assert set(BACKEND_ENV_NAME) == set(KNOWN_BACKENDS)


def test_doctor_parser_accepts_robocasa() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    register_doctor_parser(sub)
    doctor = sub.choices["doctor"]
    backend_action = next(a for a in doctor._actions if "--backend" in a.option_strings)
    assert "robocasa" in (backend_action.choices or ())


def test_mujoco_robocasa_pin_is_33x() -> None:
    assert mujoco_matches_robocasa_pin("3.3.1")
    assert mujoco_matches_robocasa_pin("3.3.0")
    assert not mujoco_matches_robocasa_pin("3.1.6")
    assert not mujoco_matches_robocasa_pin("2.3.7")
    assert not mujoco_matches_robocasa_pin("3.2.9")


def test_numpy_robocasa_pin_is_2x() -> None:
    assert numpy_matches_robocasa_pin("2.2.5")
    assert numpy_matches_robocasa_pin("2.0.0")
    assert not numpy_matches_robocasa_pin("1.26.4")


def test_robocasa_missing_light_assets_fail_with_install_fix(tmp_path: Path) -> None:
    checks = check_assets(tmp_path, "robocasa", strict=False)
    by_name = {c.name: c for c in checks}
    for name in (
        "asset_robocasa_tree",
        "asset_robocasa_tex",
        "asset_robocasa_tex_generative",
        "asset_robocasa_fixtures_lw",
        "asset_robocasa_objs_lw",
    ):
        assert name in by_name, name
        c = by_name[name]
        assert c.ok is False, name
        assert c.severity == "fail", name
        assert c.fix is not None and "install_robocasa.sh" in c.fix, name


def test_robocasa_light_assets_ok_when_present(tmp_path: Path) -> None:
    assets = tmp_path / "third_party" / "robocasa" / "robocasa" / "models" / "assets"
    (assets / "textures").mkdir(parents=True)
    (assets / "textures" / "wall.png").write_bytes(b"x")
    (assets / "generative_textures").mkdir(parents=True)
    (assets / "generative_textures" / "floor.jpg").write_bytes(b"x")
    fridges = assets / "fixtures" / "fridges"
    fridges.mkdir(parents=True)
    (fridges / "Refrigerator031").mkdir()
    lw = assets / "objects" / "lightwheel"
    lw.mkdir(parents=True)
    (lw / "stool.obj").write_text("v 0 0 0\n")
    (tmp_path / "third_party" / "robocasa" / "robocasa").mkdir(parents=True, exist_ok=True)

    checks = check_assets(tmp_path, "robocasa", strict=False)
    by_name = {c.name: c for c in checks}
    for name in (
        "asset_robocasa_tree",
        "asset_robocasa_tex",
        "asset_robocasa_tex_generative",
        "asset_robocasa_fixtures_lw",
        "asset_robocasa_objs_lw",
    ):
        assert by_name[name].ok is True, name
        assert by_name[name].severity == "fail"


def test_smoke_robocasa_suite_expands_one_l0_cell() -> None:
    configs = REPO_ROOT / "configs"
    path = resolve_suite_path("smoke_robocasa", configs_root=configs)
    suite = load_sweep_suite(path)
    jobs = expand_jobs(suite, configs_root=configs)
    assert suite.name == "smoke_robocasa"
    assert len(jobs) == 1
    job = jobs[0]
    assert job.backend == "robocasa"
    assert job.suite == "robocasa_nav"
    assert job.task_id == 0
    assert job.level == 0
    assert job.model == "random_mobile"
    assert job.episodes == 1


def test_debug_xvla_dynamic_suite_is_paper_main_slice() -> None:
    configs = REPO_ROOT / "configs"
    path = resolve_suite_path("debug_xvla_dynamic", configs_root=configs)
    suite = load_sweep_suite(path)
    jobs = expand_jobs(suite, configs_root=configs)
    assert suite.name == "debug_xvla_dynamic"
    assert suite.episodes_per_task == 20
    assert len(jobs) == 117
    assert {j.episodes for j in jobs} == {20}
    assert {j.level for j in jobs} == {1, 2, 3}
    by_backend = {}
    for job in jobs:
        by_backend.setdefault(job.backend, []).append(job)
    assert set(by_backend) == {"libero", "robotwin", "simplerenv"}
    assert len(by_backend["libero"]) == 48
    assert len(by_backend["robotwin"]) == 60
    assert len(by_backend["simplerenv"]) == 9
    assert {j.model for j in by_backend["libero"]} == {"xvla"}
    assert {j.model for j in by_backend["robotwin"]} == {"xvla_official"}
    assert {j.model for j in by_backend["simplerenv"]} == {"xvla_official"}
    assert {j.suite for j in by_backend["libero"]} == {
        "safelibero_spatial",
        "safelibero_object",
        "safelibero_goal",
        "safelibero_long",
    }
    assert {j.task_id for j in by_backend["simplerenv"]} == {0, 1, 2}


def test_paper_main_does_not_include_robocasa() -> None:
    text = (REPO_ROOT / "configs" / "suites" / "paper_main.yaml").read_text(
        encoding="utf-8"
    )
    assert "robocasa" not in text.lower()


def test_libero_pi05_oracle_swr_aegis_suite_matches_paper_main_libero() -> None:
    """Extension matrix: the safety-layer arm must cover exactly the base cells.

    Base-vs-Oracle is a paired comparison, so the extension has to line up
    cell-for-cell with the π₀.₅ arm it is compared against. paper_main runs
    LIBERO at two seeds to report hazard-realisation variance; the extension
    stays on seed 7, so the invariant is against the seed-matched slice, not
    against every paper_main π₀.₅ job.
    """
    configs = REPO_ROOT / "configs"
    path = resolve_suite_path("libero_pi05_oracle_swr_aegis", configs_root=configs)
    suite = load_sweep_suite(path)
    jobs = expand_jobs(suite, configs_root=configs)
    paper = expand_jobs(
        load_sweep_suite(resolve_suite_path("paper_main", configs_root=configs)),
        configs_root=configs,
    )
    ext_seeds = {j.seed for j in jobs}
    paper_pi05 = [
        j
        for j in paper
        if j.backend == "libero" and j.model == "pi05" and j.seed in ext_seeds
    ]
    assert suite.name == "libero_pi05_oracle_swr_aegis"
    assert suite.episodes_per_task == 50
    assert len(jobs) == 64
    assert len(paper_pi05) == 64
    assert {j.episodes for j in jobs} == {50}
    assert ext_seeds == {7}
    assert {j.model for j in jobs} == {"pi05_oracle_swr_aegis"}
    assert {j.backend for j in jobs} == {"libero"}
    ext_cells = {(j.suite, j.task_id, j.level, j.seed) for j in jobs}
    paper_cells = {(j.suite, j.task_id, j.level, j.seed) for j in paper_pi05}
    assert ext_cells == paper_cells


def test_docs_do_not_port_libero_disturbances_into_robocasa() -> None:
    """Docs must not claim LIBERO tabletop levels were copied into RoboCasa."""
    forbidden = (
        "把 LIBERO Sudden Drop / Arc-Transit / Shared Reaching 搬进 RoboCasa",
        "port LIBERO Sudden Drop into RoboCasa",
        "move LIBERO Sudden Drop into RoboCasa",
        "搬进 RoboCasa",
    )
    hits: list[str] = []
    for path in sorted((REPO_ROOT / "docs").rglob("*.md")):
        # Handoffs may quote the lock; skip experiment_upgrade/handoffs.
        rel = path.relative_to(REPO_ROOT).as_posix()
        if "/handoffs/" in rel:
            continue
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            if needle in text:
                hits.append(f"{rel}: {needle}")
    assert not hits, "forbidden LIBERO→RoboCasa porting language:\n" + "\n".join(hits)
