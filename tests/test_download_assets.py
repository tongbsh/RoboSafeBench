"""Unit tests for scripts/setup/download_assets.py (no network)."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "setup" / "download_assets.py"
MODELS = REPO / "configs" / "models"
_CKPT_MARKER = "external/checkpoints/"


def _load_mod():
    spec = importlib.util.spec_from_file_location("download_assets", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _yaml_ckpt_values(stem: str) -> list[str]:
    data = yaml.safe_load((MODELS / f"{stem}.yaml").read_text(encoding="utf-8")) or {}
    block = data.get("checkpoint") or {}
    vals: list[str] = []
    default = block.get("default")
    if isinstance(default, str):
        vals.append(default)
    per_suite = block.get("per_suite") or {}
    if isinstance(per_suite, dict):
        vals.extend(str(v) for v in per_suite.values())
    return vals


def _rel_under_checkpoints(value: str) -> str | None:
    if _CKPT_MARKER not in value:
        return None
    rel = value.split(_CKPT_MARKER, 1)[1]
    return rel.rstrip("}")


def _is_hf_repo(value: str) -> bool:
    if value.startswith(("/", ".", "~")) or "${" in value:
        return False
    return bool(re.fullmatch(r"[\w.\-]+/[\w.\-]+", value))


def _dest_covers(dest: str, yaml_rel: str) -> bool:
    return yaml_rel == dest or yaml_rel.startswith(dest + "/")


@pytest.fixture(scope="module")
def da():
    return _load_mod()


def test_catalog_names_unique(da) -> None:
    names = [spec.name for spec in da.list_catalog()]
    assert names
    assert len(names) == len(set(names))


def test_paper_main_stems_all_have_assets(da) -> None:
    mapping = da.paper_model_suites(REPO)
    catalog = da.list_catalog()
    missing: list[str] = []
    for stem, suites in sorted(mapping.items()):
        for suite in sorted(suites):
            hits = [
                spec
                for spec in catalog
                if stem in spec.models and da.spec_covers_suite(spec, suite)
            ]
            if not hits:
                missing.append(f"{stem}/{suite}")
    assert not missing, f"paper_main cells with no catalog asset: {missing}"


def test_paper_select_excludes_univla_bridge(da) -> None:
    selected = da.select_specs(workspace=REPO, paper=True)
    names = {spec.name for spec in selected}
    assert "univla-libero-spatial" in names
    assert "univla-simpler-bridge" not in names
    assert "lingbot-va-base" not in names
    assert "bridgevla-rlbench-ckpt" not in names


def test_model_starvla_suite_filter(da) -> None:
    selected = da.select_specs(workspace=REPO, models=["starvla"], suite="main20")
    assert [spec.name for spec in selected] == ["starvla-robotwin2-all"]


def test_model_xvla_official_both_paper_suites(da) -> None:
    selected = da.select_specs(workspace=REPO, models=["xvla_official"])
    assert {spec.name for spec in selected} == {
        "xvla-robotwin2",
        "xvla-simpler-widowx",
    }


def test_internvla_a1_dest_matches_yaml(da) -> None:
    spec = da.catalog_by_name("internvla-a1-robotwin")
    assert spec.relative_dest == "InternVLA/InternVLA-A1-3B-RoboTwin"
    rels = [_rel_under_checkpoints(v) for v in _yaml_ckpt_values("internvla_a1")]
    assert spec.relative_dest in rels


def test_yaml_local_paths_covered_by_catalog_dest(da) -> None:
    """Every local YAML checkpoint path is a file/dir under some catalog dest."""
    mapping = da.paper_model_suites(REPO)
    catalog = da.list_catalog()
    uncovered: list[str] = []
    for stem in mapping:
        dests = [spec.relative_dest for spec in catalog if stem in spec.models]
        repos = {spec.repo_id for spec in catalog if stem in spec.models}
        for raw in _yaml_ckpt_values(stem):
            if _is_hf_repo(raw):
                if raw not in repos:
                    uncovered.append(f"{stem} hf {raw}")
                continue
            rel = _rel_under_checkpoints(raw)
            if rel is None:
                uncovered.append(f"{stem} unparsed {raw}")
                continue
            if not any(_dest_covers(dest, rel) for dest in dests):
                uncovered.append(f"{stem} {rel} dests={dests}")
    assert not uncovered, uncovered


def test_unknown_asset_and_model(da) -> None:
    with pytest.raises(SystemExit, match="unknown asset"):
        da.catalog_by_name("not-an-asset")
    with pytest.raises(SystemExit, match="unknown model"):
        da.select_specs(workspace=REPO, models=["not_a_model"])


def test_resolve_dest_order(da, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = tmp_path / "ws"
    (ws / "src" / "dynisafety").mkdir(parents=True)
    (ws / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    monkeypatch.delenv("DYNISAFETY_CHECKPOINTS", raising=False)
    assert da.resolve_dest(cli_dest=None, env={}, workspace=ws) == (
        ws / "external" / "checkpoints"
    ).resolve()
    env_dir = tmp_path / "from-env"
    assert da.resolve_dest(
        cli_dest=None,
        env={"DYNISAFETY_CHECKPOINTS": str(env_dir)},
        workspace=ws,
    ) == env_dir.resolve()
    cli = tmp_path / "from-cli"
    assert da.resolve_dest(
        cli_dest=str(cli),
        env={"DYNISAFETY_CHECKPOINTS": str(env_dir)},
        workspace=ws,
    ) == cli.resolve()


def test_cli_paper_dry_run(da, capsys: pytest.CaptureFixture[str]) -> None:
    rc = da.main(["--paper", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "==> internvla-a1-robotwin" in out
    assert "InternVLA/InternVLA-A1-3B-RoboTwin" in out
    assert "==> xvla-simpler-widowx" in out
    assert "==> pi05-libero" in out
    assert "dry-run: skip download" in out
    assert "==> univla-simpler-bridge" not in out


def test_cli_name_alias_and_list_model(
    da, capsys: pytest.CaptureFixture[str]
) -> None:
    assert da.main(["--list", "--model", "fastwam"]) == 0
    listed = capsys.readouterr().out
    assert "fastwam-robotwin" in listed
    assert "fastwam-robotwin2-dataset" not in listed

    rc = da.main(["--name", "fastwam-robotwin", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "==> fastwam-robotwin" in out
    assert "local_dir=" in out


def test_cli_requires_selector(da) -> None:
    with pytest.raises(SystemExit):
        da.main([])
