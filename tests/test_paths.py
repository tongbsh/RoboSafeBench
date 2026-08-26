from pathlib import Path

from dynisafety.config.paths import load_paths
from dynisafety.config.resolve import find_workspace_root


def test_load_paths_reads_repo_contract(monkeypatch) -> None:
    monkeypatch.delenv("DYNI_OUT", raising=False)
    monkeypatch.delenv("DYNI_RESULTS", raising=False)
    paths = load_paths()
    assert paths.out == find_workspace_root() / "out"
    assert paths.eval == "eval/{backend}/{suite}/l{level}/{run_id}"
    assert paths.layout_root("eval") == paths.out / "eval"
    assert paths.layout_root("analysis") == paths.out / "analysis"


def test_eval_dir_and_parent_join_backend_suite_level(tmp_path: Path) -> None:
    paths = load_paths().with_out(tmp_path)
    run_dir = paths.eval_dir(
        backend="libero",
        suite="safelibero_spatial",
        level=2,
        run_id="pi05-t0-s7-20260825T000000",
    )
    assert run_dir == (
        tmp_path
        / "eval"
        / "libero"
        / "safelibero_spatial"
        / "l2"
        / "pi05-t0-s7-20260825T000000"
    )
    assert paths.eval_parent(
        backend="libero", suite="safelibero_spatial", level=2
    ) == tmp_path / "eval" / "libero" / "safelibero_spatial" / "l2"
    assert paths.videos_dir(
        backend="libero",
        suite="safelibero_spatial",
        level=2,
        run_id="pi05-t0-s7-20260825T000000",
    ) == run_dir / "videos"


def test_consumer_and_logs(tmp_path: Path) -> None:
    paths = load_paths().with_out(tmp_path)
    assert paths.consumer_dir("report", side="read") == tmp_path / "eval"
    assert paths.consumer_dir("report", side="write") == tmp_path / "reports"
    assert paths.consumer_dir("xvla_check", side="write") == (
        tmp_path / "logs" / "debug" / "xvla_walkthrough"
    )
    assert paths.logs_dir("debug", "eval_debug_libero_pi05") == (
        tmp_path / "logs" / "debug" / "eval_debug_libero_pi05"
    )


def test_dyni_out_overrides_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DYNI_OUT", str(tmp_path / "custom"))
    paths = load_paths()
    assert paths.out == tmp_path / "custom"
