from pathlib import Path

from dynisafety.cli.serve import resolve_serve_log_path


def test_resolve_serve_log_path_default_under_logs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DYNI_OUT", str(tmp_path / "out"))
    path = resolve_serve_log_path(workspace=tmp_path, model="pi05", port=8011)
    assert path == tmp_path / "out" / "logs" / "serve" / "pi05" / "port8011.log"


def test_resolve_serve_log_path_env_dir(tmp_path: Path, monkeypatch) -> None:
    log_dir = tmp_path / "artifacts" / "debug" / "pi05_libero"
    monkeypatch.setenv("DYNI_SERVE_LOG_DIR", str(log_dir))
    path = resolve_serve_log_path(workspace=tmp_path, model="pi05", port=8011)
    assert path == log_dir / "server.log"


def test_resolve_serve_log_path_explicit_file(tmp_path: Path) -> None:
    dest = tmp_path / "custom" / "serve.log"
    path = resolve_serve_log_path(
        workspace=tmp_path, model="pi05", port=8011, log_dir=str(dest)
    )
    assert path == dest
