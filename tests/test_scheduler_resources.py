"""Scheduler schedule-field lookups (M3 leftover: resources.max_jobs_per_gpu)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from dynisafety.runner.loop import make_run_dir
from dynisafety.runner.scheduler import (
    JobSpec,
    effective_max_jobs_per_gpu,
    is_job_complete,
    load_model_serve_spec,
)


def test_load_model_serve_spec_reads_resources_max_jobs(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "demo.yaml").write_text(
        yaml.dump(
            {
                "name": "demo",
                "resources": {"max_jobs_per_gpu": 2},
                "extra": {"needs_server": False, "eval_env": "demo"},
            }
        ),
        encoding="utf-8",
    )
    spec = load_model_serve_spec("demo", configs_root=tmp_path)
    assert spec.max_jobs_per_gpu == 2
    assert "max_jobs_per_gpu" not in spec.missing_fields
    assert spec.needs_server is False


def test_load_model_serve_spec_openvla_oft_canonical() -> None:
    """Regression: configs/models/openvla_oft.yaml puts the key under resources."""
    root = Path(__file__).resolve().parents[1] / "configs"
    spec = load_model_serve_spec("openvla_oft", configs_root=root)
    assert spec.max_jobs_per_gpu == 2
    assert "max_jobs_per_gpu" not in spec.missing_fields


def test_load_model_serve_spec_openpi_disables_jax_preallocate() -> None:
    """OpenPI serve must opt out of JAX's default ~75% GPU preallocation."""
    root = Path(__file__).resolve().parents[1] / "configs"
    for model in ("pi0", "pi05", "openpifast"):
        spec = load_model_serve_spec(model, configs_root=root)
        assert spec.serve_process_env.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "false"


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"resources": {"max_jobs_per_gpu": 3}}, 3),
        ({"max_jobs_per_gpu": 4}, 4),
        ({"extra": {"max_jobs_per_gpu": 5}}, 5),
        (
            {
                "resources": {"max_jobs_per_gpu": 2},
                "extra": {"max_jobs_per_gpu": 9},
                "max_jobs_per_gpu": 8,
            },
            2,
        ),
    ],
)
def test_effective_max_jobs_prefers_resources(
    payload: dict, expected: int
) -> None:
    assert effective_max_jobs_per_gpu("m", payload, None) == expected


def test_effective_max_jobs_cap() -> None:
    assert (
        effective_max_jobs_per_gpu(
            "m", {"resources": {"max_jobs_per_gpu": 4}}, cap=1
        )
        == 1
    )


def _write_server_model(
    tmp_path: Path,
    *,
    name: str = "srv",
    server: dict | None = None,
    checkpoint: object = "/ckpt",
) -> Path:
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    payload = {
        "name": name,
        "transport": "http",
        "checkpoint": checkpoint,
        "resources": {"max_jobs_per_gpu": 1},
        "server": server
        or {
            "env": "srvenv",
            "root": "/repo",
            "launch": "serve.py",
            "default_port": 9000,
            "health_path": "/health",
            "startup_timeout_s": 60,
            "args": ["--ckpt", "{checkpoint}", "--port", "{port}"],
        },
        "extra": {"needs_server": True, "server_host": "127.0.0.1"},
    }
    (models / f"{name}.yaml").write_text(yaml.dump(payload), encoding="utf-8")
    return models


def test_serve_spec_synthesizes_argv_from_server_block(tmp_path: Path) -> None:
    """M4 YAML carries `server:`, not flat serve_command; scheduler must read it."""
    _write_server_model(tmp_path)
    spec = load_model_serve_spec("srv", configs_root=tmp_path)
    assert spec.missing_fields == ()
    assert spec.serve_command == (
        "python",
        "/repo/serve.py",
        "--ckpt",
        "{checkpoint}",
        "--port",
        "{port}",
    )
    assert spec.serve_cwd == "/repo"
    assert spec.serve_env == "srvenv"
    assert spec.health_url_template == "http://127.0.0.1:{port}/health"


def test_serve_spec_resolves_per_suite_checkpoint_and_vars(tmp_path: Path) -> None:
    _write_server_model(
        tmp_path,
        checkpoint={"default": "/ckpt/a", "per_suite": {"s_a": "/ckpt/a", "s_b": "/ckpt/b"}},
        server={
            "env": "srvenv",
            "root": "/repo",
            "launch": "serve.py",
            "default_port": 9000,
            "health_path": "/health",
            "startup_timeout_s": 60,
            "args": ["--ckpt", "{checkpoint}", "--key", "{unnorm_key}"],
            "per_suite": {"s_a": {"unnorm_key": "key_a"}, "s_b": {"unnorm_key": "key_b"}},
        },
    )
    spec = load_model_serve_spec("srv", configs_root=tmp_path, suite="s_b")
    assert spec.missing_fields == ()
    assert spec.checkpoint == "/ckpt/b"
    assert spec.serve_vars == {"unnorm_key": "key_b"}


def test_serve_spec_flags_placeholder_that_needs_a_suite(tmp_path: Path) -> None:
    """Without --suite, {unnorm_key} is unresolvable — fail before conda + model load."""
    _write_server_model(
        tmp_path,
        server={
            "env": "srvenv",
            "root": "/repo",
            "launch": "serve.py",
            "default_port": 9000,
            "health_path": "/health",
            "startup_timeout_s": 60,
            "args": ["--key", "{unnorm_key}"],
            "per_suite": {"s_a": {"unnorm_key": "key_a"}},
        },
    )
    spec = load_model_serve_spec("srv", configs_root=tmp_path)
    assert any("unnorm_key" in f for f in spec.missing_fields)


def test_serve_spec_rejects_suite_outside_per_suite(tmp_path: Path) -> None:
    """Falling back to `default` here would serve the wrong fine-tune."""
    _write_server_model(
        tmp_path, checkpoint={"default": "/ckpt/a", "per_suite": {"s_a": "/ckpt/a"}}
    )
    spec = load_model_serve_spec("srv", configs_root=tmp_path, suite="s_zzz")
    assert spec.checkpoint_error is not None
    assert "s_zzz" in spec.checkpoint_error


def test_serve_spec_resolves_workspace_placeholders(tmp_path: Path) -> None:
    _write_server_model(tmp_path, checkpoint="${workspace}/external/checkpoints/demo")
    spec = load_model_serve_spec("srv", configs_root=tmp_path)
    assert "${" not in spec.checkpoint
    assert spec.checkpoint.endswith("/external/checkpoints/demo")


def test_pi05_libero_eval_env_and_complete_ckpt() -> None:
    """π₀.₅ must eval in openvla-oft and point at the nested complete orbax tree."""
    root = Path(__file__).resolve().parents[1] / "configs"
    spec = load_model_serve_spec("pi05", configs_root=root)
    pi0 = load_model_serve_spec("pi0", configs_root=root)
    assert spec.eval_env == "openvla-oft"
    assert spec.serve_env == "openpi"
    assert spec.eval_env == pi0.eval_env
    assert spec.serve_env == pi0.serve_env
    assert spec.checkpoint.endswith("pi05_libero/pi05_libero")
    assert spec.checkpoint_error is None
    assert spec.needs_server is True
    assert spec.missing_fields == ()
    assert spec.serve_command and spec.health_url_template


@pytest.mark.parametrize("model", ["univla", "starvla", "pi0", "pi05", "xvla"])
def test_paper_main_libero_servers_are_launchable(model: str) -> None:
    """Regression for P0-6: every server model in paper_main's libero row."""
    root = Path(__file__).resolve().parents[1] / "configs"
    suite = "safelibero_object" if model == "univla" else None
    spec = load_model_serve_spec(model, configs_root=root, suite=suite)
    assert spec.needs_server is True
    assert spec.missing_fields == ()
    assert spec.checkpoint_error is None
    assert spec.serve_command and spec.health_url_template


def test_univla_per_suite_selects_matching_weights() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    spatial = load_model_serve_spec("univla", configs_root=root, suite="safelibero_spatial")
    obj = load_model_serve_spec("univla", configs_root=root, suite="safelibero_object")
    assert spatial.checkpoint.endswith("univla-libero-spatial")
    assert obj.checkpoint.endswith("univla-libero-object")
    assert spatial.serve_vars["unnorm_key"] == "libero_spatial_no_noops"
    assert obj.serve_vars["unnorm_key"] == "libero_object_no_noops"


def test_starvla_eval_env_follows_per_backend() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    libero = load_model_serve_spec("starvla", configs_root=root, backend="libero")
    robotwin = load_model_serve_spec("starvla", configs_root=root, backend="robotwin")
    simpler = load_model_serve_spec("starvla", configs_root=root, backend="simplerenv")
    assert libero.eval_env == "openvla-oft"
    assert robotwin.eval_env == "internvla_a1"
    assert simpler.eval_env == "simpler_env"
    # Serve env stays shared.
    assert libero.serve_env == robotwin.serve_env == "starVLA"


def test_server_lifecycle_is_per_model_and_suite(tmp_path: Path, monkeypatch) -> None:
    """A shared server across suites would serve one suite's weights to all."""
    from dynisafety.runner import scheduler as sched_mod
    from dynisafety.runner.scheduler import (
        JobSpec,
        OnlyFilter,
        SchedulerConfig,
        ServerHandle,
        SweepScheduler,
    )

    _write_server_model(
        tmp_path,
        checkpoint={"default": "/ckpt/a", "per_suite": {"s_a": "/ckpt/a", "s_b": "/ckpt/b"}},
    )
    started: list[str] = []

    def fake_start(spec, *, gpu, port, log_path, health_timeout_s):  # noqa: ANN001
        started.append(f"{spec.name}:{spec.suite}:{spec.checkpoint}")
        return ServerHandle(
            model=spec.name, gpu=gpu, port=port, pid=-1, log_path=Path(log_path)
        )

    monkeypatch.setattr(sched_mod, "start_model_server", fake_start)
    monkeypatch.setattr(sched_mod, "stop_server", lambda handle: None)

    class _Proc:
        def wait(self) -> int:
            return 0

    monkeypatch.setattr(sched_mod.subprocess, "Popen", lambda *a, **k: _Proc())

    cfg = SchedulerConfig(
        gpu_ids=(0,),
        jobs_per_gpu_cap=None,
        resume=False,
        retry_failed=False,
        dry_run=False,
        results_root=tmp_path / "results",
        sweep_dir=tmp_path / "sweep",
        configs_root=tmp_path,
        health_timeout_s=1.0,
        only=OnlyFilter(clauses=()),
    )
    jobs = [
        JobSpec(
            backend="libero",
            suite=suite,
            task_id=0,
            task_name="t0",
            level=0,
            model="srv",
            seed=7,
            episodes=1,
        )
        for suite in ("s_a", "s_b")
    ]
    assert SweepScheduler(cfg, jobs).run() == 0
    assert started == ["srv:s_a:/ckpt/a", "srv:s_b:/ckpt/b"]


def test_eval_command_uses_bare_python_under_conda_run(tmp_path: Path) -> None:
    """conda run must not receive an absolute base-env sys.executable."""
    from dynisafety.runner.scheduler import (
        JobSpec,
        OnlyFilter,
        SchedulerConfig,
        SweepScheduler,
    )

    models = tmp_path / "models"
    models.mkdir()
    (models / "demo.yaml").write_text(
        "name: demo\nresources:\n  max_jobs_per_gpu: 1\n"
        "extra:\n  needs_server: false\n  eval_env: openvla-oft\n",
        encoding="utf-8",
    )
    cfg = SchedulerConfig(
        gpu_ids=(0,),
        jobs_per_gpu_cap=None,
        resume=False,
        retry_failed=False,
        dry_run=True,
        results_root=tmp_path / "results",
        sweep_dir=tmp_path / "sweep",
        configs_root=tmp_path,
        health_timeout_s=1.0,
        only=OnlyFilter(clauses=()),
    )
    job = JobSpec(
        backend="libero",
        suite="safelibero_spatial",
        task_id=0,
        task_name="t0",
        level=0,
        model="demo",
        seed=7,
        episodes=1,
    )
    sched = SweepScheduler(cfg, [job])
    spec = load_model_serve_spec("demo", configs_root=tmp_path)
    argv = sched._eval_command(job, spec)
    assert argv[:6] == [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "openvla-oft",
        "python",
    ]
    assert argv[6:8] == ["-m", "dynisafety.cli"]
    assert "--set" not in argv


def test_eval_command_injects_server_port_for_remote_models(tmp_path: Path) -> None:
    """Sweep-allocated port must reach eval as --set extra.server_port (not only env)."""
    from dynisafety.runner.scheduler import (
        JobSpec,
        OnlyFilter,
        SchedulerConfig,
        SweepScheduler,
        _LOOPBACK_HOST,
    )

    models = tmp_path / "models"
    models.mkdir()
    (models / "pi0demo.yaml").write_text(
        "name: pi0demo\nresources:\n  max_jobs_per_gpu: 2\n"
        "extra:\n  needs_server: true\n  eval_env: openvla-oft\n"
        "  server_host: 127.0.0.1\n"
        "server:\n  env: openpi\n  root: /tmp\n  launch: x.py\n"
        "  default_port: 8000\n  health_path: /health\n  startup_timeout_s: 10\n"
        "  args: ['--port', '{port}']\n"
        "checkpoint:\n  default: /tmp/ckpt\n",
        encoding="utf-8",
    )
    cfg = SchedulerConfig(
        gpu_ids=(0,),
        jobs_per_gpu_cap=None,
        resume=False,
        retry_failed=False,
        dry_run=True,
        results_root=tmp_path / "results",
        sweep_dir=tmp_path / "sweep",
        configs_root=tmp_path,
        health_timeout_s=1.0,
        only=OnlyFilter(clauses=()),
    )
    job = JobSpec(
        backend="libero",
        suite="safelibero_spatial",
        task_id=0,
        task_name="t0",
        level=0,
        model="pi0demo",
        seed=7,
        episodes=1,
    )
    sched = SweepScheduler(cfg, [job])
    spec = load_model_serve_spec("pi0demo", configs_root=tmp_path)
    argv = sched._eval_command(job, spec, server_port=33067)
    assert f"extra.server_host={_LOOPBACK_HOST}" in argv
    assert "extra.server_port=33067" in argv
    # default_port must not be used as the live listen port
    assert "extra.server_port=8000" not in argv


def _write_completed_summary(run_dir: Path, *, task_id: int, episodes: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "n_episodes_written": episodes,
                "task_id": task_id,
            }
        ),
        encoding="utf-8",
    )


def test_make_run_dir_uses_level_bucket(tmp_path: Path) -> None:
    path = make_run_dir(tmp_path / "results", level=2, run_id="demo-run")
    assert path == tmp_path / "results" / "l2" / "demo-run"


def test_is_job_complete_level_bucket_and_legacy_flat(tmp_path: Path) -> None:
    results = tmp_path / "results"
    job = JobSpec(
        backend="libero",
        suite="safelibero_spatial",
        task_id=0,
        task_name="t0",
        level=1,
        model="openvla_oft",
        seed=7,
        episodes=2,
    )
    assert is_job_complete(results, job) is False

    flat = (
        results
        / "libero-openvla_oft-safelibero_spatial-t0-l1-s7-20260811T000000"
    )
    _write_completed_summary(flat, task_id=0, episodes=2)
    assert is_job_complete(results, job) is True

    # Level-bucket layout is also discovered (alongside legacy flat).
    bucket = make_run_dir(
        results,
        level=1,
        run_id="libero-openvla_oft-safelibero_spatial-t0-l1-s7-20260811T010000",
    )
    _write_completed_summary(bucket, task_id=0, episodes=2)
    assert is_job_complete(results, job) is True

    # Level bucket dirs themselves must not be mistaken for run dirs.
    other = JobSpec(
        backend="libero",
        suite="safelibero_spatial",
        task_id=1,
        task_name="t1",
        level=1,
        model="openvla_oft",
        seed=7,
        episodes=2,
    )
    assert is_job_complete(results, other) is False


def test_only_filter_pipe_is_or_within_key() -> None:
    from dynisafety.runner.scheduler import OnlyFilter, parse_only_filter

    job = JobSpec(
        backend="libero",
        suite="safelibero_spatial",
        task_id=0,
        task_name="t0",
        level=2,
        model="pi05",
        seed=7,
        episodes=1,
    )
    assert parse_only_filter("backend=libero,level=1|2,model=pi05").match(job)
    assert not parse_only_filter("level=0|3").match(job)
    assert parse_only_filter(None) == OnlyFilter(clauses=())


def test_dyni_servers_per_gpu_starts_one_serve_per_slot(
    tmp_path: Path, monkeypatch
) -> None:
    from dynisafety.runner import scheduler as sched_mod
    from dynisafety.runner.scheduler import (
        JobSpec,
        OnlyFilter,
        SchedulerConfig,
        ServerHandle,
        SweepScheduler,
        servers_per_gpu_count,
    )

    assert servers_per_gpu_count(2) == 1
    monkeypatch.setenv("DYNI_SERVERS_PER_GPU", "2")
    assert servers_per_gpu_count(2) == 2
    assert servers_per_gpu_count(1) == 1

    models = tmp_path / "models"
    models.mkdir()
    (models / "srv.yaml").write_text(
        yaml.dump(
            {
                "name": "srv",
                "transport": "http",
                "checkpoint": {"default": "/ckpt"},
                "resources": {"max_jobs_per_gpu": 2},
                "server": {
                    "env": "srvenv",
                    "root": "/repo",
                    "launch": "serve.py",
                    "default_port": 9000,
                    "health_path": "/health",
                    "startup_timeout_s": 60,
                    "args": ["--port", "{port}"],
                },
                "extra": {"needs_server": True, "server_host": "127.0.0.1"},
            }
        ),
        encoding="utf-8",
    )
    started_ports: list[int] = []

    def fake_start(spec, *, gpu, port, log_path, health_timeout_s):  # noqa: ANN001
        del spec, gpu, log_path, health_timeout_s
        started_ports.append(int(port))
        return ServerHandle(
            model="srv", gpu=0, port=int(port), pid=-1, log_path=Path("/tmp/x")
        )

    monkeypatch.setattr(sched_mod, "start_model_server", fake_start)
    monkeypatch.setattr(sched_mod, "stop_server", lambda handle: None)

    class _Proc:
        def wait(self) -> int:
            return 0

    monkeypatch.setattr(sched_mod.subprocess, "Popen", lambda *a, **k: _Proc())

    cfg = SchedulerConfig(
        gpu_ids=(0,),
        jobs_per_gpu_cap=None,
        resume=False,
        retry_failed=False,
        dry_run=False,
        results_root=tmp_path / "results",
        sweep_dir=tmp_path / "sweep",
        configs_root=tmp_path,
        health_timeout_s=1.0,
        only=OnlyFilter(clauses=()),
    )
    jobs = [
        JobSpec(
            backend="libero",
            suite="safelibero_spatial",
            task_id=tid,
            task_name=f"t{tid}",
            level=1,
            model="srv",
            seed=7,
            episodes=1,
        )
        for tid in (0, 1)
    ]
    assert SweepScheduler(cfg, jobs).run() == 0
    assert len(started_ports) == 2
    assert started_ports[0] != started_ports[1]
