"""Driver / seed-warm tests (no subprocess, no weights, no sapien)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from dynisafety.backends.robotwin.env import RoboTwinEnvHandle, expert_check_setup
from dynisafety.backends.robotwin.expert_cache import resolve_expert_check_cache_dir
from dynisafety.config.schema import ContactConfig, MetricsConfig, OutputConfig, RunConfig
from dynisafety.core.protocols import TaskSpec
from dynisafety.core.rollout import BoundRun
from dynisafety.parallel_eval.cli import dispatch_parallel_eval, run_parallel_eval
from dynisafety.parallel_eval.config import load_parallel_eval_config
from dynisafety.parallel_eval.driver import (
    ChildProcessGroup,
    assign_jobs_to_slots,
    drive_parallel_eval,
    parse_infer_listen_endpoint,
    resolve_env_slot_count,
    terminate_child_processes,
)
from dynisafety.parallel_eval.episode import begin
from dynisafety.parallel_eval.protocol import LISTEN_LINE_PREFIX
from dynisafety.parallel_eval.seed_warm import (
    WARM_SCHEMA_VERSION,
    build_warm_document,
    jobs_from_warm_document,
    jobs_from_warm_path,
    load_warm_document,
    make_episode_uid,
    walk_seed_cursor,
    warm_with_handle,
    write_warm_document,
)
from dynisafety.parallel_eval.slot_state import SlotState
from dynisafety.policies.base import model_config_from_resolved


def _parallel_eval_block(**overrides: object) -> dict[str, object]:
    block: dict[str, object] = {
        "enabled": True,
        "num_envs": 2,
        "pipeline_stage_num": 1,
        "batch_window_ms": 15,
        "infer_conda_env": "XVLA",
        "env_conda_env": "internvla_a1",
        "infer_device": "cuda:0",
        "env_cuda_visible_devices": "0",
        "channel_host": "127.0.0.1",
        "channel_port": 18765,
        "channel_timeout_s": 120,
        "torch_dtype": "float32",
        "deterministic_rng": False,
        "rng_seed": 0,
    }
    block.update(overrides)
    return block


def _pe_cfg(**overrides: object) -> Any:
    return load_parallel_eval_config({"extra": {"parallel_eval": _parallel_eval_block(**overrides)}})


class _CacheAwareEnv:
    """Accept the first seed >= ``accept_at``; also satisfy ``begin`` / ``get_obs``."""

    def __init__(self, *, accept_at: int) -> None:
        self.accept_at = accept_at
        self.play_calls = 0
        self.setup_seeds: list[int] = []
        self.plan_success = False
        self.close_calls = 0
        self.take_action_cnt = 0
        self.eval_success = False
        self.step_lim = 32
        self.instruction: str | None = None
        self.info: dict[str, Any] = {"dynisafety": {}}
        self._left = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self._right = np.array([0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    def setup_demo(self, **kwargs: Any) -> None:
        self.setup_seeds.append(int(kwargs["seed"]))
        self.take_action_cnt = 0
        self.eval_success = False

    def play_once(self) -> dict[str, Any]:
        self.play_calls += 1
        seed = self.setup_seeds[-1]
        self.plan_success = seed >= self.accept_at
        return {"info": {"demo_seed": seed}}

    def check_success(self) -> bool:
        return self.plan_success

    def close_env(self, clear_cache: bool = False) -> None:
        del clear_cache
        self.close_calls += 1

    def set_instruction(self, instruction: str) -> None:
        self.instruction = instruction

    def get_obs(self) -> dict[str, Any]:
        return {
            "observation": {
                "head_camera": {"rgb": np.full((4, 4, 3), 10, dtype=np.uint8)},
                "left_camera": {"rgb": np.full((4, 4, 3), 20, dtype=np.uint8)},
                "right_camera": {"rgb": np.full((4, 4, 3), 30, dtype=np.uint8)},
            },
            "endpose": {
                "left_endpose": self._left.copy(),
                "right_endpose": self._right.copy(),
                "left_gripper": 0.0,
                "right_gripper": 1.0,
            },
        }


def _handle(tmp_path: Path, env: _CacheAwareEnv) -> RoboTwinEnvHandle:
    return RoboTwinEnvHandle(
        task_env=env,
        task=TaskSpec(
            task_id=8,
            task_name="handover_block",
            suite="main20",
            instruction="handover the block",
            max_steps=32,
        ),
        instruction="handover the block",
        task_args={"task_config": "demo_clean"},
        clear_cache_freq=8,
        robotwin_root=tmp_path,
    )


def _begin_kwargs(
    tmp_path: Path,
    env: _CacheAwareEnv,
    *,
    start_seed: int,
    episode_index: int = 0,
) -> dict[str, Any]:
    cfg = RunConfig(
        backend="robotwin",
        suite="main20",
        level=0,
        seed=7,
        model="xvla_official_parallel",
        checkpoint="/ckpt",
        max_steps=8,
        run_id="pe-driver-test",
        metrics=MetricsConfig(),
        contact=ContactConfig(
            topple_threshold_deg=30.0,
            force_equiv_dt="impact_window",
            force_report_window_s=0.05,
            impulse_reduce="sum",
        ),
        output=OutputConfig(results_dir=str(tmp_path), save_video=False),
    )
    bound = BoundRun(config_hash="hash-pe-driver", action_space="abs_ee_16d", chunk_size=2)
    resolved = {
        "extra": {
            "robotwin": {
                "expert_check": True,
                "expert_check_cache": True,
                "expert_check_max_attempts": 20,
                "expert_check_timeout_s": 30.0,
                "expert_check_log_every": 10,
                "prompt_mode": "task_name",
                "instruction_type": "unseen",
            }
        }
    }
    return {
        "handle": _handle(tmp_path, env),
        "cfg": cfg,
        "bound": bound,
        "slot_state": SlotState(gripper_threshold=0.7),
        "episode_index": episode_index,
        "start_seed": start_seed,
        "resolved_data": resolved,
        "workspace": tmp_path,
        "observation_needs_state": False,
        "warmup_steps": None,
        "post_warmup_space": None,
    }


def test_scheme1_begin_recovers_warmed_accepted_seed(tmp_path: Path) -> None:
    """Warm writes (start, accepted); jobs pass start; begin hits cache."""
    run_seed = 7
    seed_stride = 2000
    start = 2000 * (1 + run_seed)
    accept_at = start + 2
    cache_dir = resolve_expert_check_cache_dir(tmp_path)

    warm_env = _CacheAwareEnv(accept_at=accept_at)
    rows = warm_with_handle(
        _handle(tmp_path, warm_env),
        n_episodes=1,
        run_seed=run_seed,
        seed_stride=seed_stride,
        enabled=True,
        max_attempts=20,
        timeout_s=30.0,
        log_every=10,
        cache_dir=cache_dir,
    )
    assert rows[0].start_seed == start
    assert rows[0].accepted_seed == accept_at
    assert warm_env.play_calls == 3

    document = build_warm_document(
        task="handover_block",
        suite="main20",
        level=0,
        run_seed=run_seed,
        seed_stride=seed_stride,
        episodes=rows,
    )
    jobs = jobs_from_warm_document(document)
    assert jobs[0].start_seed == start

    begin_env = _CacheAwareEnv(accept_at=99999)
    slot = begin(**_begin_kwargs(tmp_path, begin_env, start_seed=jobs[0].start_seed))
    assert slot.accepted_seed == accept_at
    assert slot.candidate_seed_start == start
    assert slot.expert_cache_hit is True
    assert begin_env.play_calls == 0

    miss_env = _CacheAwareEnv(accept_at=99999)
    with pytest.raises(RuntimeError, match="max attempts"):
        expert_check_setup(
            _handle(tmp_path, miss_env),
            seed=accept_at,
            episode_index=0,
            enabled=True,
            max_attempts=3,
            timeout_s=5.0,
            log_every=1,
            cache_dir=cache_dir,
        )
    assert miss_env.play_calls > 0


def test_seed_warm_roundtrip_jobs_and_deterministic_uid(tmp_path: Path) -> None:
    rows = walk_seed_cursor(
        run_seed=7,
        seed_stride=2000,
        n_episodes=3,
        accept=lambda start, index: start + index,
    )
    document = build_warm_document(
        task="place_empty_cup",
        suite="main20",
        level=0,
        run_seed=7,
        seed_stride=2000,
        episodes=rows,
    )
    path = tmp_path / "accepted_seeds.json"
    write_warm_document(document, path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert set(raw["episodes"][0]) == {"episode_index", "start_seed", "accepted_seed"}
    loaded = load_warm_document(path)
    assert loaded.schema == WARM_SCHEMA_VERSION
    assert loaded.task == "place_empty_cup"
    assert [(row.episode_index, row.start_seed, row.accepted_seed) for row in loaded.episodes] == [
        (row.episode_index, row.start_seed, row.accepted_seed) for row in rows
    ]

    jobs_a = jobs_from_warm_path(path)
    jobs_b = jobs_from_warm_path(path)
    assert [job.episode_uid for job in jobs_a] == [job.episode_uid for job in jobs_b]
    assert jobs_a[0].episode_uid == make_episode_uid(
        task="place_empty_cup",
        level=0,
        run_seed=7,
        episode_index=0,
    )
    assert jobs_a[1].episode_uid == make_episode_uid(
        task="place_empty_cup",
        level=0,
        run_seed=7,
        episode_index=1,
    )
    assert jobs_a[0].start_seed == rows[0].start_seed
    assert jobs_a[0].episode_uid != jobs_a[1].episode_uid


def test_parse_listen_line_skips_conda_and_warning_noise() -> None:
    noise = """
libgomp: Invalid value for environment variable OMP_NUM_THREADS
/opt/conda/lib/python3.10/site-packages/foo.py:12: FutureWarning: x
  warnings.warn("x", FutureWarning)
parallel_eval.infer listening 127.0.0.1:18765
trailing noise
"""
    host, port = parse_infer_listen_endpoint(
        noise.splitlines(),
        prefix=LISTEN_LINE_PREFIX,
    )
    assert host == "127.0.0.1"
    assert port == 18765


def test_parse_listen_line_missing_prefix() -> None:
    with pytest.raises(ValueError, match="no line starting"):
        parse_infer_listen_endpoint(
            ["libgomp: boom", "FutureWarning: x"],
            prefix=LISTEN_LINE_PREFIX,
        )


def test_cli_rejects_when_enabled_false(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    pe = _pe_cfg(enabled=False)

    def _fail_load(_req: Any) -> Any:
        return SimpleNamespace(data={"extra": {"parallel_eval": _parallel_eval_block(enabled=False)}})

    monkeypatch.setattr("dynisafety.parallel_eval.cli.load_config", _fail_load)
    monkeypatch.setattr(
        "dynisafety.parallel_eval.cli.load_parallel_eval_config",
        lambda _src: pe,
    )
    code = run_parallel_eval(
        backend="robotwin",
        model="xvla_official_parallel",
        suite="main20",
        level=0,
        task=21,
        config_files=(),
        overrides=(),
        dry_run=False,
        seed_warm=None,
        warm_only=False,
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "opt-in" in err
    assert "enabled" in err


def test_pipeline_stage_num_gt_1_not_implemented(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="not implemented \\(P2\\)"):
        drive_parallel_eval(
            pe_cfg=_pe_cfg(pipeline_stage_num=2),
            jobs=[],
            workspace=tmp_path,
            log_dir=tmp_path / "logs",
            infer_payload={},
            env_payload_base={},
        )


def test_child_process_group_terminates_on_exception() -> None:
    fake = _FakePopen()
    with pytest.raises(RuntimeError, match="boom"):
        with ChildProcessGroup() as group:
            group.add(fake)
            raise RuntimeError("boom")
    assert fake.terminated is True


def test_terminate_child_processes_kills_if_terminate_leaves_running() -> None:
    fake = _FakePopen(sticky=True)
    terminate_child_processes([fake])
    assert fake.terminated is True
    assert fake.killed is True


def test_parallel_model_yaml_builds_model_config_and_parallel_block() -> None:
    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load(
        (root / "configs/models/xvla_official_parallel.yaml").read_text(encoding="utf-8")
    )
    cfg = model_config_from_resolved(raw)
    assert cfg.name == "xvla_official_parallel"
    assert cfg.extra.get("needs_server") is False
    loaded = load_parallel_eval_config({"extra": raw["extra"]})
    assert loaded.enabled is False
    assert loaded.rng_seed == 0
    assert loaded.infer_conda_env == "XVLA"


def test_driver_ast_imports_exclude_torch_and_infer() -> None:
    """Driver must not import Infer's torch chain. Bootstrap strings are not imports."""
    import ast

    root = Path(__file__).resolve().parents[1]
    source = (root / "src/dynisafety/parallel_eval/driver.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned = {"torch", "xvla_local", "infer_worker"}
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = set(alias.name.split("."))
                if parts & banned:
                    found.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            parts = set(module.split(".")) if module else set()
            for alias in node.names:
                parts.add(alias.name)
            if parts & banned:
                found.append(module or ",".join(alias.name for alias in node.names))
    assert found == []


def test_resolve_env_slot_count_clamps_to_job_count() -> None:
    assert resolve_env_slot_count(num_envs=4, n_jobs=2) == 2
    assert resolve_env_slot_count(num_envs=2, n_jobs=5) == 2
    with pytest.raises(ValueError, match="num_envs=4.*job count is 0"):
        resolve_env_slot_count(num_envs=4, n_jobs=0)


def test_assign_jobs_to_slots_rejects_empty_slots() -> None:
    jobs = [_slot_job(0), _slot_job(1)]
    with pytest.raises(ValueError, match="num_envs=4 exceeds job count 2"):
        assign_jobs_to_slots(jobs, 4)
    assigned = assign_jobs_to_slots(jobs, 2)
    assert [len(slot) for slot in assigned] == [1, 1]


def test_drive_clamps_env_count_to_job_count(tmp_path: Path) -> None:
    jobs = [_slot_job(0), _slot_job(1)]
    recorder = _PopenRecorder()
    records = drive_parallel_eval(
        pe_cfg=_pe_cfg(num_envs=4, rng_seed=20260825),
        jobs=jobs,
        workspace=tmp_path,
        log_dir=tmp_path / "logs",
        infer_payload={
            "parallel_eval": _parallel_eval_block(num_envs=4),
            "model_path": "/ckpt",
            "repo_root": "/repo",
        },
        env_payload_base={"observation_needs_state": False},
        popen=recorder,
    )
    assert records == []
    infer_calls = [item for item in recorder.calls if item.kind == "infer"]
    env_calls = [item for item in recorder.calls if item.kind == "env"]
    assert len(infer_calls) == 1
    assert len(env_calls) == 2
    written = (tmp_path / "logs" / "infer_payload.json").read_text(encoding="utf-8")
    payload = json.loads(written)
    assert payload["parallel_eval"]["num_envs"] == 2
    env0 = json.loads((tmp_path / "logs" / "env_slot0_payload.json").read_text(encoding="utf-8"))
    assert env0["rng_seed"] == 20260825
    assert not (tmp_path / "logs" / "env_slot2_payload.json").exists()


def test_drive_rejects_empty_jobs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="num_envs=4 but job count is 0"):
        drive_parallel_eval(
            pe_cfg=_pe_cfg(num_envs=4),
            jobs=[],
            workspace=tmp_path,
            log_dir=tmp_path / "logs",
            infer_payload={"parallel_eval": _parallel_eval_block(num_envs=4)},
            env_payload_base={},
        )


def test_dispatch_help_namespace_has_eval_style_flags() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    from dynisafety.parallel_eval.cli import register_parallel_eval_parser

    register_parallel_eval_parser(sub)
    args = parser.parse_args(
        [
            "parallel-eval",
            "--backend",
            "robotwin",
            "--model",
            "xvla_official_parallel",
            "--suite",
            "main20",
            "--level",
            "0",
        ]
    )
    assert args.command == "parallel-eval"
    assert callable(dispatch_parallel_eval)
    assert getattr(args, "rng_seed", "missing") is None


def test_cli_seed_and_rng_seed_map_distinct_overrides() -> None:
    """``--seed`` is runtime.seed; ``--rng-seed`` is x1. Omitting rng-seed writes nothing."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    from dynisafety.parallel_eval.cli import (
        build_parallel_eval_overrides,
        register_parallel_eval_parser,
    )

    register_parallel_eval_parser(sub)
    seed_only = parser.parse_args(
        [
            "parallel-eval",
            "--backend",
            "robotwin",
            "--model",
            "xvla_official_parallel",
            "--suite",
            "main20",
            "--level",
            "0",
            "--seed",
            "7",
        ]
    )
    seed_overrides = build_parallel_eval_overrides(seed_only)
    assert "runtime.seed=7" in seed_overrides
    assert not any(item.startswith("extra.parallel_eval.rng_seed=") for item in seed_overrides)

    both = parser.parse_args(
        [
            "parallel-eval",
            "--backend",
            "robotwin",
            "--model",
            "xvla_official_parallel",
            "--suite",
            "main20",
            "--level",
            "0",
            "--seed",
            "7",
            "--rng-seed",
            "20260825",
        ]
    )
    both_overrides = build_parallel_eval_overrides(both)
    assert "runtime.seed=7" in both_overrides
    assert "extra.parallel_eval.rng_seed=20260825" in both_overrides


def _slot_job(index: int) -> SimpleNamespace:
    return SimpleNamespace(
        episode_uid=f"ep-{index}",
        episode_index=index,
        start_seed=1000 + index,
    )


class _PopenRecorder:
    """Stand-in for ``subprocess.Popen`` that records Infer vs Env spawns."""

    def __init__(self) -> None:
        self.calls: list[_RecordedProc] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> _RecordedProc:
        proc = _RecordedProc(argv, **kwargs)
        self.calls.append(proc)
        return proc


class _RecordedProc:
    def __init__(self, argv: list[str], **kwargs: Any) -> None:
        del kwargs
        self.argv = list(argv)
        self.pid = 41000 + id(self) % 1000
        self.terminated = False
        self.killed = False
        source = argv[argv.index("-c") + 1] if "-c" in argv else ""
        self.kind = "infer" if "serve_inference" in source else "env"
        if self.kind == "infer":
            self.returncode: int | None = None
            read_fd, write_fd = os.pipe()
            os.write(write_fd, f"{LISTEN_LINE_PREFIX} 127.0.0.1:18765\n".encode())
            os.close(write_fd)
            self.stdout = os.fdopen(read_fd, "r", buffering=1)
        else:
            self.returncode = 0
            self.stdout = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int | None:
        del timeout
        if self.kind == "env" and self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15
        if self.stdout is not None:
            try:
                self.stdout.close()
            except Exception:
                pass

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class _FakePopen:
    def __init__(self, *, sticky: bool = False) -> None:
        self.pid = -1
        self.terminated = False
        self.killed = False
        self.returncode: int | None = None
        self._sticky = sticky

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        if not self._sticky:
            self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int | None:
        del timeout
        return self.returncode
