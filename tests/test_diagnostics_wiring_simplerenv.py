"""Fake-only SimplerEnv diagnostics wiring tests (no SAPIEN)."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dynisafety.backends.simplerenv.backend import (
    SimplerEnvBackend,
    _disturbance_controller,
)
from dynisafety.backends.simplerenv.env import SimplerEnvHandle
from dynisafety.config.loader import LoadRequest, load_config
from dynisafety.config.schema import (
    MISSING,
    DiagnosticsAvoidConfig,
    DiagnosticsConfig,
    DiagnosticsHazardEndConfig,
    DiagnosticsReactionConfig,
    DiagnosticsRecoveryConfig,
    DiagnosticsStopConfig,
    DiagnosticsTraceConfig,
    OutputConfig,
    RunConfig,
    RuntimeConfig,
)
from dynisafety.core.protocols import TaskSpec
from dynisafety.core.rollout import run_dir_from_cfg, run_dir_layout
from dynisafety.diagnostics import DiagnosticsSession, copy_vec
from dynisafety.diagnostics.reader import read_trace
from dynisafety.diagnostics.schema import BehaviorDiagnostics
from dynisafety.policies.base import ActionChunk

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = REPO_ROOT / "src" / "dynisafety" / "backends" / "simplerenv" / "backend.py"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _diag_cfg(*, enabled: bool = True, save: bool = True, stride: int = 1) -> DiagnosticsConfig:
    return DiagnosticsConfig(
        enabled=enabled,
        trace=DiagnosticsTraceConfig(save=save, format="jsonl_gz", stride=stride),
        reaction=DiagnosticsReactionConfig(
            window_size=10,
            sensitivity=2.0,
            min_persist_steps=3,
            heading_change_deg=15.0,
            sigma_rel_floor=0.05,
        ),
        stop=DiagnosticsStopConfig(speed_frac=0.10, speed_floor_m_per_step=0.0005, min_steps=5),
        avoid=DiagnosticsAvoidConfig(
            heading_change_deg=20.0, window_steps=30, min_speed_frac=0.05
        ),
        recovery=DiagnosticsRecoveryConfig(speed_frac=0.50, min_steps=5, min_post_hazard_steps=30),
        hazard_end=DiagnosticsHazardEndConfig(settle_quiet_steps=10, settle_quiet_m=0.002),
    )


def _run_cfg(
    tmp_path: Path | None = None,
    *,
    enabled: bool = True,
    save: bool = True,
    level: int = 0,
    backend: str = "simplerenv",
) -> RunConfig:
    output = MISSING
    if tmp_path is not None:
        output = OutputConfig(results_dir=str(tmp_path), save_video=False)
    return RunConfig(
        backend=backend,
        suite="bridge",
        level=level,
        seed=7,
        run_id="run-se",
        diagnostics=_diag_cfg(enabled=enabled, save=save),
        output=output,
        runtime=RuntimeConfig(control_freq=5),
    )


def _bound() -> SimpleNamespace:
    return SimpleNamespace(chunk_size=8, action_space="delta_ee_7d")


def _l1_ctrl(*, drop_step: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        _dropped=True,
        drop_step=drop_step,
        disturbance_onset_step=drop_step,
        settle_steps=2,
        _settle_end_step=drop_step + 2,
        obstacle_pos=(0.4, 0.0, 0.1),
    )


def _diag_session_call_kwargs(src: str, method: str) -> dict[str, str]:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != method:
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id == "diag_session"):
            continue
        return {kw.arg: ast.unparse(kw.value) for kw in node.keywords if kw.arg}
    raise AssertionError(f"diag_session.{method} not found in source")


def _rollout_fn() -> ast.FunctionDef:
    tree = ast.parse(BACKEND_SRC.read_text(encoding="utf-8"), filename=str(BACKEND_SRC))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "SimplerEnvBackend":
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "rollout":
                return item
    raise AssertionError("SimplerEnvBackend.rollout not found")


def _call_qualname(node: ast.Call) -> str:
    return ast.unparse(node.func)


def _is_level_eq_0(test: ast.expr) -> bool:
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "level"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == 0
    )


# ---------------------------------------------------------------------------
# Source contracts (static; no simulator)
# ---------------------------------------------------------------------------


def test_simplerenv_backend_wires_diagnostics_session() -> None:
    src = BACKEND_SRC.read_text(encoding="utf-8")
    assert "DiagnosticsSession.create" in src
    assert "diag_session.record_step" in src
    assert "diag_session.finalize" in src
    assert "diag_session.episode_diagnostics()" in src
    assert "copy_vec" in src
    assert "step_eef_pos = copy_vec(get_eef_pos(handle.env))" in src
    assert 'getattr(current, "_controller", None)' in src

    create_controller = None
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "create"):
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id == "DiagnosticsSession"):
            continue
        for kw in node.keywords:
            if kw.arg == "controller":
                create_controller = ast.unparse(kw.value)
            if kw.arg == "bound":
                assert ast.unparse(kw.value) == "bound"
    assert create_controller == "_disturbance_controller(handle.env)"

    record_kw = _diag_session_call_kwargs(src, "record_step")
    assert record_kw["eef_pos"] == "step_eef_pos"
    assert record_kw["eef_quat"] == "step_eef_quat"
    assert record_kw["base_xy_yaw"] == "None"
    assert record_kw["in_contact"] == "in_contact"
    assert record_kw["force_n"] == "force_n"
    assert record_kw["impulse_ns"] == "impulse_ns"

    finalize_kw = _diag_session_call_kwargs(src, "finalize")
    assert finalize_kw["metrics"] == "metrics"
    assert "first_contact_step" not in finalize_kw
    assert "contact_steps" not in finalize_kw


def test_eef_latched_once_before_env_step() -> None:
    """LIBERO pitfall: a post-step get_eef_pos is one control step late."""
    src = BACKEND_SRC.read_text(encoding="utf-8")
    # Whole module: policy-state fallback + the pre-step latch. No third call.
    assert src.count("get_eef_pos(") == 2

    rollout = _rollout_fn()
    eef_lines: list[int] = []
    step_lines: list[int] = []
    latch_ok = False
    for node in ast.walk(rollout):
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "step_eef_pos" in names:
                assert isinstance(node.value, ast.Call)
                assert ast.unparse(node.value.func) == "copy_vec"
                assert "get_eef_pos" in ast.unparse(node.value)
                latch_ok = True
        if not isinstance(node, ast.Call):
            continue
        qual = _call_qualname(node)
        if qual == "get_eef_pos":
            eef_lines.append(node.lineno)
        if qual == "handle.env.step":
            step_lines.append(node.lineno)
    assert latch_ok
    assert eef_lines == [min(eef_lines)]  # exactly one in rollout
    assert step_lines
    assert all(ln < min(step_lines) for ln in eef_lines)


def test_controller_lookup_uses_getattr_controller() -> None:
    src = BACKEND_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    helper = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_disturbance_controller":
            helper = node
            break
    assert helper is not None
    helper_src = ast.unparse(helper)
    assert "getattr" in helper_src
    assert "_controller" in helper_src
    assert "env" in helper_src
    assert "unwrapped" in helper_src


def test_l0_record_step_forces_zero_contact() -> None:
    rollout = _rollout_fn()
    assigned: dict[str, str] = {}
    for node in ast.walk(rollout):
        if not isinstance(node, ast.If) or not _is_level_eq_0(node.test):
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            names = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
            value = ast.unparse(stmt.value)
            for name in names:
                assigned[name] = value
    assert assigned["in_contact"] == "False"
    assert assigned["force_n"] == "0.0"
    assert assigned["impulse_ns"] == "0.0"


# ---------------------------------------------------------------------------
# Controller lookup (defensive; no sim)
# ---------------------------------------------------------------------------


def test_disturbance_controller_l0_and_wrapper_and_outer() -> None:
    ctrl = _l1_ctrl()
    assert _disturbance_controller(SimpleNamespace()) is None
    assert _disturbance_controller(SimpleNamespace(_controller=None)) is None
    assert _disturbance_controller(SimpleNamespace(_controller=ctrl)) is ctrl

    wrapper = SimpleNamespace(_controller=ctrl, env=SimpleNamespace(), unwrapped=SimpleNamespace())
    assert _disturbance_controller(wrapper) is ctrl

    inner = SimpleNamespace(_controller=ctrl)
    outer = SimpleNamespace(env=inner, unwrapped=SimpleNamespace())
    assert _disturbance_controller(outer) is ctrl

    unwrapped_only = SimpleNamespace(unwrapped=SimpleNamespace(_controller=ctrl))
    assert _disturbance_controller(unwrapped_only) is ctrl


def test_disturbance_controller_never_raises() -> None:
    class _Boom:
        def __getattribute__(self, name: str) -> object:
            raise RuntimeError(f"boom on {name}")

    assert _disturbance_controller(_Boom()) is None


# ---------------------------------------------------------------------------
# Fake session mini-rollout
# ---------------------------------------------------------------------------


def test_fake_session_rollout_yields_valid_diagnostics(tmp_path: Path) -> None:
    session = DiagnosticsSession.create(
        _run_cfg(tmp_path, level=0),
        episode_index=0,
        bound=_bound(),
        controller=None,
    )
    assert session.enabled is True
    live_eef = [0.11, 0.22, 0.33]
    for t in range(8):
        stashed = copy_vec(live_eef)
        live_eef[0] = 99.0
        session.record_step(
            t=t,
            eef_pos=stashed,
            eef_quat=(1.0, 0.0, 0.0, 0.0),
            action=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
            in_contact=False,
            force_n=0.0,
            impulse_ns=0.0,
            base_xy_yaw=None,
        )
        live_eef[0] = 0.11 + 0.01 * (t + 1)
    session.finalize(
        success=True,
        success_step=7,
        metrics={"first_contact_step": None, "contact_steps": 0},
    )
    diag = session.episode_diagnostics()
    assert diag is not None
    parsed = BehaviorDiagnostics.from_json_dict(diag)
    assert parsed.disturbance_triggered is False
    trace = read_trace(
        run_dir_layout(
            tmp_path,
            backend="simplerenv",
            suite="bridge",
            level=0,
            run_id="run-se",
        )
        / "traces"
        / "ep0000.jsonl.gz"
    )
    assert trace.steps[0].eef_pos == (0.11, 0.22, 0.33)
    assert all(step.base_xy_yaw is None for step in trace.steps)


# ---------------------------------------------------------------------------
# Fake backend.rollout (no SAPIEN)
# ---------------------------------------------------------------------------


class _FakePolicy:
    def reset(self, ctx: object) -> None:
        del ctx

    def act(self, obs: object) -> ActionChunk:
        del obs
        return ActionChunk(actions=np.zeros((1, 7), dtype=np.float64), space="delta_ee_7d")

    def close(self) -> None:
        return None


class _FakeManiEnv:
    """Gym-shaped stand-in: live EEF buffer overwritten in ``step``."""

    def __init__(
        self,
        *,
        n_steps: int = 4,
        dyn: dict[str, object] | None = None,
        controller: object | None = None,
    ) -> None:
        self._n = int(n_steps)
        self._i = 0
        self._dyn = dyn
        self._controller = controller
        self._episode_index = 0
        self._eef = np.array([0.11, 0.22, 0.33], dtype=np.float64)
        scene = SimpleNamespace(get_timestep=lambda: 0.05)
        self.unwrapped = SimpleNamespace(_scene=scene)
        self.scene = scene

    def reset(self, options: object = None) -> tuple[dict[str, object], dict[str, object]]:
        del options
        self._i = 0
        self._eef[:] = (0.11, 0.22, 0.33)
        return self._obs(), {}

    def _obs(self) -> dict[str, object]:
        return {
            "agent": {
                "eef_pos": np.array(
                    [0.11, 0.22, 0.33, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
                )
            }
        }

    def get_language_instruction(self) -> str:
        return "put the carrot on the plate"

    def is_final_subtask(self) -> bool:
        return True

    def step(self, action: object) -> tuple[object, float, bool, bool, dict[str, object]]:
        del action
        self._eef[:] = (99.0, 99.0, 99.0)
        self._i += 1
        info: dict[str, object] = {}
        if self._dyn is not None:
            info["dynisafety"] = dict(self._dyn)
        done = self._i >= self._n
        return self._obs(), 0.0, done, False, info


def _task() -> TaskSpec:
    return TaskSpec(
        task_id=0,
        task_name="bridge_carrot",
        suite="bridge",
        instruction="put the carrot on the plate",
        max_steps=4,
    )


def _handle(env: _FakeManiEnv) -> SimplerEnvHandle:
    return SimplerEnvHandle(
        env=env,
        base_env=env,
        task=_task(),
        instruction="put the carrot on the plate",
        control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        task_extra={
            "robot_init_x": 0.0,
            "robot_init_y": 0.0,
            "robot_init_quat": [1.0, 0.0, 0.0, 0.0],
        },
    )


def _load_se_cfg(tmp_path: Path, *, level: int, max_steps: int = 4) -> object:
    return load_config(
        LoadRequest(
            backend="simplerenv",
            level=level,
            suite="bridge",
            model="random",
            overrides=(
                f"output.results_dir={tmp_path}",
                "output.save_video=false",
                "run_id=run-se",
                f"runtime.max_steps={max_steps}",
                f"max_steps={max_steps}",
            ),
        )
    )


def _bind(backend: SimplerEnvBackend, resolved: object) -> None:
    data = resolved.data
    backend.bind_run(
        config_hash=str(resolved.config_hash),
        resolved_data=data,
        action_space=str(data["action_space"]),
        chunk_size=int(data["chunk_size"]),
        observation_needs_state=True,
    )


@pytest.fixture
def _sim_free(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dynisafety.backends.simplerenv.backend.get_primary_image",
        lambda *_a, **_k: np.zeros((8, 8, 3), dtype=np.uint8),
    )

    def _eef(env: object) -> np.ndarray:
        buf = getattr(env, "_eef", None)
        if buf is None:
            return np.array([0.11, 0.22, 0.33], dtype=np.float64)
        return buf

    monkeypatch.setattr("dynisafety.backends.simplerenv.backend.get_eef_pos", _eef)


def test_fake_rollout_l0_zero_contact_and_eef_latch(tmp_path: Path, _sim_free: None) -> None:
    """L0 must ignore info['dynisafety'] contact; EEF is the pre-step latch."""
    del _sim_free
    resolved = _load_se_cfg(tmp_path, level=0, max_steps=3)
    backend = SimplerEnvBackend()
    _bind(backend, resolved)
    env = _FakeManiEnv(
        n_steps=3,
        dyn={"in_contact": True, "force_n": 12.5, "impulse_ns": 0.4},
    )
    record = backend.rollout(env=_handle(env), policy=_FakePolicy(), cfg=resolved.run, episode_index=0)
    assert record.termination != "error"
    assert record.diagnostics is not None
    BehaviorDiagnostics.from_json_dict(record.diagnostics)
    assert record.contact_steps == 0
    assert record.first_contact_step is None

    run_dir = run_dir_from_cfg(resolved.run)
    assert run_dir is not None
    trace = read_trace(run_dir / "traces" / "ep0000.jsonl.gz")
    assert len(trace.steps) == 3
    assert trace.steps[0].eef_pos == (0.11, 0.22, 0.33)
    assert all(step.in_contact is False for step in trace.steps)
    assert all(step.force_n == 0.0 for step in trace.steps)
    assert all(step.impulse_ns == 0.0 for step in trace.steps)
    assert all(step.base_xy_yaw is None for step in trace.steps)
    assert trace.steps[0].eef_quat == (1.0, 0.0, 0.0, 0.0)


def test_fake_rollout_l1_records_dyn_contact(tmp_path: Path, _sim_free: None) -> None:
    del _sim_free
    resolved = _load_se_cfg(tmp_path, level=1, max_steps=4)
    backend = SimplerEnvBackend()
    _bind(backend, resolved)
    env = _FakeManiEnv(
        n_steps=4,
        controller=_l1_ctrl(drop_step=2),
        dyn={"in_contact": True, "force_n": 3.0, "impulse_ns": 0.1, "disturbance_onset_step": 2},
    )
    record = backend.rollout(env=_handle(env), policy=_FakePolicy(), cfg=resolved.run, episode_index=1)
    assert record.termination != "error"
    assert record.diagnostics is not None
    BehaviorDiagnostics.from_json_dict(record.diagnostics)
    assert record.contact_steps == 4
    run_dir = run_dir_from_cfg(resolved.run)
    assert run_dir is not None
    trace = read_trace(run_dir / "traces" / "ep0001.jsonl.gz")
    assert all(step.in_contact is True for step in trace.steps)
    assert trace.header.disturbance_triggered is True
    assert trace.header.t_onset == 2


def test_probe_raise_does_not_abort_rollout(tmp_path: Path, _sim_free: None, monkeypatch: pytest.MonkeyPatch) -> None:
    del _sim_free

    class _BoomProbe:
        def snapshot(self, task_step: int) -> object:
            raise RuntimeError(f"snapshot boom at t={task_step}")

    monkeypatch.setattr(
        "dynisafety.diagnostics.wiring.make_probe",
        lambda *_a, **_k: _BoomProbe(),
    )
    resolved = _load_se_cfg(tmp_path, level=1, max_steps=3)
    backend = SimplerEnvBackend()
    _bind(backend, resolved)
    env = _FakeManiEnv(n_steps=3, controller=_l1_ctrl())
    record = backend.rollout(env=_handle(env), policy=_FakePolicy(), cfg=resolved.run, episode_index=0)
    assert record.termination != "error"
    assert record.diagnostics is not None
    BehaviorDiagnostics.from_json_dict(record.diagnostics)


def test_unwritable_run_dir_does_not_abort_rollout(tmp_path: Path, _sim_free: None) -> None:
    del _sim_free
    resolved = _load_se_cfg(tmp_path, level=0, max_steps=3)
    run_dir = run_dir_from_cfg(resolved.run)
    assert run_dir is not None
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.write_text("not-a-directory\n", encoding="utf-8")
    backend = SimplerEnvBackend()
    _bind(backend, resolved)
    record = backend.rollout(
        env=_handle(_FakeManiEnv(n_steps=3)),
        policy=_FakePolicy(),
        cfg=resolved.run,
        episode_index=0,
    )
    assert record.termination != "error"
    assert record.diagnostics is not None
    assert record.diagnostics["trace_path"] is None
    assert run_dir.is_file()


def test_session_probe_and_unwritable_do_not_raise(tmp_path: Path) -> None:
    """Sidecar degrade path, same contract as the LIBERO wiring tests."""

    class _BoomProbe:
        def snapshot(self, task_step: int) -> object:
            raise RuntimeError(f"snapshot boom at t={task_step}")

    session = DiagnosticsSession.create(
        _run_cfg(tmp_path, level=1),
        episode_index=0,
        bound=_bound(),
        controller=_l1_ctrl(),
    )
    session._probe = _BoomProbe()  # noqa: SLF001
    session.record_step(
        t=0,
        eef_pos=(0.1, 0.0, 0.8),
        action=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        in_contact=False,
        force_n=0.0,
        impulse_ns=0.0,
    )
    session.finalize(success=False, success_step=None, contact_steps=0)
    assert session.episode_diagnostics() is not None

    blocked = tmp_path / "blocked"
    blocked.mkdir()
    occupied = run_dir_layout(
        blocked, backend="simplerenv", suite="bridge", level=0, run_id="run-se"
    )
    occupied.parent.mkdir(parents=True, exist_ok=True)
    occupied.write_text("blocked\n", encoding="utf-8")
    session2 = DiagnosticsSession.create(
        _run_cfg(blocked, level=0),
        episode_index=0,
        bound=_bound(),
    )
    session2.record_step(
        t=0,
        eef_pos=(0.1, 0.0, 0.8),
        action=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        in_contact=False,
        force_n=0.0,
        impulse_ns=0.0,
    )
    session2.finalize(success=False, success_step=None, contact_steps=0)
    assert session2.episode_diagnostics() is not None
    assert session2.episode_diagnostics()["trace_path"] is None
