"""Fake-only tests for diagnostics rollout wiring (no simulator)."""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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
from dynisafety.core.rollout import run_dir_layout
from dynisafety.diagnostics import DiagnosticsSession, copy_vec
from dynisafety.diagnostics.schema import BehaviorDiagnostics
from dynisafety.diagnostics.wiring import DiagnosticsSession as WiringSession

REPO_ROOT = Path(__file__).resolve().parents[1]
WIRING_SRC = REPO_ROOT / "src" / "dynisafety" / "diagnostics" / "wiring.py"
LIBERO_BACKEND_SRC = REPO_ROOT / "src" / "dynisafety" / "backends" / "libero" / "backend.py"
ROBOTWIN_BACKEND_SRC = REPO_ROOT / "src" / "dynisafety" / "backends" / "robotwin" / "backend.py"


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


def _expected_run_dir(
    tmp_path: Path,
    *,
    level: int = 0,
    run_id: str = "run-x",
    backend: str = "libero",
    suite: str = "safelibero_spatial",
) -> Path:
    return run_dir_layout(
        tmp_path, backend=backend, suite=suite, level=level, run_id=run_id
    )


def _run_cfg(
    tmp_path: Path | None = None,
    *,
    enabled: bool = True,
    save: bool = True,
    run_id: object = "run-x",
    level: int = 0,
    backend: str = "libero",
    suite: str = "safelibero_spatial",
    seed: int = 7,
    control_freq: object = 20,
    diagnostics: object = None,
) -> RunConfig:
    output = MISSING
    if tmp_path is not None:
        output = OutputConfig(results_dir=str(tmp_path), save_video=False)
    runtime = MISSING
    if control_freq is not MISSING:
        runtime = RuntimeConfig(control_freq=control_freq)
    diag = _diag_cfg(enabled=enabled, save=save) if diagnostics is None else diagnostics
    return RunConfig(
        backend=backend,
        suite=suite,
        level=level,
        seed=seed,
        run_id=run_id,  # type: ignore[arg-type]
        diagnostics=diag,  # type: ignore[arg-type]
        output=output,
        runtime=runtime,
    )


def _bound() -> SimpleNamespace:
    return SimpleNamespace(chunk_size=8, action_space="delta_ee_7d")


def _l1_ctrl(*, drop_step: int = 12) -> SimpleNamespace:
    return SimpleNamespace(
        _dropped=True,
        drop_step=drop_step,
        disturbance_onset_step=drop_step,
        settle_steps=2,
        _settle_end_step=drop_step + 2,
        obstacle_pos=(0.4, 0.0, 0.1),
    )


def _feed_steps(
    session: DiagnosticsSession,
    n: int,
    *,
    t0: int = 0,
    contact_at: frozenset[int] = frozenset(),
) -> list[list[float]]:
    """Feed ``n`` task steps. Returns the action lists so callers can assert no mutation."""
    actions: list[list[float]] = []
    for i in range(n):
        t = t0 + i
        action = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        session.record_step(
            t=t,
            eef_pos=(0.01 * i, 0.0, 0.8),
            eef_quat=(0.0, 0.0, 0.0, 1.0),
            action=action,
            in_contact=t in contact_at,
            force_n=1.5 if t in contact_at else 0.0,
            impulse_ns=0.2 if t in contact_at else 0.0,
        )
        actions.append(action)
    return actions


def _finalize(
    session: DiagnosticsSession,
    *,
    success: bool = False,
    success_step: int | None = None,
    first_contact_step: int | None = None,
    contact_steps: int = 0,
    metrics: dict[str, object] | None = None,
) -> dict[str, object] | None:
    session.finalize(
        success=success,
        success_step=success_step,
        first_contact_step=first_contact_step,
        contact_steps=contact_steps,
        metrics=metrics,
    )
    return session.episode_diagnostics()


# ---------------------------------------------------------------------------
# Disabled / missing config
# ---------------------------------------------------------------------------


def test_disabled_is_noop_and_creates_no_directories(tmp_path: Path) -> None:
    cfg = _run_cfg(tmp_path, enabled=False, level=1)
    session = DiagnosticsSession.create(
        cfg, episode_index=0, bound=_bound(), controller=_l1_ctrl()
    )
    assert session.enabled is False
    _feed_steps(session, 8)
    assert _finalize(session, success=True, success_step=3) is None
    assert list(tmp_path.iterdir()) == []


def test_missing_diagnostics_block_is_disabled(tmp_path: Path) -> None:
    cfg = RunConfig(
        backend="libero",
        level=1,
        seed=7,
        run_id="run-x",
        output=OutputConfig(results_dir=str(tmp_path), save_video=False),
    )
    session = DiagnosticsSession.create(cfg, episode_index=0, bound=_bound())
    assert session.enabled is False
    session.record_step(
        t=0,
        eef_pos=(0.0, 0.0, 0.0),
        action=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        in_contact=False,
        force_n=0.0,
        impulse_ns=0.0,
    )
    session.finalize(success=False, success_step=None, contact_steps=0)
    assert session.episode_diagnostics() is None
    assert list(tmp_path.iterdir()) == []


def test_disabled_does_not_call_make_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("make_probe must not run when diagnostics are off")

    monkeypatch.setattr("dynisafety.diagnostics.wiring.make_probe", _boom)
    session = DiagnosticsSession.create(
        _run_cfg(tmp_path, enabled=False),
        episode_index=0,
        bound=_bound(),
        controller=_l1_ctrl(),
    )
    assert session.enabled is False


# ---------------------------------------------------------------------------
# Enabled happy path
# ---------------------------------------------------------------------------


def test_enabled_l0_yields_valid_diagnostics_dict(tmp_path: Path) -> None:
    session = DiagnosticsSession.create(
        _run_cfg(tmp_path, level=0),
        episode_index=3,
        bound=_bound(),
        controller=None,
    )
    assert session.enabled is True
    actions = _feed_steps(session, 16)
    diag = _finalize(session, success=True, success_step=15)
    assert diag is not None
    parsed = BehaviorDiagnostics.from_json_dict(diag)
    # L0 has no disturbance at all, so no axis is observable. It must not share
    # a label with "the hazard came and nothing changed".
    assert parsed.behavior_class.value == "not_triggered"
    assert parsed.disturbance_triggered is False
    assert parsed.t_onset is None
    assert parsed.trace_path == "traces/ep0003.jsonl.gz"
    run_dir = _expected_run_dir(tmp_path, level=0)
    assert (run_dir / "traces" / "ep0003.jsonl.gz").is_file()
    assert (run_dir / "traces" / "index.json").is_file()
    assert actions[0] == [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def test_enabled_l1_latches_onset_from_probe(tmp_path: Path) -> None:
    drop = 12
    session = DiagnosticsSession.create(
        _run_cfg(tmp_path, level=1),
        episode_index=0,
        action_chunk_len=8,
        action_space="delta_ee_7d",
        controller=_l1_ctrl(drop_step=drop),
    )
    _feed_steps(session, 24)
    diag = _finalize(
        session,
        success=False,
        metrics={"first_contact_step": None, "contact_steps": 0},
    )
    assert diag is not None
    BehaviorDiagnostics.from_json_dict(diag)
    assert diag["disturbance_triggered"] is True
    assert diag["t_onset"] == drop
    assert diag["t_hazard_end"] == drop + 2
    assert diag["hazard_persists"] is True
    assert diag["trace_path"] == "traces/ep0000.jsonl.gz"


def test_save_false_still_analyzes_without_creating_traces(tmp_path: Path) -> None:
    session = DiagnosticsSession.create(
        _run_cfg(tmp_path, save=False, level=0),
        episode_index=0,
        bound=_bound(),
    )
    _feed_steps(session, 12)
    diag = _finalize(session, success=False)
    assert diag is not None
    assert diag["trace_path"] is None
    assert not (_expected_run_dir(tmp_path, level=0) / "traces").exists()


def test_run_dir_from_cfg_none_still_analyzes() -> None:
    cfg = _run_cfg(tmp_path=None, run_id=MISSING, level=0)
    session = DiagnosticsSession.create(cfg, episode_index=0, bound=_bound())
    assert session.enabled is True
    _feed_steps(session, 12)
    diag = _finalize(session, success=True, success_step=4)
    assert diag is not None
    BehaviorDiagnostics.from_json_dict(diag)
    assert diag["trace_path"] is None


def test_copy_vec_is_independent_of_source_buffer() -> None:
    src = [0.1, 0.2, 0.3]
    out = copy_vec(src)
    src[0] = 9.0
    assert out == (0.1, 0.2, 0.3)
    assert copy_vec(None) is None


def test_eef_stash_survives_in_place_obs_reuse(tmp_path: Path) -> None:
    """The LIBERO pitfall: env.step overwrites the same EEF buffer."""
    session = DiagnosticsSession.create(
        _run_cfg(tmp_path, level=0),
        episode_index=0,
        bound=_bound(),
    )
    live_eef = [0.11, 0.22, 0.33]
    stashed = copy_vec(live_eef)
    live_eef[0] = 99.0
    session.record_step(
        t=0,
        eef_pos=stashed,
        action=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        in_contact=False,
        force_n=0.0,
        impulse_ns=0.0,
    )
    _finalize(session, success=False)
    from dynisafety.diagnostics.reader import read_trace

    trace = read_trace(_expected_run_dir(tmp_path, level=0) / "traces" / "ep0000.jsonl.gz")
    assert trace.steps[0].eef_pos == (0.11, 0.22, 0.33)


def test_public_export_matches_wiring_module() -> None:
    assert DiagnosticsSession is WiringSession


# ---------------------------------------------------------------------------
# Degradation — never raises, rollout payload stays caller-owned
# ---------------------------------------------------------------------------


class _BoomProbe:
    def snapshot(self, task_step: int) -> object:
        raise RuntimeError(f"snapshot boom at t={task_step}")


def test_probe_snapshot_raises_does_not_escape(tmp_path: Path) -> None:
    session = DiagnosticsSession.create(
        _run_cfg(tmp_path, level=1),
        episode_index=0,
        bound=_bound(),
        controller=_l1_ctrl(),
    )
    session._probe = _BoomProbe()  # noqa: SLF001 — inject a raising probe
    actions = _feed_steps(session, 8)
    diag = _finalize(session, success=True, success_step=7)
    assert diag is not None
    BehaviorDiagnostics.from_json_dict(diag)
    assert actions[0] == [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def test_unwritable_run_dir_still_analyzes(tmp_path: Path) -> None:
    """Root ignores chmod, so occupy the run path with a file instead."""
    blocked = _expected_run_dir(tmp_path, level=0)
    blocked.parent.mkdir(parents=True, exist_ok=True)
    blocked.write_text("not-a-directory\n", encoding="utf-8")
    session = DiagnosticsSession.create(
        _run_cfg(tmp_path, level=0),
        episode_index=0,
        bound=_bound(),
    )
    assert session.enabled is True
    _feed_steps(session, 12)
    diag = _finalize(session, success=False)
    assert diag is not None
    assert diag["trace_path"] is None
    assert blocked.is_file()


def test_analyze_failure_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("analyze boom")

    monkeypatch.setattr("dynisafety.diagnostics.wiring.analyze_trace", _boom)
    session = DiagnosticsSession.create(
        _run_cfg(tmp_path, level=0),
        episode_index=0,
        bound=_bound(),
    )
    _feed_steps(session, 8)
    assert _finalize(session, success=False) is None


def test_record_failure_returns_none(tmp_path: Path) -> None:
    session = DiagnosticsSession.create(
        _run_cfg(tmp_path, level=0),
        episode_index=0,
        bound=_bound(),
    )
    session.record_step(
        t=0,
        eef_pos=(1.0, 2.0),  # not a vec3 — recorder raises
        action=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        in_contact=False,
        force_n=0.0,
        impulse_ns=0.0,
    )
    session.finalize(success=False, success_step=None, contact_steps=0)
    assert session.episode_diagnostics() is None


def test_create_failure_disables(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("make_probe boom")

    monkeypatch.setattr("dynisafety.diagnostics.wiring.make_probe", _boom)
    session = DiagnosticsSession.create(
        _run_cfg(tmp_path, level=1),
        episode_index=0,
        bound=_bound(),
        controller=_l1_ctrl(),
    )
    assert session.enabled is False
    _feed_steps(session, 4)
    assert _finalize(session, success=False) is None


def test_warnings_are_one_shot(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    session = DiagnosticsSession.create(
        _run_cfg(tmp_path, level=1),
        episode_index=0,
        bound=_bound(),
        controller=_l1_ctrl(),
    )
    session._probe = _BoomProbe()  # noqa: SLF001
    with caplog.at_level("WARNING", logger="dynisafety.diagnostics.wiring"):
        _feed_steps(session, 6)
    shots = [r for r in caplog.records if "probe.snapshot failed" in r.getMessage()]
    assert len(shots) == 1


def test_fake_rollout_keeps_caller_state_on_degrade(tmp_path: Path) -> None:
    """Sidecar failures must not rewrite the caller's action / success flag."""
    blocked = _expected_run_dir(tmp_path, level=0)
    blocked.parent.mkdir(parents=True, exist_ok=True)
    blocked.write_text("blocked\n", encoding="utf-8")
    session = DiagnosticsSession.create(
        _run_cfg(tmp_path, level=0),
        episode_index=0,
        bound=_bound(),
    )
    session._probe = _BoomProbe()  # noqa: SLF001
    success = False
    action = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    eef = [0.3, 0.0, 0.7]
    stashed = copy_vec(eef)
    eef[0] = -1.0
    session.record_step(
        t=0,
        eef_pos=stashed,
        action=action,
        in_contact=False,
        force_n=0.0,
        impulse_ns=0.0,
    )
    session.finalize(success=success, success_step=None, contact_steps=0)
    assert success is False
    assert action == [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    assert eef[0] == -1.0
    assert session.episode_diagnostics() is not None


# ---------------------------------------------------------------------------
# Isolation / source contracts
# ---------------------------------------------------------------------------


def test_record_step_forwards_secondary_eef(tmp_path: Path) -> None:
    """Dual-arm backends must get the second arm into the trace."""
    session = DiagnosticsSession.create(
        _run_cfg(tmp_path, level=1, backend="robotwin"),
        episode_index=0,
        bound=_bound(),
        controller=_l1_ctrl(),
    )
    session.record_step(
        t=0,
        eef_pos=(0.1, 0.0, 0.8),
        eef_pos_secondary=(-0.1, 0.0, 0.8),
        action=[0.0] * 7,
        in_contact=False,
        force_n=0.0,
        impulse_ns=0.0,
    )
    assert session._recorder is not None  # noqa: SLF001
    trace = session._recorder.finalize(  # noqa: SLF001
        t_hazard_end=None, disturbance_triggered=True, hazard_persists=True
    )
    assert trace.steps[0].eef_pos_secondary == (-0.1, 0.0, 0.8)


def test_record_step_ignores_unknown_fields(tmp_path: Path) -> None:
    """Unknown keywords must degrade, not abort the episode.

    A ``TypeError`` for an unexpected keyword fires at the call site, outside
    the session's own ``try``, so the rollout's except would turn it into
    ``termination="error"``. Diagnostics must never cost an episode.
    """
    session = DiagnosticsSession.create(
        _run_cfg(tmp_path, level=1),
        episode_index=0,
        bound=_bound(),
        controller=_l1_ctrl(),
    )
    session.record_step(
        t=0,
        eef_pos=(0.1, 0.0, 0.8),
        action=[0.0] * 7,
        in_contact=False,
        force_n=0.0,
        impulse_ns=0.0,
        some_future_field=123,
    )
    assert session.enabled
    diag = _finalize(session)
    assert diag is not None


def test_session_consumes_no_rng(tmp_path: Path) -> None:
    """Determinism red line (docs/user/behavior_diagnostics.md §0.2).

    Toggling diagnostics must not shift any rollout that draws from the global
    RNG streams, so the whole capture path has to leave both the legacy and the
    Generator-backed states byte-identical.
    """
    import random

    import numpy as np

    session = DiagnosticsSession.create(
        _run_cfg(tmp_path, level=1),
        episode_index=0,
        bound=_bound(),
        controller=_l1_ctrl(),
    )
    assert session.enabled

    np.random.seed(1234)
    random.seed(1234)
    np_before = np.random.get_state()
    py_before = random.getstate()

    _feed_steps(session, 40, contact_at=frozenset({30}))
    diag = _finalize(session, success=True, success_step=38, first_contact_step=30, contact_steps=1)
    assert diag is not None

    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    assert np.array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    assert random.getstate() == py_before


def test_session_does_not_mutate_inputs(tmp_path: Path) -> None:
    """The capture path is read-only: caller buffers must survive untouched."""
    import numpy as np

    session = DiagnosticsSession.create(
        _run_cfg(tmp_path, level=1),
        episode_index=0,
        bound=_bound(),
        controller=_l1_ctrl(),
    )
    eef = np.array([0.1, 0.2, 0.8], dtype=np.float64)
    action = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)
    eef_copy = eef.copy()
    action_copy = action.copy()

    session.record_step(
        t=0, eef_pos=eef, action=action, in_contact=False, force_n=0.0, impulse_ns=0.0
    )

    assert np.array_equal(eef, eef_copy)
    assert np.array_equal(action, action_copy)


def test_wiring_module_does_not_import_backends() -> None:
    banned = ("dynisafety.backends", "mujoco", "sapien", "robosuite", "robocasa", "libero")
    tree = ast.parse(WIRING_SRC.read_text(encoding="utf-8"), filename=str(WIRING_SRC))
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules = [node.module]
        for mod in modules:
            for prefix in banned:
                assert not mod.startswith(prefix), f"wiring.py imports {mod}"


def test_import_diagnostics_does_not_load_simulators_or_backends() -> None:
    script = """
import sys
from dynisafety import diagnostics
_ = diagnostics.DiagnosticsSession
banned_roots = {"mujoco", "sapien", "robosuite", "libero", "robocasa"}
leaked = sorted(
    {
        name
        for name in sys.modules
        if name.split(".")[0] in banned_roots or name.startswith("dynisafety.backends")
    }
)
print("LEAKED:", leaked or "none")
if leaked:
    raise SystemExit(2)
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "LEAKED: none" in proc.stdout


def test_libero_backend_wires_diagnostics_session() -> None:
    src = LIBERO_BACKEND_SRC.read_text(encoding="utf-8")
    assert "DiagnosticsSession.create" in src
    assert "diag_session.record_step" in src
    assert "diag_session.finalize" in src
    assert "diag_session.episode_diagnostics()" in src
    assert "step_eef_pos" in src
    assert "step_eef_quat" in src
    assert "copy_vec(obs[\"robot0_eef_pos\"])" in src


def _robotwin_manager(*, drop_step: int = 12) -> SimpleNamespace:
    """DisturbanceManager-shaped wrapper: ``controller`` + ``dynamic_level``."""
    inner = SimpleNamespace(
        _dropped=True,
        drop_step=drop_step,
        _placed_sim_step=drop_step,
        disturbance_onset_step=drop_step,
        settle_steps=2,
        _settle_end_step=drop_step + 2,
        obstacle_pos=(0.4, 0.0, 0.1),
    )
    return SimpleNamespace(controller=inner, dynamic_level=1)


def test_robotwin_shaped_manager_unwraps_and_records(tmp_path: Path) -> None:
    """Fake RoboTwin: manager wraps the controller; probe must unwrap it."""
    from dynisafety.diagnostics.probes import unwrap_manager
    from dynisafety.diagnostics.reader import read_trace

    manager = _robotwin_manager(drop_step=12)
    inner = manager.controller
    assert unwrap_manager(manager) is inner

    session = DiagnosticsSession.create(
        _run_cfg(tmp_path, level=1, backend="robotwin"),
        episode_index=0,
        bound=_bound(),
        controller=manager,
    )
    assert session.enabled is True
    assert session._probe is not None  # noqa: SLF001
    assert session._probe._ctrl is inner  # noqa: SLF001 — unwrap_manager ran

    left = [0.11, 0.22, 0.33]
    right = [0.41, 0.52, 0.63]
    stashed_left = copy_vec(left)
    stashed_right = copy_vec(right)
    left[0] = 99.0
    right[0] = 99.0
    session.record_step(
        t=0,
        eef_pos=stashed_left,
        eef_quat=None,
        eef_pos_secondary=stashed_right,
        action=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
        in_contact=False,
        force_n=0.0,
        impulse_ns=0.0,
    )
    _feed_steps(session, 23, t0=1)
    diag = _finalize(
        session,
        success=False,
        metrics={"first_contact_step": None, "contact_steps": 0},
    )
    assert diag is not None
    assert diag["disturbance_triggered"] is True
    assert diag["t_onset"] == 12
    assert diag["t_hazard_end"] == 14
    assert stashed_right == (0.41, 0.52, 0.63)

    trace = read_trace(
        _expected_run_dir(tmp_path, level=1, backend="robotwin")
        / "traces"
        / "ep0000.jsonl.gz"
    )
    assert trace.steps[0].eef_pos == (0.11, 0.22, 0.33)
    assert trace.steps[0].eef_pos_secondary == (0.41, 0.52, 0.63)


def test_record_step_forwards_secondary_and_tolerates_unknown() -> None:
    """``record_step`` must forward the second arm and swallow unknown fields.

    ``**kwargs`` is load-bearing: an unexpected keyword raises ``TypeError`` at
    the call site, outside the session's fail-soft ``try``, which would let a
    diagnostics detail abort the episode with ``termination="error"``.
    """
    sig = inspect.signature(DiagnosticsSession.record_step)
    assert "eef_pos_secondary" in sig.parameters
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def test_eef_pos_secondary_saved_on_recorder_trace(tmp_path: Path) -> None:
    """Right-arm xyz survives recorder → writer → reader (session cannot pass it)."""
    from dynisafety.diagnostics.reader import read_trace
    from dynisafety.diagnostics.recorder import StepTraceRecorder
    from dynisafety.diagnostics.schema import TRACE_SCHEMA_VERSION, TraceHeader
    from dynisafety.diagnostics.writer import TraceWriter

    header = TraceHeader(
        trace_schema_version=TRACE_SCHEMA_VERSION,
        run_id="run-x",
        episode_index=0,
        backend="robotwin",
        level=1,
        seed=7,
        disturbance_triggered=False,
        t_onset=None,
        t_hazard_end=None,
        hazard_persists=None,
        action_chunk_len=8,
        action_space="qpos_14d",
        control_freq_hz=20.0,
        stride=1,
        n_steps=0,
    )
    rec = StepTraceRecorder(header, _diag_cfg())
    live_right = [0.41, 0.52, 0.63]
    rec.record(
        t=0,
        eef_pos=(0.11, 0.22, 0.33),
        eef_pos_secondary=live_right,
        action=(0.0,) * 14,
        in_contact=False,
        force_n=0.0,
        impulse_ns=0.0,
    )
    live_right[0] = 99.0
    trace = rec.finalize(
        t_hazard_end=None,
        disturbance_triggered=False,
        hazard_persists=None,
    )
    assert trace.steps[0].eef_pos_secondary == (0.41, 0.52, 0.63)

    rel = TraceWriter(tmp_path, _diag_cfg().trace).write(0, trace)
    assert rel is not None
    loaded = read_trace(tmp_path / rel)
    assert loaded.steps[0].eef_pos == (0.11, 0.22, 0.33)
    assert loaded.steps[0].eef_pos_secondary == (0.41, 0.52, 0.63)


def test_robotwin_must_pass_control_step_first_contact_not_sim_step() -> None:
    """Trap 1: sim-step first_contact systematically over-counts preemptive reaction.

    DisturbanceManager records ``first_collision_step`` on the physical clock
    (dt=1/250). Trace ``t`` and EpisodeMetricsBuilder contact clocks are
    control steps (``take_action_cnt``). Feeding the 250×-inflated sim-step
    value into ``analyze_trace`` makes ``t_react < first_contact_step`` almost
    always true — the silent, fatal over-count this test locks out.

    Profile matches ``test_reaction_after_contact_is_not_preemptive``:
    cruise until t=25, then stop; t_react=26, control-step first contact=25.
    """
    from dynisafety.diagnostics.analyze import analyze_trace
    from dynisafety.diagnostics.schema import (
        TRACE_SCHEMA_VERSION,
        StepRecord,
        Trace,
        TraceHeader,
    )

    cruise = 0.01
    positions = [(0.0, 0.0, 0.0)]
    x = 0.0
    for i in range(50):
        x += cruise if i < 25 else 0.0
        positions.append((x, 0.0, 0.0))
    hits = {25, 26}
    steps = tuple(
        StepRecord(
            t=i,
            eef_pos=pos,
            eef_quat=None,
            base_xy_yaw=None,
            action=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            gripper_cmd=0.0,
            obstacle_pos=None,
            d_eef_obstacle=None,
            in_contact=i in hits,
            force_n=1.0 if i in hits else 0.0,
            impulse_ns=0.0,
            phase=None,
        )
        for i, pos in enumerate(positions)
    )
    header = TraceHeader(
        trace_schema_version=TRACE_SCHEMA_VERSION,
        run_id="trap1",
        episode_index=0,
        backend="robotwin",
        level=1,
        seed=0,
        disturbance_triggered=True,
        t_onset=20,
        t_hazard_end=40,
        hazard_persists=True,
        action_chunk_len=8,
        action_space="delta_ee_7d",
        control_freq_hz=20.0,
        stride=1,
        n_steps=len(steps),
    )
    trace = Trace(header=header, steps=steps)
    cfg = _diag_cfg()
    common = dict(success=False, success_step=None, contact_steps=2, trace_path=None)

    control_first = 25
    sim_first = 25 * 250  # physical-step clock (dt=1/250)

    control_diag = analyze_trace(trace, cfg, first_contact_step=control_first, **common)
    sim_diag = analyze_trace(trace, cfg, first_contact_step=sim_first, **common)

    assert control_diag.reacted is False
    assert control_diag.reaction_post_contact is True
    assert sim_diag.reacted is True
    assert sim_diag.reaction_post_contact is False
    assert control_diag.reacted != sim_diag.reacted
    assert control_diag.reaction_post_contact != sim_diag.reaction_post_contact


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


def test_robotwin_backend_wires_diagnostics_session() -> None:
    src = ROBOTWIN_BACKEND_SRC.read_text(encoding="utf-8")
    assert "DiagnosticsSession.create" in src
    assert "diag_session.record_step" in src
    assert "diag_session.finalize" in src
    assert "diag_session.episode_diagnostics()" in src
    assert "copy_vec" in src
    assert 'dynisafety_manager' in src
    # EEF + policy images come from the already-fetched obs; no extra camera render.
    assert src.count("task_env.get_obs()") == 2
    assert "raw_obs=raw_obs" in src
    assert "_copy_endpose_xyz(raw_obs, \"left_endpose\")" in src
    assert "_copy_endpose_xyz(raw_obs, \"right_endpose\")" in src

    tree = ast.parse(src)
    create_controller = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "create":
            if isinstance(func.value, ast.Name) and func.value.id == "DiagnosticsSession":
                for kw in node.keywords:
                    if kw.arg == "controller":
                        create_controller = ast.unparse(kw.value)
    assert create_controller is not None
    assert "dynisafety_manager" in create_controller
    assert "task_env" in create_controller

    finalize_kw = _diag_session_call_kwargs(src, "finalize")
    assert finalize_kw["first_contact_step"] == "first_contact_control_step"
    assert finalize_kw["contact_steps"] == "contact_control_steps"
    assert finalize_kw.get("metrics") is None

    record_kw = _diag_session_call_kwargs(src, "record_step")
    # Dual-arm: the right arm must reach the trace even though the analyzer
    # only scores the left one.
    assert record_kw["eef_pos_secondary"] == "step_eef_pos_secondary"
