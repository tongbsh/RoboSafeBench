"""Fake-only tests for RoboCasa diagnostics rollout wiring (no simulator)."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOCASA_BACKEND_SRC = (
    REPO_ROOT / "src" / "dynisafety" / "backends" / "robocasa" / "backend.py"
)


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
            min_latency_chunks=1,
            min_baseline_samples=5,
            min_baseline_speed_frac=0.4,
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
    run_id: object = "run-x",
    level: int = 0,
    backend: str = "robocasa",
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
        suite="robocasa_nav",
        level=level,
        seed=seed,
        run_id=run_id,  # type: ignore[arg-type]
        diagnostics=diag,  # type: ignore[arg-type]
        output=output,
        runtime=runtime,
    )


def _bound() -> SimpleNamespace:
    return SimpleNamespace(chunk_size=8, action_space="hybrid_mobile_12d")


def _l2_pingpong_ctrl(*, onset_step: int = 10) -> SimpleNamespace:
    """RoboCasa L2 after trigger: infinite dest-apron ping-pong.

    Never ``released`` / ``blocked``. Probe must keep ``hazard_end_step=None``.
    Do not invent an end step so Recovery can take a numeric value.
    """
    return SimpleNamespace(
        _spawned=True,
        onset_step=onset_step,
        disturbance_onset_step=onset_step,
        released=False,
        _released=False,
        blocked=False,
        move_start_step=0,
        move_end_step=8,
        _one_way_steps=8,
        obstacle_pos=(1.0, 0.5, 0.1),
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


def _rollout_task_loop(tree: ast.AST) -> ast.For:
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "rollout":
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.For) and ast.unparse(child.target) == "task_step":
                return child
    raise AssertionError("rollout for task_step loop not found")


def _assign_lineno(loop: ast.AST, name: str) -> int:
    for node in ast.walk(loop):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if ast.unparse(target) == name:
                    return int(node.lineno)
    raise AssertionError(f"assignment to {name!r} not found in rollout loop")


def _call_linenos(loop: ast.AST, func_name: str) -> list[int]:
    out: list[int] = []
    for node in ast.walk(loop):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == func_name:
            out.append(int(node.lineno))
    return out


# ---------------------------------------------------------------------------
# Source contracts
# ---------------------------------------------------------------------------


def test_robocasa_backend_wires_diagnostics_session() -> None:
    src = ROBOCASA_BACKEND_SRC.read_text(encoding="utf-8")
    assert "DiagnosticsSession.create" in src
    assert "diag_session.record_step" in src
    assert "diag_session.finalize" in src
    assert "diag_session.episode_diagnostics()" in src
    assert "copy_vec" in src
    assert "copy_vec(eef_pos_from_obs(obs, inner))" in src
    assert "_latched_base_xy_yaw" in src

    record_kw = _diag_session_call_kwargs(src, "record_step")
    assert record_kw["eef_pos"] == "step_eef_pos"
    assert record_kw["eef_quat"] == "step_eef_quat"
    assert record_kw["base_xy_yaw"] == "step_base_xy_yaw"
    assert record_kw["action"] == "action"
    assert record_kw["in_contact"] == "contact.in_contact"

    finalize_kw = _diag_session_call_kwargs(src, "finalize")
    assert finalize_kw["metrics"] == "metrics"
    assert finalize_kw.get("first_contact_step") is None
    assert finalize_kw.get("contact_steps") is None

    tree = ast.parse(src)
    create_kw: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "create":
            if isinstance(func.value, ast.Name) and func.value.id == "DiagnosticsSession":
                create_kw = {kw.arg: ast.unparse(kw.value) for kw in node.keywords if kw.arg}
    assert create_kw.get("controller") == "controller"
    assert create_kw.get("bound") == "bound"


def test_eef_and_base_latched_before_inner_step() -> None:
    """EEF and chassis must be copied before ``inner.step`` (LIBERO pitfall).

    ``record_step`` runs after contact, when ``obs`` is already t+1. Re-reading
    EEF / base there would shift the pose by one action and break every axis.
    """
    src = ROBOCASA_BACKEND_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    loop = _rollout_task_loop(tree)
    step_lines = _call_linenos(loop, "inner.step")
    assert step_lines == [min(step_lines)], "diagnostics must not add a second inner.step"
    step_lineno = step_lines[0]

    eef_lineno = _assign_lineno(loop, "step_eef_pos")
    base_lineno = _assign_lineno(loop, "step_base_xy_yaw")
    assert eef_lineno < step_lineno
    assert base_lineno < step_lineno

    banned_after = {
        "eef_pos_from_obs",
        "read_base_xy_yaw",
        "_latched_base_xy_yaw",
    }
    after: list[str] = []
    for node in ast.walk(loop):
        if not isinstance(node, ast.Call) or int(node.lineno) <= step_lineno:
            continue
        name = ast.unparse(node.func)
        if name in banned_after:
            after.append(f"{name}:{node.lineno}")
    assert after == [], f"post-step pose reads would desync the trace: {after}"


# ---------------------------------------------------------------------------
# Fake rollout — base_xy_yaw in the trace
# ---------------------------------------------------------------------------


def test_fake_rollout_persists_latched_base_xy_yaw(tmp_path: Path) -> None:
    """Mimic the RoboCasa loop: latch EEF+base, then mutate the live buffers."""
    from dynisafety.diagnostics.reader import read_trace

    session = DiagnosticsSession.create(
        _run_cfg(tmp_path, level=0),
        episode_index=0,
        bound=_bound(),
        controller=None,
    )
    assert session.enabled is True

    live_eef = [0.10, 0.20, 0.80]
    live_base = [1.50, 2.50, 0.40]
    for t in range(4):
        step_eef_pos = copy_vec(live_eef)
        step_base_xy_yaw = copy_vec(live_base)
        live_eef[0] += 0.50
        live_base[0] += 0.30
        session.record_step(
            t=t,
            eef_pos=step_eef_pos,
            eef_quat=None,
            base_xy_yaw=step_base_xy_yaw,
            action=[0.0] * 12,
            in_contact=False,
            force_n=0.0,
            impulse_ns=0.0,
        )
    session.finalize(success=False, success_step=None, contact_steps=0)
    diag = session.episode_diagnostics()
    assert diag is not None
    BehaviorDiagnostics.from_json_dict(diag)

    trace = read_trace(
        run_dir_layout(
            tmp_path,
            backend="robocasa",
            suite="robocasa_nav",
            level=0,
            run_id="run-x",
        )
        / "traces"
        / "ep0000.jsonl.gz"
    )
    assert trace.steps[0].eef_pos == (0.10, 0.20, 0.80)
    assert trace.steps[0].base_xy_yaw == (1.50, 2.50, 0.40)
    assert trace.steps[1].base_xy_yaw == (1.80, 2.50, 0.40)
    assert live_eef[0] == pytest.approx(2.10)
    assert live_base[0] == pytest.approx(2.70)


# ---------------------------------------------------------------------------
# L2 n/a Recovery — intentional, do not "fix"
# ---------------------------------------------------------------------------


def test_robocasa_l2_hazard_end_none_recovery_not_applicable(tmp_path: Path) -> None:
    """Intentional: RoboCasa L2 has no hazard-end. Do not invent ``t_hazard_end``.

    The dest-apron box ping-pongs for the rest of a clean episode (never
    ``released`` / ``blocked``). Probe keeps ``hazard_end_step=None``, so the
    Recovery axis is **n/a**. Tables must render n/a, never 0.0.

    The trajectory below cruises, then stops, then continues long enough that
    Recovery *would* enter the denominator if someone fabricated an end step.
    The n/a result must come from ``t_hazard_end is None``, not a short horizon.
    """
    onset = 10
    session = DiagnosticsSession.create(
        _run_cfg(tmp_path, level=2),
        episode_index=0,
        bound=_bound(),
        controller=_l2_pingpong_ctrl(onset_step=onset),
    )
    assert session.enabled is True
    assert session._probe is not None  # noqa: SLF001

    x = 0.0
    for t in range(80):
        session.record_step(
            t=t,
            eef_pos=(x, 0.0, 0.8),
            eef_quat=None,
            base_xy_yaw=(x, 0.1, 0.0),
            action=[0.0] * 12,
            in_contact=False,
            force_n=0.0,
            impulse_ns=0.0,
        )
        if t < 20:
            x += 0.01

    session.finalize(
        success=False,
        success_step=None,
        metrics={"first_contact_step": None, "contact_steps": 0},
    )
    diag = session.episode_diagnostics()
    assert diag is not None
    parsed = BehaviorDiagnostics.from_json_dict(diag)
    assert parsed.disturbance_triggered is True
    assert parsed.t_onset == onset
    assert parsed.t_hazard_end is None
    assert parsed.recovery_applicable is False
    assert parsed.stopped is True


# ---------------------------------------------------------------------------
# Chassis read fail-soft
# ---------------------------------------------------------------------------


def test_base_xy_yaw_read_failure_degrades_to_none() -> None:
    from dynisafety.backends.robocasa.backend import _latched_base_xy_yaw

    class _BoomInner:
        @property
        def sim(self) -> object:
            raise RuntimeError("sim boom")

    assert _latched_base_xy_yaw(_BoomInner()) is None
    assert _latched_base_xy_yaw(None) is None


def test_latched_base_xy_yaw_packs_xy_and_yaw(monkeypatch: pytest.MonkeyPatch) -> None:
    import dynisafety.backends.robocasa.env as env_mod
    from dynisafety.backends.robocasa.backend import _latched_base_xy_yaw

    def _fake_read(_inner: object) -> tuple[np.ndarray, float]:
        return np.array([1.25, 3.50], dtype=np.float64), 0.75

    monkeypatch.setattr(env_mod, "read_base_xy_yaw", _fake_read)
    assert _latched_base_xy_yaw(object()) == (1.25, 3.50, 0.75)


# ---------------------------------------------------------------------------
# Degradation — never raises
# ---------------------------------------------------------------------------


class _BoomProbe:
    def snapshot(self, task_step: int) -> object:
        raise RuntimeError(f"snapshot boom at t={task_step}")


def test_probe_snapshot_raises_does_not_escape(tmp_path: Path) -> None:
    session = DiagnosticsSession.create(
        _run_cfg(tmp_path, level=2),
        episode_index=0,
        bound=_bound(),
        controller=_l2_pingpong_ctrl(),
    )
    session._probe = _BoomProbe()  # noqa: SLF001
    action = [0.0] * 12
    session.record_step(
        t=0,
        eef_pos=(0.1, 0.0, 0.8),
        base_xy_yaw=(1.0, 2.0, 0.0),
        action=action,
        in_contact=False,
        force_n=0.0,
        impulse_ns=0.0,
    )
    session.finalize(success=False, success_step=None, contact_steps=0)
    diag = session.episode_diagnostics()
    assert diag is not None
    BehaviorDiagnostics.from_json_dict(diag)
    assert action == [0.0] * 12


def test_unwritable_run_dir_still_analyzes(tmp_path: Path) -> None:
    """Root ignores chmod, so occupy the run path with a file instead."""
    blocked = run_dir_layout(
        tmp_path, backend="robocasa", suite="robocasa_nav", level=0, run_id="run-x"
    )
    blocked.parent.mkdir(parents=True, exist_ok=True)
    blocked.write_text("not-a-directory\n", encoding="utf-8")
    session = DiagnosticsSession.create(
        _run_cfg(tmp_path, level=0),
        episode_index=0,
        bound=_bound(),
    )
    assert session.enabled is True
    session.record_step(
        t=0,
        eef_pos=(0.1, 0.0, 0.8),
        base_xy_yaw=(1.0, 2.0, 0.3),
        action=[0.0] * 12,
        in_contact=False,
        force_n=0.0,
        impulse_ns=0.0,
    )
    session.finalize(success=False, success_step=None, contact_steps=0)
    diag = session.episode_diagnostics()
    assert diag is not None
    assert diag["trace_path"] is None
    assert blocked.is_file()
