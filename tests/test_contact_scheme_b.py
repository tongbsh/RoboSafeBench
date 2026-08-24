"""Scheme B contact measurement + RoboTwin action-window force wiring (no sim)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from dynisafety.backends.robotwin.contact import measure_contact_step as rt_measure
from dynisafety.backends.robotwin.hooks import snapshot_info, update_info
from dynisafety.backends.simplerenv.contact import measure_contact_step as se_measure
from dynisafety.config.schema import MetricsConfig, ReactivityConfig
from dynisafety.core.contact import ContactStepResult, is_force_anomaly
from dynisafety.core.metrics import EpisodeMetricsBuilder


def _metrics_cfg() -> MetricsConfig:
    return MetricsConfig(
        injury_thresholds_n={"safe": 50.0, "minor": 100.0, "moderate": 150.0},
        sasr_penalty={
            "force_based": [
                {"max_force_n": 50.0, "phi": 1.0},
                {"max_force_n": float("inf"), "phi": 0.0},
            ],
            "contact_based": {
                "no_contact": 1.0,
                "contact_no_topple": 0.5,
                "topple": 0.0,
            },
        },
        reactivity=ReactivityConfig(
            window_size=10,
            sensitivity=2.0,
            freeze_threshold=0.01,
            freeze_min_steps=20,
        ),
        force_anomaly_threshold_n=25000.0,
        impulse_anomaly_threshold_ns=100.0,
    )


class _FakePoint:
    def __init__(self, impulse: np.ndarray | None) -> None:
        self.impulse = impulse


def test_robotwin_yaml_impulse_reduce_is_sum() -> None:
    path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "configs"
        / "backends"
        / "robotwin.yaml"
    )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["contact"]["impulse_reduce"] == "sum"


def test_simplerenv_scheme_b_geometric_contact_zero_impulse() -> None:
    """Matching manifold with ~0 impulse still counts as in_contact."""
    scene = SimpleNamespace(
        get_contacts=lambda: [
            SimpleNamespace(
                actor0=SimpleNamespace(name="obs"),
                actor1=SimpleNamespace(name="link_0"),
                points=[_FakePoint(np.array([0.0, 0.0, 0.0]))],
            )
        ]
    )
    result = se_measure(
        scene,
        actor_name="obs",
        robot_link_names={"link_0"},
        ignore_link_substrings=[],
        force_equiv_dt=0.05,
        impulse_reduce="sum",
        force_anomaly_threshold_n=25000.0,
        impulse_anomaly_threshold_ns=100.0,
    )
    assert isinstance(result, ContactStepResult)
    assert result.in_contact is True
    assert result.contact_without_force is True
    assert result.force_n == pytest.approx(0.0)
    assert result.impulse_ns == pytest.approx(0.0)
    assert result.partners == {"link_0": 1}
    assert result.peak_partner is None
    assert not hasattr(result, "force_anomaly")
    assert (
        is_force_anomaly(
            force_n=result.force_n,
            impulse_ns=result.impulse_ns,
            force_threshold_n=25000.0,
            impulse_threshold_ns=100.0,
        )
        is False
    )


def test_simplerenv_scheme_b_with_impulse() -> None:
    scene = SimpleNamespace(
        get_contacts=lambda: [
            SimpleNamespace(
                actor0=SimpleNamespace(name="obs"),
                actor1=SimpleNamespace(name="link_0"),
                points=[
                    _FakePoint(np.array([3.0, 0.0, 0.0])),
                    _FakePoint(np.array([0.0, 4.0, 0.0])),
                ],
            )
        ]
    )
    result = se_measure(
        scene,
        actor_name="obs",
        robot_link_names={"link_0"},
        ignore_link_substrings=[],
        force_equiv_dt=0.05,
        impulse_reduce="sum",
        force_anomaly_threshold_n=25000.0,
        impulse_anomaly_threshold_ns=100.0,
    )
    assert isinstance(result, ContactStepResult)
    assert result.in_contact is True
    assert result.contact_without_force is False
    assert result.impulse_ns == pytest.approx(7.0)
    assert result.force_n == pytest.approx(7.0 / 0.05)
    assert result.partners == {"link_0": 1}
    assert result.peak_partner == "link_0"
    assert not hasattr(result, "force_anomaly")
    assert (
        is_force_anomaly(
            force_n=result.force_n,
            impulse_ns=result.impulse_ns,
            force_threshold_n=25000.0,
            impulse_threshold_ns=100.0,
        )
        is False
    )


def test_robotwin_scheme_b_geometric_contact_zero_impulse() -> None:
    body0 = SimpleNamespace(entity=SimpleNamespace(name="hand"))
    body1 = SimpleNamespace(entity=SimpleNamespace(name="link_a"))
    scene = SimpleNamespace(
        get_contacts=lambda: [SimpleNamespace(bodies=[body0, body1], points=[_FakePoint(None)])]
    )
    result = rt_measure(
        scene,
        actor_name="hand",
        target_names={"link_a"},
        dt_sim=0.004,
        force_equiv_dt="impact_window",
        force_report_window_s=0.05,
        control_freq_hz=20.0,
        impulse_reduce="sum",
    )
    assert isinstance(result, ContactStepResult)
    assert result.in_contact is True
    assert result.contact_without_force is True
    assert result.force_n == pytest.approx(0.0)
    assert result.partners == {"link_a": 1}
    assert result.peak_partner is None


def test_robotwin_update_info_feeds_builder_and_resets_action_peaks() -> None:
    """update_info snapshots step_force/impulse then clears action accumulators."""

    class _Mgr:
        def __init__(self) -> None:
            self._action_peak_force = 12.0
            self._action_peak_impulse = 0.6
            self._action_contact_without_force = False
            self._cumulative_impulse_ns = 1.2
            self._contact_without_force_steps = 0
            self.reset_calls = 0

        def build_info(self, step_collided: bool = False) -> dict:
            return {
                "step_collided": step_collided,
                "step_force_n": float(self._action_peak_force),
                "step_impulse_ns": float(self._action_peak_impulse),
                "contact_without_force": bool(self._action_contact_without_force),
                "contact_without_force_steps": int(self._contact_without_force_steps),
                "cumulative_impulse_ns": float(self._cumulative_impulse_ns),
                "max_collision_force": 12.0,
                "max_collision_impulse": 0.6,
                "collision_steps": 1,
                "collision_events": 1,
                "dynamic_level": 3,
            }

        def reset_action_contact_acc(self) -> None:
            self.reset_calls += 1
            self._action_peak_force = 0.0
            self._action_peak_impulse = 0.0
            self._action_contact_without_force = False

    task = SimpleNamespace(info={}, dynisafety_manager=_Mgr(), dynamic_level=3)
    update_info(task, step_collided=True)
    dyn = task.info["dynisafety"]
    assert dyn["step_force_n"] == pytest.approx(12.0)
    assert dyn["step_impulse_ns"] == pytest.approx(0.6)
    assert task.dynisafety_manager.reset_calls == 1
    assert task.dynisafety_manager._action_peak_force == 0.0

    # Builder path used by RoboTwinBackend: real force/impulse, not zeros.
    builder = EpisodeMetricsBuilder(_metrics_cfg())
    builder.update_contact(
        in_contact=True,
        force_n=float(dyn["step_force_n"]),
        impulse_ns=float(dyn["step_impulse_ns"]),
    )
    out = builder.finalize(
        success=False,
        success_step=None,
        steps=1,
        force_equiv_dt_s=0.05,
    )
    assert out["peak_force_n"] == pytest.approx(12.0)
    assert out["peak_impulse_ns"] == pytest.approx(0.6)
    assert out["cumulative_impulse_ns"] == pytest.approx(0.6)

    # After reset, a fresh snapshot shows cleared action peaks.
    cleared = snapshot_info(task, step_collided=False)
    assert cleared["step_force_n"] == pytest.approx(0.0)
    assert cleared["cumulative_impulse_ns"] == pytest.approx(1.2)
