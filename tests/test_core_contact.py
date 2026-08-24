"""Pure-logic tests for ``dynisafety.core.contact`` (no simulator)."""

from __future__ import annotations

import numpy as np
import pytest

from dynisafety.core.contact import (
    is_force_anomaly,
    is_toppled,
    quat_to_matrix,
    reduce_impulse_norms,
    resolve_force_equiv_dt,
    tilt_deg_from_body_quat,
    tilt_deg_from_reference,
)

_NONE_KW = {
    "sim_timestep_s": None,
    "control_freq_hz": None,
    "force_report_window_s": None,
}


def test_resolve_force_equiv_dt_sim() -> None:
    assert resolve_force_equiv_dt(
        "sim",
        sim_timestep_s=0.004,
        control_freq_hz=20.0,
        force_report_window_s=0.05,
    ) == pytest.approx(0.004)


def test_resolve_force_equiv_dt_control() -> None:
    assert resolve_force_equiv_dt(
        "control",
        sim_timestep_s=0.004,
        control_freq_hz=20.0,
        force_report_window_s=0.05,
    ) == pytest.approx(0.05)


def test_resolve_force_equiv_dt_impact_window() -> None:
    assert resolve_force_equiv_dt(
        "impact_window",
        sim_timestep_s=0.004,
        control_freq_hz=20.0,
        force_report_window_s=0.05,
    ) == pytest.approx(0.05)


def test_resolve_force_equiv_dt_blend() -> None:
    expected = float(np.sqrt(0.004 / 25.0))
    assert resolve_force_equiv_dt(
        "blend",
        sim_timestep_s=0.004,
        control_freq_hz=25.0,
        force_report_window_s=0.05,
    ) == pytest.approx(expected)


def test_resolve_force_equiv_dt_numeric_seconds() -> None:
    assert resolve_force_equiv_dt("0.05", **_NONE_KW) == pytest.approx(0.05)
    assert resolve_force_equiv_dt(0.02, **_NONE_KW) == pytest.approx(0.02)


def test_resolve_force_equiv_dt_mode_is_stripped_and_lowercased() -> None:
    assert resolve_force_equiv_dt(
        " SIM ",
        sim_timestep_s=0.004,
        control_freq_hz=None,
        force_report_window_s=None,
    ) == pytest.approx(0.004)


@pytest.mark.parametrize(
    ("mode", "kwargs", "needle"),
    [
        ("sim", {"control_freq_hz": 20.0, "force_report_window_s": 0.05}, "sim_timestep_s"),
        ("control", {"sim_timestep_s": 0.004, "force_report_window_s": 0.05}, "control_freq_hz"),
        (
            "impact_window",
            {"sim_timestep_s": 0.004, "control_freq_hz": 20.0},
            "force_report_window_s",
        ),
        ("blend", {"control_freq_hz": 20.0, "force_report_window_s": 0.05}, "sim_timestep_s"),
        ("blend", {"sim_timestep_s": 0.004, "force_report_window_s": 0.05}, "control_freq_hz"),
    ],
)
def test_resolve_force_equiv_dt_missing_param_raises(
    mode: str, kwargs: dict[str, float], needle: str
) -> None:
    full = {
        "sim_timestep_s": None,
        "control_freq_hz": None,
        "force_report_window_s": None,
        **kwargs,
    }
    with pytest.raises(ValueError, match=needle):
        resolve_force_equiv_dt(mode, **full)


def test_resolve_force_equiv_dt_unknown_mode_lists_supported() -> None:
    with pytest.raises(ValueError, match="sim|control|impact_window|blend") as exc:
        resolve_force_equiv_dt("not-a-mode", **_NONE_KW)
    msg = str(exc.value)
    assert "sim" in msg and "blend" in msg


def test_resolve_force_equiv_dt_non_positive_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        resolve_force_equiv_dt(
            "sim",
            sim_timestep_s=0.0,
            control_freq_hz=20.0,
            force_report_window_s=0.05,
        )
    with pytest.raises(ValueError, match="positive"):
        resolve_force_equiv_dt(-0.01, **_NONE_KW)
    with pytest.raises(ValueError, match="control_freq_hz"):
        resolve_force_equiv_dt(
            "control",
            sim_timestep_s=0.004,
            control_freq_hz=0.0,
            force_report_window_s=0.05,
        )


def test_reduce_impulse_norms_sum_max_empty_invalid() -> None:
    norms = [3.0, 4.0]
    assert reduce_impulse_norms(norms, impulse_reduce="sum") == pytest.approx(7.0)
    assert reduce_impulse_norms(norms, impulse_reduce="max") == pytest.approx(4.0)
    assert reduce_impulse_norms([], impulse_reduce="sum") == 0.0
    assert reduce_impulse_norms([], impulse_reduce="bogus") == 0.0
    with pytest.raises(ValueError, match="impulse_reduce"):
        reduce_impulse_norms(norms, impulse_reduce="mean")


def test_is_toppled_zero_threshold_upright_is_false() -> None:
    assert is_toppled(0.0, threshold_deg=0.0) is False
    assert is_toppled(0.0, threshold_deg=15.0) is False
    assert is_toppled(15.0, threshold_deg=15.0) is False
    assert is_toppled(15.1, threshold_deg=15.0) is True


def test_is_toppled_negative_threshold_raises() -> None:
    with pytest.raises(ValueError, match="threshold_deg"):
        is_toppled(1.0, threshold_deg=-1.0)


def test_tilt_deg_from_reference_pure_yaw_is_zero() -> None:
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    half = np.sqrt(0.5)
    yaw90 = np.array([half, 0.0, 0.0, half])
    assert tilt_deg_from_reference(ref_quat=identity, current_quat=yaw90) == pytest.approx(
        0.0, abs=1e-6
    )


def test_tilt_deg_from_reference_roll_x_90() -> None:
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    half = np.sqrt(0.5)
    roll90 = np.array([half, half, 0.0, 0.0])
    assert tilt_deg_from_reference(ref_quat=identity, current_quat=roll90) == pytest.approx(
        90.0, abs=1e-6
    )


def test_tilt_deg_from_body_quat_world_z_and_relative() -> None:
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    half = np.sqrt(0.5)
    roll90 = np.array([half, half, 0.0, 0.0])
    yaw90 = np.array([half, 0.0, 0.0, half])
    assert tilt_deg_from_body_quat(identity) == pytest.approx(0.0, abs=1e-6)
    assert tilt_deg_from_body_quat(roll90) == pytest.approx(90.0, abs=1e-6)
    assert tilt_deg_from_body_quat(yaw90) == pytest.approx(0.0, abs=1e-6)
    assert tilt_deg_from_body_quat(roll90, ref_quat=identity) == pytest.approx(90.0, abs=1e-6)
    assert tilt_deg_from_body_quat(yaw90, ref_quat=identity) == pytest.approx(90.0, abs=1e-6)


def test_is_force_anomaly_each_threshold() -> None:
    assert (
        is_force_anomaly(
            force_n=10.0,
            impulse_ns=1.0,
            force_threshold_n=5.0,
            impulse_threshold_ns=100.0,
        )
        is True
    )
    assert (
        is_force_anomaly(
            force_n=1.0,
            impulse_ns=10.0,
            force_threshold_n=5.0,
            impulse_threshold_ns=5.0,
        )
        is True
    )
    assert (
        is_force_anomaly(
            force_n=5.0,
            impulse_ns=5.0,
            force_threshold_n=5.0,
            impulse_threshold_ns=5.0,
        )
        is False
    )


def test_quat_to_matrix_near_zero_is_identity() -> None:
    mat = quat_to_matrix(np.array([0.0, 0.0, 0.0, 0.0]))
    assert mat == pytest.approx(np.eye(3))
