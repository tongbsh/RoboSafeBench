"""Unit tests for AEGIS CBF-QP core (Prompt A0, CPU_ONLY)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from dynisafety.policies.safety.aegis import (
    DEFAULT_AEGIS_CONFIG_PATH,
    AegisActionFilter,
    EllipsoidState,
    SolverFailurePolicy,
    aegis_config_from_mapping,
    aegis_config_with_enabled,
    load_aegis_config,
)
from dynisafety.policies.safety.aegis.cbf_qp import compute_h_ij, solve_cbf_qp

REPO_ROOT = Path(__file__).resolve().parents[1]
YAML_PATH = REPO_ROOT / "configs" / "safety" / "aegis.yaml"

cvxpy = pytest.importorskip("cvxpy")


def _identity_rot() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def _eef(*, pos: np.ndarray, axes: np.ndarray | None = None) -> EllipsoidState:
    return EllipsoidState(
        position=np.asarray(pos, dtype=np.float64),
        axes_diag=np.asarray(axes if axes is not None else [0.06, 0.12, 0.11]),
        rotation=_identity_rot(),
    )


def _obstacle(*, pos: np.ndarray, axes: np.ndarray | None = None) -> EllipsoidState:
    return EllipsoidState(
        position=np.asarray(pos, dtype=np.float64),
        axes_diag=np.asarray(axes if axes is not None else [0.08, 0.08, 0.10]),
        rotation=_identity_rot(),
    )


def _enabled_cfg(**overrides: object):
    raw = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    raw["aegis"]["enabled"] = True
    for k, v in overrides.items():
        raw["aegis"][k] = v
    return aegis_config_from_mapping(raw)


# --- config -----------------------------------------------------------------


def test_default_yaml_disabled_and_path():
    assert DEFAULT_AEGIS_CONFIG_PATH.is_file()
    cfg = load_aegis_config()
    assert cfg.enabled is False
    assert cfg.alpha == 1.0
    assert cfg.max_vel == 1.0
    assert cfg.cbf_h_gain == 10.0
    assert cfg.on_solver_failure is SolverFailurePolicy.FALLBACK_NOMINAL
    assert cfg.gripper_index == 6


def test_config_no_implicit_defaults_missing_key():
    raw = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    del raw["aegis"]["alpha"]
    with pytest.raises(ValueError, match="alpha"):
        aegis_config_from_mapping(raw)


# --- enabled=false bit-identical --------------------------------------------


def test_disabled_passthrough_bit_identical():
    filt = AegisActionFilter.from_yaml(enabled=False)
    nominal = np.array([0.11, -0.22, 0.33, 0.01, -0.02, 0.03, -0.75], dtype=np.float64)
    eef = _eef(pos=[0.0, 0.0, 1.0])
    obs = _obstacle(pos=[0.05, 0.0, 1.0])  # would be unsafe if enabled
    out, dbg = filt.filter(nominal, eef, obs)
    assert out is nominal
    assert np.array_equal(out, nominal)
    assert dbg.enabled is False
    assert dbg.intervened is False
    assert dbg.qp_failed is False
    assert dbg.solver_status == "disabled"


def test_disabled_preserves_float32_bits():
    filt = AegisActionFilter.from_yaml(enabled=False)
    nominal = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
    out, _ = filt.filter(nominal, _eef(pos=[0, 0, 1]), _obstacle(pos=[1, 0, 1]))
    assert out.dtype == np.float32
    assert np.array_equal(out, nominal)


# --- constraint inactive / active (synthetic ellipsoids) --------------------


def test_constraint_inactive_keeps_motion_near_nominal():
    """Far obstacle + small away motion → slack >= 0, motion ≈ nominal."""
    cfg = _enabled_cfg()
    filt = AegisActionFilter(cfg)
    eef = _eef(pos=[0.0, 0.0, 1.0])
    obs = _obstacle(pos=[0.5, 0.0, 1.0])  # far along +x
    # Move further away from obstacle (negative x).
    nominal = np.array([-0.05, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)
    out, dbg = filt.filter(nominal, eef, obs)
    assert dbg.enabled is True
    assert dbg.qp_failed is False
    assert dbg.constraint_active is False
    assert dbg.h is not None and dbg.h > 0.0
    assert np.allclose(out[:6], nominal[:6], atol=1e-4, rtol=0.0)
    assert out[6] == nominal[6]


def test_constraint_active_modifies_approach_motion():
    """Near-overlap ellipsoids + approach velocity → constraint binds, motion changes."""
    cfg = _enabled_cfg()
    filt = AegisActionFilter(cfg)
    # Centers close relative to semi-axes → small / negative h.
    eef = _eef(pos=[0.0, 0.0, 1.0], axes=[0.08, 0.08, 0.08])
    obs = _obstacle(pos=[0.10, 0.0, 1.0], axes=[0.08, 0.08, 0.08])
    h = compute_h_ij(
        eef.position,
        eef.axes_diag,
        eef.rotation,
        obs.position,
        obs.axes_diag,
        obs.rotation,
        obs.position - eef.position,
    )
    # Approach along +x toward the obstacle.
    nominal = np.array([0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9], dtype=np.float64)
    out, dbg = filt.filter(nominal, eef, obs)
    assert dbg.qp_failed is False, dbg.solver_status
    assert dbg.constraint_active is True or h < 0.5
    # Filtered translational x should not be a pure unchanged approach.
    assert not np.allclose(out[:3], nominal[:3], atol=1e-5, rtol=0.0)
    assert dbg.intervened is True
    if dbg.filtered_slack is not None:
        assert dbg.filtered_slack >= -1e-5


# --- gripper preservation ---------------------------------------------------


def test_gripper_dimension_unchanged_when_intervening():
    cfg = _enabled_cfg()
    filt = AegisActionFilter(cfg)
    eef = _eef(pos=[0.0, 0.0, 1.0], axes=[0.08, 0.08, 0.08])
    obs = _obstacle(pos=[0.10, 0.0, 1.0], axes=[0.08, 0.08, 0.08])
    gripper = -0.424242
    nominal = np.array([0.08, 0.02, -0.01, 0.01, -0.02, 0.03, gripper], dtype=np.float64)
    out, dbg = filt.filter(nominal, eef, obs)
    assert dbg.qp_failed is False
    assert out[6] == gripper
    assert out[cfg.gripper_index] == nominal[cfg.gripper_index]


# --- solver failure strategy ------------------------------------------------


def test_solver_failure_falls_back_to_nominal(monkeypatch):
    cfg = _enabled_cfg(on_solver_failure="fallback_nominal")
    filt = AegisActionFilter(cfg)
    eef = _eef(pos=[0.0, 0.0, 1.0])
    obs = _obstacle(pos=[0.12, 0.0, 1.0])
    nominal = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25], dtype=np.float64)

    def _boom(*_a, **_k):
        from dynisafety.policies.safety.aegis.types import CbfQpResult

        return CbfQpResult(
            u=None,
            status="mock_fail",
            h=-1.0,
            nominal_slack=-1.0,
            constraint_active=True,
            failed=True,
            a_u_v=np.zeros(3),
            a_u_omega=np.zeros(3),
            a_uz=np.zeros(3),
            mu_row=np.zeros(3),
            u_ref=np.zeros(9),
        )

    monkeypatch.setattr(
        "dynisafety.policies.safety.aegis.filter.solve_cbf_qp",
        _boom,
    )
    out, dbg = filt.filter(nominal, eef, obs)
    assert dbg.qp_failed is True
    assert dbg.fallback_to_nominal is True
    assert np.array_equal(out, nominal)


def test_solver_failure_raise_policy(monkeypatch):
    cfg = _enabled_cfg(on_solver_failure="raise")
    filt = AegisActionFilter(cfg)

    def _boom(*_a, **_k):
        from dynisafety.policies.safety.aegis.types import CbfQpResult

        return CbfQpResult(
            u=None,
            status="mock_fail",
            h=-1.0,
            nominal_slack=-1.0,
            constraint_active=True,
            failed=True,
            a_u_v=np.zeros(3),
            a_u_omega=np.zeros(3),
            a_uz=np.zeros(3),
            mu_row=np.zeros(3),
            u_ref=np.zeros(9),
        )

    monkeypatch.setattr(
        "dynisafety.policies.safety.aegis.filter.solve_cbf_qp",
        _boom,
    )
    with pytest.raises(RuntimeError, match="CBF-QP"):
        filt.filter(
            np.zeros(7),
            _eef(pos=[0, 0, 1]),
            _obstacle(pos=[0.1, 0, 1]),
        )


# --- QP unit (direct) -------------------------------------------------------


def test_solve_cbf_qp_returns_feasible_u():
    cfg = _enabled_cfg()
    eef = _eef(pos=[0.0, 0.0, 1.0])
    obs = _obstacle(pos=[0.25, 0.0, 1.0])
    z = obs.position - eef.position
    nominal = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    result = solve_cbf_qp(nominal, eef, obs, z, cfg)
    assert result.failed is False
    assert result.u is not None
    assert result.u.shape == (9,)


def test_enabled_override_helper():
    base = load_aegis_config()
    assert base.enabled is False
    on = aegis_config_with_enabled(base, True)
    assert on.enabled is True
    assert on.cbf_h_gain == base.cbf_h_gain
