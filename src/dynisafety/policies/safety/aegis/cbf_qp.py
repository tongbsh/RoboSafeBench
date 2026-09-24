"""CBF geometry + QP solve extracted from vlsa-aegis ``utils.py`` / ``main_aegis.py``.

Upstream sources (read-only reference):
- ``third_party/vlsa-aegis/main/utils.py``: ``compute_h_ij``, ``compute_h_coeffs_3d``
- ``third_party/vlsa-aegis/main/main_aegis.py``: OSQP block under ``flag_safety_control``
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dynisafety.policies.safety.aegis.config import AegisConfig
from dynisafety.policies.safety.aegis.types import CbfQpResult, EllipsoidState, MotionMode

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover - exercised only in bare envs
    cp = None  # type: ignore[assignment]


def vector_hat(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric hat operator: R^3 -> so(3)."""
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )


def project_matrix(z: np.ndarray) -> np.ndarray:
    """Compute ``(I - z z^T)`` with normalized ``z``."""
    z = np.asarray(z, dtype=np.float64).reshape(3)
    z = z / (np.linalg.norm(z) + 1e-12)
    return np.eye(3) - np.outer(z, z)


def compute_h_ij(
    p_i: np.ndarray,
    q_i_diag: np.ndarray,
    r_i: np.ndarray,
    p_j: np.ndarray,
    q_j_diag: np.ndarray,
    r_j: np.ndarray,
    z_ij: np.ndarray,
) -> float:
    """Ellipsoid CBF value ``h_ij`` (upstream ``compute_h_ij``)."""
    q_i = np.diag(np.asarray(q_i_diag, dtype=np.float64).reshape(3))
    q_j = np.diag(np.asarray(q_j_diag, dtype=np.float64).reshape(3))
    qbar_i = r_i @ q_i @ r_i.T
    qbar_j = r_j @ q_j @ r_j.T
    qbar_i_inv = np.linalg.inv(qbar_i)
    z = np.asarray(z_ij, dtype=np.float64).reshape(3)
    z = z / np.linalg.norm(z)
    term1 = np.linalg.norm(qbar_j @ qbar_i_inv @ z)
    term2 = (p_j - p_i).T @ qbar_i_inv @ z
    denom = np.linalg.norm(qbar_i_inv @ z)
    return float((-term1 + term2 - 1.0) / denom)


def compute_h_coeffs_3d(
    p_i: np.ndarray,
    q_i_diag: np.ndarray,
    r_i: np.ndarray,
    p_j: np.ndarray,
    q_j_diag: np.ndarray,
    r_j: np.ndarray,
    z: np.ndarray,
    *,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """CBF Lie-derivative coefficients (upstream ``compute_h_coeffs_3d``).

    Returns
    -------
    a_v, a_omega, a_uz, h, mu_row
    """
    q_i = np.diag(np.asarray(q_i_diag, dtype=np.float64).reshape(3))
    q_j = np.diag(np.asarray(q_j_diag, dtype=np.float64).reshape(3))
    qbar_i = r_i @ q_i @ r_i.T
    qbar_j = r_j @ q_j @ r_j.T
    qbar_i_inv = np.linalg.inv(qbar_i)
    qbar_i_inv2 = qbar_i_inv @ qbar_i_inv
    qbar_j2 = qbar_j @ qbar_j

    z = np.asarray(z, dtype=np.float64).reshape(3)
    z = z / (np.linalg.norm(z) + eps)
    a_vec = qbar_i_inv @ z
    denom = np.linalg.norm(a_vec) + eps
    b_vec = qbar_j @ a_vec
    term1 = np.linalg.norm(b_vec) + eps
    sigma = term1 * denom + eps
    rho = 1.0 - (p_j - p_i).T @ a_vec + term1

    eta_row = -(1.0 / denom) * (z.T @ qbar_i_inv)
    term_mu_1 = (rho / (denom**3 + eps)) * (z.T @ qbar_i_inv2)
    term_mu_2 = (1.0 / denom) * ((p_j - p_i).T @ qbar_i_inv)
    term_mu_3 = (1.0 / sigma) * (z.T @ qbar_i_inv @ qbar_j2 @ qbar_i_inv)
    mu_row = term_mu_1 + term_mu_2 - term_mu_3

    tmp1 = z.T @ qbar_i_inv2 @ vector_hat(z)
    left_vec = z.T @ qbar_i_inv @ qbar_j2
    ja_vec = vector_hat(a_vec)
    tmp2 = left_vec @ (ja_vec - qbar_i_inv @ vector_hat(z))
    part_a = (p_j - p_i).T @ qbar_i_inv @ vector_hat(z)
    part_b = z.T @ qbar_i_inv @ vector_hat(p_j - p_i)
    tmp3 = part_a + part_b
    zeta_tilde = (
        rho * (1.0 / (denom**3 + eps)) * tmp1
        + (1.0 / sigma) * tmp2
        + (1.0 / denom) * tmp3
    )

    a_v = (eta_row @ r_i).ravel()
    a_omega = np.asarray(zeta_tilde @ r_i, dtype=np.float64).ravel()
    a_uz = (mu_row @ project_matrix(z)).ravel()
    h = compute_h_ij(p_i, q_i_diag, r_i, p_j, q_j_diag, r_j, z)
    return a_v, a_omega, a_uz, float(h), np.asarray(mu_row, dtype=np.float64).ravel()


def _solver_objects(cfg: AegisConfig) -> list[Any]:
    if cp is None:
        raise ImportError(
            "cvxpy is required for AEGIS CBF-QP. Install with: "
            'pip install "cvxpy>=1.4" osqp'
        )
    names = (cfg.solver, *cfg.solver_fallback)
    installed = set(cp.installed_solvers())
    out: list[Any] = []
    for name in names:
        key = name.upper()
        if key in installed:
            out.append(getattr(cp, key))
    return out


def solve_cbf_qp(
    nominal_7d: np.ndarray,
    eef: EllipsoidState,
    obstacle: EllipsoidState,
    z_direction: np.ndarray,
    cfg: AegisConfig,
) -> CbfQpResult:
    """Solve the upstream-style 9-D CBF-QP for one control step.

    Decision variable ``u = [u_v(3), u_omega(3), u_z(3)]``.
    Constraint: ``a_u_v·u_v + a_u_omega·u_omega + a_uz·u_z + cbf_h_gain * h >= 0``.
    """
    if cp is None:
        raise ImportError(
            "cvxpy is required for AEGIS CBF-QP. Install with: "
            'pip install "cvxpy>=1.4" osqp'
        )

    if cfg.motion_mode is MotionMode.TRANSLATIONAL:
        return _solve_cbf_qp_translational(
            nominal_7d, eef, obstacle, z_direction, cfg
        )

    action = np.asarray(nominal_7d, dtype=np.float64).reshape(7)
    r1 = eef.rotation
    z = np.asarray(z_direction, dtype=np.float64).reshape(3)
    z_norm = np.linalg.norm(z)
    if z_norm < 1e-12:
        raise ValueError("z_direction must be non-zero")
    z = z / z_norm

    a_v, a_omega, a_uz, h, mu_row = compute_h_coeffs_3d(
        eef.position,
        eef.axes_diag,
        r1,
        obstacle.position,
        obstacle.axes_diag,
        obstacle.rotation,
        z,
    )
    scale_u = float(cfg.u_to_action_scale)
    scale_a = float(cfg.action_to_u_scale)
    a_u_v = scale_u * a_v
    a_u_omega = scale_u * a_omega

    v_ref = r1.T @ action[:3]
    u_v_ref = scale_a * v_ref
    u_omega_ref = scale_a * action[3:6]
    u_z_nom = float(cfg.cbf_h_gain) * mu_row
    u_ref = np.hstack([u_v_ref, u_omega_ref, u_z_nom])

    nominal_slack = float(
        a_u_v @ u_v_ref
        + a_u_omega @ u_omega_ref
        + a_uz @ u_z_nom
        + float(cfg.cbf_h_gain) * h
    )
    constraint_active = nominal_slack < -float(cfg.slack_active_eps)

    w = np.diag(
        [float(cfg.cost_weight_motion)] * 6 + [float(cfg.cost_weight_z)] * 3
    )
    u = cp.Variable(9)
    objective = cp.Minimize(cp.quad_form(u - u_ref, w))
    constraints = [
        a_u_v @ u[:3] + a_u_omega @ u[3:6] + a_uz @ u[6:] + float(cfg.cbf_h_gain) * h
        >= 0
    ]
    problem = cp.Problem(objective, constraints)

    status = "solver_unavailable"
    solved_u: np.ndarray | None = None
    last_err: str | None = None
    for solver in _solver_objects(cfg):
        try:
            problem.solve(solver=solver, warm_start=False)
            status = str(problem.status)
            if u.value is not None and status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                solved_u = np.asarray(u.value, dtype=np.float64).reshape(9)
                break
            if u.value is not None:
                # Accept a numeric value even if status is atypical (OSQP quirks).
                solved_u = np.asarray(u.value, dtype=np.float64).reshape(9)
                break
        except Exception as exc:  # noqa: BLE001 — surface as qp_failed
            last_err = f"{type(exc).__name__}: {exc}"
            status = f"error:{last_err}"
            continue

    if solved_u is None and not _solver_objects(cfg):
        status = "solver_unavailable"

    failed = solved_u is None
    return CbfQpResult(
        u=solved_u,
        status=status,
        h=h,
        nominal_slack=nominal_slack,
        constraint_active=constraint_active,
        failed=failed,
        a_u_v=np.asarray(a_u_v, dtype=np.float64),
        a_u_omega=np.asarray(a_u_omega, dtype=np.float64),
        a_uz=np.asarray(a_uz, dtype=np.float64),
        mu_row=mu_row,
        u_ref=u_ref,
    )


def _solve_cbf_qp_translational(
    nominal_7d: np.ndarray,
    eef: EllipsoidState,
    obstacle: EllipsoidState,
    z_direction: np.ndarray,
    cfg: AegisConfig,
) -> CbfQpResult:
    """Upstream ``main_aegis_translational.py``: QP over ``[u_v, u_z]`` only."""
    if cp is None:
        raise ImportError(
            "cvxpy is required for AEGIS CBF-QP. Install with: "
            'pip install "cvxpy>=1.4" osqp'
        )

    action = np.asarray(nominal_7d, dtype=np.float64).reshape(7)
    r1 = eef.rotation
    z = np.asarray(z_direction, dtype=np.float64).reshape(3)
    z_norm = np.linalg.norm(z)
    if z_norm < 1e-12:
        raise ValueError("z_direction must be non-zero")
    z = z / z_norm

    a_v, _a_omega, a_uz, h, mu_row = compute_h_coeffs_3d(
        eef.position,
        eef.axes_diag,
        r1,
        obstacle.position,
        obstacle.axes_diag,
        obstacle.rotation,
        z,
    )
    scale_u = float(cfg.u_to_action_scale)
    scale_a = float(cfg.action_to_u_scale)
    a_u_v = scale_u * a_v

    v_ref = r1.T @ action[:3]
    u_v_ref = scale_a * v_ref
    u_z_nom = float(cfg.cbf_h_gain) * mu_row
    u_ref6 = np.hstack([u_v_ref, u_z_nom])
    u_ref = np.hstack([u_v_ref, np.zeros(3, dtype=np.float64), u_z_nom])

    nominal_slack = float(a_u_v @ u_v_ref + a_uz @ u_z_nom + float(cfg.cbf_h_gain) * h)
    constraint_active = nominal_slack < -float(cfg.slack_active_eps)

    w = np.diag([float(cfg.cost_weight_motion)] * 3 + [float(cfg.cost_weight_z)] * 3)
    u = cp.Variable(6)
    objective = cp.Minimize(cp.quad_form(u - u_ref6, w))
    constraints = [a_u_v @ u[:3] + a_uz @ u[3:] + float(cfg.cbf_h_gain) * h >= 0]
    problem = cp.Problem(objective, constraints)

    status = "solver_unavailable"
    solved_u: np.ndarray | None = None
    last_err: str | None = None
    for solver in _solver_objects(cfg):
        try:
            problem.solve(solver=solver, warm_start=False)
            status = str(problem.status)
            if u.value is not None and status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                raw = np.asarray(u.value, dtype=np.float64).reshape(6)
                solved_u = np.hstack([raw[:3], np.zeros(3, dtype=np.float64), raw[3:]])
                break
            if u.value is not None:
                raw = np.asarray(u.value, dtype=np.float64).reshape(6)
                solved_u = np.hstack([raw[:3], np.zeros(3, dtype=np.float64), raw[3:]])
                break
        except Exception as exc:  # noqa: BLE001 — surface as qp_failed
            last_err = f"{type(exc).__name__}: {exc}"
            status = f"error:{last_err}"
            continue

    if solved_u is None and not _solver_objects(cfg):
        status = "solver_unavailable"

    return CbfQpResult(
        u=solved_u,
        status=status,
        h=h,
        nominal_slack=nominal_slack,
        constraint_active=constraint_active,
        failed=solved_u is None,
        a_u_v=np.asarray(a_u_v, dtype=np.float64),
        a_u_omega=np.zeros(3, dtype=np.float64),
        a_uz=np.asarray(a_uz, dtype=np.float64),
        mu_row=mu_row,
        u_ref=u_ref,
    )


def map_u_to_action7(
    u: np.ndarray,
    *,
    rotation: np.ndarray,
    gripper: float,
    cfg: AegisConfig,
) -> np.ndarray:
    """Map QP ``u`` back to a 7D EE action; gripper copied verbatim."""
    u = np.asarray(u, dtype=np.float64).reshape(9)
    out = np.zeros(7, dtype=np.float64)
    scale = float(cfg.u_to_action_scale)
    out[:3] = scale * (rotation @ u[:3])
    out[3:6] = scale * u[3:6]
    gi = int(cfg.gripper_index)
    if gi < 0 or gi >= 7:
        raise ValueError(f"gripper_index out of range for 7D action: {gi}")
    out[gi] = float(gripper)
    return out


def update_z_direction(
    z_direction: np.ndarray,
    u_z: np.ndarray,
    *,
    dt: float,
) -> np.ndarray:
    """Project ``u_z`` onto the tangent plane and integrate (upstream ``z_fixed``)."""
    z = np.asarray(z_direction, dtype=np.float64).reshape(3)
    z = z / (np.linalg.norm(z) + 1e-12)
    u_z = np.asarray(u_z, dtype=np.float64).reshape(3)
    eye = np.eye(3)
    dz = (eye - np.outer(z, z)) @ u_z
    z_next = z + dz * float(dt)
    return z_next / (np.linalg.norm(z_next) + 1e-12)


def default_z_direction(eef: EllipsoidState, obstacle: EllipsoidState) -> np.ndarray:
    """Initialize separating direction as ``(p_obstacle - p_eef)`` (upstream)."""
    delta = obstacle.position - eef.position
    n = np.linalg.norm(delta)
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return (delta / n).astype(np.float64)
