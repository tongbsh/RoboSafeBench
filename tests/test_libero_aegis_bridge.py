"""LIBERO AEGIS bridge tests (Prompt A1) — CPU_ONLY behavior matrix."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml

from dynisafety.backends.libero.safety.aegis_bridge import (
    LiberoAegisSession,
    aegis_enabled_from_extra,
    build_eef_ellipsoid,
    resolve_eef_axes_diag,
)
from dynisafety.backends.libero.safety.phases import LiberoOraclePhase
from dynisafety.backends.libero.safety.runtime import (
    LiberoOracleSession,
    oracle_enabled_from_extra,
)
from dynisafety.config.schema import MISSING
from dynisafety.policies.base import ActionChunk, Observation
from dynisafety.policies.safety.aegis import EllipsoidState

cvxpy = pytest.importorskip("cvxpy")

REPO = __import__("pathlib").Path(__file__).resolve().parents[1]
ENABLE_YAML = REPO / "configs" / "backends" / "libero" / "aegis" / "enable.yaml"


def _cfg(level: int = 2) -> MagicMock:
    cfg = MagicMock()
    cfg.level = level
    cfg.disturbance = MISSING
    return cfg


def _aegis_extra(
    *,
    enabled: bool = True,
    activate_after: str = "settled_or_done",
) -> dict:
    return {
        "aegis": {
            "enabled": enabled,
            "activate_after": activate_after,
            "obstacle_source": "sim_gt",
            "config_path": "configs/safety/aegis.yaml",
            "eef_offset_local": [0.0, 0.0, -0.08],
            "eef_axes_diag": [0.06, 0.12, 0.11],
            "obstacle_axes_diag_default": [0.06, 0.06, 0.08],
            "use_geom_size_for_obstacle": False,
        }
    }


def _obs_dict(
    *,
    eef_pos=(0.0, 0.0, 1.0),
    eef_quat=(0.0, 0.0, 0.0, 1.0),
) -> dict:
    return {
        "robot0_eef_pos": np.asarray(eef_pos, dtype=np.float64),
        "robot0_eef_quat": np.asarray(eef_quat, dtype=np.float64),
        "robot0_gripper_qpos": np.asarray([-0.7, 0.7], dtype=np.float64),
    }


def _fake_env_with_obstacle(
    *,
    name: str = "box_obstacle_1",
    pos=(0.12, 0.0, 1.0),
    quat_wxyz=(1.0, 0.0, 0.0, 0.0),
) -> MagicMock:
    """Minimal MuJoCo-like env: objects_dict + body xpos/xquat."""
    body_id = 3
    sim = SimpleNamespace(
        model=SimpleNamespace(
            ngeom=0,
            body_name2id=lambda _n: body_id,
            geom_bodyid=np.zeros(0, dtype=int),
            geom_size=np.zeros((0, 3)),
        ),
        data=SimpleNamespace(
            body_xpos={body_id: np.asarray(pos, dtype=np.float64)},
            body_xquat={body_id: np.asarray(quat_wxyz, dtype=np.float64)},
        ),
    )
    # body_xpos/xquat accessed as arrays in real mujoco; emulate indexing
    sim.data.body_xpos = np.zeros((body_id + 1, 3), dtype=np.float64)
    sim.data.body_xpos[body_id] = np.asarray(pos, dtype=np.float64)
    sim.data.body_xquat = np.zeros((body_id + 1, 4), dtype=np.float64)
    sim.data.body_xquat[body_id] = np.asarray(quat_wxyz, dtype=np.float64)

    obj = SimpleNamespace(root_body="obstacle_root")
    inner = SimpleNamespace(objects_dict={name: obj}, sim=sim)
    env = SimpleNamespace(env=inner)
    return env


def _l2_controller(*, released: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        move_start_step=10,
        move_end_step=30,
        _settle_end_step=45,
        released=released,
        active_obstacle_name="box_obstacle_1",
    )


# --- enable defaults --------------------------------------------------------


def test_aegis_disabled_by_default() -> None:
    assert aegis_enabled_from_extra({}) is False
    assert aegis_enabled_from_extra({"aegis": {"enabled": False}}) is False
    assert aegis_enabled_from_extra({"aegis": {"enabled": True}}) is True
    # Base libero.yaml must not flip aegis on (overlay-only enable).
    base = yaml.safe_load((REPO / "configs" / "backends" / "libero.yaml").read_text())
    libero = (base.get("extra") or {}).get("libero") or {}
    assert not bool((libero.get("aegis") or {}).get("enabled", False))


def test_enable_yaml_overlay_keys() -> None:
    assert ENABLE_YAML.is_file()
    raw = yaml.safe_load(ENABLE_YAML.read_text(encoding="utf-8"))
    block = raw["extra"]["libero"]["aegis"]
    assert block["enabled"] is True
    assert block["activate_after"] == "settled_or_done"
    assert block["obstacle_source"] == "sim_gt"
    assert block["eef_axes_diag"] == [0.06, 0.12, 0.11]
    assert block["eef_axes_diag_by_instruction_substr"]["milk"] == [0.06, 0.12, 0.2]


def test_eef_axes_follow_official_instruction_substrings() -> None:
    default = np.array([0.06, 0.12, 0.11])
    tall = np.array([0.06, 0.12, 0.2])
    overrides = (
        ("orange juice", tall),
        ("milk", tall),
        ("alphabet soup", tall),
    )
    assert np.allclose(
        resolve_eef_axes_diag(default, overrides, "pick up the milk and place it in the basket"),
        tall,
    )
    assert np.allclose(
        resolve_eef_axes_diag(default, overrides, "pick up the black bowl between the plate"),
        default,
    )


# --- behavior matrix --------------------------------------------------------


def test_matrix_both_off_matches_raw_vla() -> None:
    """两关 → 与未合入前 VLA 一致（session disabled, action untouched）。"""
    assert oracle_enabled_from_extra({}, level=2) is False
    assert aegis_enabled_from_extra({}) is False
    session = LiberoAegisSession.create(
        cfg=_cfg(2), extra_libero={}, oracle_enabled=False
    )
    assert session.enabled is False
    nominal = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)
    out = session.maybe_filter_action(
        nominal,
        obs=_obs_dict(),
        env=_fake_env_with_obstacle(),
        controller=_l2_controller(),
        task_step=50,
    )
    assert out is nominal
    extra = session.episode_extra()
    assert extra["aegis_enabled"] is False
    assert extra["aegis_intervened_steps"] == 0
    assert extra["aegis_qp_fail_count"] == 0
    assert extra["aegis_obstacle_source"] is None


def test_matrix_oracle_only_pause_unchanged() -> None:
    """只 oracle → 与 Prompt 11 LiberoOracleSession 一致（AEGIS 不介入）。"""
    extra_libero = {"oracle": {"enabled": True}}
    assert oracle_enabled_from_extra(extra_libero, level=2) is True
    assert aegis_enabled_from_extra(extra_libero) is False

    base = MagicMock()
    base.act.return_value = ActionChunk(
        actions=np.array([[0.1, 0, 0, 0, 0, 0, 0.5]], dtype=np.float64),
        space="delta_ee_7d",
    )
    from dynisafety.policies.safety import wrap_policy
    from dynisafety.policies.base import EpisodeContext

    wrapped = wrap_policy(base, action_space="delta_ee_7d", enabled=True)
    wrapped.reset(
        EpisodeContext(
            instruction="t",
            task_id=0,
            task_name="t",
            episode_index=0,
            seed=0,
            level=2,
            backend="libero",
            suite="s",
            extra={},
        )
    )
    oracle = LiberoOracleSession(
        enabled=True, level=2, settle_steps=15, policy=wrapped
    )
    aegis = LiberoAegisSession.create(
        cfg=_cfg(2), extra_libero=extra_libero, oracle_enabled=True
    )
    ctrl = _l2_controller()
    # Active window: oracle clears queue / pauses; aegis must not filter.
    o = oracle.inject_observation(
        Observation(
            images={"primary": np.zeros((2, 2, 3), dtype=np.uint8)},
            state=None,
            instruction="t",
            step=15,
            extra={},
        ),
        controller=ctrl,
        task_step=15,
    )
    assert oracle.should_clear_action_queue(o) is True
    assert aegis.enabled is False
    nominal = np.zeros(7)
    assert (
        aegis.maybe_filter_action(
            nominal, obs=_obs_dict(), env=_fake_env_with_obstacle(),
            controller=ctrl, task_step=15,
        )
        is nominal
    )


def test_matrix_oracle_plus_aegis_filters_only_after_settle() -> None:
    """oracle+aegis：动态相不滤；settled_or_done 后调用 filter。"""
    extra = _aegis_extra(enabled=True)
    extra["oracle"] = {"enabled": True}
    session = LiberoAegisSession.create(
        cfg=_cfg(2), extra_libero=extra, oracle_enabled=True
    )
    assert session.enabled is True
    env = _fake_env_with_obstacle()
    ctrl = _l2_controller()
    nominal = np.array([0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float64)

    # Dynamic phase (task_step=15): must pass through without filter call.
    out_dyn = session.maybe_filter_action(
        nominal,
        obs=_obs_dict(),
        env=env,
        controller=ctrl,
        task_step=15,
        probe_phase=LiberoOraclePhase.ENTERING_OR_ACTIVE,
    )
    assert np.array_equal(out_dyn, nominal)
    assert session.filter_calls == 0

    # Settled: filter may run (and possibly intervene).
    calls = {"n": 0}
    real_filter = session.filter.filter

    def _counting_filter(*args, **kwargs):
        calls["n"] += 1
        return real_filter(*args, **kwargs)

    session.filter.filter = _counting_filter  # type: ignore[method-assign]
    _ = session.maybe_filter_action(
        nominal,
        obs=_obs_dict(),
        env=env,
        controller=ctrl,
        task_step=50,
        probe_phase=LiberoOraclePhase.SETTLED_OR_DONE,
    )
    assert calls["n"] == 1
    assert session.filter_calls == 1
    assert session.should_filter_phase(LiberoOraclePhase.ENTERING_OR_ACTIVE) is False
    assert session.should_filter_phase(LiberoOraclePhase.SETTLED_OR_DONE) is True
    assert session.should_filter_phase(LiberoOraclePhase.WITHDRAWN) is True


def test_matrix_oracle_aegis_l3_withdrawn_is_clearance_noop() -> None:
    """L3 WITHDRAWN is post-hazard eligible but AEGIS no-ops (scene cleared)."""
    extra = _aegis_extra(enabled=True, activate_after="settled_or_done")
    session = LiberoAegisSession.create(
        cfg=_cfg(3), extra_libero=extra, oracle_enabled=True
    )
    assert session.should_filter_phase(LiberoOraclePhase.WITHDRAWN) is True
    nominal = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, -0.7], dtype=np.float64)
    out = session.maybe_filter_action(
        nominal,
        obs=_obs_dict(),
        env=_fake_env_with_obstacle(),
        controller=SimpleNamespace(active_obstacle_name="hand_pusher_1"),
        task_step=80,
        probe_phase=LiberoOraclePhase.WITHDRAWN,
        action_space="delta_ee_7d",
    )
    assert np.array_equal(out, nominal)
    assert session.filter_calls == 0
    assert session.skipped_withdrawn_cleared == 1
    assert "L3" in session.episode_extra()["aegis_note"]


def test_abs_ee_converted_before_cbf() -> None:
    """abs_ee_7d must not be fed to CBF as if it were a world velocity."""
    extra = _aegis_extra(enabled=True, activate_after="always")
    session = LiberoAegisSession.create(
        cfg=_cfg(2), extra_libero=extra, oracle_enabled=False
    )
    env = _fake_env_with_obstacle(pos=(0.20, 0.0, 1.0))
    ctrl = _l2_controller(released=True)
    # Far absolute target ≈ current EEF → near-zero delta → filter should run.
    eef_pos = (0.0, 0.0, 1.0)
    abs_cmd = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -0.7], dtype=np.float64)
    out = session.maybe_filter_action(
        abs_cmd,
        obs=_obs_dict(eef_pos=eef_pos),
        env=env,
        controller=ctrl,
        task_step=50,
        probe_phase=LiberoOraclePhase.SETTLED_OR_DONE,
        action_space="abs_ee_7d",
    )
    assert out.shape == (7,)
    assert session.filter_calls == 1
    # Gripper preserved; absolute pose stays near EEF (not treated as +1m/s etc.).
    assert out[6] == abs_cmd[6]
    assert np.linalg.norm(out[:3] - np.asarray(eef_pos)) < 0.5


def test_ellipsoid_build_error_counted_separately() -> None:
    session = LiberoAegisSession.create(
        cfg=_cfg(2),
        extra_libero=_aegis_extra(enabled=True),
        oracle_enabled=False,
    )
    # objects_dict missing the named obstacle → KeyError path.
    env = SimpleNamespace(env=SimpleNamespace(objects_dict={}, sim=None))
    ctrl = SimpleNamespace(active_obstacle_name="missing_obstacle")
    nominal = np.ones(7)
    out = session.maybe_filter_action(
        nominal,
        obs=_obs_dict(),
        env=env,
        controller=ctrl,
        task_step=1,
        probe_phase=LiberoOraclePhase.SETTLED_OR_DONE,
        action_space="delta_ee_7d",
    )
    assert np.array_equal(out, nominal)
    assert session.skipped_ellipsoid_build_error == 1
    assert session.skipped_no_obstacle == 0
    assert session.last_ellipsoid_build_error is not None


def test_matrix_aegis_only_filters_every_step() -> None:
    """只 aegis → 每步过滤（静态对照；oracle 关时 settled_or_done ≡ always）。"""
    extra = _aegis_extra(enabled=True, activate_after="settled_or_done")
    session = LiberoAegisSession.create(
        cfg=_cfg(2), extra_libero=extra, oracle_enabled=False
    )
    assert session.oracle_enabled is False
    assert session.should_filter_phase(LiberoOraclePhase.INACTIVE) is True
    assert session.should_filter_phase(LiberoOraclePhase.ENTERING_OR_ACTIVE) is True
    assert session.should_filter_phase(LiberoOraclePhase.SETTLED_OR_DONE) is True

    env = _fake_env_with_obstacle()
    ctrl = _l2_controller()
    nominal = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, -0.7], dtype=np.float64)
    _ = session.maybe_filter_action(
        nominal,
        obs=_obs_dict(),
        env=env,
        controller=ctrl,
        task_step=0,
        probe_phase=LiberoOraclePhase.INACTIVE,
    )
    assert session.filter_calls == 1


# --- geometry / extras / gripper --------------------------------------------


def test_eef_ellipsoid_applies_offset() -> None:
    obs = _obs_dict(eef_pos=(1.0, 2.0, 3.0), eef_quat=(0.0, 0.0, 0.0, 1.0))
    ell = build_eef_ellipsoid(
        obs,
        axes_diag=np.array([0.06, 0.12, 0.11]),
        offset_local=np.array([0.0, 0.0, -0.08]),
        ellipsoid_cls=EllipsoidState,
    )
    assert np.allclose(ell.position, [1.0, 2.0, 2.92])


def test_episode_extra_keys_and_source() -> None:
    session = LiberoAegisSession.create(
        cfg=_cfg(1),
        extra_libero=_aegis_extra(enabled=True),
        oracle_enabled=False,
    )
    env = _fake_env_with_obstacle(name="moka_pot_obstacle_1")
    ctrl = SimpleNamespace(active_obstacle_name="moka_pot_obstacle_1")
    nominal = np.array([0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25], dtype=np.float64)
    out = session.maybe_filter_action(
        nominal,
        obs=_obs_dict(),
        env=env,
        controller=ctrl,
        task_step=10,
        probe_phase=LiberoOraclePhase.SETTLED_OR_DONE,
    )
    assert out.shape == (7,)
    assert out[6] == nominal[6]  # gripper never rewritten by A0 filter
    extra = session.episode_extra()
    assert extra["aegis_enabled"] is True
    assert extra["aegis_obstacle_source"] == "sim_gt"
    assert "aegis_intervened_steps" in extra
    assert "aegis_qp_fail_count" in extra
    assert extra["aegis_last_obstacle_name"] == "moka_pot_obstacle_1"


def test_no_obstacle_skips_without_fail() -> None:
    session = LiberoAegisSession.create(
        cfg=_cfg(2),
        extra_libero=_aegis_extra(enabled=True),
        oracle_enabled=False,
    )
    ctrl = SimpleNamespace(active_obstacle_name=None)
    nominal = np.ones(7)
    out = session.maybe_filter_action(
        nominal,
        obs=_obs_dict(),
        env=_fake_env_with_obstacle(),
        controller=ctrl,
        task_step=1,
        probe_phase=LiberoOraclePhase.SETTLED_OR_DONE,
    )
    assert np.array_equal(out, nominal)
    assert session.skipped_no_obstacle == 1
    assert session.filter_calls == 0


def test_create_requires_geometry_when_enabled() -> None:
    with pytest.raises(ValueError, match="eef_offset_local"):
        LiberoAegisSession.create(
            cfg=_cfg(2),
            extra_libero={
                "aegis": {
                    "enabled": True,
                    "activate_after": "settled_or_done",
                    "obstacle_source": "sim_gt",
                }
            },
            oracle_enabled=False,
        )


def test_backend_import_hook_present() -> None:
    """Smoke: backend imports LiberoAegisSession (no sim)."""
    from dynisafety.backends.libero import backend as be

    assert hasattr(be, "LiberoAegisSession")
    src = (REPO / "src/dynisafety/backends/libero/backend.py").read_text(encoding="utf-8")
    assert "aegis_session.enabled" in src
    assert "maybe_filter_action" in src
    assert "action_space_now" in src
    # No global X-VLA-only hard force of use_delta at task_step==0.
    assert "_set_use_delta(handle, use_delta=False)" not in src
    # Disturbance files untouched by this prompt's intent — spot-check no aegis in l1/l2/l3.
    for name in ("l1.py", "l2.py", "l3.py"):
        text = (REPO / "src/dynisafety/backends/libero/disturbance" / name).read_text()
        assert "aegis" not in text.lower()
