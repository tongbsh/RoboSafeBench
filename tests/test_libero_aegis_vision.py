"""LIBERO AEGIS vision source tests (Prompt A3a) — CPU_ONLY, mock estimator."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml

from dynisafety.backends.libero.safety.aegis_bridge import LiberoAegisSession
from dynisafety.backends.libero.safety.aegis_vision import (
    FixtureVisionEstimator,
    UnavailableVisionEstimator,
    VisionEstimateFailure,
    VisionPromptMode,
    fixture_ellipsoid,
    prompt_from_obstacle_name,
    vision_config_from_mapping,
)
from dynisafety.backends.libero.safety.phases import LiberoOraclePhase
from dynisafety.config.schema import MISSING

cvxpy = pytest.importorskip("cvxpy")

REPO = Path(__file__).resolve().parents[1]
VISION_YAML = REPO / "configs" / "backends" / "libero" / "aegis" / "vision.yaml"
ENABLE_YAML = REPO / "configs" / "backends" / "libero" / "aegis" / "enable.yaml"
LIBERO_YAML = REPO / "configs" / "backends" / "libero.yaml"


def _cfg(level: int = 2) -> MagicMock:
    cfg = MagicMock()
    cfg.level = level
    cfg.disturbance = MISSING
    return cfg


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
    name: str = "moka_pot_obstacle_1",
    pos=(0.12, 0.0, 1.0),
    quat_wxyz=(1.0, 0.0, 0.0, 0.0),
) -> SimpleNamespace:
    body_id = 3
    sim = SimpleNamespace(
        model=SimpleNamespace(
            ngeom=0,
            body_name2id=lambda _n: body_id,
            geom_bodyid=np.zeros(0, dtype=int),
            geom_size=np.zeros((0, 3)),
        ),
        data=SimpleNamespace(),
    )
    sim.data.body_xpos = np.zeros((body_id + 1, 3), dtype=np.float64)
    sim.data.body_xpos[body_id] = np.asarray(pos, dtype=np.float64)
    sim.data.body_xquat = np.zeros((body_id + 1, 4), dtype=np.float64)
    sim.data.body_xquat[body_id] = np.asarray(quat_wxyz, dtype=np.float64)
    obj = SimpleNamespace(root_body="obstacle_root")
    inner = SimpleNamespace(objects_dict={name: obj}, sim=sim)
    return SimpleNamespace(env=inner)


def _ctrl(name: str = "moka_pot_obstacle_1") -> SimpleNamespace:
    return SimpleNamespace(active_obstacle_name=name)


def _vision_block_from_yaml() -> dict:
    raw = yaml.safe_load(VISION_YAML.read_text(encoding="utf-8"))
    return dict(raw["extra"]["libero"]["aegis"]["vision"])


def _vision_extra(
    *,
    enabled: bool = True,
    activate_after: str = "settled_or_done",
    on_failure: str | None = None,
    refresh: str | None = None,
    refresh_every_n: int | None = None,
    text_prompt_mode: str | None = None,
    text_prompt: str | None = None,
    prompt_map: dict[str, str] | None = None,
) -> dict:
    vision = _vision_block_from_yaml()
    if on_failure is not None:
        vision["on_failure"] = on_failure
    if refresh is not None:
        vision["refresh"] = refresh
    if refresh_every_n is not None:
        vision["refresh_every_n"] = refresh_every_n
    if text_prompt_mode is not None:
        vision["text_prompt_mode"] = text_prompt_mode
    if text_prompt is not None:
        vision["text_prompt"] = text_prompt
    if prompt_map is not None:
        vision["prompt_map"] = prompt_map
    return {
        "aegis": {
            "enabled": enabled,
            "activate_after": activate_after,
            "obstacle_source": "vision",
            "config_path": "configs/safety/aegis.yaml",
            "eef_offset_local": [0.0, 0.0, -0.08],
            "eef_axes_diag": [0.06, 0.12, 0.11],
            "obstacle_axes_diag_default": [0.06, 0.06, 0.08],
            "use_geom_size_for_obstacle": False,
            "vision": vision,
        }
    }


def _filter(
    session: LiberoAegisSession,
    *,
    task_step: int = 50,
    phase: LiberoOraclePhase = LiberoOraclePhase.SETTLED_OR_DONE,
    env: SimpleNamespace | None = None,
    ctrl: SimpleNamespace | None = None,
    nominal: np.ndarray | None = None,
) -> np.ndarray:
    if nominal is None:
        nominal = np.array([0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float64)
    return session.maybe_filter_action(
        nominal,
        obs=_obs_dict(),
        env=env if env is not None else _fake_env_with_obstacle(),
        controller=ctrl if ctrl is not None else _ctrl(),
        task_step=task_step,
        probe_phase=phase,
        action_space="delta_ee_7d",
    )


# --- YAML contract ----------------------------------------------------------


def test_vision_yaml_is_overlay_only() -> None:
    assert VISION_YAML.is_file()
    raw = yaml.safe_load(VISION_YAML.read_text(encoding="utf-8"))
    block = raw["extra"]["libero"]["aegis"]
    assert block["obstacle_source"] == "vision"
    assert "enabled" not in block
    vision = block["vision"]
    vision_config_from_mapping(vision)
    assert vision["on_failure"] == "fallback_sim_gt"
    assert vision["refresh"] == "on_settle"
    assert vision["views"] == ["agentview", "backview"]
    assert vision["cluster"] == "dbscan"
    assert vision["dbscan_eps"] == 0.0001
    assert vision["dbscan_min_points"] == 50
    assert vision["prompt_map"]["moka_pot_obstacle_1"] == "blue moka pot"

    enable = yaml.safe_load(ENABLE_YAML.read_text(encoding="utf-8"))
    assert enable["extra"]["libero"]["aegis"]["obstacle_source"] == "sim_gt"

    base = yaml.safe_load(LIBERO_YAML.read_text(encoding="utf-8"))
    libero = (base.get("extra") or {}).get("libero") or {}
    assert not bool((libero.get("aegis") or {}).get("enabled", False))
    assert (libero.get("aegis") or {}).get("obstacle_source") != "vision"


def test_prompt_from_obstacle_name() -> None:
    assert prompt_from_obstacle_name("moka_pot_obstacle_1") == "moka pot"
    assert prompt_from_obstacle_name("white_storage_box_obstacle_1") == "white storage box"
    assert prompt_from_obstacle_name(
        "moka_pot_obstacle_1", {"moka_pot_obstacle_1": "blue moka pot"}
    ) == "blue moka pot"
    assert prompt_from_obstacle_name(None) is None


def test_vision_source_requires_vision_block() -> None:
    with pytest.raises(ValueError, match="vision mapping is required"):
        LiberoAegisSession.create(
            cfg=_cfg(2),
            extra_libero={
                "aegis": {
                    "enabled": True,
                    "activate_after": "settled_or_done",
                    "obstacle_source": "vision",
                    "eef_offset_local": [0.0, 0.0, -0.08],
                    "eef_axes_diag": [0.06, 0.12, 0.11],
                    "obstacle_axes_diag_default": [0.06, 0.06, 0.08],
                }
            },
            oracle_enabled=False,
        )


def test_unknown_obstacle_source_rejected() -> None:
    with pytest.raises(ValueError, match="obstacle_source"):
        LiberoAegisSession.create(
            cfg=_cfg(2),
            extra_libero={"aegis": {"enabled": True, "obstacle_source": "lidar"}},
            oracle_enabled=False,
        )


# --- fixture estimator success / cache --------------------------------------


def test_fixture_vision_filters_and_records_source() -> None:
    estimator = FixtureVisionEstimator(ellipsoid=fixture_ellipsoid())
    session = LiberoAegisSession.create(
        cfg=_cfg(2),
        extra_libero=_vision_extra(),
        oracle_enabled=True,
        vision_estimator=estimator,
    )
    out = _filter(session)
    assert out.shape == (7,)
    assert out[6] == 0.5
    extra = session.episode_extra()
    assert extra["aegis_obstacle_source"] == "vision"
    assert extra["aegis_obstacle_source_requested"] == "vision"
    assert extra["aegis_vision_failed"] is False
    assert extra["aegis_vision_calls"] == 1
    assert extra["aegis_vision_last_prompt"] == "blue moka pot"
    assert extra["aegis_filter_calls"] == 1
    assert extra["aegis_last_obstacle_name"] == "moka_pot_obstacle_1"


def test_on_settle_caches_successful_estimate() -> None:
    estimator = FixtureVisionEstimator(ellipsoid=fixture_ellipsoid())
    session = LiberoAegisSession.create(
        cfg=_cfg(2),
        extra_libero=_vision_extra(refresh="on_settle"),
        oracle_enabled=True,
        vision_estimator=estimator,
    )
    _filter(session, task_step=50)
    _filter(session, task_step=51)
    assert estimator.calls == 1
    extra = session.episode_extra()
    assert extra["aegis_vision_calls"] == 1
    assert extra["aegis_vision_cache_hits"] == 1
    assert extra["aegis_filter_calls"] == 2


def test_refresh_always_reestimates() -> None:
    estimator = FixtureVisionEstimator(ellipsoid=fixture_ellipsoid())
    session = LiberoAegisSession.create(
        cfg=_cfg(2),
        extra_libero=_vision_extra(refresh="always"),
        oracle_enabled=True,
        vision_estimator=estimator,
    )
    _filter(session, task_step=50)
    _filter(session, task_step=51)
    assert estimator.calls == 2
    assert session.vision_cache_hits == 0


def test_refresh_every_n_steps() -> None:
    estimator = FixtureVisionEstimator(ellipsoid=fixture_ellipsoid())
    session = LiberoAegisSession.create(
        cfg=_cfg(2),
        extra_libero=_vision_extra(refresh="every_n_steps", refresh_every_n=2),
        oracle_enabled=True,
        vision_estimator=estimator,
    )
    _filter(session, task_step=50)
    _filter(session, task_step=51)
    _filter(session, task_step=52)
    assert estimator.calls == 2
    assert session.vision_cache_hits == 1


def test_prompt_map_and_fixed_prompt() -> None:
    estimator = FixtureVisionEstimator(ellipsoid=fixture_ellipsoid())
    session = LiberoAegisSession.create(
        cfg=_cfg(2),
        extra_libero=_vision_extra(
            prompt_map={"moka_pot_obstacle_1": "blue moka pot"}
        ),
        oracle_enabled=True,
        vision_estimator=estimator,
    )
    _filter(session)
    assert estimator.last_request is not None
    assert estimator.last_request.text_prompt == "blue moka pot"

    estimator2 = FixtureVisionEstimator(ellipsoid=fixture_ellipsoid())
    session2 = LiberoAegisSession.create(
        cfg=_cfg(2),
        extra_libero=_vision_extra(
            text_prompt_mode="fixed", text_prompt="red milk carton"
        ),
        oracle_enabled=True,
        vision_estimator=estimator2,
    )
    _filter(session2)
    assert estimator2.last_request is not None
    assert estimator2.last_request.text_prompt == "red milk carton"
    assert estimator2.last_request.prompt_mode is VisionPromptMode.FIXED


def test_dynamic_phase_does_not_call_vision() -> None:
    estimator = FixtureVisionEstimator(ellipsoid=fixture_ellipsoid())
    session = LiberoAegisSession.create(
        cfg=_cfg(2),
        extra_libero=_vision_extra(),
        oracle_enabled=True,
        vision_estimator=estimator,
    )
    nominal = np.ones(7)
    out = _filter(
        session,
        task_step=15,
        phase=LiberoOraclePhase.ENTERING_OR_ACTIVE,
        nominal=nominal,
    )
    assert np.array_equal(out, nominal)
    assert estimator.calls == 0
    assert session.filter_calls == 0


def test_reset_clears_vision_cache() -> None:
    estimator = FixtureVisionEstimator(ellipsoid=fixture_ellipsoid())
    session = LiberoAegisSession.create(
        cfg=_cfg(2),
        extra_libero=_vision_extra(),
        oracle_enabled=True,
        vision_estimator=estimator,
    )
    _filter(session, task_step=50)
    session.reset()
    _filter(session, task_step=50)
    assert estimator.calls == 2


# --- failure policies -------------------------------------------------------


def test_fallback_sim_gt_uses_mujoco_ellipsoid() -> None:
    estimator = FixtureVisionEstimator(fail_reason=VisionEstimateFailure.NO_BOX)
    session = LiberoAegisSession.create(
        cfg=_cfg(2),
        extra_libero=_vision_extra(on_failure="fallback_sim_gt"),
        oracle_enabled=True,
        vision_estimator=estimator,
    )
    out = _filter(session)
    assert out.shape == (7,)
    extra = session.episode_extra()
    assert extra["aegis_vision_failed"] is True
    assert extra["aegis_obstacle_source"] == "sim_gt"
    assert extra["aegis_obstacle_source_requested"] == "vision"
    assert extra["aegis_vision_last_reason"] == "no_box"
    assert extra["aegis_vision_fallback_count"] == 1
    assert extra["aegis_filter_calls"] == 1
    assert extra["aegis_vision_fail_count"] == 1


def test_skip_filter_does_not_call_qp() -> None:
    estimator = FixtureVisionEstimator(fail_reason=VisionEstimateFailure.EMPTY_CLOUD)
    session = LiberoAegisSession.create(
        cfg=_cfg(2),
        extra_libero=_vision_extra(on_failure="skip_filter"),
        oracle_enabled=True,
        vision_estimator=estimator,
    )
    nominal = np.array([0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float64)
    out = _filter(session, nominal=nominal)
    assert np.array_equal(out, nominal)
    extra = session.episode_extra()
    assert extra["aegis_filter_calls"] == 0
    assert extra["aegis_skipped_vision_failure"] == 1
    assert extra["aegis_vision_failed"] is True
    assert extra["aegis_obstacle_source"] is None
    assert extra["aegis_obstacle_source_requested"] == "vision"


def test_on_failure_raise() -> None:
    estimator = FixtureVisionEstimator(fail_reason=VisionEstimateFailure.FIT_FAILED)
    session = LiberoAegisSession.create(
        cfg=_cfg(2),
        extra_libero=_vision_extra(on_failure="raise"),
        oracle_enabled=True,
        vision_estimator=estimator,
    )
    with pytest.raises(RuntimeError, match="vision obstacle estimate failed"):
        _filter(session)


def test_hold_aliases_skip_filter() -> None:
    estimator = FixtureVisionEstimator(fail_reason=VisionEstimateFailure.NO_BOX)
    session = LiberoAegisSession.create(
        cfg=_cfg(2),
        extra_libero=_vision_extra(on_failure="hold"),
        oracle_enabled=True,
        vision_estimator=estimator,
    )
    nominal = np.ones(7)
    out = _filter(session, nominal=nominal)
    assert np.array_equal(out, nominal)
    assert session.skipped_vision_failure == 1


def test_unavailable_estimator_missing_weights_fallback() -> None:
    extra = _vision_extra(on_failure="fallback_sim_gt")
    extra["aegis"]["vision"]["grounding_dino_checkpoint"] = "does_not_exist.pth"
    extra["aegis"]["vision"]["grounding_dino_config"] = "does_not_exist.py"
    cfg = vision_config_from_mapping(extra["aegis"]["vision"])
    estimator = UnavailableVisionEstimator(cfg, repo_root=REPO)
    session = LiberoAegisSession.create(
        cfg=_cfg(2),
        extra_libero=extra,
        oracle_enabled=True,
        vision_estimator=estimator,
    )
    _filter(session)
    extra_ep = session.episode_extra()
    assert extra_ep["aegis_vision_failed"] is True
    assert extra_ep["aegis_vision_last_reason"] == "missing_weights"
    assert extra_ep["aegis_obstacle_source"] == "sim_gt"
    assert extra_ep["aegis_filter_calls"] == 1


def test_from_llm_unavailable_without_calling_dino() -> None:
    extra = _vision_extra(text_prompt_mode="from_llm", on_failure="skip_filter")
    cfg = vision_config_from_mapping(extra["aegis"]["vision"])
    estimator = UnavailableVisionEstimator(cfg, repo_root=REPO)
    session = LiberoAegisSession.create(
        cfg=_cfg(2),
        extra_libero=extra,
        oracle_enabled=True,
        vision_estimator=estimator,
    )
    _filter(session)
    assert session.last_vision_reason == "from_llm_unavailable"
    assert session.filter_calls == 0


def test_default_placeholder_without_injected_estimator() -> None:
    """No injected estimator: real DINO path cannot see cameras on the fake env."""
    session = LiberoAegisSession.create(
        cfg=_cfg(2),
        extra_libero=_vision_extra(on_failure="fallback_sim_gt"),
        oracle_enabled=True,
    )
    _filter(session)
    extra = session.episode_extra()
    assert extra["aegis_vision_failed"] is True
    assert extra["aegis_vision_last_reason"] in {
        "missing_depth",
        "backend_unavailable",
        "missing_weights",
        "render_failed",
    }
    assert extra["aegis_obstacle_source"] == "sim_gt"
    assert extra["aegis_filter_calls"] == 1


def test_cached_failure_does_not_recall_estimator() -> None:
    estimator = FixtureVisionEstimator(fail_reason=VisionEstimateFailure.NO_BOX)
    session = LiberoAegisSession.create(
        cfg=_cfg(2),
        extra_libero=_vision_extra(on_failure="fallback_sim_gt", refresh="on_settle"),
        oracle_enabled=True,
        vision_estimator=estimator,
    )
    _filter(session, task_step=50)
    _filter(session, task_step=60)
    assert estimator.calls == 1
    assert session.vision_fail_count == 1
    assert session.vision_fallback_count == 2
    assert session.filter_calls == 2
