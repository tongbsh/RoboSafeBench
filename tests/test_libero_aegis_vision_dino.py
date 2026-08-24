"""A3c GroundingDINO geometry + mock-predictor tests (CPU, no DINO weights load)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from dynisafety.backends.libero.safety.aegis_vision import (
    VisionEstimateRequest,
    VisionPromptMode,
    make_default_vision_estimator,
    vision_config_from_mapping,
)
from dynisafety.backends.libero.safety.aegis_vision_dino import GroundingDinoVisionEstimator
from dynisafety.backends.libero.safety.aegis_vision_geometry import (
    CameraFrame,
    dino_box_to_native_xyxy,
    filter_points,
    fit_mvee_ellipsoid,
    unproject_depth_crop,
)
from dynisafety.policies.safety.aegis import EllipsoidState

cvxpy = pytest.importorskip("cvxpy")

REPO = Path(__file__).resolve().parents[1]
VISION_YAML = REPO / "configs" / "backends" / "libero" / "aegis" / "vision.yaml"


def _cfg_from_yaml():
    raw = yaml.safe_load(VISION_YAML.read_text(encoding="utf-8"))
    return vision_config_from_mapping(raw["extra"]["libero"]["aegis"]["vision"])


def test_yaml_parses_a3c_keys() -> None:
    cfg = _cfg_from_yaml()
    assert cfg.device == "cuda"
    assert cfg.flip_rgb_for_dino is True
    assert cfg.views == ("agentview", "backview")
    assert cfg.cluster == "dbscan"
    assert cfg.dbscan_eps == 0.0001
    assert cfg.min_cloud_points >= 1
    assert "default" in cfg.table_bounds


def test_factory_uses_dino_when_files_present() -> None:
    cfg = _cfg_from_yaml()
    est = make_default_vision_estimator(cfg, repo_root=REPO)
    assert isinstance(est, GroundingDinoVisionEstimator)


def test_dino_box_matches_official_astype_int_flip() -> None:
    """Parity with utils.get_point_cloud: scale, astype(int), then w-1-x2."""
    crop = dino_box_to_native_xyxy(
        np.array([0.5, 0.5, 0.2, 0.2]), width=100, height=80, flip_180=True
    )
    assert crop == (39, 31, 59, 47)


def test_unproject_center_pixel() -> None:
    h = w = 5
    depth = np.full((h, w), 2.0)
    k = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])
    t = np.eye(4)
    pts = unproject_depth_crop(
        depth, k, t, xmin=2, xmax=3, ymin=2, ymax=3, v_flip=False
    )
    assert pts.shape == (1, 3)
    assert np.allclose(pts[0], [0.0, 0.0, 2.0], atol=1e-6)


def test_filter_and_mvee_on_blob() -> None:
    rng = np.random.default_rng(0)
    pts = rng.normal(loc=[0.0, 0.0, 1.1], scale=[0.02, 0.03, 0.015], size=(200, 3))
    cfg = _cfg_from_yaml()
    filtered = filter_points(
        pts,
        cfg.table_bounds["default"],
        keep_nearest_frac=0.8,
        cluster="dbscan",
        dbscan_eps=0.0001,
        dbscan_min_points=50,
    )
    assert filtered.shape[0] >= 16
    ell = fit_mvee_ellipsoid(filtered, EllipsoidState)
    assert np.allclose(ell.position, [0.0, 0.0, 1.1], atol=0.08)
    assert np.all(ell.axes_diag > 0)


def _bump_frame(size: int = 48) -> CameraFrame:
    yy, xx = np.mgrid[0:size, 0:size]
    cx = cy = (size - 1) * 0.5
    depth = 1.10 + 0.05 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 40.0)
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[..., 1] = 180
    f = float(size)
    k = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    return CameraFrame(
        rgb=rgb,
        depth=depth,
        K=k,
        T_cam_to_world=np.eye(4),
        camera_name="agentview",
    )


def test_mock_predictor_fits_ellipsoid() -> None:
    cfg = _cfg_from_yaml()
    frame = _bump_frame()

    def predictor(rgb, prompt):
        del rgb
        assert "moka" in prompt
        return np.array([[0.5, 0.5, 0.7, 0.7]], dtype=np.float64), np.array([0.9])

    def loader(request, camera_name):
        del request, camera_name
        return frame

    est = GroundingDinoVisionEstimator(
        cfg, repo_root=REPO, predictor=predictor, frame_loader=loader
    )
    request = VisionEstimateRequest(
        obs={},
        env=None,
        controller=None,
        task_step=50,
        obstacle_name="moka_pot_obstacle_1",
        text_prompt="moka pot",
        prompt_mode=VisionPromptMode.FROM_CONTROLLER_NAME,
        views=("agentview",),
        suite="safelibero_spatial",
    )
    out = est.estimate(request)
    assert out.failed is False, out.reason
    assert out.ellipsoid is not None
    assert out.extra.get("backend") == "grounding_dino"
    assert np.isfinite(out.ellipsoid.position).all()


def test_mock_predictor_no_box() -> None:
    cfg = _cfg_from_yaml()
    frame = _bump_frame()

    def predictor(rgb, prompt):
        del rgb, prompt
        return np.zeros((0, 4)), None

    est = GroundingDinoVisionEstimator(
        cfg,
        repo_root=REPO,
        predictor=predictor,
        frame_loader=lambda *_a, **_k: frame,
    )
    request = VisionEstimateRequest(
        obs={},
        env=None,
        controller=None,
        task_step=0,
        obstacle_name="moka_pot_obstacle_1",
        text_prompt="moka pot",
        prompt_mode=VisionPromptMode.FROM_CONTROLLER_NAME,
        views=("agentview",),
        suite="safelibero_spatial",
    )
    out = est.estimate(request)
    assert out.failed is True
    assert out.reason == "no_box"


def test_session_mock_dino_sets_effective_source_vision() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from dynisafety.backends.libero.safety.aegis_bridge import LiberoAegisSession
    from dynisafety.backends.libero.safety.phases import LiberoOraclePhase
    from dynisafety.config.schema import MISSING

    cfg_vis = _cfg_from_yaml()
    frame = _bump_frame()
    est = GroundingDinoVisionEstimator(
        cfg_vis,
        repo_root=REPO,
        predictor=lambda _rgb, _prompt: (
            np.array([[0.5, 0.5, 0.7, 0.7]]),
            np.array([0.9]),
        ),
        frame_loader=lambda *_a, **_k: frame,
    )
    raw = yaml.safe_load(VISION_YAML.read_text(encoding="utf-8"))
    extra = {
        "aegis": {
            "enabled": True,
            "activate_after": "settled_or_done",
            "obstacle_source": "vision",
            "config_path": "configs/safety/aegis.yaml",
            "eef_offset_local": [0.0, 0.0, -0.08],
            "eef_axes_diag": [0.06, 0.12, 0.11],
            "obstacle_axes_diag_default": [0.06, 0.06, 0.08],
            "use_geom_size_for_obstacle": False,
            "vision": raw["extra"]["libero"]["aegis"]["vision"],
        }
    }
    run = MagicMock()
    run.level = 2
    run.disturbance = MISSING
    run.suite = "safelibero_spatial"
    session = LiberoAegisSession.create(
        cfg=run, extra_libero=extra, oracle_enabled=True, vision_estimator=est
    )
    obs = {
        "robot0_eef_pos": np.array([0.0, 0.0, 1.0]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([-0.7, 0.7]),
    }
    out = session.maybe_filter_action(
        np.array([0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]),
        obs=obs,
        env=SimpleNamespace(env=SimpleNamespace(objects_dict={}, sim=None)),
        controller=SimpleNamespace(active_obstacle_name="moka_pot_obstacle_1"),
        task_step=50,
        probe_phase=LiberoOraclePhase.SETTLED_OR_DONE,
    )
    assert out.shape == (7,)
    extra_ep = session.episode_extra()
    assert extra_ep["aegis_obstacle_source"] == "vision"
    assert extra_ep["aegis_vision_failed"] is False
    assert extra_ep["aegis_filter_calls"] == 1
