"""Fake-only tests for capture camera K / T_cw extractors (no simulator)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from dynisafety.backends.libero.env import apply_rot180_opencv, get_capture_cameras
from dynisafety.backends.robotwin.env import extract_capture_cameras
from dynisafety.backends.simplerenv.env import get_capture_cameras as se_get_capture_cameras
from dynisafety.backends.simplerenv.env import resolve_primary_camera_name


def test_libero_rot180_matches_flipped_pixels() -> None:
    K = np.array([[80.0, 0.0, 8.0], [0.0, 80.0, 8.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[2, 3] = 1.0
    K2, T2 = apply_rot180_opencv(K, T, width=16, height=16)
    np.testing.assert_allclose(K2[0, 2], 7.0)
    np.testing.assert_allclose(K2[1, 2], 7.0)
    np.testing.assert_allclose(K2[0, 0], 80.0)
    np.testing.assert_allclose(T2[0, 0], -1.0)
    np.testing.assert_allclose(T2[1, 1], -1.0)
    np.testing.assert_allclose(T2[2, 2], 1.0)
    np.testing.assert_allclose(T2[2, 3], 1.0)


class _FakeSim:
    class model:
        cam_fovy = np.array([90.0])

        @staticmethod
        def camera_name2id(name: str) -> int:
            del name
            return 0

    class data:
        cam_xpos = np.array([[0.0, 0.0, 1.0]])
        cam_xmat = np.eye(3).reshape(1, 9)


def test_libero_get_capture_cameras_from_fake_sim() -> None:
    env = SimpleNamespace(sim=_FakeSim())
    obs = {
        "agentview_image": np.zeros((16, 16, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((16, 16, 3), dtype=np.uint8),
    }
    cameras = get_capture_cameras(env, obs)
    assert set(cameras) == {"primary", "wrist"}
    primary = cameras["primary"]
    assert primary["name"] == "agentview"
    assert primary["image_transform"] == "rot180"
    assert primary["K"].shape == (3, 3)
    assert primary["T_cw"].shape == (4, 4)
    np.testing.assert_allclose(primary["K"][0, 2], 7.0)
    assert cameras["wrist"]["name"] == "robot0_eye_in_hand"


def test_simplerenv_get_capture_cameras_from_obs() -> None:
    K = np.array([[623.5, 0.0, 319.5], [0.0, 623.5, 239.5], [0.0, 0.0, 1.0]])
    T = np.eye(4, dtype=np.float64)
    T[2, 3] = 0.5
    env = SimpleNamespace(robot_uid="widowx")
    obs = {
        "camera_param": {
            "3rd_view_camera": {"intrinsic_cv": K, "extrinsic_cv": T},
        },
        "image": {"3rd_view_camera": {"rgb": np.zeros((480, 640, 3), dtype=np.uint8)}},
    }
    cameras = se_get_capture_cameras(env, obs, camera_name=None)
    assert set(cameras) == {"primary"}
    primary = cameras["primary"]
    assert primary["name"] == "3rd_view_camera"
    assert primary["width"] == 640
    assert primary["height"] == 480
    np.testing.assert_allclose(primary["K"], K)
    np.testing.assert_allclose(primary["T_cw"], T)
    assert resolve_primary_camera_name(env, None) == "3rd_view_camera"


def test_robotwin_extract_capture_cameras_from_obs() -> None:
    K = np.diag([200.0, 200.0, 1.0])
    T = np.eye(4, dtype=np.float64)
    raw = {
        "observation": {
            "head_camera": {
                "rgb": np.zeros((8, 8, 3), dtype=np.uint8),
                "intrinsic_cv": K,
                "extrinsic_cv": T,
            },
            "left_camera": {
                "rgb": np.zeros((8, 8, 3), dtype=np.uint8),
                "intrinsic_cv": K,
                "extrinsic_cv": T + 0.0,
            },
            "right_camera": {
                "rgb": np.zeros((8, 8, 3), dtype=np.uint8),
                "intrinsic_cv": K,
                "extrinsic_cv": np.eye(4, dtype=np.float64),
            },
        }
    }
    cameras = extract_capture_cameras(raw)
    assert set(cameras) == {"primary", "wrist", "wrist_right"}
    assert cameras["primary"]["name"] == "head_camera"
    assert cameras["wrist"]["name"] == "left_camera"
    assert cameras["wrist_right"]["name"] == "right_camera"
    np.testing.assert_allclose(cameras["primary"]["K"], K)
    assert cameras["primary"]["width"] == 8
