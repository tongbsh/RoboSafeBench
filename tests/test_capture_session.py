"""Fake-only tests for the optional WRM CaptureSession sidecar (no simulator)."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dynisafety.capture import CaptureSession
from dynisafety.capture.session import CaptureSession as SessionCls
from dynisafety.config.schema import MISSING, CaptureConfig, OutputConfig, RunConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
SESSION_SRC = REPO_ROOT / "src" / "dynisafety" / "capture" / "session.py"
LIBERO_BACKEND_SRC = REPO_ROOT / "src" / "dynisafety" / "backends" / "libero" / "backend.py"
SIMPLERENV_BACKEND_SRC = (
    REPO_ROOT / "src" / "dynisafety" / "backends" / "simplerenv" / "backend.py"
)
ROBOTWIN_BACKEND_SRC = REPO_ROOT / "src" / "dynisafety" / "backends" / "robotwin" / "backend.py"
ROBOCASA_BACKEND_SRC = REPO_ROOT / "src" / "dynisafety" / "backends" / "robocasa" / "backend.py"
PARALLEL_EPISODE_SRC = REPO_ROOT / "src" / "dynisafety" / "parallel_eval" / "episode.py"


def _cap_cfg(*, enabled: bool, dataset_root: Path | str | object = MISSING) -> CaptureConfig:
    return CaptureConfig(enabled=enabled, dataset_root=dataset_root)


def _run_cfg(
    tmp_path: Path | None = None,
    *,
    enabled: bool = False,
    dataset_root: Path | str | object | None = None,
    run_id: object = "run-cap",
    level: int = 0,
    backend: str = "libero",
    suite: str = "safelibero_spatial",
    seed: int = 7,
    model: str = "random",
    capture: object | None = None,
) -> RunConfig:
    output = MISSING
    if tmp_path is not None:
        output = OutputConfig(results_dir=str(tmp_path), save_video=False)
    if capture is None:
        root: object = MISSING
        if dataset_root is not None:
            root = str(dataset_root)
        elif tmp_path is not None:
            root = str(tmp_path / "capture")
        capture = _cap_cfg(enabled=enabled, dataset_root=root)
    return RunConfig(
        backend=backend,
        suite=suite,
        level=level,
        seed=seed,
        model=model,
        run_id=run_id,  # type: ignore[arg-type]
        capture=capture,  # type: ignore[arg-type]
        output=output,
    )


def _bound() -> SimpleNamespace:
    return SimpleNamespace(chunk_size=8, action_space="delta_ee_7d", config_hash="abc123def456")


def _feed(session: CaptureSession, n: int = 3, *, with_images: bool = False) -> None:
    for t in range(n):
        primary = np.zeros((16, 16, 3), dtype=np.uint8) + t if with_images else None
        session.record_step(
            t=t,
            eef_pos=(0.1 * t, 0.0, 0.2),
            eef_quat=(0.0, 0.0, 0.0, 1.0),
            action=(0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5),
            in_contact=t == 1,
            force_n=2.0 if t == 1 else 0.0,
            reward=0.0,
            joint_position=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            primary=primary,
            wrist=primary,
        )


def test_disabled_create_is_noop(tmp_path: Path) -> None:
    dest = tmp_path / "capture"
    session = CaptureSession.create(
        _run_cfg(tmp_path, enabled=False, dataset_root=dest),
        episode_index=0,
        bound=_bound(),
    )
    assert session.enabled is False
    assert session.wants_images is False
    _feed(session, with_images=True)
    session.finalize(
        success=True,
        success_step=2,
        termination="success",
        task_name="pick",
        instruction="pick the bowl",
    )
    assert session.package_dir is None
    assert not dest.exists()
    assert list(tmp_path.iterdir()) == [] or all(p.name != "capture" for p in tmp_path.iterdir())


def test_missing_capture_block_is_disabled() -> None:
    cfg = RunConfig(backend="libero", suite="s", level=0, seed=7, run_id="r")
    session = CaptureSession.create(cfg, episode_index=0, bound=_bound())
    assert session.enabled is False
    assert session.wants_images is False


def test_record_step_ignores_unknown_fields(tmp_path: Path) -> None:
    session = CaptureSession.create(
        _run_cfg(tmp_path, enabled=True),
        episode_index=0,
        bound=_bound(),
    )
    session.record_step(
        t=0,
        eef_pos=(0.0, 0.0, 0.0),
        action=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        in_contact=False,
        force_n=0.0,
        not_a_real_field=1,
    )
    session.finalize(success=False, success_step=None, termination="max_steps", task_name="t")
    assert session.enabled is True


def test_disabled_session_does_not_retain_caller_buffers(tmp_path: Path) -> None:
    session = CaptureSession.create(
        _run_cfg(tmp_path, enabled=False),
        episode_index=0,
        bound=_bound(),
    )
    buf = np.array([1.0, 2.0, 3.0])
    img = np.ones((2, 2, 3), dtype=np.uint8)
    session.record_step(
        t=0,
        eef_pos=buf,
        action=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        in_contact=False,
        force_n=0.0,
        primary=img,
    )
    buf[:] = 9.0
    img[:] = 7
    session.finalize(success=False, success_step=None, termination="max_steps")
    assert session.package_dir is None
    assert not list(session._steps)  # noqa: SLF001


@pytest.mark.skipif(
    not (REPO_ROOT / "A_robosafe_capture_datasets" / "writer" / "package.py").is_file(),
    reason="isolated capture writer package is not present",
)
def test_enabled_writes_episode_package(tmp_path: Path) -> None:
    dest = tmp_path / "wrm"
    session = CaptureSession.create(
        _run_cfg(tmp_path, enabled=True, dataset_root=dest, model="pi05"),
        episode_index=3,
        bound=_bound(),
    )
    assert session.enabled is True
    assert session.wants_images is True
    _feed(session, n=4, with_images=True)
    session.finalize(
        success=False,
        success_step=None,
        termination="max_steps",
        contact_steps=1,
        first_contact_step=1,
        toppled=False,
        task_id=2,
        task_name="pick the bowl",
        instruction="pick the bowl",
        checkpoint={"default": "/weights/xvla-libero-full"},
        config_hash="abc123def456",
        steps=4,
    )
    assert session.package_dir is not None
    pkg = session.package_dir
    assert pkg.is_dir()
    annotation = json.loads((pkg / "annotation.json").read_text(encoding="utf-8"))
    assert annotation["schema_version"] == "robosafe_capture_1.0"
    assert annotation["failure"]["failure_type"] == "CONTACT"
    assert annotation["spavr"]["task_success"] is False
    assert annotation["source_traceability"]["checkpoint"] == "/weights/xvla-libero-full"
    assert (pkg / "trajectory" / "tcp_pose.npy").is_file()
    assert (pkg / "trajectory" / "action_native.npy").is_file()
    assert (pkg / "trajectory" / "contact.npy").is_file()
    tcp = np.load(pkg / "trajectory" / "tcp_pose.npy")
    assert tcp.shape == (4, 7)
    contact = np.load(pkg / "trajectory" / "contact.npy")
    assert int(contact[1]) == 1


def test_session_src_does_not_import_backends_or_writer() -> None:
    """Top-level imports only. Writer is loaded lazily inside ``_load_writer``."""
    banned = (
        "dynisafety.backends",
        "A_robosafe_capture_datasets",
        "mujoco",
        "sapien",
        "robosuite",
        "robocasa",
        "libero",
    )
    tree = ast.parse(SESSION_SRC.read_text(encoding="utf-8"), filename=str(SESSION_SRC))

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.fn_depth = 0
            self.hits: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.fn_depth += 1
            self.generic_visit(node)
            self.fn_depth -= 1

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.fn_depth += 1
            self.generic_visit(node)
            self.fn_depth -= 1

        def visit_Import(self, node: ast.Import) -> None:
            if self.fn_depth:
                return
            for alias in node.names:
                self.hits.append(alias.name)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if self.fn_depth or not node.module:
                return
            self.hits.append(node.module)

    visitor = _Visitor()
    visitor.visit(tree)
    for mod in visitor.hits:
        for prefix in banned:
            assert not mod.startswith(prefix), f"session.py top-level import {mod}"


def test_import_capture_does_not_load_simulators_or_backends() -> None:
    script = """
import sys
from dynisafety import capture
_ = capture.CaptureSession
banned_roots = {"mujoco", "sapien", "robosuite", "libero", "robocasa"}
leaked = sorted(
    {
        name
        for name in sys.modules
        if name.split(".")[0] in banned_roots
        or name.startswith("dynisafety.backends")
        or name.startswith("A_robosafe_capture_datasets")
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


def _assert_capture_wired(src: str, *, session_name: str = "capture_session") -> None:
    assert "CaptureSession.create" in src
    assert f"{session_name}.record_step" in src
    assert f"{session_name}.finalize" in src
    assert f"{session_name}.wants_images" in src
    assert f"{session_name}.enabled" in src


def test_libero_backend_wires_capture_session() -> None:
    src = LIBERO_BACKEND_SRC.read_text(encoding="utf-8")
    _assert_capture_wired(src)
    assert "get_capture_cameras" in src
    assert "cameras=step_cameras" in src


def test_simplerenv_backend_wires_capture_session() -> None:
    src = SIMPLERENV_BACKEND_SRC.read_text(encoding="utf-8")
    _assert_capture_wired(src)
    assert "get_capture_cameras" in src
    assert "cameras=step_cameras" in src


def test_robotwin_backend_wires_capture_session() -> None:
    src = ROBOTWIN_BACKEND_SRC.read_text(encoding="utf-8")
    _assert_capture_wired(src)
    # Capture extras must reuse the already-fetched obs; no extra camera render.
    assert src.count("task_env.get_obs()") == 2
    assert "extract_capture_cameras" in src
    assert "cameras=step_cameras" in src


def test_robocasa_backend_wires_capture_session() -> None:
    _assert_capture_wired(ROBOCASA_BACKEND_SRC.read_text(encoding="utf-8"))


def test_parallel_eval_wires_capture_session() -> None:
    src = PARALLEL_EPISODE_SRC.read_text(encoding="utf-8")
    _assert_capture_wired(src, session_name="slot.capture_session")
    assert "CaptureSession.create" in src
    assert "extract_capture_cameras" in src
    assert "cameras=slot.step_cameras" in src
    assert "get_obs()" not in src.split("def collect_obs")[1].split("def apply_chunk")[0]


def test_libero_gates_image_copies_on_wants_images() -> None:
    """Primary/wrist copies for capture must sit behind wants_images."""
    src = LIBERO_BACKEND_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    found_gate = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Attribute)
            and test.attr == "wants_images"
            and isinstance(test.value, ast.Name)
            and test.value.id == "capture_session"
        ):
            body_src = ast.get_source_segment(src, node)
            assert body_src is not None
            assert "get_primary_image" in body_src
            assert "get_wrist_image" in body_src
            found_gate = True
    assert found_gate, "expected if capture_session.wants_images: around image copies"


def _wants_images_bodies(src: str, *, session_expr: str) -> list[str]:
    tree = ast.parse(src)
    bodies: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test_src = ast.get_source_segment(src, node.test) or ""
        if f"{session_expr}.wants_images" not in test_src:
            continue
        body_src = ast.get_source_segment(src, node)
        if body_src:
            bodies.append(body_src)
    return bodies


def test_other_backends_gate_capture_images() -> None:
    simpler = SIMPLERENV_BACKEND_SRC.read_text(encoding="utf-8")
    bodies = _wants_images_bodies(simpler, session_expr="capture_session")
    assert any("get_primary_image" in body for body in bodies)
    assert all("get_wrist_image" not in body for body in bodies)

    robotwin = ROBOTWIN_BACKEND_SRC.read_text(encoding="utf-8")
    bodies = _wants_images_bodies(robotwin, session_expr="capture_session")
    assert any("_copy_camera_rgb" in body for body in bodies)
    assert any("right_camera" in body for body in bodies)
    assert any("left_camera" in body for body in bodies)

    robocasa = ROBOCASA_BACKEND_SRC.read_text(encoding="utf-8")
    bodies = _wants_images_bodies(robocasa, session_expr="capture_session")
    assert any('images["primary"]' in body or "images['primary']" in body for body in bodies)

    parallel = PARALLEL_EPISODE_SRC.read_text(encoding="utf-8")
    bodies = _wants_images_bodies(parallel, session_expr="slot.capture_session")
    assert any("_copy_camera_rgb" in body for body in bodies)
    assert any("right_camera" in body for body in bodies)


@pytest.mark.skipif(
    not (REPO_ROOT / "A_robosafe_capture_datasets" / "writer" / "package.py").is_file(),
    reason="isolated capture writer package is not present",
)
def test_enabled_writes_right_tcp_and_base(tmp_path: Path) -> None:
    dest = tmp_path / "wrm"
    session = CaptureSession.create(
        _run_cfg(tmp_path, enabled=True, dataset_root=dest, backend="robotwin"),
        episode_index=0,
        bound=_bound(),
        fps=30,
    )
    left = np.full((8, 8, 3), 11, dtype=np.uint8)
    right = np.full((8, 8, 3), 22, dtype=np.uint8)
    session.record_step(
        t=0,
        eef_pos=(0.1, 0.0, 0.2),
        eef_quat=(0.0, 0.0, 0.0, 1.0),
        action=(0.0,) * 14,
        in_contact=False,
        force_n=0.0,
        wrist=left,
        wrist_right=right,
        tcp_pose_right=(0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0),
        base_xy_yaw=(1.0, 2.0, 0.5),
    )
    session.finalize(
        success=True,
        success_step=0,
        termination="success",
        task_name="handover",
        instruction="handover the block",
    )
    assert session.package_dir is not None
    right = np.load(session.package_dir / "trajectory" / "tcp_pose_right.npy")
    assert right.shape == (1, 7)
    base = np.load(session.package_dir / "trajectory" / "base_pose.npy")
    assert base.shape == (1, 7)
    annotation = json.loads(
        (session.package_dir / "annotation.json").read_text(encoding="utf-8")
    )
    assert annotation["data"]["video"]["wrist"] == "video/episode_wrist.mp4"
    assert annotation["data"]["video"]["wrist_right"] == "video/episode_wrist_right.mp4"
    assert (session.package_dir / "video" / "episode_wrist.mp4").is_file()
    assert (session.package_dir / "video" / "episode_wrist_right.mp4").is_file()


@pytest.mark.skipif(
    not (REPO_ROOT / "A_robosafe_capture_datasets" / "writer" / "package.py").is_file(),
    reason="isolated capture writer package is not present",
)
def test_enabled_writes_camera_params(tmp_path: Path) -> None:
    dest = tmp_path / "wrm"
    session = CaptureSession.create(
        _run_cfg(tmp_path, enabled=True, dataset_root=dest),
        episode_index=0,
        bound=_bound(),
    )
    K = np.array([[100.0, 0.0, 8.0], [0.0, 100.0, 8.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    T0 = np.eye(4, dtype=np.float64)
    T1 = np.eye(4, dtype=np.float64)
    T1[0, 3] = 0.1
    for t, T_cw in enumerate((T0, T1)):
        session.record_step(
            t=t,
            eef_pos=(0.0, 0.0, 0.0),
            action=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            in_contact=False,
            force_n=0.0,
            primary=np.zeros((16, 16, 3), dtype=np.uint8),
            cameras={
                "primary": {
                    "name": "agentview",
                    "K": K,
                    "T_cw": T_cw,
                    "width": 16,
                    "height": 16,
                    "convention": "opencv",
                    "image_transform": "rot180",
                }
            },
        )
    session.finalize(
        success=True,
        success_step=1,
        termination="success",
        task_name="pick",
        instruction="pick",
    )
    assert session.package_dir is not None
    annotation = json.loads(
        (session.package_dir / "annotation.json").read_text(encoding="utf-8")
    )
    assert annotation["data"]["camera"]["primary"] == "cameras/primary.json"
    cam = json.loads(
        (session.package_dir / "cameras" / "primary.json").read_text(encoding="utf-8")
    )
    assert cam["name"] == "agentview"
    assert cam["image_transform"] == "rot180"
    np.testing.assert_allclose(cam["K"], K)
    stacked = np.load(session.package_dir / "cameras" / "primary_T_cw.npy")
    assert stacked.shape == (2, 4, 4)
    np.testing.assert_allclose(stacked[1, 0, 3], 0.1)


def test_capture_session_symbol_is_session_class() -> None:
    assert CaptureSession is SessionCls
