"""Per-episode WRM capture sidecar. Fail-soft; no simulator / backend imports.

Mirrors ``DiagnosticsSession``: ``create`` returns a live object or a disabled
no-op, per-step hooks are silent when off, and ``finalize`` writes once.

This module must not import ``dynisafety.backends`` or the isolated
``A_robosafe_capture_datasets`` package at import time. The writer is loaded
lazily only when a session is enabled and actually writing.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dynisafety.config.schema import MISSING, CaptureConfig, RunConfig
from dynisafety.core.episode import format_checkpoint_label

_LOG = logging.getLogger(__name__)

_LIBERO_FPS = 30
_DEFAULT_FPS = {
    "libero": 30,
    "simplerenv": 5,
    "robotwin": 30,
    "robocasa": 20,
}


def _capture_requested(cfg: RunConfig) -> bool:
    cap = getattr(cfg, "capture", MISSING)
    if cap is MISSING or cap is None:
        return False
    flag = getattr(cap, "enabled", MISSING)
    if flag is MISSING or flag is None:
        return False
    return bool(flag)


def _cfg_str(value: Any, fallback: str = "") -> str:
    if value is MISSING or value is None:
        return fallback
    return str(value)


def _as_float_tuple(value: Sequence[Any] | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    return tuple(float(v) for v in value)


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _copy_image(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        return np.asarray(value).copy()
    except (TypeError, ValueError):
        return None


_CAMERA_SLOTS = ("primary", "wrist", "wrist_right")


def _as_matrix(value: Any, shape: tuple[int, ...]) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.shape != shape or not np.all(np.isfinite(arr)):
        return None
    return arr.copy()


def _copy_cameras(value: Any) -> dict[str, dict[str, Any]] | None:
    if not isinstance(value, Mapping):
        return None
    out: dict[str, dict[str, Any]] = {}
    for slot, raw in value.items():
        if slot not in _CAMERA_SLOTS or not isinstance(raw, Mapping):
            continue
        item: dict[str, Any] = {}
        name = raw.get("name")
        if name:
            item["name"] = str(name)
        K = _as_matrix(raw.get("K"), (3, 3))
        T_cw = _as_matrix(raw.get("T_cw"), (4, 4))
        if K is not None:
            item["K"] = K
        if T_cw is not None:
            item["T_cw"] = T_cw
        if "K" not in item and "T_cw" not in item:
            continue
        for key in ("width", "height"):
            if raw.get(key) is None:
                continue
            try:
                item[key] = int(raw[key])
            except (TypeError, ValueError):
                pass
        item["convention"] = str(raw.get("convention") or "opencv")
        item["image_transform"] = str(raw.get("image_transform") or "identity")
        out[str(slot)] = item
    return out or None


def _tcp_pose_7d(
    eef_pos: Sequence[Any] | None,
    eef_quat: Sequence[Any] | None,
) -> np.ndarray:
    pos = np.zeros(3, dtype=np.float64)
    quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if eef_pos is not None:
        arr = np.asarray(eef_pos, dtype=np.float64).reshape(-1)
        n = min(3, int(arr.size))
        pos[:n] = arr[:n]
    if eef_quat is not None:
        arr = np.asarray(eef_quat, dtype=np.float64).reshape(-1)
        n = min(4, int(arr.size))
        quat[:n] = arr[:n]
        norm = float(np.linalg.norm(quat))
        if norm >= 1e-12:
            quat = quat / norm
    return np.concatenate([pos, quat])


def _yaw_to_quat_xyzw(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw)
    return (0.0, 0.0, float(np.sin(half)), float(np.cos(half)))


def _as_pose7(value: Sequence[Any] | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        arr = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if len(arr) >= 7:
        return arr[:7]
    if len(arr) >= 3:
        return arr[:3] + (0.0, 0.0, 0.0, 1.0)
    return None


def _base_pose_7d(
    base_pose: Sequence[Any] | None,
    base_xy_yaw: Sequence[Any] | None,
) -> tuple[float, ...] | None:
    posed = _as_pose7(base_pose)
    if posed is not None:
        return posed
    if base_xy_yaw is None:
        return None
    try:
        xy_yaw = tuple(float(v) for v in base_xy_yaw)
    except (TypeError, ValueError):
        return None
    if len(xy_yaw) < 3:
        return None
    qx, qy, qz, qw = _yaw_to_quat_xyzw(xy_yaw[2])
    return (xy_yaw[0], xy_yaw[1], 0.0, qx, qy, qz, qw)


def _gripper_from_action(action: Sequence[Any], action_space: str) -> float:
    try:
        values = tuple(float(v) for v in action)
    except (TypeError, ValueError):
        return 0.0
    if not values:
        return 0.0
    if action_space in {"delta_ee_7d", "abs_ee_7d", "hybrid_mobile_12d"}:
        idx = 6 if action_space == "hybrid_mobile_12d" else 6
        return values[idx] if idx < len(values) else values[-1]
    return values[-1]


@dataclass
class _PendingStep:
    t: int
    action: tuple[float, ...]
    eef_pos: tuple[float, ...]
    eef_quat: tuple[float, ...] | None
    in_contact: bool
    force_n: float
    reward: float
    joint_position: tuple[float, ...] | None
    primary: np.ndarray | None
    wrist: np.ndarray | None
    wrist_right: np.ndarray | None
    gripper: float
    tcp_pose_right: tuple[float, ...] | None = None
    base_pose: tuple[float, ...] | None = None
    cameras: dict[str, dict[str, Any]] | None = None


class CaptureSession:
    """Per-episode recorder + package writer, fail-soft for the rollout.

    ``create(cfg, ...)`` is the only constructor backends should call.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        dataset_root: Path | None = None,
        action_space: str = "",
        backend: str = "",
        suite: str = "",
        level: int = 0,
        seed: int = 0,
        model: str = "",
        run_id: str = "",
        config_hash: str = "",
        episode_index: int = 0,
        fps: int = _LIBERO_FPS,
    ) -> None:
        self.enabled = bool(enabled)
        self._dataset_root = dataset_root
        self._action_space = str(action_space)
        self._backend = str(backend)
        self._suite = str(suite)
        self._level = int(level)
        self._seed = int(seed)
        self._model = str(model)
        self._run_id = str(run_id)
        self._config_hash = str(config_hash)
        self._episode_index = int(episode_index)
        self._fps = int(fps)
        self._steps: list[_PendingStep] = []
        self._warned = False
        self._dead = False
        self._finalized = False
        self._package_dir: Path | None = None

    @property
    def wants_images(self) -> bool:
        """True only when this session will persist camera frames."""
        return self.enabled and not self._dead

    @property
    def package_dir(self) -> Path | None:
        return self._package_dir

    @classmethod
    def create(
        cls,
        cfg: RunConfig,
        *,
        episode_index: int,
        bound: Any | None = None,
        action_space: str | None = None,
        fps: int | None = None,
    ) -> CaptureSession:
        """Build a live session or a disabled no-op. Never raises."""
        if not _capture_requested(cfg):
            return cls(enabled=False)
        try:
            return cls._create_enabled(
                cfg,
                episode_index=episode_index,
                bound=bound,
                action_space=action_space,
                fps=fps,
            )
        except Exception:
            _LOG.warning(
                "WRM capture session create failed; disabling for this episode",
                exc_info=True,
            )
            return cls(enabled=False)

    @classmethod
    def _create_enabled(
        cls,
        cfg: RunConfig,
        *,
        episode_index: int,
        bound: Any | None,
        action_space: str | None,
        fps: int | None,
    ) -> CaptureSession:
        if bound is not None and action_space is None:
            action_space = str(bound.action_space)
        if action_space is None:
            raise ValueError("CaptureSession.create needs bound= or action_space")

        cap = cfg.capture
        if not isinstance(cap, CaptureConfig):
            raise TypeError(f"cfg.capture must be CaptureConfig, got {type(cap)!r}")
        raw_root = cap.dataset_root
        if raw_root is MISSING or raw_root is None or not str(raw_root).strip():
            raise ValueError("capture.dataset_root must be set when capture.enabled=true")

        backend = _cfg_str(cfg.backend)
        resolved_fps = _LIBERO_FPS
        if fps is not None:
            try:
                parsed = int(fps)
                if parsed > 0:
                    resolved_fps = parsed
            except (TypeError, ValueError):
                resolved_fps = _DEFAULT_FPS.get(backend, _LIBERO_FPS)
        else:
            resolved_fps = _DEFAULT_FPS.get(backend, _LIBERO_FPS)
        return cls(
            enabled=True,
            dataset_root=Path(str(raw_root)),
            action_space=str(action_space),
            backend=backend,
            suite=_cfg_str(cfg.suite),
            level=int(cfg.level),
            seed=int(cfg.seed),
            model=_cfg_str(cfg.model),
            run_id=_cfg_str(cfg.run_id),
            config_hash=_cfg_str(getattr(bound, "config_hash", "") if bound is not None else ""),
            episode_index=int(episode_index),
            fps=resolved_fps,
        )

    def record_step(
        self,
        *,
        t: int,
        action: Sequence[Any],
        eef_pos: Sequence[Any],
        in_contact: bool,
        force_n: float,
        eef_quat: Sequence[Any] | None = None,
        reward: Any = 0.0,
        joint_position: Sequence[Any] | None = None,
        primary: Any = None,
        wrist: Any = None,
        wrist_right: Any = None,
        gripper: float | None = None,
        tcp_pose_right: Sequence[Any] | None = None,
        eef_pos_secondary: Sequence[Any] | None = None,
        base_pose: Sequence[Any] | None = None,
        base_xy_yaw: Sequence[Any] | None = None,
        cameras: Mapping[str, Any] | None = None,
        **unused: Any,
    ) -> None:
        """Copy one already-computed task step. No-op when disabled or dead.

        ``**unused`` drops unknown fields with a one-shot warning so a backend
        passing a future keyword cannot abort the episode.
        """
        if not self.enabled or self._dead:
            return
        if unused:
            self._warn_once(
                f"WRM capture ignoring unknown record_step fields: {sorted(unused)}"
            )
        try:
            action_t = tuple(float(v) for v in action)
            grip = (
                float(gripper)
                if gripper is not None
                else _gripper_from_action(action_t, self._action_space)
            )
            self._steps.append(
                _PendingStep(
                    t=int(t),
                    action=action_t,
                    eef_pos=tuple(float(v) for v in eef_pos),
                    eef_quat=_as_float_tuple(eef_quat),
                    in_contact=bool(in_contact),
                    force_n=float(force_n),
                    reward=_as_float(reward),
                    joint_position=_as_float_tuple(joint_position),
                    primary=_copy_image(primary) if self.wants_images else None,
                    wrist=_copy_image(wrist) if self.wants_images else None,
                    wrist_right=_copy_image(wrist_right) if self.wants_images else None,
                    gripper=grip,
                    tcp_pose_right=_as_pose7(tcp_pose_right) or _as_pose7(eef_pos_secondary),
                    base_pose=_base_pose_7d(base_pose, base_xy_yaw),
                    cameras=_copy_cameras(cameras),
                )
            )
        except Exception:
            self._warn_once(
                "WRM capture record_step failed; dropping capture for this episode"
            )
            self._dead = True
            self._steps.clear()

    def finalize(
        self,
        *,
        success: bool,
        success_step: int | None,
        termination: str,
        first_contact_step: int | None = None,
        contact_steps: int | None = None,
        toppled: bool | None = None,
        metrics: Mapping[str, Any] | None = None,
        task_id: int = 0,
        task_name: str = "",
        instruction: str = "",
        checkpoint: str = "",
        config_hash: str | None = None,
        steps: int | None = None,
    ) -> None:
        """Write one episode package. No-op when disabled, dead, or empty."""
        if not self.enabled or self._dead or self._finalized:
            return
        self._finalized = True
        if not self._steps or self._dataset_root is None:
            return
        try:
            if metrics is not None:
                if first_contact_step is None:
                    first_contact_step = metrics.get("first_contact_step")
                if contact_steps is None:
                    contact_steps = int(metrics["contact_steps"])
                if toppled is None and "toppled" in metrics:
                    toppled = metrics.get("toppled")
                if steps is None and "steps" in metrics:
                    steps = int(metrics["steps"])
            if contact_steps is None:
                contact_steps = 0
            if config_hash:
                self._config_hash = str(config_hash)
            self._write_package(
                success=bool(success),
                success_step=success_step,
                termination=str(termination),
                first_contact_step=first_contact_step,
                contact_steps=int(contact_steps),
                toppled=toppled,
                task_id=int(task_id),
                task_name=str(task_name),
                instruction=str(instruction),
                checkpoint=str(checkpoint),
                steps=int(steps) if steps is not None else len(self._steps),
            )
        except Exception:
            self._warn_once(
                "WRM capture finalize failed; skipping package for this episode"
            )
            self._package_dir = None

    def _write_package(
        self,
        *,
        success: bool,
        success_step: int | None,
        termination: str,
        first_contact_step: int | None,
        contact_steps: int,
        toppled: bool | None,
        task_id: int,
        task_name: str,
        instruction: str,
        checkpoint: str,
        steps: int,
    ) -> None:
        writer = _load_writer()
        if writer is None:
            self._warn_once(
                "WRM capture writer unavailable (A_robosafe_capture_datasets "
                "not importable); skipping package"
            )
            return

        samples = [self._to_sample(writer.StepSample, pending) for pending in self._steps]
        label = writer.label_failure(
            samples=samples,
            success=success,
            termination=termination,
            contact_steps=contact_steps,
            first_contact_step=first_contact_step,
            toppled=toppled,
            trust_record_first_contact=self._backend != "robotwin",
        )
        ep_id = writer.episode_dir_name(
            task_name=task_name or "episode",
            level=self._level,
            model=self._model or "model",
            seed=self._seed,
            episode_index=self._episode_index,
        )
        dest = (
            self._dataset_root
            / self._backend
            / f"s{self._seed}"
            / writer.sanitize_token(self._model or "model")
            / f"l{self._level}"
            / writer.sanitize_token(task_name or "episode")
            / ep_id
        )
        source = {
            "backend": self._backend,
            "suite": self._suite,
            "task_id": int(task_id),
            "task_name": task_name,
            "level": self._level,
            "model": self._model,
            "checkpoint": format_checkpoint_label(checkpoint, suite=self._suite),
            "seed": self._seed,
            "episode_index": self._episode_index,
            "run_id": self._run_id,
            "config_hash": self._config_hash,
            "termination": termination,
            "steps": int(steps),
        }
        writer.write_episode_package(
            root=dest,
            samples=samples,
            label=label,
            fps=self._fps,
            action_space=self._action_space,
            task_name=task_name,
            instruction=instruction,
            source=source,
            episode_id=ep_id,
            apply_robotwin_sparse_reward=self._backend == "robotwin",
            success_step=success_step,
        )
        self._package_dir = dest

    def _to_sample(self, sample_cls: Any, pending: _PendingStep) -> Any:
        joints = (
            np.asarray(pending.joint_position, dtype=np.float64)
            if pending.joint_position is not None
            else np.zeros(0, dtype=np.float64)
        )
        return sample_cls(
            step=int(pending.t),
            action_native=np.asarray(pending.action, dtype=np.float64),
            tcp_pose=_tcp_pose_7d(pending.eef_pos, pending.eef_quat),
            joint_position=joints,
            reward=float(pending.reward),
            contact=1.0 if pending.in_contact else 0.0,
            force=float(pending.force_n),
            gripper=float(pending.gripper),
            primary=pending.primary,
            wrist=pending.wrist,
            wrist_right=pending.wrist_right,
            tcp_pose_right=(
                np.asarray(pending.tcp_pose_right, dtype=np.float64)
                if pending.tcp_pose_right is not None
                else None
            ),
            base_pose=(
                np.asarray(pending.base_pose, dtype=np.float64)
                if pending.base_pose is not None
                else None
            ),
            cameras=pending.cameras,
        )

    def _warn_once(self, message: str) -> None:
        if self._warned:
            return
        self._warned = True
        _LOG.warning(message, exc_info=True)


@dataclass
class _WriterApi:
    StepSample: Any
    label_failure: Any
    episode_dir_name: Any
    sanitize_token: Any
    write_episode_package: Any


def _load_writer() -> _WriterApi | None:
    """Lazy-load the isolated capture writer. None if the package is absent."""
    try:
        from A_robosafe_capture_datasets.hooks.recorder import StepSample
        from A_robosafe_capture_datasets.labels.heuristic import label_failure
        from A_robosafe_capture_datasets.writer.package import (
            episode_dir_name,
            sanitize_token,
            write_episode_package,
        )
    except ImportError:
        return None
    return _WriterApi(
        StepSample=StepSample,
        label_failure=label_failure,
        episode_dir_name=episode_dir_name,
        sanitize_token=sanitize_token,
        write_episode_package=write_episode_package,
    )


__all__ = ["CaptureSession"]
