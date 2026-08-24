"""GroundingDINO obstacle ellipsoid estimator (Prompt A3c).

Lazy-imports torch / groundingdino. Tests inject ``predictor`` + ``frame_loader``.
Never writes detection PNGs. Does not modify the CBF-QP core.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from dynisafety.backends.libero.safety.aegis_vision import (
    VisionConfig,
    VisionEstimate,
    VisionEstimateFailure,
    VisionEstimateRequest,
    VisionPromptMode,
    bounds_for_suite,
    resolve_repo_path,
)
from dynisafety.backends.libero.safety.aegis_vision_geometry import (
    CameraFrame,
    dino_box_to_native_xyxy,
    filter_points,
    fit_mvee_ellipsoid,
    load_camera_frame,
    unproject_depth_crop,
)

Predictor = Callable[[np.ndarray, str], tuple[np.ndarray, np.ndarray | None]]
FrameLoader = Callable[[VisionEstimateRequest, str], CameraFrame | None]


class GroundingDinoVisionEstimator:
    """RGB-D + GroundingDINO → ``EllipsoidState``."""

    def __init__(
        self,
        cfg: VisionConfig,
        *,
        repo_root: Path | None = None,
        predictor: Predictor | None = None,
        frame_loader: FrameLoader | None = None,
        ellipsoid_cls: Any | None = None,
    ) -> None:
        self._cfg = cfg
        self._repo_root = repo_root
        self._predictor = predictor
        self._frame_loader = frame_loader
        self._ellipsoid_cls = ellipsoid_cls
        self._model: Any | None = None
        self._device: str | None = None

    def estimate(self, request: VisionEstimateRequest) -> VisionEstimate:
        prompt = request.text_prompt
        if request.prompt_mode is VisionPromptMode.FROM_LLM:
            return VisionEstimate.failure(
                VisionEstimateFailure.FROM_LLM_UNAVAILABLE,
                prompt=prompt,
            )
        try:
            return self._estimate_inner(request)
        except Exception as exc:  # noqa: BLE001 — eval loop must not die
            return VisionEstimate.failure(
                VisionEstimateFailure.BACKEND_UNAVAILABLE,
                prompt=prompt,
                extra={"error": f"{type(exc).__name__}: {exc}"},
            )

    def _estimate_inner(self, request: VisionEstimateRequest) -> VisionEstimate:
        from dynisafety.policies.safety.aegis import EllipsoidState

        ellipsoid_cls = self._ellipsoid_cls or EllipsoidState
        views: Sequence[str] = request.views or self._cfg.views
        clouds: list[np.ndarray] = []
        view_extra: list[dict[str, Any]] = []
        saw_frame = False
        saw_box = False

        for camera_name in views:
            frame = self._load_frame(request, camera_name)
            if frame is None:
                view_extra.append({"camera": camera_name, "status": "no_frame"})
                continue
            saw_frame = True
            rgb_for_dino = frame.rgb[::-1, ::-1] if self._cfg.flip_rgb_for_dino else frame.rgb
            boxes, scores = self._predict_boxes(rgb_for_dino, request.text_prompt)
            if boxes.size == 0:
                view_extra.append({"camera": camera_name, "status": "no_box"})
                continue
            saw_box = True
            height, width = int(frame.rgb.shape[0]), int(frame.rgb.shape[1])
            crop = dino_box_to_native_xyxy(
                boxes[0],
                width=width,
                height=height,
                flip_180=self._cfg.flip_rgb_for_dino,
            )
            if crop is None:
                view_extra.append({"camera": camera_name, "status": "empty_crop"})
                continue
            xmin, ymin, xmax, ymax = crop
            pts = unproject_depth_crop(
                frame.depth,
                frame.K,
                frame.T_cam_to_world,
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                v_flip=self._cfg.unproject_v_flip,
            )
            view_extra.append(
                {
                    "camera": camera_name,
                    "status": "ok",
                    "n_raw": int(pts.shape[0]),
                    "score": None if scores is None or scores.size == 0 else float(scores[0]),
                }
            )
            if pts.shape[0] > 0:
                clouds.append(pts)

        if not saw_frame:
            return VisionEstimate.failure(
                VisionEstimateFailure.MISSING_DEPTH,
                prompt=request.text_prompt,
                extra={"views": view_extra},
            )
        if not saw_box or not clouds:
            reason = (
                VisionEstimateFailure.NO_BOX if not saw_box else VisionEstimateFailure.EMPTY_CLOUD
            )
            return VisionEstimate.failure(
                reason,
                prompt=request.text_prompt,
                extra={"views": view_extra},
            )

        merged = np.concatenate(clouds, axis=0)
        bounds = bounds_for_suite(self._cfg, request.suite)
        filtered = filter_points(
            merged,
            bounds,
            keep_nearest_frac=self._cfg.keep_nearest_frac,
            cluster=self._cfg.cluster,
            dbscan_eps=self._cfg.dbscan_eps,
            dbscan_min_points=self._cfg.dbscan_min_points,
        )
        if filtered.shape[0] < int(self._cfg.min_cloud_points):
            return VisionEstimate.failure(
                VisionEstimateFailure.EMPTY_CLOUD,
                prompt=request.text_prompt,
                extra={
                    "views": view_extra,
                    "n_merged": int(merged.shape[0]),
                    "n_filtered": int(filtered.shape[0]),
                },
            )
        try:
            ellipsoid = fit_mvee_ellipsoid(filtered, ellipsoid_cls)
        except Exception as exc:  # noqa: BLE001
            return VisionEstimate.failure(
                VisionEstimateFailure.FIT_FAILED,
                prompt=request.text_prompt,
                extra={
                    "error": f"{type(exc).__name__}: {exc}",
                    "n_filtered": int(filtered.shape[0]),
                },
            )
        return VisionEstimate.ok(
            ellipsoid,
            prompt=request.text_prompt,
            extra={
                "views": view_extra,
                "n_merged": int(merged.shape[0]),
                "n_filtered": int(filtered.shape[0]),
                "backend": "grounding_dino",
            },
        )

    def _load_frame(self, request: VisionEstimateRequest, camera_name: str) -> CameraFrame | None:
        if self._frame_loader is not None:
            return self._frame_loader(request, camera_name)
        return load_camera_frame(env=request.env, obs=request.obs, camera_name=camera_name)

    def _predict_boxes(self, rgb: np.ndarray, prompt: str) -> tuple[np.ndarray, np.ndarray | None]:
        if self._predictor is not None:
            boxes, scores = self._predictor(rgb, prompt)
            arr = np.asarray(boxes, dtype=np.float64)
            if arr.size == 0:
                return np.zeros((0, 4), dtype=np.float64), None
            return arr.reshape(-1, 4), None if scores is None else np.asarray(
                scores, dtype=np.float64
            )
        return self._predict_grounding_dino(rgb, prompt)

    def _resolve_device(self) -> str:
        requested = self._cfg.device
        if requested != "cuda":
            return requested
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        if self._cfg.device_fallback == "cpu":
            return "cpu"
        raise RuntimeError("CUDA requested but unavailable and device_fallback='none'")

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        from groundingdino.util.inference import load_model

        ckpt = resolve_repo_path(self._cfg.grounding_dino_checkpoint, repo_root=self._repo_root)
        cfg_path = resolve_repo_path(self._cfg.grounding_dino_config, repo_root=self._repo_root)
        if not ckpt.is_file() or not cfg_path.is_file():
            raise FileNotFoundError(f"GroundingDINO files missing: {ckpt} / {cfg_path}")
        self._device = self._resolve_device()
        try:
            self._model = load_model(str(cfg_path), str(ckpt), device=self._device)
        except TypeError:
            self._model = load_model(str(cfg_path), str(ckpt), cpu_only=(self._device != "cuda"))
        return self._model

    def _predict_grounding_dino(
        self, rgb: np.ndarray, prompt: str
    ) -> tuple[np.ndarray, np.ndarray | None]:
        import groundingdino.datasets.transforms as dino_t
        from groundingdino.util.inference import predict
        from PIL import Image

        model = self._load_model()
        device = self._device or self._resolve_device()
        image_source = Image.fromarray(np.asarray(rgb)).convert("RGB")
        transform = dino_t.Compose(
            [
                dino_t.RandomResize([800], max_size=1333),
                dino_t.ToTensor(),
                dino_t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_transformed, _ = transform(image_source, None)
        boxes, logits, _phrases = predict(
            model=model,
            image=image_transformed,
            caption=str(prompt),
            box_threshold=float(self._cfg.box_threshold),
            text_threshold=float(self._cfg.text_threshold),
            device=device,
        )
        if boxes is None or getattr(boxes, "shape", [0])[0] == 0:
            return np.zeros((0, 4), dtype=np.float64), None
        boxes_np = np.asarray(boxes.detach().cpu().numpy() if hasattr(boxes, "detach") else boxes)
        scores = None
        if logits is not None:
            scores = np.asarray(
                logits.detach().cpu().numpy() if hasattr(logits, "detach") else logits
            )
        return boxes_np.reshape(-1, 4), scores
