"""HTTP serve for VLA-Adapter-Pro LIBERO (DyniSafety examples contract).

Launched by the scheduler as::

    python <workspace>/scripts/vla_adapter/deploy_vla_adapter_dynisafety_server.py \\
        --pretrained_checkpoint ... --repo_root ... \\
        --task_suite_name libero_spatial --unnorm_key libero_spatial_no_noops \\
        --model_family openvla --use_pro_version true --use_minivlm true \\
        --center_crop true --num_images_in_input 2 --num_open_loop_steps 8 \\
        --port {port}

``server.root`` is the DyniSafety workspace, so this file lives in-tree.
``--repo_root`` is prepended to ``sys.path`` *before* importing upstream
``experiments.robot`` so ``prismatic`` comes from VLA-Adapter, not openvla-oft.

LIBERO images from DyniSafety are already 180°-rotated and 256×256.
Do **not** rotate them again. Resize + center-crop happen inside upstream
``get_vla_action``. Gripper binarize+invert happens here once, matching
official ``process_action`` for ``model_family=openvla``.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import numpy as np

# Official LIBERO Pro protocol. Reject anything else at argv parse time so a
# YAML typo cannot silently become L0-all-fail.
_ALLOWED_TASK_SUITES = {
    "libero_spatial": "libero_spatial_no_noops",
    "libero_object": "libero_object_no_noops",
    "libero_goal": "libero_goal_no_noops",
    "libero_10": "libero_10_no_noops",
}
_ACTION_DIM = 7
_PROPRIO_DIM = 8
_NUM_OPEN_LOOP_STEPS = 8
_REQUIRED_SPACE = "delta_ee_7d"


def _json_bytes(body: dict[str, Any]) -> bytes:
    return json.dumps(body).encode("utf-8")


def _as_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def _require_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be int, got {value!r}")
    return value


def _as_hwc_uint8(array: Any, *, name: str) -> np.ndarray:
    image = np.asarray(array)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{name} must be HWC RGB, got shape {image.shape}")
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and float(image.max()) <= 1.0:
            image = np.clip(image * 255.0, 0, 255)
        image = image.astype(np.uint8)
    return np.ascontiguousarray(image)


def normalize_gripper_action(action: np.ndarray, *, binarize: bool = True) -> np.ndarray:
    """Official ``robot_utils.normalize_gripper_action`` (0/1 → [-1, +1])."""
    normalized = np.asarray(action, dtype=np.float32).copy()
    orig_low, orig_high = 0.0, 1.0
    normalized[..., -1] = (
        2.0 * (normalized[..., -1] - orig_low) / (orig_high - orig_low) - 1.0
    )
    if binarize:
        normalized[..., -1] = np.sign(normalized[..., -1])
    return normalized


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """Official ``robot_utils.invert_gripper_action`` (RLDS 0=close vs LIBERO -1=open)."""
    inverted = np.asarray(action, dtype=np.float32).copy()
    inverted[..., -1] *= -1.0
    return inverted


def process_openvla_gripper(action: np.ndarray, *, model_family: str) -> np.ndarray:
    """Official ``process_action`` for ``model_family == "openvla"``."""
    if model_family != "openvla":
        raise ValueError(
            f"LIBERO Pro gripper invert is defined only for model_family='openvla', "
            f"got {model_family!r}"
        )
    return invert_gripper_action(normalize_gripper_action(action, binarize=True))


def resolve_unnorm_key(
    unnorm_key: str,
    task_suite_name: str,
    stats: Mapping[str, Any],
) -> str:
    """Official ``check_unnorm_key`` plus a hard map for the four Pro suites."""
    if task_suite_name not in _ALLOWED_TASK_SUITES:
        raise ValueError(
            f"task_suite_name must be one of {sorted(_ALLOWED_TASK_SUITES)}, "
            f"got {task_suite_name!r} (long suite is libero_10, not libero_long)"
        )
    expected = _ALLOWED_TASK_SUITES[task_suite_name]
    want = str(unnorm_key).strip() or task_suite_name
    resolved = None
    for candidate in (want, f"{want}_no_noops"):
        if candidate in stats:
            resolved = candidate
            break
    if resolved is None and len(stats) == 1:
        resolved = next(iter(stats))
    if resolved is None:
        raise ValueError(
            f"unnorm_key {want!r} (and {want + '_no_noops'!r}) not found in "
            f"norm_stats. Available: {sorted(stats)}"
        )
    if resolved != expected:
        raise ValueError(
            f"resolved unnorm_key {resolved!r} does not match official "
            f"{task_suite_name} → {expected}. Available: {sorted(stats)}"
        )
    return resolved


def observation_for_vla(
    primary: np.ndarray,
    wrist: np.ndarray,
    state: np.ndarray,
) -> dict[str, Any]:
    """Map DyniSafety keys onto official ``get_vla_action`` obs keys.

    Does not rotate. Upstream ``get_vla_action`` looks up ``full_image`` and
    any key containing ``wrist``.
    """
    return {
        "full_image": primary,
        "wrist_image": wrist,
        "state": state,
    }


class _EvalConfig:
    """Minimal stand-in for official ``GenerateConfig`` (fields get_vla reads)."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class _InferenceState:
    def __init__(self, cfg: _EvalConfig, *, repo_root: Path) -> None:
        root = str(repo_root.resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        from experiments.robot.openvla_utils import (  # noqa: PLC0415
            get_action_head,
            get_processor,
            get_proprio_projector,
        )
        from experiments.robot.robot_utils import get_action, get_model  # noqa: PLC0415

        self._lock = threading.Lock()
        self.act_hits = 0
        self.cfg = cfg
        self._get_action = get_action
        print(f"[vla-adapter-serve] loading ckpt={cfg.pretrained_checkpoint}", flush=True)
        self.model = get_model(cfg)
        stats = getattr(self.model, "norm_stats", None) or {}
        cfg.unnorm_key = resolve_unnorm_key(
            str(cfg.unnorm_key), str(cfg.task_suite_name), stats
        )
        print(f"[vla-adapter-serve] unnorm_key={cfg.unnorm_key}", flush=True)
        self.processor = get_processor(cfg)
        if not bool(cfg.use_l1_regression):
            raise ValueError("Pro eval requires use_l1_regression=true and an action head")
        self.action_head = get_action_head(cfg, self.model.llm_dim)
        if self.action_head is None:
            raise ValueError("action_head is None; Pro weights need the L1 head")
        self.proprio_projector = None
        if bool(cfg.use_proprio):
            self.proprio_projector = get_proprio_projector(
                cfg, self.model.llm_dim, _PROPRIO_DIM
            )

    def plan(self, payload: dict[str, Any]) -> tuple[np.ndarray, str]:
        space = payload.get("space")
        if space != _REQUIRED_SPACE:
            raise ValueError(
                f"space must be {_REQUIRED_SPACE!r} (OSC_POSE delta EE, not joints), "
                f"got {space!r}"
            )
        horizon = _require_int(payload, "horizon")
        if horizon != int(self.cfg.num_open_loop_steps):
            raise ValueError(
                f"horizon must equal num_open_loop_steps="
                f"{self.cfg.num_open_loop_steps}, got {horizon}"
            )
        instruction = payload.get("instruction")
        if not isinstance(instruction, str):
            raise ValueError(f"instruction must be str, got {instruction!r}")
        raw_images = payload.get("images")
        if not isinstance(raw_images, dict):
            raise ValueError("images must be a JSON object")
        if "primary" not in raw_images or "wrist" not in raw_images:
            raise KeyError(
                f"images must include primary and wrist; have {sorted(raw_images)}"
            )
        primary = _as_hwc_uint8(raw_images["primary"], name="primary")
        wrist = _as_hwc_uint8(raw_images["wrist"], name="wrist")
        raw_state = payload.get("state")
        if raw_state is None:
            raise ValueError("state is required")
        state = np.asarray(raw_state, dtype=np.float32).reshape(-1)
        if state.size != _PROPRIO_DIM:
            raise ValueError(
                f"state must be {_PROPRIO_DIM}D eef_pos+axis-angle+gripper_qpos, "
                f"got shape {state.shape}"
            )
        obs = observation_for_vla(primary, wrist, state)
        with self._lock:
            actions = self._get_action(
                self.cfg,
                self.model,
                obs,
                instruction,
                processor=self.processor,
                action_head=self.action_head,
                proprio_projector=self.proprio_projector,
                use_film=bool(self.cfg.use_film),
                use_minivlm=bool(self.cfg.use_minivlm),
            )
        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != _ACTION_DIM:
            raise ValueError(f"model returned action dim {arr.shape[1]}, expected {_ACTION_DIM}")
        if arr.shape[0] < horizon:
            raise ValueError(f"model returned {arr.shape[0]} steps, need horizon={horizon}")
        out = np.stack(
            [process_openvla_gripper(row, model_family=str(self.cfg.model_family)) for row in arr[:horizon]],
            axis=0,
        )
        return out.astype(np.float32), space


def build_server(host: str, port: int, *, cfg: _EvalConfig, repo_root: Path) -> ThreadingHTTPServer:
    state = _InferenceState(cfg, repo_root=repo_root)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _write_json(self, code: int, body: dict[str, Any]) -> None:
            payload = _json_bytes(body)
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length else b"{}"
            if not raw:
                return {}
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("JSON body must be an object")
            return data

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/health":
                self._write_json(
                    200, {"ok": True, "status": "healthy", "model": "vla_adapter"}
                )
                return
            self._write_json(404, {"ok": False, "error": f"unknown path {path}"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                body = self._read_json()
            except (ValueError, json.JSONDecodeError) as exc:
                self._write_json(400, {"ok": False, "error": str(exc)})
                return
            if path == "/reset":
                self._write_json(200, {"ok": True})
                return
            if path == "/act":
                state.act_hits += 1
                try:
                    actions, space = state.plan(body)
                except Exception as exc:
                    self._write_json(400, {"ok": False, "error": str(exc)})
                    return
                self._write_json(
                    200,
                    {
                        "ok": True,
                        "actions": np.asarray(actions, dtype=np.float32).tolist(),
                        "space": space,
                    },
                )
                return
            self._write_json(404, {"ok": False, "error": f"unknown path {path}"})

    server = ThreadingHTTPServer((host, port), Handler)
    server.inference_state = state  # type: ignore[attr-defined]
    return server


def _lock_protocol(args: argparse.Namespace) -> None:
    if args.model_family != "openvla":
        raise SystemExit(
            f"--model_family must be 'openvla' (upstream get_model rejects "
            f"openvla_oft); got {args.model_family!r}"
        )
    if args.task_suite_name not in _ALLOWED_TASK_SUITES:
        raise SystemExit(
            f"--task_suite_name must be one of {sorted(_ALLOWED_TASK_SUITES)} "
            f"(long = libero_10, not libero_long); got {args.task_suite_name!r}"
        )
    if args.num_open_loop_steps != _NUM_OPEN_LOOP_STEPS:
        raise SystemExit(
            f"--num_open_loop_steps must be {_NUM_OPEN_LOOP_STEPS} "
            f"(NUM_ACTIONS_CHUNK); got {args.num_open_loop_steps}"
        )
    if args.num_images_in_input != 2:
        raise SystemExit(
            f"--num_images_in_input must be 2 (primary+wrist); got {args.num_images_in_input}"
        )
    if not args.use_pro_version:
        raise SystemExit("--use_pro_version must be true for LIBERO-*-Pro weights")
    if not args.use_minivlm:
        raise SystemExit("--use_minivlm must be true (Qwen prompt; default False is L0-fail)")
    if not args.use_l1_regression:
        raise SystemExit("--use_l1_regression must be true")
    if args.use_film:
        raise SystemExit("--use_film must be false for Pro eval")
    if not args.use_proprio:
        raise SystemExit("--use_proprio must be true")
    if not args.center_crop:
        raise SystemExit("--center_crop must be true (trained with random crop)")
    if not args.load_in_8bit and not args.load_in_4bit:
        return
    raise SystemExit("--load_in_8bit / --load_in_4bit must be false")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLA-Adapter-Pro DyniSafety HTTP server")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--pretrained_checkpoint", type=Path, required=True)
    parser.add_argument("--repo_root", type=Path, required=True)
    parser.add_argument("--task_suite_name", type=str, required=True)
    parser.add_argument("--unnorm_key", type=str, required=True)
    parser.add_argument("--model_family", type=str, required=True)
    parser.add_argument("--use_pro_version", type=_as_bool, required=True)
    parser.add_argument("--use_minivlm", type=_as_bool, required=True)
    parser.add_argument("--use_l1_regression", type=_as_bool, required=True)
    parser.add_argument("--use_film", type=_as_bool, required=True)
    parser.add_argument("--use_proprio", type=_as_bool, required=True)
    parser.add_argument("--center_crop", type=_as_bool, required=True)
    parser.add_argument("--num_images_in_input", type=int, required=True)
    parser.add_argument("--num_open_loop_steps", type=int, required=True)
    parser.add_argument("--load_in_8bit", type=_as_bool, default=False)
    parser.add_argument("--load_in_4bit", type=_as_bool, default=False)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    if args.port < 0:
        raise SystemExit(f"--port must be >= 0, got {args.port}")
    _lock_protocol(args)
    return args


def serve_forever(args: argparse.Namespace) -> None:
    cfg = _EvalConfig(
        pretrained_checkpoint=str(args.pretrained_checkpoint),
        model_family=args.model_family,
        task_suite_name=args.task_suite_name,
        unnorm_key=args.unnorm_key,
        use_pro_version=args.use_pro_version,
        use_minivlm=args.use_minivlm,
        use_l1_regression=args.use_l1_regression,
        use_diffusion=False,
        use_film=args.use_film,
        use_proprio=args.use_proprio,
        center_crop=args.center_crop,
        num_images_in_input=args.num_images_in_input,
        num_open_loop_steps=args.num_open_loop_steps,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )
    server = build_server(args.host, args.port, cfg=cfg, repo_root=args.repo_root)
    bound = int(server.server_address[1])
    print(f"BOUND_PORT={bound}", flush=True)
    print(f"vla_adapter_server listening on {args.host}:{bound}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.pretrained_checkpoint.exists():
        raise SystemExit(f"--pretrained_checkpoint does not exist: {args.pretrained_checkpoint}")
    if not args.repo_root.exists():
        raise SystemExit(f"--repo_root does not exist: {args.repo_root}")
    serve_forever(args)


if __name__ == "__main__":
    main()
