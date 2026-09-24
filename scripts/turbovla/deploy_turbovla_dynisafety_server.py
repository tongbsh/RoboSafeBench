"""HTTP serve for TurboVLA LIBERO (DyniSafety examples contract).

Launched by the scheduler as::

    python <workspace>/scripts/turbovla/deploy_turbovla_dynisafety_server.py \\
        --ckpt_path ... --dinov3_path ... --bert_path ... \\
        --stats_path ... --stats_key ... --repo_root ... \\
        --precision bf16 --port {port}

``server.root`` is the DyniSafety workspace, so this file lives in-tree.
``--repo_root`` is prepended to ``sys.path`` so ``TurboVLAPolicy`` imports
from the cloned upstream tree.

LIBERO images from DyniSafety are already 180°-rotated and 256×256.
Do **not** call ``rotate_libero_image`` again.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np


def _json_bytes(body: dict[str, Any]) -> bytes:
    return json.dumps(body).encode("utf-8")


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


def _ensure_256(image: np.ndarray, *, name: str) -> np.ndarray:
    if image.shape[0] == 256 and image.shape[1] == 256:
        return image
    from PIL import Image

    resized = Image.fromarray(image).resize((256, 256), Image.Resampling.BILINEAR)
    if resized.size != (256, 256):
        raise ValueError(f"{name} resize failed: {resized.size}")
    return np.asarray(resized, dtype=np.uint8)


class _InferenceState:
    def __init__(
        self,
        *,
        ckpt_path: Path,
        dinov3_path: Path,
        bert_path: Path,
        stats_path: Path,
        stats_key: str,
        repo_root: Path,
        precision: str,
    ) -> None:
        root = str(repo_root.resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        from turbovla.evaluation.suite_policy import TurboVLAPolicy

        self._lock = threading.Lock()
        self.act_hits = 0
        self.policy = TurboVLAPolicy(
            ckpt_path=str(ckpt_path),
            dinov3_path=str(dinov3_path),
            bert_path=str(bert_path),
            stats_path=str(stats_path),
            stats_key=stats_key,
            precision=precision,
            allow_hf_download=False,
            verbose=True,
        )

    def plan(self, payload: dict[str, Any]) -> tuple[np.ndarray, str]:
        space = payload.get("space")
        if not isinstance(space, str) or not space.strip():
            raise ValueError(f"space must be a non-empty str, got {space!r}")
        horizon = _require_int(payload, "horizon")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
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
        primary = _ensure_256(
            _as_hwc_uint8(raw_images["primary"], name="primary"), name="primary"
        )
        wrist = _ensure_256(
            _as_hwc_uint8(raw_images["wrist"], name="wrist"), name="wrist"
        )
        raw_state = payload.get("state")
        if raw_state is None:
            raise ValueError("state is required")
        state = np.asarray(raw_state, dtype=np.float32).reshape(-1)
        with self._lock:
            actions = self.policy.predict_env_action_chunk(
                primary,
                wrist,
                instruction,
                state,
                execute_steps=horizon,
            )
        return np.asarray(actions, dtype=np.float32), space


def build_server(
    host: str,
    port: int,
    *,
    ckpt_path: Path,
    dinov3_path: Path,
    bert_path: Path,
    stats_path: Path,
    stats_key: str,
    repo_root: Path,
    precision: str,
) -> ThreadingHTTPServer:
    state = _InferenceState(
        ckpt_path=ckpt_path,
        dinov3_path=dinov3_path,
        bert_path=bert_path,
        stats_path=stats_path,
        stats_key=stats_key,
        repo_root=repo_root,
        precision=precision,
    )

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
                self._write_json(200, {"ok": True, "status": "healthy", "model": "turbovla"})
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


def serve_forever(
    host: str,
    port: int,
    *,
    ckpt_path: Path,
    dinov3_path: Path,
    bert_path: Path,
    stats_path: Path,
    stats_key: str,
    repo_root: Path,
    precision: str,
) -> None:
    print(f"[turbovla-serve] loading ckpt={ckpt_path}", flush=True)
    server = build_server(
        host,
        port,
        ckpt_path=ckpt_path,
        dinov3_path=dinov3_path,
        bert_path=bert_path,
        stats_path=stats_path,
        stats_key=stats_key,
        repo_root=repo_root,
        precision=precision,
    )
    bound = int(server.server_address[1])
    print(f"BOUND_PORT={bound}", flush=True)
    print(f"turbovla_server listening on {host}:{bound}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TurboVLA DyniSafety HTTP server")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--dinov3_path", type=Path, required=True)
    parser.add_argument("--bert_path", type=Path, required=True)
    parser.add_argument("--stats_path", type=Path, required=True)
    parser.add_argument("--stats_key", type=str, required=True)
    parser.add_argument("--repo_root", type=Path, required=True)
    parser.add_argument("--precision", type=str, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    if args.port < 0:
        raise SystemExit(f"--port must be >= 0, got {args.port}")
    if args.precision not in {"bf16", "fp32"}:
        raise SystemExit(f"--precision must be bf16 or fp32, got {args.precision!r}")
    for label, path in (
        ("ckpt_path", args.ckpt_path),
        ("dinov3_path", args.dinov3_path),
        ("bert_path", args.bert_path),
        ("stats_path", args.stats_path),
        ("repo_root", args.repo_root),
    ):
        if not path.exists():
            raise SystemExit(f"--{label} does not exist: {path}")
    serve_forever(
        args.host,
        args.port,
        ckpt_path=args.ckpt_path,
        dinov3_path=args.dinov3_path,
        bert_path=args.bert_path,
        stats_path=args.stats_path,
        stats_key=args.stats_key,
        repo_root=args.repo_root,
        precision=args.precision,
    )


if __name__ == "__main__":
    main()
