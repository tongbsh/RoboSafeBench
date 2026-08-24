"""Dependency-free toy VLA HTTP server (stdlib + numpy).

The scheduler prepends ``server.root`` (``examples/example_agent``) to
``PYTHONPATH``, not ``src/``. Bootstrap the src layout so ``MockBackbone``
imports when launched as ``python server/toy_vla_server.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _json_bytes(body: dict[str, Any]) -> bytes:
    return json.dumps(body).encode("utf-8")


def _require_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be int, got {value!r}")
    return value


def _require_float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be float, got {value!r}")
    return float(value)


class _InferenceState:
    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint
        self.act_hits = 0
        self._backbone: Any = None
        self._key: tuple[Any, ...] | None = None

    def plan(self, payload: dict[str, Any]) -> tuple[np.ndarray, str]:
        from dyni_example_agent.mock_backbone import MockBackbone

        space = payload.get("space")
        if not isinstance(space, str) or not space.strip():
            raise ValueError(f"space must be a non-empty str, got {space!r}")
        horizon = _require_int(payload, "horizon")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        action_dim = _require_int(payload, "action_dim")
        seed = _require_int(payload, "seed")
        feature_grid = _require_int(payload, "feature_grid")
        action_scale = _require_float(payload, "action_scale")
        episode_index = _require_int(payload, "episode_index")
        if episode_index < 0:
            raise ValueError(f"episode_index must be >= 0, got {episode_index}")

        key = (action_dim, feature_grid, action_scale, seed)
        if self._backbone is None or self._key != key:
            self._backbone = MockBackbone.from_checkpoint(
                self.checkpoint,
                action_dim=action_dim,
                feature_grid=feature_grid,
                action_scale=action_scale,
                seed=seed,
            )
            self._key = key

        raw_images = payload.get("images")
        if not isinstance(raw_images, dict):
            raise ValueError("images must be a JSON object")
        images = {str(name): np.asarray(array) for name, array in raw_images.items()}
        raw_state = payload.get("state")
        state = None if raw_state is None else np.asarray(raw_state, dtype=np.float32)
        instruction = payload.get("instruction")
        if not isinstance(instruction, str):
            raise ValueError(f"instruction must be str, got {instruction!r}")
        step = _require_int(payload, "step")
        actions = self._backbone.plan(
            images,
            state,
            instruction,
            step,
            horizon,
            episode_index=episode_index,
        )
        return actions, space


def build_server(host: str, port: int, checkpoint: str | Path) -> ThreadingHTTPServer:
    """Bind a silent ``ThreadingHTTPServer`` (``port=0`` → ephemeral)."""
    state = _InferenceState(Path(checkpoint))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            return

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
                self._write_json(200, {"ok": True, "status": "healthy"})
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
                except (ValueError, FileNotFoundError, KeyError, TypeError) as exc:
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


def serve_forever(host: str, port: int, checkpoint: str | Path) -> None:
    server = build_server(host, port, checkpoint)
    bound = int(server.server_address[1])
    print(f"BOUND_PORT={bound}", flush=True)
    print(f"toy_vla_server listening on {host}:{bound}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Toy DyniSafety VLA HTTP server")
    parser.add_argument("--port", type=int, required=True, help="Listen port; 0 = ephemeral")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    serve_forever(args.host, args.port, args.checkpoint)


if __name__ == "__main__":
    main()
