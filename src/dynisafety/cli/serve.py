"""``dynisafety serve`` — start a model inference server (replaces start_*_server.sh)."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

from dynisafety.config.resolve import find_workspace_root
from dynisafety.runner.scheduler import (
    allocate_port,
    detect_gpu_ids,
    load_model_serve_spec,
    start_model_server,
    stop_server,
)


def register_serve_parser(subparsers: argparse._SubParsersAction) -> None:
    """Frozen flags: ``--model`` (required) ``--suite`` ``--port`` ``--gpu``."""
    p = subparsers.add_parser(
        "serve",
        help="Start model inference server (from configs/models/*.yaml)",
    )
    p.add_argument("--model", required=True, help="Model id / configs/models/{model}.yaml")
    p.add_argument(
        "--suite",
        default=None,
        help=(
            "Task suite the server will be asked to run. Required for models "
            "with per-suite weights (checkpoint.per_suite / server.per_suite), "
            "since the server loads its checkpoint at startup."
        ),
    )
    p.add_argument(
        "--port",
        type=int,
        default=None,
        help="Listen port (omit → allocate free TCP port)",
    )
    p.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU index for CUDA_VISIBLE_DEVICES (omit → first discovered GPU)",
    )
    p.set_defaults(serve_handler=dispatch_serve)


def resolve_serve_log_path(
    *,
    workspace: Path,
    model: str,
    port: int,
    log_dir: str | None = None,
) -> Path:
    """Process log for a standalone ``dynisafety serve``.

    ``DYNI_SERVE_LOG_DIR`` (or ``log_dir``) pins the directory; otherwise
    ``{out}/logs/serve/{model}/port{port}.log``. Sweep still writes server
    logs under ``{out}/_sweep/`` via an explicit path.
    """
    raw = log_dir if log_dir is not None else os.environ.get("DYNI_SERVE_LOG_DIR", "")
    raw = str(raw or "").strip()
    if raw:
        path = Path(raw).expanduser()
        if path.suffix == ".log":
            return path
        return path / "server.log"
    from dynisafety.config.paths import load_paths

    del workspace
    return load_paths().logs_dir("serve", model) / f"port{int(port)}.log"


def run_serve(
    *,
    model: str,
    suite: str | None,
    port: int | None,
    gpu: int | None,
) -> int:
    """Load ModelServeSpec → allocate port/gpu → start → block until SIGINT."""
    workspace = find_workspace_root()
    configs_root = workspace / "configs"
    spec = load_model_serve_spec(model, configs_root=configs_root, suite=suite)

    if "model_yaml" in spec.missing_fields:
        print(
            f"[serve] ERROR: missing configs/models/{model}.yaml. "
            "Add the file with needs_server and a server: block "
            "(migrate argv from start_*_server.sh / launch_server.sh).",
            file=sys.stderr,
        )
        return 2
    if not spec.needs_server:
        print(
            f"[serve] ERROR: model {model!r} has needs_server=false "
            "(local in-process policy; no server to start).",
            file=sys.stderr,
        )
        return 2
    if spec.checkpoint_error:
        print(f"[serve] ERROR: model {model!r}: {spec.checkpoint_error}", file=sys.stderr)
        return 2
    if spec.missing_fields or not spec.serve_command or not spec.health_url_template:
        detail = ", ".join(spec.missing_fields) or "serve_command/health_url_template"
        hint = ""
        if suite is None and any(
            "per_suite" in f or "placeholders" in f for f in spec.missing_fields
        ):
            hint = " Pass --suite for models with per-suite weights."
        print(
            f"[serve] ERROR: model {model!r} incomplete serve config; "
            f"missing {detail}. Fill the server: block in "
            f"configs/models/{model}.yaml (env/root/launch/args/health_path)."
            f"{hint}",
            file=sys.stderr,
        )
        return 2

    if gpu is None:
        gpus = detect_gpu_ids(None)
        if not gpus:
            print(
                "[serve] ERROR: no GPU discovered; pass --gpu N explicitly.",
                file=sys.stderr,
            )
            return 2
        gpu_id = int(gpus[0])
    else:
        gpu_id = int(gpu)

    listen_port = int(port) if port is not None else allocate_port()

    health_raw = os.environ.get("DYNISAFETY_SERVER_WAIT_TIMEOUT")
    health_timeout_s = float(health_raw) if health_raw else 600.0

    log_path = resolve_serve_log_path(
        workspace=workspace, model=model, port=listen_port
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[serve] model      = {spec.name}")
    print(f"[serve] suite      = {spec.suite or '(none — using checkpoint.default)'}")
    print(f"[serve] checkpoint = {spec.checkpoint or '(none)'}")
    print(f"[serve] gpu        = {gpu_id}")
    print(f"[serve] port       = {listen_port}")
    print(f"[serve] serve_env  = {spec.serve_env or '(current python)'}")
    print(f"[serve] cwd        = {spec.serve_cwd or '(inherit)'}")
    print(f"[serve] command    = {list(spec.serve_command)}")
    print(f"[serve] health     = {spec.health_url_template}")
    print(f"[serve] log        = {log_path}")
    print("[serve] starting… (Ctrl+C to stop)")

    try:
        handle = start_model_server(
            spec,
            gpu=gpu_id,
            port=listen_port,
            log_path=log_path,
            health_timeout_s=health_timeout_s,
        )
    except Exception as exc:  # noqa: BLE001 — surface startup failures to CLI
        print(f"[serve] ERROR: failed to start server: {exc}", file=sys.stderr)
        if log_path.is_file():
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
                if tail:
                    print("[serve] --- server log (tail) ---", file=sys.stderr)
                    print("\n".join(tail), file=sys.stderr)
            except OSError:
                pass
        return 1

    print(f"[serve] ready pid={handle.pid} port={handle.port}")

    stopping = {"done": False}

    def _stop(signum: int, frame: object) -> None:  # noqa: ARG001
        if stopping["done"]:
            return
        stopping["done"] = True
        print(f"[serve] signal {signum}; shutting down…", file=sys.stderr)
        stop_server(handle)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        if handle.process is not None:
            rc = int(handle.process.wait())
        else:
            while not stopping["done"]:
                try:
                    os.kill(handle.pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.5)
            rc = 0
    except KeyboardInterrupt:
        stop_server(handle)
        rc = 0
    finally:
        if not stopping["done"]:
            stop_server(handle)

    print(f"[serve] exited rc={rc}")
    if stopping["done"]:
        # Operator asked for the shutdown; the child's non-zero rc is our own
        # SIGTERM, not a failure. `scripts/eval/serve.sh` runs under `set -e`.
        return 0
    return 0 if rc == 0 else 1


def dispatch_serve(args: argparse.Namespace) -> int:
    return run_serve(
        model=args.model,
        suite=args.suite,
        port=args.port,
        gpu=args.gpu,
    )
