#!/usr/bin/env python3
"""Print the conda env a model needs, read from configs/models/<model>.yaml.

The YAML stays the single source of truth for per-model environments; shell
wrappers query this instead of hardcoding names.

Roles:
    eval    client-side env that runs `dynisafety eval` (extra.eval_env)
    serve   server-side env that runs the inference server (server.env)

Usage:
    python tools/model_env.py openvla_oft eval      -> openvla-oft
    python tools/model_env.py starvla serve         -> starVLA
    python tools/model_env.py starvla eval --backend robotwin -> internvla_a1
    python tools/model_env.py --audit               -> table of all models
    python tools/model_env.py --needs-server starvla; echo $?   -> 0 if server
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "configs" / "models"

# Keep import path working when invoked as a script without PYTHONPATH=src.
sys.path.insert(0, str(REPO / "src"))
from dynisafety.policies.base import apply_per_backend  # noqa: E402


def load(model: str) -> dict:
    path = MODELS / f"{model}.yaml"
    if not path.exists():
        avail = ", ".join(sorted(p.stem for p in MODELS.glob("*.yaml")))
        sys.exit(f"unknown model '{model}'. available: {avail}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def env_for(cfg: dict, role: str, *, backend: str | None = None) -> str | None:
    if role == "eval" and backend:
        cfg = apply_per_backend(cfg, backend)
    extra = cfg.get("extra") or {}
    if role == "serve":
        return (cfg.get("server") or {}).get("env") or extra.get("serve_env")
    return extra.get("eval_env")


def needs_server(cfg: dict) -> bool:
    extra = cfg.get("extra") or {}
    if "needs_server" in extra:
        return bool(extra["needs_server"])
    return cfg.get("transport", "local") != "local"


def existing_envs() -> set[str]:
    try:
        out = subprocess.run(
            ["conda", "env", "list"], capture_output=True, text=True, timeout=30
        ).stdout
    except Exception:
        return set()
    names = set()
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(line.split()[0])
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("role", nargs="?", default="eval", choices=["eval", "serve"])
    ap.add_argument(
        "--backend",
        default=None,
        help="Eval backend id; applies configs/models per_backend overlays "
        "(e.g. starvla eval --backend robotwin → internvla_a1)",
    )
    ap.add_argument("--audit", action="store_true", help="table of all models vs disk")
    ap.add_argument("--needs-server", metavar="MODEL", help="exit 0 if model needs a server")
    args = ap.parse_args()

    if args.needs_server:
        return 0 if needs_server(load(args.needs_server)) else 1

    if args.audit:
        have = existing_envs()
        print(f"{'MODEL':<22}{'EVAL ENV':<18}{'SERVE ENV':<18}STATUS")
        bad = 0
        for path in sorted(MODELS.glob("*.yaml")):
            cfg = load(path.stem)
            ev = env_for(cfg, "eval") or "-"
            sv = env_for(cfg, "serve") if needs_server(cfg) else None
            missing = [
                e for e in (ev, sv)
                if e and e not in ("-", "base") and have and e not in have
            ]
            if missing:
                bad += 1
            status = f"MISSING: {' '.join(missing)}" if missing else "ok"
            print(f"{path.stem:<22}{ev:<18}{(sv or '-'):<18}{status}")
        print(f"\n{bad} model(s) reference a conda env that does not exist on this machine")
        return 1 if bad else 0

    if not args.model:
        ap.print_help()
        return 2
    env = env_for(load(args.model), args.role, backend=args.backend)
    if not env:
        sys.exit(f"model '{args.model}' declares no {args.role} env in configs/models/{args.model}.yaml")
    print(env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
