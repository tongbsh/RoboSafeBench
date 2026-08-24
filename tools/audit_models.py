#!/usr/bin/env python3
"""Pre-rerun audit: does every model's checkpoint / repo / server root exist?

Resolves ``configs/models/*.yaml`` through the real placeholder resolver (so
``${workspace}`` / ``${env:VAR|default}`` behave exactly as at eval time), then
checks each path on disk. Hugging Face repo ids (``org/name``, no separator
ambiguity) are reported separately since they are fetched at load time.

Usage:
    python tools/audit_models.py                 # all models
    python tools/audit_models.py --backend robotwin
    python tools/audit_models.py --paper         # only models in paper_main suite
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import yaml  # noqa: E402

from dynisafety.config.resolve import (  # noqa: E402
    default_third_party_root,
    resolve_placeholders,
)

MODELS = REPO / "configs" / "models"
SUITES = REPO / "configs" / "suites"

# Keys worth checking, in (dotted path, label) form.
CHECK_KEYS = [
    ("checkpoint.default", "ckpt"),
    ("extra.repo_root", "repo"),
    ("server.root", "srv_root"),
    ("extra.dataset_stats_path", "stats"),
    ("extra.action_decoder_path", "decoder"),
    ("extra.openpi_client_src", "pi_client"),
    ("hand.mesh_path", "mesh"),
]


def dig(data: dict, dotted: str):
    cur = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def looks_like_hf_repo(value: str) -> bool:
    """`org/name` with no path separators beyond one, not absolute."""
    if value.startswith(("/", ".", "~")):
        return False
    return bool(re.fullmatch(r"[\w.\-]+/[\w.\-]+", value))


def paper_models() -> set[str]:
    path = SUITES / "paper_main.yaml"
    if not path.exists():
        return set()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: set[str] = set()
    for entry in data.get("matrix") or []:
        for m in entry.get("models") or []:
            out.add(str(m))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", help="only models declaring this backend")
    ap.add_argument("--paper", action="store_true", help="only paper_main models")
    args = ap.parse_args()

    wanted = paper_models() if args.paper else None
    if args.paper and not wanted:
        print("paper_main.yaml has no models; aborting", file=sys.stderr)
        return 2

    rows: list[tuple[str, str, str, str]] = []
    missing = 0
    for path in sorted(MODELS.glob("*.yaml")):
        name = path.stem
        if wanted is not None and name not in wanted:
            continue
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if args.backend and args.backend not in (raw.get("backends") or []):
            continue
        data = resolve_placeholders(
            raw, workspace=REPO, third_party=default_third_party_root(REPO)
        )
        for dotted, label in CHECK_KEYS:
            val = dig(data, dotted)
            if val is None:
                continue
            val = str(val)
            if "${" in val:
                status = "UNRESOLVED"
                missing += 1
            elif looks_like_hf_repo(val):
                status = "hf-repo"
            elif Path(val).exists():
                status = "ok"
            else:
                status = "MISSING"
                missing += 1
            rows.append((name, label, status, val))

    width = max((len(r[0]) for r in rows), default=10)
    cur = None
    for name, label, status, val in rows:
        if name != cur:
            cur = name
            print()
        shown = val if len(val) <= 84 else "…" + val[-83:]
        print(f"{name:<{width}}  {label:<9} {status:<11} {shown}")

    print(f"\n{missing} path(s) MISSING or UNRESOLVED out of {len(rows)} checked")
    if missing:
        print("Fix: python scripts/install/download_assets.py --list   (or export the env var)")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
