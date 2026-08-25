#!/usr/bin/env python3
"""Repair editable (``pip install -e``) installs that point at moved paths.

Editable installs record **absolute** source paths in
``site-packages/*.pth`` and ``site-packages/__editable__*_finder.py``
(``MAPPING`` / ``NAMESPACES`` dicts). Moving the source tree silently breaks
``import`` for those packages — the dist-info still exists, so the package looks
installed while importing raises ``ModuleNotFoundError``.

This scans every conda env, rewrites stale paths according to KNOWN_MOVES, and
verifies the result. Metadata-only files (``dist-info/direct_url.json``) are
rewritten too so provenance stays truthful, but they never affect imports.

Usage:
    python scripts/setup/repair_editable_installs.py              # dry-run report
    python scripts/setup/repair_editable_installs.py --apply      # rewrite in place
    python scripts/setup/repair_editable_installs.py --verify     # import-test only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# old path fragment -> new path fragment, relative to the repo root.
# Extend this when directories move again.
KNOWN_MOVES: list[tuple[str, str]] = [
    ("dynamic_SimplerEnv/SimplerEnv-OpenVLA", "third_party/SimplerEnv-OpenVLA"),
    ("dynamic_RoboTwin/RoboTwin", "external/simulators/RoboTwin"),
    ("simulator_env", "external/simulators"),
    ("vlaModelHome", "external/model_repos"),
    ("download_checkpoints", "external/checkpoints"),
    ("download_datasets", "external/datasets"),
]

# Files whose content is functional (affects sys.path / imports).
FUNCTIONAL_GLOBS = ("*.pth", "__editable__*.py", "*.egg-link")
# Files that are provenance metadata only.
METADATA_GLOBS = ("*.dist-info/direct_url.json",)

# env name -> modules that must import for that env to be usable
VERIFY_MATRIX: dict[str, list[str]] = {
    "simpler_env": ["simpler_env", "mani_skill2_real2sim"],
    "openpi": ["openpi", "openpi_client"],
    "openvla-oft": ["openpi_client"],
    "internvla_a1": ["lerobot", "curobo"],
    "fastwam": ["fastwam"],
    "isaac-gr00t": ["gr00t"],
    "starVLA": ["deployment"],
    "univla": ["prismatic"],
    "internvla-m1": ["InternVLA"],
}


def conda_envs() -> dict[str, Path]:
    """env name -> site-packages dir, discovered via ``conda info --json``.

    No hardcoded install location: honours whatever conda is on PATH (and thus
    the user's ``envs_dirs`` configuration).
    """
    try:
        res = subprocess.run(
            ["conda", "info", "--json"], capture_output=True, text=True, timeout=120
        )
        info = json.loads(res.stdout)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"cannot query conda: {exc}", file=sys.stderr)
        return {}

    out: dict[str, Path] = {}
    for raw in info.get("envs", []):
        env = Path(raw)
        if not env.is_dir():
            continue
        sp = sorted(env.glob("lib/python*/site-packages"))
        if sp:
            out[env.name] = sp[-1]
    return out


def rewrite(text: str) -> tuple[str, list[str]]:
    """Apply KNOWN_MOVES to *text*; return (new_text, descriptions)."""
    changes: list[str] = []
    new = text
    for old, dst in KNOWN_MOVES:
        needle = f"{REPO}/{old}"
        if needle in new:
            new = new.replace(needle, f"{REPO}/{dst}")
            changes.append(f"{old} -> {dst}")
    return new, changes


def scan(apply: bool) -> int:
    envs = conda_envs()
    print(f"scanning {len(envs)} conda env(s) against {len(KNOWN_MOVES)} known moves\n")
    total_files = 0
    total_envs = 0
    for name, sp in envs.items():
        hits: list[tuple[Path, list[str], bool]] = []
        for pattern, functional in [(g, True) for g in FUNCTIONAL_GLOBS] + [
            (g, False) for g in METADATA_GLOBS
        ]:
            for path in sp.glob(pattern):
                try:
                    text = path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                new, changes = rewrite(text)
                if changes:
                    hits.append((path, changes, functional))
                    if apply:
                        path.write_text(new, encoding="utf-8")
        if not hits:
            continue
        total_envs += 1
        print(f"--- {name}")
        for path, changes, functional in hits:
            total_files += 1
            tag = "FUNCTIONAL" if functional else "metadata  "
            print(f"    [{tag}] {path.name}: {', '.join(changes)}")
    verb = "rewrote" if apply else "would rewrite"
    print(f"\n{verb} {total_files} file(s) across {total_envs} env(s)")
    if not apply and total_files:
        print("re-run with --apply to fix, then --verify")
    return 0


def verify() -> int:
    envs = conda_envs()
    print("import check (LD_LIBRARY_PATH is prefixed with the env's lib)\n")
    bad = 0
    for name, mods in VERIFY_MATRIX.items():
        if name not in envs:
            print(f"{name:<16} SKIP (env not present)")
            continue
        script = (
            "import importlib,sys\n"
            f"mods={mods!r}\n"
            "for m in mods:\n"
            "    try:\n"
            "        importlib.import_module(m); print('OK:'+m)\n"
            "    except Exception as e:\n"
            "        print('FAIL:'+m+' ('+type(e).__name__+')')\n"
        )
        # Pass the probe via a temp file: quoting a multi-line script through
        # `bash -c "... python -c ..."` is too fragile.
        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(script)
            probe = fh.name
        # `conda run` avoids hardcoding a conda install location. The inner
        # bash prefixes LD_LIBRARY_PATH because the dynamic linker reads it
        # before the interpreter starts (tokenizers/ICU need the env's libstdc++).
        inner = (
            'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" && '
            f"python {probe}"
        )
        try:
            res = subprocess.run(
                ["conda", "run", "--no-capture-output", "-n", name, "bash", "-c", inner],
                capture_output=True,
                text=True,
                timeout=900,
            )
        finally:
            os.unlink(probe)
        line = " ".join(
            t for t in res.stdout.split() if t.startswith(("OK:", "FAIL:"))
        )
        if "FAIL:" in line:
            bad += 1
        print(f"{name:<16} {line or res.stderr.strip().splitlines()[-1:] or '(no output)'}")
    print(f"\n{bad} env(s) still have failing imports")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="rewrite files in place")
    ap.add_argument("--verify", action="store_true", help="only run the import check")
    args = ap.parse_args()
    if args.verify:
        return verify()
    return scan(apply=args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
