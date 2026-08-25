"""Infer-side import isolation. Must run in a fresh interpreter.

The pytest process already imports ``backends.robotwin`` / ``imageio`` in
Env-side tests. An in-process ``name not in sys.modules`` check would
false-pass after those tests (or false-fail if they ran first).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _src_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(REPO_ROOT / "src")
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not prev else f"{src}{os.pathsep}{prev}"
    return env


def test_infer_bootstrap_does_not_import_robotwin_or_imageio() -> None:
    """Package + Infer entry must stay loadable in the XVLA conda (no imageio)."""
    script = r"""
import sys

import dynisafety.parallel_eval as pe
from dynisafety.parallel_eval.infer_worker import serve_inference

assert callable(serve_inference)
assert "drive_parallel_eval" in pe.__all__
assert "drive_parallel_eval" in dir(pe)
# dir() / __all__ must not pull the lazy driver.
assert "dynisafety.parallel_eval.driver" not in sys.modules
assert "dynisafety.parallel_eval.seed_warm" not in sys.modules

leaked = [
    name
    for name in sys.modules
    if name == "imageio"
    or name.startswith("imageio.")
    or name == "sapien"
    or name.startswith("sapien.")
    or name.startswith("dynisafety.backends.robotwin")
]
if leaked:
    raise SystemExit("LEAKED " + ",".join(sorted(leaked)))
print("ISOLATION_OK")
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_src_env(),
    )
    assert proc.returncode == 0, (
        f"infer bootstrap leaked Env deps\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "ISOLATION_OK" in proc.stdout


def test_package_lazy_export_drive_parallel_eval() -> None:
    """Env / parent still gets ``from dynisafety.parallel_eval import drive_parallel_eval``."""
    from dynisafety.parallel_eval import drive_parallel_eval

    assert callable(drive_parallel_eval)
