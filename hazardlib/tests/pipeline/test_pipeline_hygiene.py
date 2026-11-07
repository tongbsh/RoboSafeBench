"""The pipeline modules import no simulator and parse as Python 3.8 (T06 acceptance 8)."""
import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "hazardlib"
PIPELINE = ["driving.py", "_settings.py", "pipeline_io.py", "cli.py", "calibrate/protocol.py"] + sorted(
    p.relative_to(SRC).as_posix() for d in ("probe", "instantiate", "gates", "manifest") for p in (SRC / d).glob("*.py"))
ALLOWED_THIRD_PARTY = {"numpy", "yaml", "hazardlib"}
SIMULATORS = {"mujoco", "pybullet", "sapien", "isaacgym", "isaacsim", "robosuite", "calvin", "libero", "torch",
              "scipy", "dynisafety", "hazardhosts", "gym", "gymnasium"}


def _roots(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0]


def test_pipeline_module_list_is_complete():
    assert len(PIPELINE) >= 20 and all((SRC / rel).is_file() for rel in PIPELINE)


@pytest.mark.parametrize("rel", PIPELINE)
def test_no_simulator_imports_and_py38_syntax(rel):
    text = (SRC / rel).read_text(encoding="utf-8")
    tree = ast.parse(text, feature_version=(3, 8))
    roots = set(_roots(tree))
    assert not roots & SIMULATORS, roots & SIMULATORS
    stdlib = set(getattr(sys, "stdlib_module_names", ())) or None
    if stdlib is not None:
        assert roots <= stdlib | ALLOWED_THIRD_PARTY, roots - stdlib - ALLOWED_THIRD_PARTY


def test_importing_the_pipeline_loads_no_simulator():
    """In a fresh interpreter: other tests import ``hazardhosts`` into this one."""
    code = "import sys, hazardlib.cli; print('\\n'.join(sorted({n.split('.')[0] for n in sys.modules})))"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SRC.parent), os.environ.get("PYTHONPATH", "")]))
    proc = subprocess.run([sys.executable, "-c", code], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)
    assert proc.returncode == 0, proc.stderr
    loaded = set(proc.stdout.split())
    assert "hazardlib" in loaded and not loaded & SIMULATORS, loaded & SIMULATORS
