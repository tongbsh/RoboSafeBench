"""The pipeline modules import no simulator and parse as Python 3.8 (T06 acceptance 8)."""
import ast
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
    import hazardlib.cli  # noqa: F401  (imports every stage)
    loaded = {name.split(".")[0] for name in sys.modules}
    assert not loaded & SIMULATORS, loaded & SIMULATORS
