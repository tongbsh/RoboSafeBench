"""Static checks on hazardlib/src (acceptance 6 and 7 of T01, plus rule R0/R4)."""
import ast
import importlib
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
CORE = SRC / "hazardlib"
FORBIDDEN_IMPORTS = {"torch", "scipy", "mujoco", "pybullet", "sapien", "robosuite", "dynisafety",
                     "hazardhosts", "typing_extensions"}
FORBIDDEN_PREFIXES = ("isaac",)
BENCH_PACKAGES = ["libero", "simplerenv", "robotwin", "robocasa", "robodojo", "calvin"]
# Builtins that are only subscriptable at runtime from Python 3.9.
GENERIC_BUILTINS = {"list", "dict", "tuple", "set", "frozenset", "type"}


def _sources(root):
    return sorted(root.rglob("*.py"))


def _parse(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


def test_source_files_exist():
    assert len(_sources(CORE)) >= 10


@pytest.mark.parametrize("path", _sources(CORE), ids=lambda p: str(p.relative_to(SRC)))
def test_core_imports_no_simulator_or_dynisafety(path):
    for module in _imports(_parse(path)):
        root = module.split(".")[0]
        assert root not in FORBIDDEN_IMPORTS, "%s imports %s" % (path.name, module)
        assert not root.startswith(FORBIDDEN_PREFIXES), "%s imports %s" % (path.name, module)


def test_core_dependencies_are_stdlib_numpy_yaml_or_itself_only():
    import sys
    stdlib = set(getattr(sys, "stdlib_module_names", ())) or {
        "dataclasses", "enum", "functools", "hashlib", "json", "math", "numbers", "os", "pathlib",
        "re", "typing", "__future__"}
    allowed = stdlib | {"numpy", "yaml", "hazardlib"}
    for path in _sources(CORE):
        for module in _imports(_parse(path)):
            assert module.split(".")[0] in allowed, "%s imports %s" % (path.name, module)


@pytest.mark.parametrize("path", _sources(SRC), ids=lambda p: str(p.relative_to(SRC)))
def test_python38_compatible_syntax(path):
    tree = _parse(path)
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"slots\s*=\s*True", text), "dataclass(slots=True)"
    match_node = getattr(ast, "Match", None)
    for node in ast.walk(tree):
        if match_node is not None:
            assert not isinstance(node, match_node), "match statement"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "zip":
            assert all(kw.arg != "strict" for kw in node.keywords), "zip(strict=)"
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            pytest.fail("%s:%d uses '|' (X | Y unions are unavailable on 3.8)" % (path.name, node.lineno))
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            assert node.value.id not in GENERIC_BUILTINS, (
                "%s:%d subscripts builtin %s (needs 3.9)" % (path.name, node.lineno, node.value.id))
        if isinstance(node, ast.Attribute) and node.attr in ("removeprefix", "removesuffix"):
            pytest.fail("%s:%d uses str.%s (3.9+)" % (path.name, node.lineno, node.attr))


@pytest.mark.parametrize("path", _sources(CORE), ids=lambda p: str(p.relative_to(SRC)))
def test_no_numeric_default_arguments(path):
    """Rule: functions take numeric parameters explicitly; defaults live in YAML."""
    for node in ast.walk(_parse(path)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            defaults = list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]
            for default in defaults:
                if isinstance(default, ast.Constant):
                    value = default.value
                    assert isinstance(value, bool) or not isinstance(value, (int, float)), (
                        "%s:%d numeric default argument %r" % (path.name, node.lineno, value))


def test_module_level_numeric_constants_are_the_documented_ones():
    """Numbers that may exist at module level: hash-format constants and a display cap."""
    found = {}
    for path in _sources(CORE):
        for node in _parse(path).body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, (int, float)) \
                    and not isinstance(node.value.value, bool):
                for target in node.targets:
                    found[(path.name, target.id)] = node.value.value
    assert set(found) == {("hashing.py", "HASH_FORMAT_VERSION"), ("hashing.py", "FLOAT_SIG_DIGITS"),
                          ("config.py", "_SHOW_AT_MOST")}


@pytest.mark.parametrize("bench", BENCH_PACKAGES)
def test_bench_adapter_packages_exist_and_are_empty(bench):
    module = importlib.import_module("hazardhosts." + bench)
    public = [n for n in vars(module) if not n.startswith("_")]
    assert public == [], "hazardhosts.%s should be empty in T01, has %s" % (bench, public)
    assert (SRC / "hazardhosts" / bench / "__init__.py").read_text() == ""


def test_pyproject_declares_the_required_metadata():
    text = (SRC.parent / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.8"' in text
    assert 'name = "hazardlib"' in text
    deps = re.search(r"dependencies = \[(.*?)\]", text, re.S).group(1)
    assert sorted(re.findall(r'"([^"]+)"', deps)) == ["numpy", "pyyaml"]
