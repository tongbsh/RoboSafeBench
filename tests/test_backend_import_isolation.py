"""Registration / backend import must not load simulator packages.

These checks run in a fresh interpreter so a prior test that imported
``mujoco`` / ``sapien`` cannot mask a leak. They do not require the
simulators to be installed.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKENDS_ROOT = REPO_ROOT / "src" / "dynisafety" / "backends"
_BACKEND_PACKAGES = ("libero", "robocasa", "robotwin", "simplerenv")
_BANNED_SIMULATORS = frozenset({"mujoco", "sapien", "robosuite", "robocasa", "libero"})


def _is_type_checking_if(node: ast.If) -> bool:
    test = node.test
    return isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"


def _imported_roots(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name.split(".")[0] for alias in node.names]
    if node.module is None:
        return []
    return [node.module.split(".")[0]]


def _toplevel_simulator_imports(path: Path) -> list[tuple[int, str]]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    hits: list[tuple[int, str]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self._fn_depth = 0
            self._type_checking = 0

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._fn_depth += 1
            self.generic_visit(node)
            self._fn_depth -= 1

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._fn_depth += 1
            self.generic_visit(node)
            self._fn_depth -= 1

        def visit_If(self, node: ast.If) -> None:
            if _is_type_checking_if(node):
                self._type_checking += 1
                self.generic_visit(node)
                self._type_checking -= 1
                return
            self.generic_visit(node)

        def visit_Import(self, node: ast.Import) -> None:
            self._check(node)
            self.generic_visit(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            self._check(node)
            self.generic_visit(node)

        def _check(self, node: ast.Import | ast.ImportFrom) -> None:
            if self._fn_depth or self._type_checking:
                return
            for root in _imported_roots(node):
                if root in _BANNED_SIMULATORS:
                    hits.append((node.lineno, ast.get_source_segment(src, node) or root))

    Visitor().visit(tree)
    return hits


def _run_isolation_script(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_backend_packages_have_no_toplevel_simulator_imports() -> None:
    """Static guard: no module-level mujoco/sapien/robosuite/robocasa/libero."""
    offenders: list[str] = []
    for pkg in _BACKEND_PACKAGES:
        root = BACKENDS_ROOT / pkg
        for path in sorted(root.rglob("*.py")):
            for lineno, text in _toplevel_simulator_imports(path):
                rel = path.relative_to(REPO_ROOT).as_posix()
                offenders.append(f"{rel}:{lineno}: {text}")
    assert not offenders, "top-level simulator imports leak at registration:\n  " + "\n  ".join(
        offenders
    )


def test_populate_backends_does_not_leak_simulators() -> None:
    script = """
import sys
banned = {"mujoco", "sapien", "robosuite"}
before = set(sys.modules)
from dynisafety.core.registry import Registry
from dynisafety.backends.registry import populate_backends, BACKEND_IMPORT_ERRORS
r = Registry("backend")
populate_backends(r)
leaked = sorted({k.split(".")[0] for k in set(sys.modules) - before} & banned)
print("registered backends:", r.available())
print("import errors:", {k: type(v).__name__ for k, v in BACKEND_IMPORT_ERRORS.items()})
print("LEAKED SIMULATORS:", leaked or "none")
if leaked:
    raise SystemExit(2)
"""
    proc = _run_isolation_script(script)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "LEAKED SIMULATORS: none" in proc.stdout


def test_import_libero_backend_does_not_load_mujoco() -> None:
    script = """
import sys
from dynisafety.backends.libero.backend import LiberoBackend
print("mujoco leaked by importing LiberoBackend:", "mujoco" in sys.modules)
if "mujoco" in sys.modules:
    raise SystemExit(2)
_ = LiberoBackend
"""
    proc = _run_isolation_script(script)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "mujoco leaked by importing LiberoBackend: False" in proc.stdout
