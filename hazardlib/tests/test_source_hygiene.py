"""Static checks on hazardlib/src (acceptance 6 and 7 of T01, plus rule R0/R4)."""
import ast
import importlib
import re
import sys
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
CORE = SRC / "hazardlib"
FORBIDDEN_IMPORTS = {"torch", "scipy", "mujoco", "pybullet", "sapien", "robosuite", "dynisafety",
                     "hazardhosts", "typing_extensions"}
FORBIDDEN_PREFIXES = ("isaac",)
BENCH_PACKAGES = ["libero", "simplerenv", "robotwin", "robocasa", "robodojo", "calvin"]
# The pytest plugin of the conformance suite (T05) is the only core file that may import pytest; the suite
# itself (python -m hazardlib.conformance) runs without it.
PYTEST_PLUGIN = CORE / "conformance" / "conftest.py"
# Builtins that are only subscriptable at runtime from Python 3.9.
GENERIC_BUILTINS = {"list", "dict", "tuple", "set", "frozenset", "type"}
# Frozen copy of sys.stdlib_module_names from CPython 3.11.18 (305 names); Python 3.8 has no such attribute.
STDLIB_MODULE_NAMES_PY311 = frozenset({
    "__future__", "_abc", "_aix_support", "_ast", "_asyncio", "_bisect", "_blake2", "_bootsubprocess", "_bz2",
    "_codecs", "_codecs_cn", "_codecs_hk", "_codecs_iso2022", "_codecs_jp", "_codecs_kr", "_codecs_tw",
    "_collections", "_collections_abc", "_compat_pickle", "_compression", "_contextvars", "_crypt", "_csv",
    "_ctypes", "_curses", "_curses_panel", "_datetime", "_dbm", "_decimal", "_elementtree", "_frozen_importlib",
    "_frozen_importlib_external", "_functools", "_gdbm", "_hashlib", "_heapq", "_imp", "_io", "_json",
    "_locale", "_lsprof", "_lzma", "_markupbase", "_md5", "_msi", "_multibytecodec", "_multiprocessing",
    "_opcode", "_operator", "_osx_support", "_overlapped", "_pickle", "_posixshmem", "_posixsubprocess",
    "_py_abc", "_pydecimal", "_pyio", "_queue", "_random", "_scproxy", "_sha1", "_sha256", "_sha3", "_sha512",
    "_signal", "_sitebuiltins", "_socket", "_sqlite3", "_sre", "_ssl", "_stat", "_statistics", "_string",
    "_strptime", "_struct", "_symtable", "_thread", "_threading_local", "_tkinter", "_tokenize", "_tracemalloc",
    "_typing", "_uuid", "_warnings", "_weakref", "_weakrefset", "_winapi", "_zoneinfo", "abc", "aifc",
    "antigravity", "argparse", "array", "ast", "asynchat", "asyncio", "asyncore", "atexit", "audioop", "base64",
    "bdb", "binascii", "bisect", "builtins", "bz2", "cProfile", "calendar", "cgi", "cgitb", "chunk", "cmath",
    "cmd", "code", "codecs", "codeop", "collections", "colorsys", "compileall", "concurrent", "configparser",
    "contextlib", "contextvars", "copy", "copyreg", "crypt", "csv", "ctypes", "curses", "dataclasses",
    "datetime", "dbm", "decimal", "difflib", "dis", "distutils", "doctest", "email", "encodings", "ensurepip",
    "enum", "errno", "faulthandler", "fcntl", "filecmp", "fileinput", "fnmatch", "fractions", "ftplib",
    "functools", "gc", "genericpath", "getopt", "getpass", "gettext", "glob", "graphlib", "grp", "gzip",
    "hashlib", "heapq", "hmac", "html", "http", "idlelib", "imaplib", "imghdr", "imp", "importlib", "inspect",
    "io", "ipaddress", "itertools", "json", "keyword", "lib2to3", "linecache", "locale", "logging", "lzma",
    "mailbox", "mailcap", "marshal", "math", "mimetypes", "mmap", "modulefinder", "msilib", "msvcrt",
    "multiprocessing", "netrc", "nis", "nntplib", "nt", "ntpath", "nturl2path", "numbers", "opcode", "operator",
    "optparse", "os", "ossaudiodev", "pathlib", "pdb", "pickle", "pickletools", "pipes", "pkgutil", "platform",
    "plistlib", "poplib", "posix", "posixpath", "pprint", "profile", "pstats", "pty", "pwd", "py_compile",
    "pyclbr", "pydoc", "pydoc_data", "pyexpat", "queue", "quopri", "random", "re", "readline", "reprlib",
    "resource", "rlcompleter", "runpy", "sched", "secrets", "select", "selectors", "shelve", "shlex", "shutil",
    "signal", "site", "smtpd", "smtplib", "sndhdr", "socket", "socketserver", "spwd", "sqlite3", "sre_compile",
    "sre_constants", "sre_parse", "ssl", "stat", "statistics", "string", "stringprep", "struct", "subprocess",
    "sunau", "symtable", "sys", "sysconfig", "syslog", "tabnanny", "tarfile", "telnetlib", "tempfile",
    "termios", "textwrap", "this", "threading", "time", "timeit", "tkinter", "token", "tokenize", "tomllib",
    "trace", "traceback", "tracemalloc", "tty", "turtle", "turtledemo", "types", "typing", "unicodedata",
    "unittest", "urllib", "uu", "uuid", "venv", "warnings", "wave", "weakref", "webbrowser", "winreg",
    "winsound", "wsgiref", "xdrlib", "xml", "xmlrpc", "zipapp", "zipfile", "zipimport", "zlib", "zoneinfo",
})


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
    stdlib = set(getattr(sys, "stdlib_module_names", ())) or STDLIB_MODULE_NAMES_PY311
    allowed = stdlib | {"numpy", "yaml", "hazardlib"}
    for path in _sources(CORE):
        extra = {"pytest"} if path == PYTEST_PLUGIN else set()
        for module in _imports(_parse(path)):
            assert module.split(".")[0] in allowed | extra, "%s imports %s" % (path.name, module)


def test_only_the_conformance_pytest_plugin_imports_pytest():
    importers = [path for path in _sources(CORE)
                 if any(module.split(".")[0] == "pytest" for module in _imports(_parse(path)))]
    assert importers == [PYTEST_PLUGIN]


@pytest.mark.skipif(not hasattr(sys, "stdlib_module_names"), reason="sys.stdlib_module_names needs 3.10+")
def test_stdlib_imports_of_the_sources_are_in_the_frozen_copy():
    """Keeps the 3.8 fallback above honest: what the running interpreter calls stdlib must be in the copy."""
    imported = {module.split(".")[0] for path in _sources(SRC) for module in _imports(_parse(path))}
    stdlib_used = imported & set(sys.stdlib_module_names)
    assert stdlib_used, "no stdlib import found; the scan is broken"
    assert stdlib_used <= STDLIB_MODULE_NAMES_PY311, sorted(stdlib_used - STDLIB_MODULE_NAMES_PY311)


def test_the_frozen_stdlib_copy_does_not_admit_forbidden_packages():
    assert len(STDLIB_MODULE_NAMES_PY311) == 305
    assert not (STDLIB_MODULE_NAMES_PY311 & (FORBIDDEN_IMPORTS | {"numpy", "yaml", "hazardlib"}))
    assert not any(name.startswith(FORBIDDEN_PREFIXES) for name in STDLIB_MODULE_NAMES_PY311)


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
    # Importing an adapter submodule attaches it to its parent package. That
    # does not mean __init__.py eagerly imports or exports simulator objects.
    public = [n for n, value in vars(module).items() if not n.startswith("_")
              and not (isinstance(value, types.ModuleType)
                       and value.__name__.startswith(module.__name__ + "."))]
    assert public == [], "hazardhosts.%s should be empty in T01, has %s" % (bench, public)
    assert (SRC / "hazardhosts" / bench / "__init__.py").read_text() == ""


def test_pyproject_declares_the_required_metadata():
    text = (SRC.parent / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.8"' in text
    assert 'name = "hazardlib"' in text
    deps = re.search(r"dependencies = \[(.*?)\]", text, re.S).group(1)
    assert sorted(re.findall(r'"([^"]+)"', deps)) == ["numpy", "pyyaml"]
