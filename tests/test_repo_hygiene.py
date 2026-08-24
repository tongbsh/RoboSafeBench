"""Repository hygiene guards (M12) — grep-class prohibitions as pytest.

Covers S1 H1–H20 / S2 matrix. Failures print path:line for PR checklists.
No simulator required.
"""

from __future__ import annotations

import ast
import re
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path

import pytest

from dynisafety.config.schema import assert_no_embedded_defaults

# ---------------------------------------------------------------------------
# Helpers (S2 signatures)
# ---------------------------------------------------------------------------


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def iter_src_py(relative_under: str = "src") -> Iterable[Path]:
    root = repo_root() / relative_under
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def file_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def rg_paths(
    pattern: str,
    roots: Sequence[Path],
    *,
    glob: str = "*.py",
) -> list[tuple[Path, int, str]]:
    """Return (path, 1-based line, line text) for regex hits under *roots*."""
    cre = re.compile(pattern)
    hits: list[tuple[Path, int, str]] = []
    for root in roots:
        if root.is_file():
            candidates = [root]
        elif root.is_dir():
            candidates = sorted(root.rglob(glob))
        else:
            continue
        for path in candidates:
            if not path.is_file():
                continue
            try:
                lines = file_text(path).splitlines()
            except UnicodeDecodeError:
                continue
            for i, line in enumerate(lines, start=1):
                if cre.search(line):
                    hits.append((path, i, line.rstrip()))
    return hits


def assert_no_matches(hits: Sequence[tuple[Path, int, str]], *, msg: str) -> None:
    if not hits:
        return
    root = repo_root()
    detail = "\n".join(
        f"  {path.relative_to(root)}:{lineno}: {text}" for path, lineno, text in hits
    )
    pytest.fail(f"{msg}\n{detail}")


def readme_relative_targets(readme: Path) -> list[str]:
    """Extract relative markdown link targets (no http/mailto/#-only)."""
    text = file_text(readme)
    targets: list[str] = []
    for _label, url in re.findall(r"\[([^\]]*)\]\(([^)]+)\)", text):
        url = url.strip()
        if url.startswith(("http://", "https://", "mailto:")):
            continue
        path_part = url.split("#", 1)[0].strip()
        if not path_part:
            continue
        targets.append(path_part)
    return targets


def git_ls_files() -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root(),
        check=True,
        capture_output=True,
    )
    if not proc.stdout:
        return []
    return [p.decode("utf-8", errors="replace") for p in proc.stdout.split(b"\0") if p]


# Q1: only ImportError diagnostic regex in registry (not log reverse-parsing).
_RE_ALLOWLIST: frozenset[tuple[str, str]] = frozenset(
    {
        ("src/dynisafety/core/registry.py", "_MISSING_MOD_RE"),
    }
)

_MEDIA_OR_CKPT_SUFFIXES: frozenset[str] = frozenset(
    {
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".onnx",
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".bin",
    }
)

# H20: per-model eval shell names (install_* / apply_patches allowed).
_EVAL_SHELL_NAME_RE = re.compile(
    r"(?i)(^|[_\-/])(eval|run_eval|inference)([_\-.]|$)|dynisafety_eval|main_inference"
)


# ---------------------------------------------------------------------------
# H7 schema
# ---------------------------------------------------------------------------


def test_schema_defaults_are_missing_only() -> None:
    assert_no_embedded_defaults()


# ---------------------------------------------------------------------------
# H2 / H5 / H6 paths & conda
# ---------------------------------------------------------------------------


def test_no_conda_activate_in_src() -> None:
    hits = rg_paths(r"conda activate", [repo_root() / "src"])
    assert_no_matches(hits, msg="H2: src/ must not contain 'conda activate'")


def test_no_miniconda_hardcode() -> None:
    roots = [repo_root() / d for d in ("src", "scripts", "configs")]
    hits = rg_paths(r"/root/miniconda3", roots, glob="*")
    assert_no_matches(
        hits, msg="H5: src/scripts/configs must not hardcode /root/miniconda3"
    )


def test_no_autodl_tmp_hardcode() -> None:
    roots = [repo_root() / d for d in ("src", "scripts", "configs")]
    hits = rg_paths(r"/root/autodl-tmp", roots, glob="*")
    assert_no_matches(
        hits, msg="H6: src/scripts/configs must not hardcode /root/autodl-tmp"
    )


# ---------------------------------------------------------------------------
# H4 libero magic constants
# ---------------------------------------------------------------------------


def test_no_libero_magic_constants() -> None:
    hits = rg_paths(
        r"\b(TASK_MAX_STEPS|LEVEL1_DROP_Z_BIAS|PER_TASK_OBSTACLE_RESCALE)\b",
        [repo_root() / "src"],
    )
    assert_no_matches(hits, msg="H4: banned LIBERO magic constants in src/")


# ---------------------------------------------------------------------------
# H3 / H10 adapters: no raw transport / PORT env defaults
# ---------------------------------------------------------------------------


def test_adapters_have_no_raw_transport() -> None:
    adapters = repo_root() / "src" / "dynisafety" / "policies" / "adapters"
    # Literal API forms from M4 acceptance + H10 PORT getenv.
    hits = rg_paths(
        r"requests\.post|websockets\.connect|zmq\.Context|"
        r"os\.environ\.get\(\s*[\"']PORT[\"']|os\.getenv\(\s*[\"']PORT[\"']",
        [adapters],
    )
    assert_no_matches(
        hits,
        msg="H3/H10: adapters must not use raw transport APIs or PORT env defaults",
    )


# ---------------------------------------------------------------------------
# H1 / H17: no log-regex re.* in core/report (whitelist registry diagnostic)
# ---------------------------------------------------------------------------


def test_core_report_no_log_regex_recompile() -> None:
    root = repo_root()
    roots = [root / "src" / "dynisafety" / "core", root / "src" / "dynisafety" / "report"]
    raw_hits = rg_paths(r"\bre\.(compile|search|match|findall)\b", roots)
    offenders: list[tuple[Path, int, str]] = []
    for path, lineno, text in raw_hits:
        rel = path.relative_to(root).as_posix()
        allowed = False
        for allow_rel, symbol in _RE_ALLOWLIST:
            if rel == allow_rel and symbol in text:
                allowed = True
                break
        if not allowed:
            offenders.append((path, lineno, text))
    assert_no_matches(
        offenders,
        msg=(
            "H1/H17: core/ and report/ must not use re.compile/search/match/findall "
            "except allowlisted ImportError diagnostics"
        ),
    )


# ---------------------------------------------------------------------------
# H8 display names in core
# ---------------------------------------------------------------------------


def test_core_has_no_display_metric_names() -> None:
    core = repo_root() / "src" / "dynisafety" / "core"
    hits = rg_paths(r"\bSR\b|FPeakMean", [core])
    assert_no_matches(
        hits, msg="H8: display names SR / FPeakMean must not appear in core/"
    )


# ---------------------------------------------------------------------------
# H9 no backends import in core/config/report
# ---------------------------------------------------------------------------


def test_core_config_report_no_backends_import() -> None:
    root = repo_root() / "src" / "dynisafety"
    roots = [root / "core", root / "config", root / "report"]
    hits = rg_paths(
        r"from\s+dynisafety\.backends|import\s+dynisafety\.backends",
        roots,
    )
    assert_no_matches(
        hits, msg="H9: core/config/report must not import dynisafety.backends"
    )


# ---------------------------------------------------------------------------
# H11 cli: no if backend == "..."
# ---------------------------------------------------------------------------


def test_cli_has_no_backend_string_switch() -> None:
    cli = repo_root() / "src" / "dynisafety" / "cli"
    hits = rg_paths(r"""if\s+backend\s*==\s*['\"]""", [cli])
    assert_no_matches(
        hits,
        msg='H11: cli/ must not use literal if backend == "..." (use registry)',
    )


# ---------------------------------------------------------------------------
# H12 single episode orchestrator
# ---------------------------------------------------------------------------


def test_single_episode_orchestrator() -> None:
    root = repo_root()
    src = root / "src" / "dynisafety"
    loop = src / "runner" / "loop.py"
    assert loop.is_file(), "runner/loop.py missing"
    loop_text = file_text(loop)
    assert re.search(r"^def run_episodes\(", loop_text, re.M), (
        "H12: runner/loop.py must define run_episodes"
    )

    for backend in ("libero", "simplerenv", "robotwin", "robocasa"):
        backend_py = src / "backends" / backend / "backend.py"
        assert backend_py.is_file(), f"missing {backend_py.relative_to(root)}"
        assert re.search(r"^\s*def rollout\(", file_text(backend_py), re.M), (
            f"H12: {backend}/backend.py must define rollout"
        )

    offenders: list[tuple[Path, int, str]] = []
    for path in iter_src_py("src/dynisafety"):
        rel = path.relative_to(root).as_posix()
        if rel == "src/dynisafety/runner/loop.py":
            # Only run_episodes is the allowed orchestrator name here;
            # DummyBackend.rollout for dry-run is fine (not run_episode*).
            for i, line in enumerate(file_text(path).splitlines(), start=1):
                if re.search(r"\bdef run_episode\(", line):
                    offenders.append((path, i, line.rstrip()))
            continue
        for i, line in enumerate(file_text(path).splitlines(), start=1):
            if re.search(r"\bdef run_episodes?\(", line):
                offenders.append((path, i, line.rstrip()))
    assert_no_matches(
        offenders,
        msg=(
            "H12: only runner/loop.py may define run_episodes; "
            "no def run_episode(s) elsewhere under src/dynisafety"
        ),
    )


# ---------------------------------------------------------------------------
# H14 nested .git / H15 MANO / H19 media
# ---------------------------------------------------------------------------


def test_no_nested_git_in_index() -> None:
    bad = [p for p in git_ls_files() if "/.git/" in p or p.endswith("/.git")]
    assert bad == [], "H14: nested .git paths must not be tracked:\n  " + "\n  ".join(
        bad
    )


def test_no_mano_assets_tracked() -> None:
    files = git_ls_files()
    bad = [
        p
        for p in files
        if re.search(r"(?i)mano", p)
        and not p.startswith("docs/")
        and "THIRD_PARTY" not in p
        and not p.endswith(".md")
    ]
    # Allow mentioning MANO only in docs/notices; block assets and convert scripts.
    assert bad == [], (
        "H15: MANO-restricted assets/scripts must not be git-tracked:\n  "
        + "\n  ".join(bad)
    )


def test_no_tracked_media_or_checkpoints() -> None:
    bad = [
        p
        for p in git_ls_files()
        if Path(p).suffix.lower() in _MEDIA_OR_CKPT_SUFFIXES
        and not p.startswith("docs/")
    ]
    assert bad == [], (
        "H19: checkpoints/videos must not be tracked:\n  " + "\n  ".join(bad)
    )


# ---------------------------------------------------------------------------
# H16 README links
# ---------------------------------------------------------------------------


def test_readme_relative_links_exist() -> None:
    readme = repo_root() / "README.md"
    assert readme.is_file()
    missing: list[str] = []
    for target in readme_relative_targets(readme):
        path = (repo_root() / target).resolve()
        try:
            path.relative_to(repo_root().resolve())
        except ValueError:
            missing.append(f"{target} (escapes repo root)")
            continue
        if not path.exists():
            missing.append(target)
    assert not missing, "H16: README relative links missing:\n  " + "\n  ".join(missing)


# ---------------------------------------------------------------------------
# H20 no per-model eval shells (install scripts exempt)
# ---------------------------------------------------------------------------


def test_scripts_no_per_model_eval_shell() -> None:
    scripts = repo_root() / "scripts"
    if not scripts.is_dir():
        return
    offenders: list[str] = []
    for path in sorted(scripts.rglob("*.sh")):
        rel = path.relative_to(repo_root()).as_posix()
        name = path.name
        if name.startswith("install_") or name in {
            "apply_patches.sh",
            "install_backend.sh",
        }:
            continue
        # Official wrappers and authorized experiment runners are not
        # per-model paper eval entrypoints (H20 targets leftover one-off shells).
        if rel.startswith((
            "scripts/debug/",
            "scripts/run/",
            "scripts/feasibility/",
        )) or name == "debug_eval.sh":
            continue
        if _EVAL_SHELL_NAME_RE.search(name) or _EVAL_SHELL_NAME_RE.search(rel):
            offenders.append(rel)
    assert not offenders, (
        "H20: per-model eval shells are forbidden under scripts/ "
        "(install_* exempt):\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Extra: adapters AST scan mirrors test_policies (defense in depth)
# ---------------------------------------------------------------------------


def test_adapters_ast_no_forbidden_transport_attrs() -> None:
    """AST-level twin of source grep for requests.post / websockets.connect / zmq.Context."""
    adapters = repo_root() / "src" / "dynisafety" / "policies" / "adapters"
    forbidden_mod = frozenset({"requests", "websockets", "zmq"})
    forbidden_attr = frozenset({"post", "connect", "Context"})
    offenders: list[str] = []
    for path in sorted(adapters.glob("*.py")):
        if path.name.startswith("_"):
            continue
        tree = ast.parse(file_text(path), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id in forbidden_mod and node.attr in forbidden_attr:
                    offenders.append(f"{path.name}:{node.lineno} {node.value.id}.{node.attr}")
    assert not offenders, "H3 AST: " + "; ".join(offenders)
