"""``dynisafety doctor`` — environment self-check (M10)."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.error import URLError
from urllib.request import Request, urlopen

from dynisafety.config.schema import KNOWN_BACKENDS

BackendName = Literal["libero", "simplerenv", "robotwin", "robocasa"]
Severity = Literal["fail", "warn"]

SIMPLERENV_SHA = "ccfe3809766839a2fcfb7a3d3c9abff585189188"
ROBOTWIN_SHA = "afd5f1bbc5759ac1512dc42b3cdaf8a218fb7c00"
MANISKILL2_SHA = "91d154bfd864577f8d2e80f3fc2f8b4d9df9ae5c"
LIBERO_UPSTREAM_SHA = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
ROBOCASA_SHA = "921c9a5736a8d0ea5589657898aadcfa55a6a195"
ROBOSUITE_SHA = "5ce6643f3092639d08f7b0f90ed1c6a84f50552c"

BACKEND_ENV_NAME = {
    "libero": "dynisafety-libero",
    "simplerenv": "dynisafety-simplerenv",
    "robotwin": "dynisafety-robotwin",
    "robocasa": "dynisafety-robocasa",
}

# RoboCasa kitchen assets (lightweight set). Missing → FAIL + install_robocasa.sh.
# Paths match scripts/install/install_robocasa.sh / robocasa DOWNLOAD_ASSET_REGISTRY.
_ROBOCASA_LIGHT_ASSET_SPECS: tuple[tuple[str, str, str], ...] = (
    ("asset_robocasa_tex", "textures", "tex"),
    ("asset_robocasa_tex_generative", "generative_textures", "tex_generative"),
    ("asset_robocasa_fixtures_lw", "fixtures/fridges", "fixtures_lw"),
    ("asset_robocasa_objs_lw", "objects/lightwheel", "objs_lw"),
)


def mujoco_matches_robocasa_pin(version: str) -> bool:
    """RoboCasa asserts mujoco 3.3.x (official pin 3.3.1)."""
    parts = str(version).split(".")
    return len(parts) >= 2 and parts[0] == "3" and parts[1] == "3"


def numpy_matches_robocasa_pin(version: str) -> bool:
    """RoboCasa asserts numpy 2.x (official pin 2.2.5); mutex vs other backends."""
    return str(version).split(".", 1)[0] == "2"


def robocasa_assets_root(workspace: Path) -> Path:
    return workspace / "third_party" / "robocasa" / "robocasa" / "models" / "assets"


def _dir_has_images(folder: Path) -> bool:
    if not folder.is_dir():
        return False
    return any(
        p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        for p in folder.rglob("*")
        if p.is_file()
    )


def _dir_has_meshes(folder: Path) -> bool:
    if not folder.is_dir():
        return False
    return any(
        p.suffix.lower() in {".obj", ".stl"} for p in folder.rglob("*") if p.is_file()
    )


def _robocasa_light_asset_ok(name: str, folder: Path) -> bool:
    if name == "fixtures_lw":
        return folder.is_dir() and any(folder.iterdir())
    if name == "objs_lw":
        return _dir_has_meshes(folder)
    if name in {"tex", "tex_generative"}:
        return _dir_has_images(folder)
    return folder.is_dir() and any(folder.iterdir())


_HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    ok: bool
    detail: str
    fix: str | None
    severity: Severity


def find_workspace_root(start: Path | None = None) -> Path:
    """Locate repo root (pyproject.toml + src/dynisafety)."""
    cur = (start or Path.cwd()).resolve()
    for candidate in (cur, *cur.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "dynisafety"
        ).is_dir():
            return candidate
    here = Path(__file__).resolve().parents[3]
    if (here / "pyproject.toml").is_file() and (here / "src" / "dynisafety").is_dir():
        return here
    raise FileNotFoundError(
        "Could not find dynisafety workspace root "
        "(need pyproject.toml and src/dynisafety/)"
    )


def register_doctor_parser(subparsers: argparse._SubParsersAction) -> None:
    """Flags: ``--backend`` (optional), ``--strict``, ``--online`` (store_true)."""
    p = subparsers.add_parser(
        "doctor",
        help="Environment self-check (submodule/patch/simulator/assets/…)",
    )
    p.add_argument(
        "--backend",
        default=None,
        choices=KNOWN_BACKENDS,
        help="Limit checks to one backend (omit = all)",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Treat optional asset/mirror/GPU gaps as FAIL",
    )
    p.add_argument(
        "--online",
        action="store_true",
        help="Probe Hugging Face reachability for checkpoint repo ids",
    )
    p.set_defaults(doctor_handler=dispatch_doctor)


def _optional_severity(strict: bool) -> Severity:
    return "fail" if strict else "warn"


def _backends(backend: BackendName | None) -> tuple[BackendName, ...]:
    if backend is None:
        return KNOWN_BACKENDS
    return (backend,)


def _git_head(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def check_python_torch_cuda() -> DoctorCheck:
    py = sys.version_info
    detail_parts = [f"python {py.major}.{py.minor}.{py.micro}"]
    if py < (3, 10):
        return DoctorCheck(
            name="python_torch_cuda",
            ok=False,
            detail="; ".join(detail_parts) + " (need >=3.10)",
            fix="conda create -n dynisafety-libero python=3.10 -y  # or use environment/*.yml",
            severity="fail",
        )
    torch_detail = "torch not installed"
    try:
        import torch

        torch_detail = f"torch {torch.__version__}"
        if torch.cuda.is_available():
            torch_detail += (
                f"; cuda {torch.version.cuda}; "
                f"device0={torch.cuda.get_device_name(0)}"
            )
        else:
            torch_detail += "; cuda unavailable"
    except ImportError:
        return DoctorCheck(
            name="python_torch_cuda",
            ok=False,
            detail="; ".join(detail_parts + [torch_detail]),
            fix=(
                "pip install torch --index-url "
                "https://download.pytorch.org/whl/cu124  "
                "# pick CUDA tag from requirements/README.md"
            ),
            severity="fail",
        )
    return DoctorCheck(
        name="python_torch_cuda",
        ok=True,
        detail="; ".join(detail_parts + [torch_detail]),
        fix=None,
        severity="fail",
    )


def check_submodule_pins(
    workspace: Path, backend: BackendName | None
) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    for b in _backends(backend):
        if b == "simplerenv":
            tree = workspace / "third_party" / "SimplerEnv-OpenVLA"
            head = _git_head(tree)
            ok = head == SIMPLERENV_SHA
            checks.append(
                DoctorCheck(
                    name="submodule_simplerenv",
                    ok=ok,
                    detail=(
                        f"path={tree} head={head or 'MISSING'} "
                        f"pin={SIMPLERENV_SHA[:7]}"
                    ),
                    fix=(
                        None
                        if ok
                        else f"bash {workspace}/scripts/install/install_simplerenv.sh"
                    ),
                    severity="fail",
                )
            )
            ms = tree / "ManiSkill2_real2sim"
            ms_head = _git_head(ms)
            ms_ok = ms_head == MANISKILL2_SHA
            checks.append(
                DoctorCheck(
                    name="submodule_maniskill2",
                    ok=ms_ok,
                    detail=(
                        f"path={ms} head={ms_head or 'MISSING'} "
                        f"pin={MANISKILL2_SHA[:7]}"
                    ),
                    fix=(
                        None
                        if ms_ok
                        else (
                            f"bash {workspace}/scripts/install/apply_patches.sh simplerenv"
                        )
                    ),
                    severity="fail",
                )
            )
        elif b == "robotwin":
            tree = workspace / "third_party" / "RoboTwin"
            head = _git_head(tree)
            ok = head == ROBOTWIN_SHA
            checks.append(
                DoctorCheck(
                    name="submodule_robotwin",
                    ok=ok,
                    detail=(
                        f"path={tree} head={head or 'MISSING'} "
                        f"pin={ROBOTWIN_SHA[:7]}"
                    ),
                    fix=(
                        None
                        if ok
                        else f"bash {workspace}/scripts/install/install_robotwin.sh"
                    ),
                    severity="fail",
                )
            )
            mirror = workspace / "mirrors" / "RoboTwin.git"
            # Optional unless --strict (private mirror / hosted URL).
            # Severity applied by caller via strict flag in check_assets-like path;
            # keep pin check required above; mirror is separate optional below.
            _ = mirror
        elif b == "libero":
            dest = workspace / "third_party" / "dyniSafety_libero"
            local = workspace / "external" / "simulators" / "dyniSafety_libero"
            present = dest.is_dir() or local.is_dir()
            where = str(dest if dest.is_dir() else local if local.is_dir() else dest)
            checks.append(
                DoctorCheck(
                    name="libero_tree",
                    ok=present,
                    detail=(
                        f"tree={where}; upstream_ref={LIBERO_UPSTREAM_SHA[:7]} "
                        f"(D2=B install, not submodule)"
                    ),
                    fix=(
                        None
                        if present
                        else f"bash {workspace}/scripts/install/install_libero.sh"
                    ),
                    severity="fail",
                )
            )
        elif b == "robocasa":
            tree = workspace / "third_party" / "robocasa"
            head = _git_head(tree)
            present = tree.is_dir()
            checks.append(
                DoctorCheck(
                    name="robocasa_tree",
                    ok=present,
                    detail=(
                        f"path={tree} head={head or 'MISSING'} "
                        f"pin={ROBOCASA_SHA[:7]} (install clone, not submodule)"
                    ),
                    fix=(
                        None
                        if present
                        else f"bash {workspace}/scripts/install/install_robocasa.sh"
                    ),
                    severity="fail",
                )
            )
            rs = workspace / "third_party" / "robosuite"
            rs_head = _git_head(rs)
            rs_ok = rs.is_dir()
            checks.append(
                DoctorCheck(
                    name="robosuite_tree",
                    ok=rs_ok,
                    detail=(
                        f"path={rs} head={rs_head or 'MISSING'} "
                        f"pin={ROBOSUITE_SHA[:7]} (git master, not PyPI 1.4)"
                    ),
                    fix=(
                        None
                        if rs_ok
                        else f"bash {workspace}/scripts/install/install_robocasa.sh"
                    ),
                    severity="fail",
                )
            )
    return checks


def check_robotwin_mirror(workspace: Path, *, strict: bool) -> DoctorCheck:
    mirror = workspace / "mirrors" / "RoboTwin.git"
    ok = mirror.is_dir()
    sev = _optional_severity(strict)
    return DoctorCheck(
        name="robotwin_private_mirror",
        ok=ok,
        detail=f"path={mirror} ({'present' if ok else 'missing'})",
        fix=(
            None
            if ok
            else (
                f"bash {workspace}/scripts/install/install_robotwin.sh  "
                "# rebuilds mirror if vendored tree exists"
            )
        ),
        severity=sev,
    )


def check_patches(
    workspace: Path, backend: BackendName | None
) -> list[DoctorCheck]:
    script = workspace / "scripts" / "apply_patches.sh"
    target = backend if backend is not None else "all"
    if not script.is_file():
        return [
            DoctorCheck(
                name="patches",
                ok=False,
                detail=f"missing {script}",
                fix="git checkout HEAD -- scripts/install/apply_patches.sh",
                severity="fail",
            )
        ]
    try:
        proc = subprocess.run(
            ["bash", str(script), "--check", target],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(workspace),
        )
    except OSError as exc:
        return [
            DoctorCheck(
                name="patches",
                ok=False,
                detail=f"failed to run apply_patches.sh: {exc}",
                fix=f"bash {script} --check {target}",
                severity="fail",
            )
        ]
    combined = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode == 0:
        return [
            DoctorCheck(
                name="patches",
                ok=True,
                detail=f"apply_patches.sh --check {target} → APPLIED (exit 0)",
                fix=None,
                severity="fail",
            )
        ]
    # exit 2 = MISSING/DRIFTED; other = error
    snippet = " | ".join(
        line.strip()
        for line in combined.splitlines()
        if line.strip()
        and any(k in line for k in ("MISSING", "DRIFTED", "ERROR", "APPLIED"))
    ) or combined.strip()[:240]
    return [
        DoctorCheck(
            name="patches",
            ok=False,
            detail=f"exit={proc.returncode}; {snippet}",
            fix=f"bash {workspace}/scripts/install/apply_patches.sh {target}",
            severity="fail",
        )
    ]


def check_simulator_imports(backend: BackendName | None) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    for b in _backends(backend):
        if b == "libero":
            try:
                import mujoco

                ver = getattr(mujoco, "__version__", "?")
                checks.append(
                    DoctorCheck(
                        name="sim_mujoco",
                        ok=True,
                        detail=f"mujoco {ver}",
                        fix=None,
                        severity="fail",
                    )
                )
            except ImportError as exc:
                checks.append(
                    DoctorCheck(
                        name="sim_mujoco",
                        ok=False,
                        detail=f"import mujoco failed: {exc}",
                        fix=(
                            "pip install -r requirements/libero.txt && "
                            'pip install -e ".[libero]"'
                        ),
                        severity="fail",
                    )
                )
        elif b == "simplerenv":
            try:
                import sapien

                ver = getattr(sapien, "__version__", "?")
                major = str(ver).split(".", 1)[0]
                ok = major == "2"
                checks.append(
                    DoctorCheck(
                        name="sim_sapien_simplerenv",
                        ok=ok,
                        detail=f"sapien {ver} (need 2.2.x)",
                        fix=(
                            None
                            if ok
                            else (
                                "pip install -r requirements/simplerenv.txt && "
                                'pip install -e ".[simplerenv]"  '
                                "# do not mix with robotwin sapien 3"
                            )
                        ),
                        severity="fail",
                    )
                )
            except ImportError as exc:
                checks.append(
                    DoctorCheck(
                        name="sim_sapien_simplerenv",
                        ok=False,
                        detail=f"import sapien failed: {exc}",
                        fix=(
                            "pip install -r requirements/simplerenv.txt && "
                            'pip install -e ".[simplerenv]"'
                        ),
                        severity="fail",
                    )
                )
            try:
                import mani_skill2  # noqa: F401

                checks.append(
                    DoctorCheck(
                        name="sim_maniskill2",
                        ok=True,
                        detail="mani_skill2 importable",
                        fix=None,
                        severity="fail",
                    )
                )
            except ImportError as exc:
                checks.append(
                    DoctorCheck(
                        name="sim_maniskill2",
                        ok=False,
                        detail=f"import mani_skill2 failed: {exc}",
                        fix=(
                            "bash scripts/install/install_simplerenv.sh && "
                            "bash scripts/install/apply_patches.sh simplerenv && "
                            "pip install -e "
                            "third_party/SimplerEnv-OpenVLA/ManiSkill2_real2sim"
                        ),
                        severity="fail",
                    )
                )
        elif b == "robotwin":
            try:
                import sapien

                ver = getattr(sapien, "__version__", "?")
                ok = str(ver).startswith("3.")
                checks.append(
                    DoctorCheck(
                        name="sim_sapien_robotwin",
                        ok=ok,
                        detail=f"sapien {ver} (need 3.x)",
                        fix=(
                            None
                            if ok
                            else (
                                "pip install -r requirements/robotwin.txt && "
                                'pip install -e ".[robotwin]"  '
                                "# do not mix with simplerenv sapien 2"
                            )
                        ),
                        severity="fail",
                    )
                )
            except ImportError as exc:
                checks.append(
                    DoctorCheck(
                        name="sim_sapien_robotwin",
                        ok=False,
                        detail=f"import sapien failed: {exc}",
                        fix=(
                            "pip install -r requirements/robotwin.txt && "
                            'pip install -e ".[robotwin]"'
                        ),
                        severity="fail",
                    )
                )
        elif b == "robocasa":
            install_fix = (
                f"bash {find_workspace_root()}/scripts/install/install_robocasa.sh  "
                "# env: dynisafety-robocasa (numpy 2)"
            )
            try:
                import robocasa

                ver = getattr(robocasa, "__version__", "?")
                checks.append(
                    DoctorCheck(
                        name="sim_robocasa",
                        ok=True,
                        detail=f"robocasa {ver}",
                        fix=None,
                        severity="fail",
                    )
                )
            except ImportError as exc:
                checks.append(
                    DoctorCheck(
                        name="sim_robocasa",
                        ok=False,
                        detail=f"import robocasa failed: {exc}",
                        fix=install_fix,
                        severity="fail",
                    )
                )
            try:
                import robosuite

                rs_ver = getattr(robosuite, "__version__", "?")
                checks.append(
                    DoctorCheck(
                        name="sim_robosuite",
                        ok=True,
                        detail=f"robosuite {rs_ver} (git master, not PyPI 1.4)",
                        fix=None,
                        severity="fail",
                    )
                )
            except ImportError as exc:
                checks.append(
                    DoctorCheck(
                        name="sim_robosuite",
                        ok=False,
                        detail=f"import robosuite failed: {exc}",
                        fix=install_fix,
                        severity="fail",
                    )
                )
            try:
                import mujoco

                ver = str(getattr(mujoco, "__version__", "?"))
                ok = mujoco_matches_robocasa_pin(ver)
                checks.append(
                    DoctorCheck(
                        name="sim_mujoco_robocasa",
                        ok=ok,
                        detail=f"mujoco {ver} (need 3.3.x; robocasa pin 3.3.1)",
                        fix=None if ok else install_fix,
                        severity="fail",
                    )
                )
            except ImportError as exc:
                checks.append(
                    DoctorCheck(
                        name="sim_mujoco_robocasa",
                        ok=False,
                        detail=f"import mujoco failed: {exc}",
                        fix=install_fix,
                        severity="fail",
                    )
                )
    return checks


def check_extras_mutex() -> DoctorCheck | None:
    """Detect impossible sapien 2+3 coexistence in the *current* interpreter."""
    try:
        import sapien

        ver = str(getattr(sapien, "__version__", ""))
    except ImportError:
        return None
    # Single installed sapien cannot be both; mutex is about wrong extra combo
    # across intended backends. If we somehow see an unexpected version, flag it.
    if ver.startswith("2.") or ver.startswith("3."):
        return DoctorCheck(
            name="extras_mutex",
            ok=True,
            detail=(
                f"current env has sapien {ver}; "
                "use separate conda envs for simplerenv vs robotwin; "
                "robocasa is a fourth env (numpy 2, no sapien mix)"
            ),
            fix=None,
            severity="fail",
        )
    return DoctorCheck(
        name="extras_mutex",
        ok=False,
        detail=f"unexpected sapien version {ver!r}",
        fix=(
            "conda deactivate; create env from "
            "environment/dynisafety-simplerenv.yml OR "
            "environment/dynisafety-robotwin.yml (not both); "
            "robocasa uses environment/dynisafety-robocasa.yml"
        ),
        severity="fail",
    )


def check_numpy_mutex(backend: BackendName | None) -> DoctorCheck:
    """NumPy 2 (robocasa) cannot share an env with the other backends."""
    try:
        import numpy as np

        ver = str(getattr(np, "__version__", "?"))
    except ImportError:
        return DoctorCheck(
            name="numpy_mutex",
            ok=False,
            detail="numpy not installed",
            fix="pip install -e .  # or env dynisafety-robocasa + install_robocasa.sh",
            severity="fail",
        )
    backends = _backends(backend)
    only_robocasa = backends == ("robocasa",)
    if only_robocasa:
        ok = numpy_matches_robocasa_pin(ver)
        return DoctorCheck(
            name="numpy_mutex",
            ok=ok,
            detail=(
                f"numpy {ver} (robocasa pin 2.2.5; "
                "mutually exclusive with libero/simplerenv/robotwin)"
            ),
            fix=(
                None
                if ok
                else (
                    "bash "
                    f"{find_workspace_root()}/scripts/install/install_robocasa.sh  "
                    "# env: dynisafety-robocasa; numpy 2; "
                    "do not pip install -e . without --no-deps"
                )
            ),
            severity="fail",
        )
    return DoctorCheck(
        name="numpy_mutex",
        ok=True,
        detail=(
            f"numpy {ver}; robocasa needs numpy 2 in dynisafety-robocasa "
            "(mutually exclusive with this interpreter)"
        ),
        fix=None,
        severity="fail",
    )


def check_assets(
    workspace: Path, backend: BackendName | None, *, strict: bool
) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    sev_opt = _optional_severity(strict)
    mano = workspace / "assets" / "mano_v1_2"
    mano_ok = (mano / "LICENSE.txt").is_file() or (mano / "models").is_dir()
    checks.append(
        DoctorCheck(
            name="asset_mano",
            ok=mano_ok,
            detail=f"MANO dir {mano} ({'present' if mano_ok else 'missing'})",
            fix=(
                None
                if mano_ok
                else f"python {workspace}/scripts/install/download_assets.py --mano-notice"
            ),
            severity=sev_opt,
        )
    )
    for b in _backends(backend):
        if b == "robotwin":
            mesh = (
                workspace
                / "src"
                / "dynisafety"
                / "backends"
                / "robotwin"
                / "assets"
                / "hand_objects"
                / "hand_pusher"
                / "visual"
                / "hand_pusher_vis.stl"
            )
            mesh_ok = mesh.is_file()
            checks.append(
                DoctorCheck(
                    name="asset_robotwin_hand_mesh",
                    ok=mesh_ok,
                    detail=f"hand mesh {mesh}",
                    fix=(
                        None
                        if mesh_ok
                        else (
                            "ensure src/dynisafety/backends/robotwin/assets/hand_objects is present "
                            "(tracked hand_pusher assets)"
                        )
                    ),
                    severity="fail",
                )
            )
            assets_root = workspace / "external" / "simulators" / "RoboTwin" / "assets"
            # Large sim assets are optional for clone hygiene but needed for live eval.
            big_ok = assets_root.is_dir() and any(assets_root.iterdir())
            checks.append(
                DoctorCheck(
                    name="asset_robotwin_sim_assets",
                    ok=big_ok,
                    detail=f"RoboTwin assets root {assets_root}",
                    fix=(
                        None
                        if big_ok
                        else (
                            f"bash {workspace}/scripts/install/install_robotwin.sh  "
                            "# then place upstream assets (~16GB); see docs/third_party.md"
                        )
                    ),
                    severity=sev_opt,
                )
            )
            checks.append(check_robotwin_mirror(workspace, strict=strict))
        if b == "libero":
            # Obstacle/hand assets live inside dyniSafety_libero after install.
            tree = workspace / "third_party" / "dyniSafety_libero"
            if not tree.is_dir():
                tree = workspace / "external" / "simulators" / "dyniSafety_libero"
            hint = tree / "libero"
            ok = hint.is_dir()
            checks.append(
                DoctorCheck(
                    name="asset_libero_package",
                    ok=ok,
                    detail=f"libero package under {tree}",
                    fix=(
                        None
                        if ok
                        else f"bash {workspace}/scripts/install/install_libero.sh"
                    ),
                    severity="fail",
                )
            )
        if b == "simplerenv":
            tree = workspace / "third_party" / "SimplerEnv-OpenVLA"
            ok = (tree / "simpler_env").is_dir() or (tree / "ManiSkill2_real2sim").is_dir()
            checks.append(
                DoctorCheck(
                    name="asset_simplerenv_tree",
                    ok=ok,
                    detail=f"SimplerEnv tree {tree}",
                    fix=(
                        None
                        if ok
                        else f"bash {workspace}/scripts/install/install_simplerenv.sh"
                    ),
                    severity="fail",
                )
            )
        if b == "robocasa":
            tree = workspace / "third_party" / "robocasa"
            pkg = tree / "robocasa"
            ok = pkg.is_dir()
            install_fix = f"bash {workspace}/scripts/install/install_robocasa.sh"
            checks.append(
                DoctorCheck(
                    name="asset_robocasa_tree",
                    ok=ok,
                    detail=f"RoboCasa package under {tree}",
                    fix=None if ok else install_fix,
                    severity="fail",
                )
            )
            # Lightweight kitchen assets: missing is FAIL (not a warning).
            assets = robocasa_assets_root(workspace)
            for check_name, rel, label in _ROBOCASA_LIGHT_ASSET_SPECS:
                folder = assets / rel
                ready = _robocasa_light_asset_ok(label, folder)
                checks.append(
                    DoctorCheck(
                        name=check_name,
                        ok=ready,
                        detail=(
                            f"{label} {folder} "
                            f"({'present' if ready else 'missing'})"
                        ),
                        fix=None if ready else install_fix,
                        severity="fail",
                    )
                )
    return checks


def _expand_workspace_placeholder(raw: str, workspace: Path) -> str:
    return raw.replace("${workspace}", str(workspace)).replace(
        "${third_party}", str(workspace / "third_party")
    )


def _looks_like_hf_repo(value: str) -> bool:
    return bool(_HF_REPO_RE.match(value)) and not value.startswith((".", "/", "~"))


def _hf_reachable(repo_id: str, timeout_s: float = 5.0) -> bool:
    url = f"https://huggingface.co/api/models/{repo_id}"
    try:
        req = Request(url, method="GET", headers={"User-Agent": "dynisafety-doctor"})
        with urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 — explicit HF API
            return 200 <= getattr(resp, "status", 200) < 300
    except (URLError, TimeoutError, OSError):
        return False


def check_checkpoints(
    workspace: Path,
    *,
    online: bool,
    strict: bool,
    backend: BackendName | None = None,
) -> list[DoctorCheck]:
    models_dir = workspace / "configs" / "models"
    if not models_dir.is_dir():
        return [
            DoctorCheck(
                name="checkpoints",
                ok=False,
                detail=f"missing {models_dir}",
                fix="git checkout HEAD -- configs/models",
                severity="fail",
            )
        ]
    try:
        import yaml
    except ImportError:
        return [
            DoctorCheck(
                name="checkpoints",
                ok=False,
                detail="pyyaml not installed",
                fix="pip install -e .",
                severity="fail",
            )
        ]

    checks: list[DoctorCheck] = []
    sev_opt = _optional_severity(strict)
    for path in sorted(models_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            checks.append(
                DoctorCheck(
                    name=f"checkpoint:{path.stem}",
                    ok=False,
                    detail=f"YAML parse error: {exc}",
                    fix=f"fix YAML syntax in {path}",
                    severity="fail",
                )
            )
            continue
        if backend is not None:
            backends = data.get("backends")
            if isinstance(backends, list) and backends and backend not in backends:
                continue
        ckpt_block = data.get("checkpoint") or {}
        raw = ckpt_block.get("default") if isinstance(ckpt_block, dict) else None
        if not raw or not isinstance(raw, str):
            continue
        # Shallow placeholder expand; nested ${env:...} left as-is → treat as unresolved.
        if "${env:" in raw:
            checks.append(
                DoctorCheck(
                    name=f"checkpoint:{path.stem}",
                    ok=False,
                    detail=f"unresolved env placeholder in checkpoint: {raw}",
                    fix=(
                        f"export the env var used in {path.name} or "
                        f"python {workspace}/scripts/install/download_assets.py --list"
                    ),
                    severity=sev_opt,
                )
            )
            continue
        expanded = _expand_workspace_placeholder(raw, workspace)
        if _looks_like_hf_repo(expanded):
            if online:
                ok = _hf_reachable(expanded)
                checks.append(
                    DoctorCheck(
                        name=f"checkpoint:{path.stem}",
                        ok=ok,
                        detail=(
                            f"HF repo {expanded} "
                            f"({'reachable' if ok else 'unreachable'})"
                        ),
                        fix=(
                            None
                            if ok
                            else (
                                f"python {workspace}/scripts/install/download_assets.py "
                                f"--list  # or fix network / repo id {expanded}"
                            )
                        ),
                        severity="fail",
                    )
                )
            else:
                checks.append(
                    DoctorCheck(
                        name=f"checkpoint:{path.stem}",
                        ok=True,
                        detail=(
                            f"HF repo id {expanded} "
                            "(offline; skipped reachability)"
                        ),
                        fix=None,
                        severity="fail",
                    )
                )
            continue
        local = Path(expanded).expanduser()
        ok = local.exists()
        checks.append(
            DoctorCheck(
                name=f"checkpoint:{path.stem}",
                ok=ok,
                detail=f"local path {local}",
                fix=(
                    None
                    if ok
                    else (
                        f"python {workspace}/scripts/install/download_assets.py --list && "
                        f"python {workspace}/scripts/install/download_assets.py --asset <name>"
                    )
                ),
                severity=sev_opt,
            )
        )
    if not checks:
        checks.append(
            DoctorCheck(
                name="checkpoints",
                ok=True,
                detail="no checkpoint.default entries found under configs/models",
                fix=None,
                severity="fail",
            )
        )
    return checks


def _conda_env_names() -> set[str]:
    conda = shutil.which("conda")
    if conda is None:
        return set()
    try:
        proc = subprocess.run(
            [conda, "env", "list", "--json"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return set()
    if proc.returncode != 0:
        return set()
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return set()
    names: set[str] = set()
    for entry in payload.get("envs", []):
        names.add(Path(entry).name)
    return names


def _model_env_names(
    workspace: Path, backend: BackendName | None
) -> set[str]:
    models_dir = workspace / "configs" / "models"
    names: set[str] = set()
    try:
        import yaml
    except ImportError:
        return names
    for path in models_dir.glob("*.yaml"):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        if backend is not None:
            backends = data.get("backends")
            if isinstance(backends, list) and backends and backend not in backends:
                continue
        extra = data.get("extra") or {}
        if not isinstance(extra, dict):
            continue
        for key in ("eval_env", "serve_env", "env"):
            val = extra.get(key)
            if isinstance(val, str) and val.strip():
                names.add(val.strip())
    return names


def check_conda_envs(
    workspace: Path, backend: BackendName | None
) -> list[DoctorCheck]:
    present = _conda_env_names()
    checks: list[DoctorCheck] = []
    if not present and shutil.which("conda") is None:
        return [
            DoctorCheck(
                name="conda",
                ok=False,
                detail="conda not found on PATH",
                fix="install Miniconda/Mambaforge and ensure `conda` is on PATH",
                severity="fail",
            )
        ]
    for b in _backends(backend):
        env_name = BACKEND_ENV_NAME[b]
        ok = env_name in present
        yml = workspace / "environment" / f"{env_name}.yml"
        checks.append(
            DoctorCheck(
                name=f"conda_backend:{env_name}",
                ok=ok,
                detail=f"conda env {env_name} ({'found' if ok else 'missing'})",
                fix=(
                    None
                    if ok
                    else f"conda env create -f {yml}"
                ),
                severity="fail",
            )
        )
    # YAML-literal server/eval env names (Q7: no alias mapping; M4 will fix drift).
    wanted = _model_env_names(workspace, backend)
    skip = set(BACKEND_ENV_NAME.values()) | {"base"}
    for name in sorted(wanted):
        if name in skip:
            continue
        ok = name in present
        checks.append(
            DoctorCheck(
                name=f"conda_yaml:{name}",
                ok=ok,
                detail=(
                    f"configs/models references env {name!r} "
                    f"({'found' if ok else 'missing'}); "
                    "name drift vs disk: align configs/models/*.yaml "
                    "extra.eval_env with an installed conda env name"
                ),
                fix=(
                    None
                    if ok
                    else (
                        f"# create upstream model env named {name!r}; "
                        f"see environment/README.md (server envs not shipped)"
                    )
                ),
                severity="fail",
            )
        )
    return checks


def check_gpus(*, strict: bool) -> DoctorCheck:
    sev = _optional_severity(strict)
    try:
        import torch

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            mem = []
            for i in range(n):
                props = torch.cuda.get_device_properties(i)
                mem.append(f"{i}:{props.name} {props.total_memory / (1024**3):.1f}GiB")
            return DoctorCheck(
                name="gpu",
                ok=True,
                detail=f"n={n}; " + "; ".join(mem),
                fix=None,
                severity="fail",
            )
    except ImportError:
        pass
    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        proc = subprocess.run(
            [nvidia, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return DoctorCheck(
                name="gpu",
                ok=True,
                detail=proc.stdout.strip().replace("\n", " | "),
                fix=None,
                severity="fail",
            )
    return DoctorCheck(
        name="gpu",
        ok=False,
        detail="no CUDA GPU detected",
        fix="nvidia-smi  # verify drivers; live eval needs a GPU",
        severity=sev,
    )


def run_doctor(
    *,
    backend: BackendName | None,
    workspace: Path,
    strict: bool,
    online: bool,
) -> list[DoctorCheck]:
    """Run the M10 check table; no file side effects."""
    checks: list[DoctorCheck] = []
    checks.append(check_python_torch_cuda())
    checks.extend(check_submodule_pins(workspace, backend))
    checks.extend(check_patches(workspace, backend))
    checks.extend(check_simulator_imports(backend))
    mutex = check_extras_mutex()
    if mutex is not None:
        checks.append(mutex)
    checks.append(check_numpy_mutex(backend))
    checks.extend(check_assets(workspace, backend, strict=strict))
    checks.extend(check_checkpoints(workspace, online=online, strict=strict, backend=backend))
    checks.extend(check_conda_envs(workspace, backend))
    checks.append(check_gpus(strict=strict))
    return checks


def format_doctor_table(checks: Sequence[DoctorCheck]) -> str:
    """Human-readable table (no run-log parsing)."""
    headers = ("STATUS", "SEV", "NAME", "DETAIL", "FIX")
    rows: list[tuple[str, str, str, str, str]] = []
    for c in checks:
        if c.ok:
            status = "OK"
        else:
            status = "FAIL" if c.severity == "fail" else "WARN"
        rows.append(
            (
                status,
                c.severity.upper(),
                c.name,
                c.detail.replace("\n", " "),
                (c.fix or "-") if not c.ok else "-",
            )
        )
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = min(max(widths[i], len(cell)), 100 if i >= 3 else 24)

    def fmt(row: Sequence[str]) -> str:
        parts = []
        for i, cell in enumerate(row):
            text = cell if len(cell) <= widths[i] else cell[: widths[i] - 1] + "…"
            parts.append(text.ljust(widths[i]))
        return "  ".join(parts)

    lines = [fmt(headers), fmt(tuple("-" * w for w in widths))]
    lines.extend(fmt(r) for r in rows)
    n_fail = sum(1 for c in checks if (not c.ok) and c.severity == "fail")
    n_warn = sum(1 for c in checks if (not c.ok) and c.severity == "warn")
    lines.append(f"summary: {len(checks)} checks, {n_fail} fail, {n_warn} warn")
    return "\n".join(lines)


def doctor_exit_code(checks: Sequence[DoctorCheck]) -> int:
    """Non-zero iff any severity=fail check is not ok."""
    return 1 if any((not c.ok) and c.severity == "fail" for c in checks) else 0


def dispatch_doctor(args: argparse.Namespace) -> int:
    workspace = find_workspace_root()
    backend = args.backend
    strict = bool(getattr(args, "strict", False))
    online = bool(getattr(args, "online", False))
    checks = run_doctor(
        backend=backend,
        workspace=workspace,
        strict=strict,
        online=online,
    )
    print(format_doctor_table(checks))
    code = doctor_exit_code(checks)
    if code != 0:
        print(
            "dynisafety doctor: FAILED — copy FIX column commands to remediate",
            file=sys.stderr,
        )
    else:
        warn_n = sum(1 for c in checks if (not c.ok) and c.severity == "warn")
        if warn_n:
            print(
                f"dynisafety doctor: OK with {warn_n} warning(s) "
                "(re-run with --strict to promote)",
                file=sys.stderr,
            )
        else:
            print("dynisafety doctor: OK", file=sys.stderr)
    return code
