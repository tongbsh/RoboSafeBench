#!/usr/bin/env python3
"""Download DyniSafety checkpoints / datasets without hard-coded machine paths.

Replaces root ``download_hf.py``. Destination resolution (D11=A):

1. ``--dest DIR``
2. env ``DYNISAFETY_CHECKPOINTS``
3. ``<workspace>/external/checkpoints`` (workspace-relative convention)

Local folders match ``configs/models/<stem>.yaml`` ``checkpoint.default`` /
``per_suite`` (under that dest root). ``--model <stem>`` and ``--paper``
select from this catalog using ``configs/suites/paper_main.yaml``.

π₀ / π₀.₅ live on ``gs://openpi-assets`` (not Hugging Face). MANO hand
models are **not** redistributed; use ``--mano-notice``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


LIBERO_SUITES = (
    "safelibero_spatial",
    "safelibero_object",
    "safelibero_goal",
    "safelibero_long",
)


@dataclass(frozen=True)
class AssetSpec:
    name: str
    repo_id: str
    repo_type: str  # "model" | "dataset" | "gcs"
    allow_patterns: tuple[str, ...] | None
    relative_dest: str  # under resolve_dest()
    notes: str = ""
    source: str = "hf"  # "hf" | "gcs"
    models: tuple[str, ...] = ()
    suites: tuple[str, ...] = ()


class AssetSkip(RuntimeError):
    """Requested asset cannot be fetched in this environment (e.g. no gsutil)."""


def find_workspace_root(start: Path | None = None) -> Path:
    """Locate repo root (pyproject.toml + src/dynisafety); fallback to cwd parents."""
    cur = (start or Path.cwd()).resolve()
    for candidate in (cur, *cur.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "dynisafety"
        ).is_dir():
            return candidate
    # Script lives under <root>/scripts/setup/; walk up if cwd is elsewhere.
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "dynisafety"
        ).is_dir():
            return candidate
    raise SystemExit(
        "Could not find dynisafety workspace root "
        "(need pyproject.toml and src/dynisafety/)"
    )


def resolve_dest(
    *,
    cli_dest: str | None,
    env: Mapping[str, str],
    workspace: Path,
) -> Path:
    """Prefer --dest, else DYNISAFETY_CHECKPOINTS, else workspace/external/checkpoints."""
    if cli_dest:
        return Path(cli_dest).expanduser().resolve()
    if "DYNISAFETY_CHECKPOINTS" in env and env["DYNISAFETY_CHECKPOINTS"].strip():
        return Path(env["DYNISAFETY_CHECKPOINTS"]).expanduser().resolve()
    return (workspace / "external" / "checkpoints").resolve()


def _hf(
    name: str,
    repo_id: str,
    dest: str,
    *,
    models: tuple[str, ...] = (),
    suites: tuple[str, ...] = (),
    allow_patterns: tuple[str, ...] | None = None,
    repo_type: str = "model",
    notes: str = "",
) -> AssetSpec:
    return AssetSpec(
        name=name,
        repo_id=repo_id,
        repo_type=repo_type,
        allow_patterns=allow_patterns,
        relative_dest=dest,
        notes=notes,
        source="hf",
        models=models,
        suites=suites,
    )


def _gcs(
    name: str,
    uri: str,
    dest: str,
    *,
    models: tuple[str, ...] = (),
    suites: tuple[str, ...] = (),
    notes: str = "",
) -> AssetSpec:
    return AssetSpec(
        name=name,
        repo_id=uri,
        repo_type="gcs",
        allow_patterns=None,
        relative_dest=dest,
        notes=notes,
        source="gcs",
        models=models,
        suites=suites,
    )


def list_catalog() -> list[AssetSpec]:
    """Built-in download catalog (paths relative to resolve_dest())."""
    return [
        # --- paper_main: libero / openvla_oft (YAML checkpoint is the HF repo id) ---
        _hf(
            "openvla-oft-libero-spatial",
            "moojink/openvla-7b-oft-finetuned-libero-spatial",
            "openvla-oft/openvla-7b-oft-finetuned-libero-spatial",
            models=("openvla_oft",),
            suites=("safelibero_spatial",),
            notes="YAML checkpoint is the HF repo id; adapter loads via transformers",
        ),
        _hf(
            "openvla-oft-libero-object",
            "moojink/openvla-7b-oft-finetuned-libero-object",
            "openvla-oft/openvla-7b-oft-finetuned-libero-object",
            models=("openvla_oft",),
            suites=("safelibero_object",),
            notes="YAML checkpoint is the HF repo id; adapter loads via transformers",
        ),
        _hf(
            "openvla-oft-libero-goal",
            "moojink/openvla-7b-oft-finetuned-libero-goal",
            "openvla-oft/openvla-7b-oft-finetuned-libero-goal",
            models=("openvla_oft",),
            suites=("safelibero_goal",),
            notes="YAML checkpoint is the HF repo id; adapter loads via transformers",
        ),
        _hf(
            "openvla-oft-libero-10",
            "moojink/openvla-7b-oft-finetuned-libero-10",
            "openvla-oft/openvla-7b-oft-finetuned-libero-10",
            models=("openvla_oft",),
            suites=("safelibero_long",),
            notes="YAML checkpoint is the HF repo id; adapter loads via transformers",
        ),
        # --- paper_main: libero / xvla ---
        _hf(
            "xvla-libero-full",
            "2toINF/X-VLA-Libero",
            "xVLA/xvla-libero-full",
            models=("xvla",),
            suites=LIBERO_SUITES,
            notes="HF repo 2toINF/X-VLA-Libero → YAML folder xVLA/xvla-libero-full",
        ),
        # --- paper_main: starvla (three trees; HF names ≠ local folder names) ---
        _hf(
            "starvla-libero-4in1",
            "StarVLA/Qwen3-VL-OFT-LIBERO-4in1",
            "starVLA/starvla-libero-4in1",
            models=("starvla",),
            suites=LIBERO_SUITES,
            notes="HF StarVLA/Qwen3-VL-OFT-LIBERO-4in1 → YAML starVLA/starvla-libero-4in1",
        ),
        _hf(
            "starvla-robotwin2-all",
            "StarVLA/Qwen3-VL-OFT-RoboTwin2-All",
            "starVLA/starvla-robotwin2-all",
            models=("starvla",),
            suites=("main20",),
            notes="HF StarVLA/Qwen3-VL-OFT-RoboTwin2-All → YAML starVLA/starvla-robotwin2-all",
        ),
        _hf(
            "starvla-qwen-gr00t-bridge",
            "StarVLA/Qwen-GR00T-Bridge",
            "starVLA/Qwen-GR00T-Bridge",
            models=("starvla",),
            suites=("bridge",),
        ),
        # --- paper_main: pi0 / pi05 (GCS, not HF) ---
        _gcs(
            "pi0-libero",
            "gs://openpi-assets/checkpoints/pi0_libero",
            "openpi_cache/openpi-assets/checkpoints/pi0_libero",
            models=("pi0",),
            suites=LIBERO_SUITES,
            notes=(
                "Not on Hugging Face. Needs gsutil or gcloud storage, "
                "or openpi.shared.download.maybe_download with "
                "OPENPI_DATA_HOME=<dest>/openpi_cache"
            ),
        ),
        _gcs(
            "pi05-libero",
            "gs://openpi-assets/checkpoints/pi05_libero",
            "openpi_cache/openpi-assets/checkpoints/pi05_libero",
            models=("pi05",),
            suites=LIBERO_SUITES,
            notes=(
                "YAML serve path is .../pi05_libero/pi05_libero (nested orbax). "
                "Needs gsutil / gcloud / openpi maybe_download"
            ),
        ),
        # --- extra: univla (one HF repo, four LIBERO subdirs; not paper_main) ---
        _hf(
            "univla-libero-spatial",
            "qwbu/univla-7b-224-sft-libero",
            "univla/sft-libero",
            models=("univla",),
            suites=("safelibero_spatial",),
            allow_patterns=("univla-libero-spatial/**",),
        ),
        _hf(
            "univla-libero-object",
            "qwbu/univla-7b-224-sft-libero",
            "univla/sft-libero",
            models=("univla",),
            suites=("safelibero_object",),
            allow_patterns=("univla-libero-object/**",),
        ),
        _hf(
            "univla-libero-goal",
            "qwbu/univla-7b-224-sft-libero",
            "univla/sft-libero",
            models=("univla",),
            suites=("safelibero_goal",),
            allow_patterns=("univla-libero-goal/**",),
        ),
        _hf(
            "univla-libero-10",
            "qwbu/univla-7b-224-sft-libero",
            "univla/sft-libero",
            models=("univla",),
            suites=("safelibero_long",),
            allow_patterns=("univla-libero-10/**",),
        ),
        _hf(
            "univla-simpler-bridge",
            "qwbu/univla-7b-224-sft-simpler-bridge",
            "univla/sft-simpler-bridge",
            models=("univla",),
            suites=("bridge",),
            notes="In univla.yaml per_suite.bridge; not used by paper_main",
        ),
        # --- paper_main: openvla_oft_rlinf ---
        _hf(
            "openvla-oft-rlinf-spatial",
            "RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-spatial",
            "openvla-oft_RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-spatial",
            models=("openvla_oft_rlinf",),
            suites=("safelibero_spatial",),
        ),
        _hf(
            "openvla-oft-rlinf-object",
            "RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-object",
            "openvla-oft_RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-object",
            models=("openvla_oft_rlinf",),
            suites=("safelibero_object",),
        ),
        _hf(
            "openvla-oft-rlinf-goal",
            "RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-goal",
            "openvla-oft_RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-goal",
            models=("openvla_oft_rlinf",),
            suites=("safelibero_goal",),
        ),
        _hf(
            "openvla-oft-rlinf-long",
            "RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-long",
            "openvla-oft_RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-long",
            models=("openvla_oft_rlinf",),
            suites=("safelibero_long",),
        ),
        # --- paper_main: robotwin ---
        _hf(
            "internvla-a1-robotwin",
            "InternRobotics/InternVLA-A1-3B-RoboTwin",
            "InternVLA/InternVLA-A1-3B-RoboTwin",
            models=("internvla_a1",),
            suites=("main20",),
        ),
        _hf(
            "fastwam-robotwin",
            "yuanty/fastwam",
            "fastwam/robotwin",
            models=("fastwam",),
            suites=("main20",),
            allow_patterns=(
                "robotwin_uncond_3cam_384.pt",
                "robotwin_uncond_3cam_384_dataset_stats.json",
                "README.md",
            ),
            notes="https://huggingface.co/yuanty/fastwam",
        ),
        _hf(
            "lingbot-va-posttrain-robotwin",
            "robbyant/lingbot-va-posttrain-robotwin",
            "lingbot-va/lingbot-va-posttrain-robotwin",
            models=("lingbot_va",),
            suites=("main20",),
        ),
        # --- paper_main: simplerenv / internvla_m1 + xvla_official ---
        _hf(
            "internvla-m1-bridge",
            "InternRobotics/InternVLA-M1-Pretrain-RT-1-Bridge",
            "InternVLA/InternVLA-M1-Pretrain-RT-1-Bridge",
            models=("internvla_m1",),
            suites=("bridge",),
            allow_patterns=(
                "checkpoints/*",
                "config.yaml",
                "dataset_statistics.json",
                "summary.jsonl",
                "README.md",
            ),
            notes="Must keep config.yaml + dataset_statistics.json beside checkpoints/",
        ),
        _hf(
            "xvla-robotwin2",
            "2toINF/X-VLA-RoboTwin2",
            "xVLA/X-VLA-RoboTwin2",
            models=("xvla_official",),
            suites=("main20",),
        ),
        _hf(
            "xvla-simpler-widowx",
            "2toINF/X-VLA-WidowX",
            "xVLA/xvla-simpler-widowx",
            models=("xvla_official",),
            suites=("bridge",),
            notes="HF 2toINF/X-VLA-WidowX → YAML xVLA/xvla-simpler-widowx",
        ),
        # --- extras (not paper_main; --asset only unless also --model) ---
        _hf(
            "turbovla-libero",
            "H-EmbodVis/TurboVLA",
            "turbovla",
            models=("turbovla",),
            suites=LIBERO_SUITES,
            allow_patterns=(
                "checkpoints/libero/*",
                "libero_all4_stats.json",
            ),
            notes=(
                "Not in paper_main. Unified LIBERO ckpt "
                "checkpoints/libero/turbovla_libero.pth + libero_all4_stats.json. "
                "Checkpoints follow the DINOv3 license."
            ),
        ),
        _hf(
            "turbovla-dinov3-vitb",
            "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "turbovla/backbones/dinov3-vitb16",
            models=("turbovla",),
            suites=LIBERO_SUITES,
            notes="Required DINOv3 ViT-B backbone for TurboVLA LIBERO serve",
        ),
        _hf(
            "turbovla-bert-base",
            "google-bert/bert-base-uncased",
            "turbovla/backbones/bert-base-uncased",
            models=("turbovla",),
            suites=LIBERO_SUITES,
            notes="Required BERT text encoder for TurboVLA LIBERO serve",
        ),
        _hf(
            "vla-adapter-libero-spatial-pro",
            "VLA-Adapter/LIBERO-Spatial-Pro",
            "vla_adapter/LIBERO-Spatial-Pro",
            models=("vla_adapter",),
            suites=("safelibero_spatial",),
            notes="Not in paper_main. Official VLA-Adapter-Pro spatial (~3GB, gated=false).",
        ),
        _hf(
            "vla-adapter-libero-object-pro",
            "VLA-Adapter/LIBERO-Object-Pro",
            "vla_adapter/LIBERO-Object-Pro",
            models=("vla_adapter",),
            suites=("safelibero_object",),
            notes="Not in paper_main. Official VLA-Adapter-Pro object.",
        ),
        _hf(
            "vla-adapter-libero-goal-pro",
            "VLA-Adapter/LIBERO-Goal-Pro",
            "vla_adapter/LIBERO-Goal-Pro",
            models=("vla_adapter",),
            suites=("safelibero_goal",),
            notes="Not in paper_main. Official VLA-Adapter-Pro goal.",
        ),
        _hf(
            "vla-adapter-libero-long-pro",
            "VLA-Adapter/LIBERO-Long-Pro",
            "vla_adapter/LIBERO-Long-Pro",
            models=("vla_adapter",),
            suites=("safelibero_long",),
            notes="Not in paper_main. Official VLA-Adapter-Pro long (unnorm_key=libero_10_no_noops).",
        ),
        _hf(
            "lingbot-va-base",
            "robbyant/lingbot-va-base",
            "lingbot-va/lingbot-va-base",
        ),
        _hf(
            "fastwam-robotwin2-dataset",
            "yuanty/robotwin2.0-fastwam",
            "../external/datasets/robotwin2.0-fastwam",
            repo_type="dataset",
            notes=(
                "LeRobot v2.1 split archives; after download: "
                "cd <dir> && cat robotwin2.0.tar.gz.part-* | tar -xzf -"
            ),
        ),
        _hf(
            "bridgevla-rlbench-eval",
            "LPY/BridgeVLA_RLBench_EVAL_DATA",
            "../external/datasets/rlbench_eval",
            repo_type="dataset",
        ),
        _hf(
            "bridgevla-rlbench-ckpt",
            "LPY/BridgeVLA",
            "bridgeVLA",
            repo_type="dataset",
            allow_patterns=("checkpoints/RLBench/*",),
        ),
    ]


def catalog_by_name(name: str, catalog: Sequence[AssetSpec] | None = None) -> AssetSpec:
    items = list(catalog) if catalog is not None else list_catalog()
    for spec in items:
        if spec.name == name:
            return spec
    known = ", ".join(s.name for s in items)
    raise SystemExit(f"unknown asset {name!r}; choose from: {known}")


def paper_model_suites(workspace: Path) -> dict[str, set[str]]:
    """model stem → suites listed in configs/suites/paper_main.yaml."""
    path = workspace / "configs" / "suites" / "paper_main.yaml"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("PyYAML is required to read paper_main.yaml") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[str, set[str]] = {}
    for entry in data.get("matrix") or []:
        suites = [str(s) for s in (entry.get("suites") or [])]
        for model in entry.get("models") or []:
            out.setdefault(str(model), set()).update(suites)
    if not out:
        raise SystemExit(f"{path} has no models")
    return out


def spec_covers_suite(spec: AssetSpec, suite: str) -> bool:
    return not spec.suites or suite in spec.suites


def select_specs(
    *,
    workspace: Path,
    assets: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    paper: bool = False,
    suite: str | None = None,
    catalog: Sequence[AssetSpec] | None = None,
) -> list[AssetSpec]:
    """Union of --asset / --model / --paper, optionally filtered by --suite."""
    items = list(catalog) if catalog is not None else list_catalog()
    selected: list[AssetSpec] = []
    seen: set[str] = set()

    def add(spec: AssetSpec) -> None:
        if spec.name in seen:
            return
        if suite and not spec_covers_suite(spec, suite):
            return
        seen.add(spec.name)
        selected.append(spec)

    if paper:
        mapping = paper_model_suites(workspace)
        for spec in items:
            for model in spec.models:
                wanted = mapping.get(model)
                if wanted is None:
                    continue
                if spec.suites and not wanted.intersection(spec.suites):
                    continue
                add(spec)
                break

    for stem in models or []:
        hits = [spec for spec in items if stem in spec.models]
        if not hits:
            known = sorted({m for spec in items for m in spec.models})
            raise SystemExit(
                f"unknown model {stem!r}; catalog models: {', '.join(known)}"
            )
        for spec in hits:
            add(spec)

    for name in assets or []:
        spec = catalog_by_name(name, items)
        if suite and not spec_covers_suite(spec, suite):
            raise SystemExit(
                f"asset {name!r} does not cover suite {suite!r} "
                f"(suites={spec.suites or 'any'})"
            )
        if spec.name not in seen:
            seen.add(spec.name)
            selected.append(spec)

    if not selected:
        raise SystemExit(
            "no catalog entries matched; pass --asset NAME, --model STEM, "
            "and/or --paper (optional --suite SUITE)"
        )
    return selected


def local_dir_for(dest_root: Path, spec: AssetSpec) -> Path:
    return (dest_root / spec.relative_dest).resolve()


def download_hf_snapshot(
    repo_id: str,
    *,
    dest: Path,
    repo_type: str,
    allow_patterns: Sequence[str] | None,
    max_workers: int | None,
) -> Path:
    """Wrap huggingface_hub.snapshot_download; return local directory."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "huggingface_hub is required: pip install huggingface_hub"
        ) from exc

    dest.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "local_dir": str(dest),
    }
    if max_workers is not None:
        kwargs["max_workers"] = max_workers
    if allow_patterns is not None:
        kwargs["allow_patterns"] = list(allow_patterns)
    snapshot_download(**kwargs)
    return dest


def _gcs_sync_cmd(uri: str, dest: Path) -> list[str] | None:
    gsutil = shutil.which("gsutil")
    if gsutil:
        return [gsutil, "-m", "rsync", "-r", uri.rstrip("/") + "/", str(dest)]
    gcloud = shutil.which("gcloud")
    if gcloud:
        return [gcloud, "storage", "rsync", "-r", uri.rstrip("/") + "/", str(dest)]
    return None


def download_gcs(uri: str, dest: Path) -> Path:
    """Sync a gs:// prefix into dest (openpi-assets layout)."""
    dest.mkdir(parents=True, exist_ok=True)
    cmd = _gcs_sync_cmd(uri, dest)
    if cmd is None:
        raise AssetSkip(
            "gsutil/gcloud not found. Install Google Cloud SDK, or in the "
            "openpi env:\n"
            f"  export OPENPI_DATA_HOME={dest.parents[2]}\n"
            "  python -c \"from openpi.shared import download; "
            f"download.maybe_download({uri!r})\""
        )
    print(f"    cmd={' '.join(cmd)}")
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"gcs download failed (exit {completed.returncode}): {uri}")
    return dest


def print_mano_notice(*, suggested_dest: Path) -> None:
    """Print MPI MANO license summary; do not download or copy binaries."""
    text = f"""
MANO / SMPL+H — license notice (DyniSafety does NOT redistribute these assets)
==============================================================================
The MANO/SMPL+H model is owned by Max-Planck-Gesellschaft (MPI-IS). It is
licensed for non-commercial scientific research / education / artistic projects
only. Commercial use is prohibited. You must not redistribute the model files.

Obtain the model yourself from the official MANO site after accepting their
terms, then place the files under:

  {suggested_dest}

Typical layout (illustrative):

  {suggested_dest}/LICENSE.txt
  {suggested_dest}/models/MANO_LEFT.pkl
  {suggested_dest}/models/MANO_RIGHT.pkl

Do NOT commit these files to git (workspace /assets/ is gitignored).
Runtime RoboTwin L3 currently uses the small hand_pusher mesh shipped with
DyniSafety, not MANO; keep MANO only if you explicitly need it for experiments.

Official contact for commercial licensing: ps-license@tue.mpg.de
""".strip()
    print(text)


def _print_catalog(specs: Sequence[AssetSpec]) -> None:
    print(f"{'NAME':<34} {'SRC':<4} {'MODELS':<22} RELATIVE_DEST")
    for spec in specs:
        models = ",".join(spec.models) if spec.models else "-"
        print(
            f"{spec.name:<34} {spec.source:<4} {models:<22} {spec.relative_dest}"
        )
        if spec.notes:
            print(f"  notes: {spec.notes}")


def _download_one(
    spec: AssetSpec,
    *,
    dest_root: Path,
    max_workers: int | None,
    dry_run: bool,
) -> None:
    local_dir = local_dir_for(dest_root, spec)
    print(f"==> {spec.name}")
    print(f"    source={spec.source} repo={spec.repo_id} type={spec.repo_type}")
    if spec.models:
        print(f"    models={','.join(spec.models)}")
    print(f"    local_dir={local_dir}")
    if spec.notes:
        print(f"    notes: {spec.notes}")
    if dry_run:
        print("    dry-run: skip download")
        return
    if spec.source == "gcs":
        download_gcs(spec.repo_id, local_dir)
    else:
        download_hf_snapshot(
            spec.repo_id,
            dest=local_dir,
            repo_type=spec.repo_type,
            allow_patterns=spec.allow_patterns,
            max_workers=max_workers,
        )
    print(f"    done: {local_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download DyniSafety HF / GCS assets onto YAML checkpoint paths. "
            "Destination: --dest, else DYNISAFETY_CHECKPOINTS, else "
            "<workspace>/external/checkpoints."
        )
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List catalog entries and exit (honors --model / --paper / --suite)",
    )
    parser.add_argument(
        "--asset",
        "--name",
        action="append",
        dest="assets",
        metavar="NAME",
        help="Catalog asset name to download (repeatable; --name is an alias)",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        metavar="STEM",
        help="configs/models/<stem>.yaml — download every catalog entry for it",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Download paper_main weights (all stems × suites in that suite YAML)",
    )
    parser.add_argument(
        "--suite",
        default=None,
        metavar="SUITE",
        help="Keep only assets that cover this suite (e.g. main20, bridge)",
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="Checkpoint root directory (overrides DYNISAFETY_CHECKPOINTS)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="huggingface_hub max_workers (omit for library default)",
    )
    parser.add_argument(
        "--mano-notice",
        action="store_true",
        help="Print MANO license / self-download instructions and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without calling the network",
    )
    args = parser.parse_args(argv)

    workspace = find_workspace_root()
    dest_root = resolve_dest(
        cli_dest=args.dest,
        env=os.environ,
        workspace=workspace,
    )

    if args.mano_notice:
        # Suggested place is under workspace/assets (gitignored), not dest_root.
        print_mano_notice(suggested_dest=workspace / "assets" / "mano_v1_2")
        return 0

    want_select = bool(args.assets or args.models or args.paper)
    if args.list and not want_select:
        print(f"workspace: {workspace}")
        print(f"dest_root: {dest_root}")
        _print_catalog(list_catalog())
        return 0

    if not want_select:
        parser.error(
            "pass --asset NAME, --model STEM, or --paper "
            "(or --list / --mano-notice)"
        )

    selected = select_specs(
        workspace=workspace,
        assets=args.assets,
        models=args.models,
        paper=args.paper,
        suite=args.suite,
    )

    if args.list:
        print(f"workspace: {workspace}")
        print(f"dest_root: {dest_root}")
        _print_catalog(selected)
        return 0

    skipped = 0
    for spec in selected:
        try:
            _download_one(
                spec,
                dest_root=dest_root,
                max_workers=args.max_workers,
                dry_run=args.dry_run,
            )
        except AssetSkip as exc:
            skipped += 1
            print(f"    skip: {exc}", file=sys.stderr)
    if skipped:
        print(f"{skipped} asset(s) skipped (see messages above)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
