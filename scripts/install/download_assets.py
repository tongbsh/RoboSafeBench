#!/usr/bin/env python3
"""Download DyniSafety checkpoints / datasets without hard-coded machine paths.

Replaces root ``download_hf.py``. Destination resolution (D11=A):

1. ``--dest DIR``
2. env ``DYNISAFETY_CHECKPOINTS``
3. ``<workspace>/external/checkpoints`` (workspace-relative convention)

MANO hand models are **not** redistributed. Use ``--mano-notice`` to print the
MPI/MPG license summary and where to place a user-obtained copy.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class AssetSpec:
    name: str
    repo_id: str
    repo_type: str  # "model" | "dataset"
    allow_patterns: tuple[str, ...] | None
    relative_dest: str  # under resolve_dest()
    notes: str = ""


def find_workspace_root(start: Path | None = None) -> Path:
    """Locate repo root (pyproject.toml + src/dynisafety); fallback to cwd parents."""
    cur = (start or Path.cwd()).resolve()
    for candidate in (cur, *cur.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "dynisafety"
        ).is_dir():
            return candidate
    # Script lives under <root>/scripts/install/; walk up if cwd is elsewhere.
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


def list_catalog() -> list[AssetSpec]:
    """Built-in download catalog (paths relative to resolve_dest())."""
    return [
        AssetSpec(
            name="fastwam-robotwin",
            repo_id="yuanty/fastwam",
            repo_type="model",
            allow_patterns=(
                "robotwin_uncond_3cam_384.pt",
                "robotwin_uncond_3cam_384_dataset_stats.json",
                "README.md",
            ),
            relative_dest="fastwam/robotwin",
            notes="https://huggingface.co/yuanty/fastwam",
        ),
        AssetSpec(
            name="openvla-oft-rlinf-object",
            repo_id="RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-object",
            repo_type="model",
            allow_patterns=None,
            relative_dest="openvla-oft_RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-object",
        ),
        AssetSpec(
            name="openvla-oft-rlinf-goal",
            repo_id="RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-goal",
            repo_type="model",
            allow_patterns=None,
            relative_dest="openvla-oft_RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-goal",
        ),
        AssetSpec(
            name="lingbot-va-posttrain-robotwin",
            repo_id="robbyant/lingbot-va-posttrain-robotwin",
            repo_type="model",
            allow_patterns=None,
            relative_dest="lingbot-va/lingbot-va-posttrain-robotwin",
        ),
        AssetSpec(
            name="lingbot-va-base",
            repo_id="robbyant/lingbot-va-base",
            repo_type="model",
            allow_patterns=None,
            relative_dest="lingbot-va/lingbot-va-base",
        ),
        AssetSpec(
            name="internvla-m1-bridge",
            repo_id="InternRobotics/InternVLA-M1-Pretrain-RT-1-Bridge",
            repo_type="model",
            allow_patterns=(
                "checkpoints/*",
                "config.yaml",
                "dataset_statistics.json",
                "summary.jsonl",
                "README.md",
            ),
            relative_dest="InternVLA/InternVLA-M1-Pretrain-RT-1-Bridge",
            notes="Must keep config.yaml + dataset_statistics.json beside checkpoints/",
        ),
        AssetSpec(
            name="starvla-qwen-gr00t-bridge",
            repo_id="StarVLA/Qwen-GR00T-Bridge",
            repo_type="model",
            allow_patterns=None,
            relative_dest="starVLA/Qwen-GR00T-Bridge",
        ),
        AssetSpec(
            name="xvla-robotwin2",
            repo_id="2toINF/X-VLA-RoboTwin2",
            repo_type="model",
            allow_patterns=None,
            relative_dest="xVLA/X-VLA-RoboTwin2",
        ),
        AssetSpec(
            name="internvla-a1-robotwin",
            repo_id="InternRobotics/InternVLA-A1-3B-RoboTwin",
            repo_type="model",
            allow_patterns=None,
            relative_dest="InternVLA-A1-3B-RoboTwin",
        ),
        AssetSpec(
            name="fastwam-robotwin2-dataset",
            repo_id="yuanty/robotwin2.0-fastwam",
            repo_type="dataset",
            allow_patterns=None,
            relative_dest="../external/datasets/robotwin2.0-fastwam",
            notes=(
                "LeRobot v2.1 split archives; after download: "
                "cd <dir> && cat robotwin2.0.tar.gz.part-* | tar -xzf -"
            ),
        ),
        AssetSpec(
            name="bridgevla-rlbench-eval",
            repo_id="LPY/BridgeVLA_RLBench_EVAL_DATA",
            repo_type="dataset",
            allow_patterns=None,
            relative_dest="../external/datasets/rlbench_eval",
        ),
        AssetSpec(
            name="bridgevla-rlbench-ckpt",
            repo_id="LPY/BridgeVLA",
            repo_type="dataset",
            allow_patterns=("checkpoints/RLBench/*",),
            relative_dest="bridgeVLA",
        ),
    ]


def catalog_by_name(name: str) -> AssetSpec:
    for spec in list_catalog():
        if spec.name == name:
            return spec
    known = ", ".join(s.name for s in list_catalog())
    raise SystemExit(f"unknown asset {name!r}; choose from: {known}")


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


def _print_catalog() -> None:
    print(f"{'NAME':<32} {'TYPE':<8} RELATIVE_DEST")
    for spec in list_catalog():
        print(f"{spec.name:<32} {spec.repo_type:<8} {spec.relative_dest}")
        if spec.notes:
            print(f"  notes: {spec.notes}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download DyniSafety HF assets. Destination: --dest, else "
            "DYNISAFETY_CHECKPOINTS, else <workspace>/external/checkpoints."
        )
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List catalog entries and exit",
    )
    parser.add_argument(
        "--asset",
        action="append",
        dest="assets",
        metavar="NAME",
        help="Catalog asset name to download (repeatable)",
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

    if args.list:
        print(f"workspace: {workspace}")
        print(f"dest_root: {dest_root}")
        _print_catalog()
        return 0

    if not args.assets:
        parser.error("pass --asset NAME (or --list / --mano-notice)")

    for name in args.assets:
        spec = catalog_by_name(name)
        local_dir = (dest_root / spec.relative_dest).resolve()
        # Guard: relative_dest may use ../external/datasets — still under workspace.
        print(f"==> {spec.name}")
        print(f"    repo={spec.repo_id} type={spec.repo_type}")
        print(f"    local_dir={local_dir}")
        if args.dry_run:
            print("    dry-run: skip download")
            continue
        download_hf_snapshot(
            spec.repo_id,
            dest=local_dir,
            repo_type=spec.repo_type,
            allow_patterns=spec.allow_patterns,
            max_workers=args.max_workers,
        )
        print(f"    done: {local_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
