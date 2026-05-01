#!/usr/bin/env python
"""Thin InternVLA-A1 entrypoint for DyniSafety-RoboTwin.

Keeps a model-specific executable so shell launchers do not need to know
whether the implementation lives in ``inference.py`` or is later refactored
behind another shared library.
"""

from __future__ import annotations

import tyro

import inference as canonical_inference


def main(args: canonical_inference.InferenceArgs) -> None:
    canonical_inference.main(args)


if __name__ == "__main__":
    tyro.cli(main)
