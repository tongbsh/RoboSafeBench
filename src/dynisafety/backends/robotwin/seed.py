"""Episode seed search helpers (§0.3 — unified seed_stride formula)."""

from __future__ import annotations


def episode_search_start(*, run_seed: int, seed_stride: int) -> int:
    """§0.3: seed_stride * (1 + run_seed). Never use 100000 formula."""
    if seed_stride <= 0:
        raise ValueError(f"seed_stride must be > 0, got {seed_stride!r}")
    if run_seed < 0:
        raise ValueError(f"run_seed must be >= 0, got {run_seed!r}")
    return seed_stride * (1 + run_seed)


def next_candidate_seed(current: int) -> int:
    """Expert-check rejection or between-episode advance: ``current + 1``."""
    return int(current) + 1


def seed_formula_repr(seed_stride: int) -> str:
    """Audit string, e.g. ``'seed_stride*(1+seed)'``."""
    if seed_stride <= 0:
        raise ValueError(f"seed_stride must be > 0, got {seed_stride!r}")
    return "seed_stride*(1+seed)"
