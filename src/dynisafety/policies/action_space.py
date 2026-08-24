"""Declarative action-space validation and conversion (M4).

Backends call :func:`convert_action` after ``Policy.act``; adapters must not
embed backend-specific controller switches.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from dynisafety.policies.base import ActionChunk

ACTION_SPACES: frozenset[str] = frozenset(
    {"delta_ee_7d", "abs_ee_7d", "abs_ee_16d", "qpos_14d", "hybrid_mobile_12d"}
)

_EE_DIM = 7
_EE_DUAL_DIM = 16
_QPOS_DIM = 14
_HYBRID_MOBILE_DIM = 12


def dim_for_space(space: str) -> int:
    """Return action dimensionality for *space*."""
    if space in ("delta_ee_7d", "abs_ee_7d"):
        return _EE_DIM
    if space == "abs_ee_16d":
        return _EE_DUAL_DIM
    if space == "qpos_14d":
        return _QPOS_DIM
    if space == "hybrid_mobile_12d":
        return _HYBRID_MOBILE_DIM
    raise ValueError(
        f"unknown action space {space!r}; expected one of {sorted(ACTION_SPACES)}"
    )


def validate_chunk(chunk: ActionChunk) -> None:
    """Raise ``ValueError`` if *chunk* space/shape is inconsistent."""
    if chunk.space not in ACTION_SPACES:
        raise ValueError(
            f"unknown ActionChunk.space {chunk.space!r}; "
            f"expected one of {sorted(ACTION_SPACES)}"
        )
    actions = np.asarray(chunk.actions)
    if actions.ndim != 2:
        raise ValueError(
            f"ActionChunk.actions must have shape (chunk, dim), got ndim={actions.ndim}"
        )
    expected = dim_for_space(chunk.space)
    if actions.shape[1] != expected:
        raise ValueError(
            f"ActionChunk.actions dim mismatch for space={chunk.space!r}: "
            f"expected {expected}, got {actions.shape[1]}"
        )
    if actions.shape[0] < 1:
        raise ValueError("ActionChunk.actions chunk size must be >= 1")


def effective_action_space(
    *,
    configured_space: str,
    step: int,
    warmup_steps: int | None,
    post_warmup_space: str | None,
) -> str:
    """Resolve the active space for *step* given optional warm-up switch.

    When ``warmup_steps`` is ``None``, never switches (YAML omitted warm-up).
    When ``step < warmup_steps``, returns *configured_space*; otherwise
    *post_warmup_space* if set, else *configured_space*.
    """
    if configured_space not in ACTION_SPACES:
        raise ValueError(
            f"unknown configured_space {configured_space!r}; "
            f"expected one of {sorted(ACTION_SPACES)}"
        )
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError(f"step must be a non-negative int, got {step!r}")
    if warmup_steps is None:
        return configured_space
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise ValueError(f"warmup_steps must be int >= 0 or None, got {warmup_steps!r}")
    if step < warmup_steps:
        return configured_space
    if post_warmup_space is None:
        return configured_space
    if post_warmup_space not in ACTION_SPACES:
        raise ValueError(
            f"unknown post_warmup_space {post_warmup_space!r}; "
            f"expected one of {sorted(ACTION_SPACES)}"
        )
    return post_warmup_space


def _eef_state_from_context(context: Mapping[str, Any]) -> np.ndarray:
    if "eef_state" not in context:
        raise ValueError(
            "convert_action delta_ee_7d↔abs_ee_7d requires context['eef_state'] "
            "with shape (7,)"
        )
    eef = np.asarray(context["eef_state"], dtype=np.float64)
    if eef.shape != (_EE_DIM,):
        raise ValueError(
            f"context['eef_state'] must have shape (7,), got {eef.shape}"
        )
    return eef


def _convert_row(
    row: np.ndarray,
    *,
    source: str,
    target: str,
    context: Mapping[str, Any],
) -> np.ndarray:
    if source == target:
        return np.asarray(row, dtype=np.float64).copy()

    if {source, target} == {"delta_ee_7d", "abs_ee_7d"}:
        eef = _eef_state_from_context(context)
        out = np.asarray(row, dtype=np.float64).copy()
        if source == "delta_ee_7d" and target == "abs_ee_7d":
            out[:6] = eef[:6] + out[:6]
            return out
        # abs → delta
        out[:6] = out[:6] - eef[:6]
        return out

    raise ValueError(
        f"unsupported action-space conversion {source!r} → {target!r}; "
        "supported: identity, delta_ee_7d↔abs_ee_7d (needs context['eef_state']). "
        "abs_ee_16d is dual-arm identity only. "
        "hybrid_mobile_12d is PandaOmron identity only. "
        "ee↔qpos requires Backend-side FK/IK and is not performed here."
    )


def convert_action(
    chunk: ActionChunk,
    *,
    target_space: str,
    context: Mapping[str, Any],
) -> np.ndarray:
    """Convert *chunk* actions to *target_space*.

    ``context`` is supplied by the Backend (e.g. current EEF pose). Unsupported
    pairs raise ``ValueError`` (never silent identity).
    """
    validate_chunk(chunk)
    if target_space not in ACTION_SPACES:
        raise ValueError(
            f"unknown target_space {target_space!r}; "
            f"expected one of {sorted(ACTION_SPACES)}"
        )
    if not isinstance(context, Mapping):
        raise TypeError(f"context must be a mapping, got {type(context)!r}")

    actions = np.asarray(chunk.actions, dtype=np.float64)
    if chunk.space == target_space:
        return actions.copy()

    rows = [
        _convert_row(actions[i], source=chunk.space, target=target_space, context=context)
        for i in range(actions.shape[0])
    ]
    return np.stack(rows, axis=0)
