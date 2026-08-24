"""In-memory per-step trace accumulator. Zero simulator / RNG / render side effects.

The recorder only stores values the rollout loop has already computed. It must
never step the env, consume RNG, trigger rendering, or mutate caller objects.

Rollout observation buffers are commonly reused in place; every sequence is
copied into a new ``tuple[float, ...]`` so later in-place writes cannot corrupt
earlier steps.

``stride > 1`` records only ``t % stride == 0``. That undersamples the EEF
velocity series and therefore **harms reaction-latency precision** (a reaction
that starts on a dropped step is reported late or missed). Default stride is 1.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from dynisafety.config.schema import MISSING, DiagnosticsConfig, DiagnosticsTraceConfig
from dynisafety.diagnostics.schema import (
    TRACE_SCHEMA_VERSION,
    DisturbancePhase,
    StepRecord,
    Trace,
    TraceHeader,
    Vec3,
)

# Gripper slices implied by ``policies/action_space.py`` layouts (and matching
# ``configs/safety/oracle_stop_wait_resume.yaml`` ``gripper_indices_by_space``):
#   delta_ee_7d / abs_ee_7d : last dim
#   abs_ee_16d              : last dim of each arm (xyz+quat+grip) x 2
#   qpos_14d                : last dim of each 7-D arm
#   hybrid_mobile_12d       : [6:7]  (PandaOmron right_gripper)
#
# ``StepRecord.gripper_cmd`` is a single float, so dual-arm spaces store the
# first (left) gripper. The other arm remains available on ``action``.
GRIPPER_INDICES: dict[str, tuple[int, ...]] = {
    "delta_ee_7d": (6,),
    "abs_ee_7d": (6,),
    "abs_ee_16d": (7, 15),
    "qpos_14d": (6, 13),
    "hybrid_mobile_12d": (6,),
}

_UNSET: Any = object()


def extract_gripper_cmd(action: Sequence[Any], action_space: str) -> float | None:
    """Return the primary gripper component of ``action``, or None.

    Unknown action spaces, non-numeric input, and out-of-range slices return
    None instead of raising — diagnostics must not crash a rollout.
    """
    indices = GRIPPER_INDICES.get(action_space)
    if not indices:
        return None
    try:
        values = tuple(float(v) for v in action)
    except (TypeError, ValueError):
        return None
    idx = indices[0]
    if idx >= len(values):
        return None
    return values[idx]


def _as_float_tuple(value: Sequence[Any]) -> tuple[float, ...]:
    """Defensive copy: a new tuple of Python floats, never the caller's object."""
    return tuple(float(v) for v in value)


def _as_vec3(value: Sequence[Any], *, field: str) -> Vec3:
    out = _as_float_tuple(value)
    if len(out) != 3:
        raise ValueError(f"{field} must have 3 components, got {len(out)}")
    return (out[0], out[1], out[2])


def _as_vec3_or_none(value: Sequence[Any] | None, *, field: str) -> Vec3 | None:
    return None if value is None else _as_vec3(value, field=field)


def _as_phase(value: DisturbancePhase | str | None) -> DisturbancePhase | None:
    if value is None:
        return None
    if isinstance(value, DisturbancePhase):
        return value
    return DisturbancePhase(value)


def _clearance(eef_pos: Vec3, obstacle_pos: Vec3 | None) -> float | None:
    if obstacle_pos is None:
        return None
    return math.hypot(
        eef_pos[0] - obstacle_pos[0],
        eef_pos[1] - obstacle_pos[1],
        eef_pos[2] - obstacle_pos[2],
    )


def _trace_cfg(
    cfg: DiagnosticsTraceConfig | DiagnosticsConfig | None,
) -> DiagnosticsTraceConfig | None:
    if cfg is None:
        return None
    if isinstance(cfg, DiagnosticsTraceConfig):
        return cfg
    if isinstance(cfg, DiagnosticsConfig):
        inner = cfg.trace
        if isinstance(inner, DiagnosticsTraceConfig):
            return inner
    raise TypeError(
        "cfg must be DiagnosticsTraceConfig, DiagnosticsConfig, or None, "
        f"got {type(cfg)!r}"
    )


def _resolve_stride(
    cfg: DiagnosticsTraceConfig | None,
    stride: int | None,
) -> int:
    if stride is not None:
        resolved = int(stride)
    elif cfg is not None and cfg.stride is not MISSING and cfg.stride is not None:
        resolved = int(cfg.stride)
    else:
        resolved = 1
    if resolved < 1:
        raise ValueError(f"diagnostics.trace.stride must be >= 1, got {resolved}")
    return resolved


class StepTraceRecorder:
    """Accumulate a ``Trace`` from values the rollout already computed.

    Parameters
    ----------
    header:
        Static episode identity. ``n_steps``, ``stride``, ``t_hazard_end``,
        ``disturbance_triggered``, and ``hazard_persists`` are placeholders
        overwritten by :meth:`finalize`. ``t_onset`` is kept unless finalize
        overrides it (L1 distance-trigger is often known only at episode end).
    cfg:
        ``DiagnosticsTraceConfig`` or the parent ``DiagnosticsConfig``. Only
        ``trace.stride`` is read. May be omitted when ``stride`` is passed.
    stride:
        Explicit record stride; wins over ``cfg``. See module docstring.
    """

    def __init__(
        self,
        header: TraceHeader,
        cfg: DiagnosticsTraceConfig | DiagnosticsConfig | None = None,
        *,
        stride: int | None = None,
    ) -> None:
        self._header = header
        self._action_space = str(header.action_space)
        self._stride = _resolve_stride(_trace_cfg(cfg), stride)
        self._steps: list[StepRecord] = []
        self._finalized = False

    @property
    def stride(self) -> int:
        return self._stride

    @property
    def n_recorded(self) -> int:
        return len(self._steps)

    def record(
        self,
        *,
        t: int,
        eef_pos: Sequence[Any],
        action: Sequence[Any],
        in_contact: bool,
        force_n: float,
        impulse_ns: float,
        eef_quat: Sequence[Any] | None = None,
        eef_pos_secondary: Sequence[Any] | None = None,
        base_xy_yaw: Sequence[Any] | None = None,
        obstacle_pos: Sequence[Any] | None = None,
        gripper_cmd: float | None = None,
        phase: DisturbancePhase | str | None = None,
    ) -> None:
        """Copy one already-computed step. Keyword-only to match rollout wiring.

        Sequences (numpy arrays, lists, tuples) are copied element-wise. The
        recorder never retains a reference to a caller-owned buffer.

        ``d_eef_obstacle`` is the Euclidean distance between the copied
        ``eef_pos`` and ``obstacle_pos`` (None when the obstacle is hidden).

        When ``gripper_cmd`` is omitted, it is sliced from the copied ``action``
        using :data:`GRIPPER_INDICES`. Unknown spaces yield None.
        """
        if self._finalized:
            raise RuntimeError("StepTraceRecorder.record called after finalize")
        if int(t) % self._stride != 0:
            return

        eef = _as_vec3(eef_pos, field="eef_pos")
        action_t = _as_float_tuple(action)
        obstacle = _as_vec3_or_none(obstacle_pos, field="obstacle_pos")
        if gripper_cmd is None:
            gripper_cmd = extract_gripper_cmd(action_t, self._action_space)
        else:
            gripper_cmd = float(gripper_cmd)

        self._steps.append(
            StepRecord(
                t=int(t),
                eef_pos=eef,
                eef_quat=_as_float_tuple(eef_quat) if eef_quat is not None else None,
                eef_pos_secondary=_as_vec3_or_none(
                    eef_pos_secondary, field="eef_pos_secondary"
                ),
                base_xy_yaw=_as_vec3_or_none(base_xy_yaw, field="base_xy_yaw"),
                action=action_t,
                gripper_cmd=gripper_cmd,
                obstacle_pos=obstacle,
                d_eef_obstacle=_clearance(eef, obstacle),
                in_contact=bool(in_contact),
                force_n=float(force_n),
                impulse_ns=float(impulse_ns),
                phase=_as_phase(phase),
            )
        )

    def finalize(
        self,
        *,
        t_hazard_end: int | None,
        disturbance_triggered: bool,
        hazard_persists: bool | None,
        t_onset: Any = _UNSET,
    ) -> Trace:
        """Freeze the accumulated steps into an immutable ``Trace``.

        Overwrites the placeholder header fields that are only known after the
        rollout (and optionally ``t_onset``, which L1 learns at trigger time).
        ``stride`` is the *effective* recording stride (constructor ``stride=``
        wins over ``cfg``). ``n_steps`` is the number of *recorded* rows, which
        is smaller than the task-step count when ``stride > 1``.
        """
        if self._finalized:
            raise RuntimeError("StepTraceRecorder.finalize called more than once")
        self._finalized = True
        header = replace(
            self._header,
            trace_schema_version=TRACE_SCHEMA_VERSION,
            stride=self._stride,
            n_steps=len(self._steps),
            t_hazard_end=t_hazard_end if t_hazard_end is None else int(t_hazard_end),
            disturbance_triggered=bool(disturbance_triggered),
            hazard_persists=hazard_persists if hazard_persists is None else bool(hazard_persists),
            t_onset=(
                self._header.t_onset
                if t_onset is _UNSET
                else (None if t_onset is None else int(t_onset))
            ),
        )
        return Trace(header=header, steps=tuple(self._steps))


__all__ = [
    "GRIPPER_INDICES",
    "StepTraceRecorder",
    "extract_gripper_cmd",
]
