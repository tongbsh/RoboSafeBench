"""Shared rollout sidecar for behavior diagnostics.

Four backends share one control-flow shape (see ``LiberoOracleSession`` /
``LiberoAegisSession``): ``create`` returns a live object or a disabled
no-op, per-step hooks are silent when off, and the episode payload is read
once at the end.

This module must not import ``dynisafety.backends`` — simulators are
mutually exclusive and ``tests/test_backend_import_isolation.py`` guards
that. Thresholds and identity come from ``RunConfig``; traces land next to
``episodes.jsonl`` via ``run_dir_from_cfg``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from dynisafety.config.schema import MISSING, DiagnosticsConfig, RunConfig
from dynisafety.core.rollout import run_dir_from_cfg
from dynisafety.diagnostics.analyze import analyze_trace
from dynisafety.diagnostics.probes import make_probe
from dynisafety.diagnostics.recorder import StepTraceRecorder
from dynisafety.diagnostics.schema import (
    TRACE_SCHEMA_VERSION,
    DisturbanceProbe,
    DisturbanceState,
    TraceHeader,
)
from dynisafety.diagnostics.writer import TraceWriter

_LOG = logging.getLogger(__name__)


def copy_vec(value: Sequence[Any] | None) -> tuple[float, ...] | None:
    """Defensive copy of a pose / action vector.

    Rollout observation buffers are commonly reused in place. Stash the
    pre-``env.step`` EEF (and any other pose) with this helper so a later
    ``record_step`` cannot pick up the *next* observation.
    """
    if value is None:
        return None
    return tuple(float(v) for v in value)


def _diagnostics_requested(cfg: RunConfig) -> bool:
    diag = cfg.diagnostics
    if diag is MISSING or diag is None:
        return False
    flag = getattr(diag, "enabled", MISSING)
    if flag is MISSING or flag is None:
        return False
    return bool(flag)


def _control_freq_hz(cfg: RunConfig) -> float | None:
    runtime = cfg.runtime
    if runtime is MISSING or runtime is None:
        return None
    freq = getattr(runtime, "control_freq", MISSING)
    if freq is MISSING or freq is None:
        return None
    return float(freq)


def _stride(diag: DiagnosticsConfig) -> int:
    trace = diag.trace
    if trace is MISSING or trace is None:
        return 1
    raw = getattr(trace, "stride", MISSING)
    if raw is MISSING or raw is None:
        return 1
    return int(raw)


def _cfg_str(value: Any, fallback: str = "") -> str:
    if value is MISSING or value is None:
        return fallback
    return str(value)


class DiagnosticsSession:
    """Per-episode recorder + probe + writer, fail-soft for the rollout.

    ``create(cfg, ...)`` is the only constructor backends should call.

    * ``cfg.diagnostics`` missing or ``enabled=False`` → disabled no-op;
      ``episode_diagnostics()`` is ``None`` and no directories are created.
    * Every public method swallows exceptions, logs a **one-shot** warning,
      and degrades. A failed diagnostic path never changes obs / action /
      control flow, and never aborts evaluation.
    * ``TraceWriter.write`` returning ``None`` (save off, or no run dir)
      still runs ``analyze_trace``; only persist failures drop ``trace_path``.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        diag_cfg: DiagnosticsConfig | None = None,
        recorder: StepTraceRecorder | None = None,
        probe: DisturbanceProbe | None = None,
        writer: TraceWriter | None = None,
        episode_index: int = 0,
    ) -> None:
        self.enabled = bool(enabled)
        self._diag_cfg = diag_cfg
        self._recorder = recorder
        self._probe = probe
        self._writer = writer
        self._episode_index = int(episode_index)
        self._warned = False
        self._dead = False
        self._finalized = False
        self._last_state: DisturbanceState | None = None
        self._last_t: int | None = None
        self._diagnostics: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        cfg: RunConfig,
        *,
        episode_index: int,
        controller: object | None = None,
        bound: Any | None = None,
        action_chunk_len: int | None = None,
        action_space: str | None = None,
    ) -> DiagnosticsSession:
        """Build a live session or a disabled no-op. Never raises."""
        if not _diagnostics_requested(cfg):
            return cls(enabled=False)
        try:
            return cls._create_enabled(
                cfg,
                episode_index=episode_index,
                controller=controller,
                bound=bound,
                action_chunk_len=action_chunk_len,
                action_space=action_space,
            )
        except Exception:
            _LOG.warning(
                "behavior diagnostics session create failed; disabling for this episode",
                exc_info=True,
            )
            return cls(enabled=False)

    @classmethod
    def _create_enabled(
        cls,
        cfg: RunConfig,
        *,
        episode_index: int,
        controller: object | None,
        bound: Any | None,
        action_chunk_len: int | None,
        action_space: str | None,
    ) -> DiagnosticsSession:
        if bound is not None:
            if action_chunk_len is None:
                action_chunk_len = int(bound.chunk_size)
            if action_space is None:
                action_space = str(bound.action_space)
        if action_chunk_len is None or action_space is None:
            raise ValueError(
                "DiagnosticsSession.create needs bound= or action_chunk_len+action_space"
            )

        diag_cfg = cfg.diagnostics
        if not isinstance(diag_cfg, DiagnosticsConfig):
            raise TypeError(f"cfg.diagnostics must be DiagnosticsConfig, got {type(diag_cfg)!r}")

        level = int(cfg.level)
        header = TraceHeader(
            trace_schema_version=TRACE_SCHEMA_VERSION,
            run_id=_cfg_str(cfg.run_id),
            episode_index=int(episode_index),
            backend=_cfg_str(cfg.backend),
            level=level,
            seed=int(cfg.seed),
            disturbance_triggered=False,
            t_onset=None,
            t_hazard_end=None,
            hazard_persists=None,
            action_chunk_len=int(action_chunk_len),
            action_space=str(action_space),
            control_freq_hz=_control_freq_hz(cfg),
            stride=_stride(diag_cfg),
            n_steps=0,
        )
        recorder = StepTraceRecorder(header, cfg=diag_cfg)
        probe = make_probe(_cfg_str(cfg.backend), controller, level=level)

        writer: TraceWriter | None = None
        run_dir = run_dir_from_cfg(cfg)
        if run_dir is not None:
            try:
                writer = TraceWriter(run_dir, cfg=diag_cfg)
            except Exception:
                _LOG.warning(
                    "behavior diagnostics TraceWriter init failed; analysis will continue",
                    exc_info=True,
                )

        return cls(
            enabled=True,
            diag_cfg=diag_cfg,
            recorder=recorder,
            probe=probe,
            writer=writer,
            episode_index=episode_index,
        )

    def record_step(
        self,
        *,
        t: int,
        eef_pos: Sequence[Any],
        action: Sequence[Any],
        in_contact: bool,
        force_n: float,
        impulse_ns: float,
        eef_quat: Sequence[Any] | None = None,
        base_xy_yaw: Sequence[Any] | None = None,
        eef_pos_secondary: Sequence[Any] | None = None,
        **unused: Any,
    ) -> None:
        """Copy one already-computed task step. No-op when disabled or dead.

        ``**unused`` is deliberate. An unexpected keyword raises ``TypeError``
        at the call site, *before* the body's ``try`` can demote it, so a
        backend passing a field this version does not know would abort the
        episode with ``termination="error"``. Diagnostics must never do that;
        unknown fields are dropped with a one-shot warning instead.
        """
        if not self.enabled or self._dead or self._recorder is None:
            return
        if unused:
            self._warn_once(
                f"behavior diagnostics ignoring unknown record_step fields: {sorted(unused)}"
            )
        try:
            obstacle_pos, phase = self._snapshot_fields(int(t))
            self._recorder.record(
                t=int(t),
                eef_pos=eef_pos,
                action=action,
                in_contact=in_contact,
                force_n=force_n,
                impulse_ns=impulse_ns,
                eef_quat=eef_quat,
                base_xy_yaw=base_xy_yaw,
                eef_pos_secondary=eef_pos_secondary,
                obstacle_pos=obstacle_pos,
                phase=phase,
            )
            self._last_t = int(t)
        except Exception:
            self._warn_once(
                "behavior diagnostics record_step failed; dropping diagnostics for this episode"
            )
            self._dead = True

    def finalize(
        self,
        *,
        success: bool,
        success_step: int | None,
        first_contact_step: int | None = None,
        contact_steps: int | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        """Freeze the trace, optionally persist it, then analyze.

        ``first_contact_step`` / ``contact_steps`` may be passed explicitly or
        taken from ``metrics`` (the dict ``EpisodeMetricsBuilder.finalize``
        returns). Persist failures leave ``trace_path=None`` but still analyze.
        """
        if not self.enabled or self._dead or self._finalized:
            return
        self._finalized = True
        if self._recorder is None or self._diag_cfg is None:
            return
        try:
            if metrics is not None:
                if first_contact_step is None:
                    first_contact_step = metrics.get("first_contact_step")
                if contact_steps is None:
                    contact_steps = int(metrics["contact_steps"])
            if contact_steps is None:
                contact_steps = 0

            self._refresh_snapshot()
            state = self._last_state
            t_onset = None if state is None else state.onset_step
            t_hazard_end = None if state is None else state.hazard_end_step
            hazard_persists = None if state is None else state.hazard_persists
            triggered = t_onset is not None

            trace = self._recorder.finalize(
                t_hazard_end=t_hazard_end,
                disturbance_triggered=triggered,
                hazard_persists=hazard_persists,
                t_onset=t_onset,
            )
            trace_path = self._write_trace(trace)
            diag = analyze_trace(
                trace,
                self._diag_cfg,
                success=bool(success),
                success_step=success_step,
                first_contact_step=first_contact_step,
                contact_steps=int(contact_steps),
                trace_path=trace_path,
            )
            self._diagnostics = diag.to_json_dict()
        except Exception:
            self._warn_once(
                "behavior diagnostics finalize failed; dropping diagnostics for this episode"
            )
            self._diagnostics = None

    def episode_diagnostics(self) -> dict[str, Any] | None:
        """``EpisodeRecord.diagnostics`` payload, or ``None`` when off / degraded."""
        if not self.enabled or self._dead:
            return None
        return self._diagnostics

    def _snapshot_fields(self, task_step: int) -> tuple[Any, Any]:
        if self._probe is None:
            return None, None
        try:
            state = self._probe.snapshot(task_step)
            self._last_state = state
            return state.obstacle_pos, state.phase
        except Exception:
            self._warn_once(
                "behavior diagnostics probe.snapshot failed; recording without obstacle/phase"
            )
            return None, None

    def _refresh_snapshot(self) -> None:
        if self._probe is None or self._last_t is None:
            return
        try:
            self._last_state = self._probe.snapshot(int(self._last_t))
        except Exception:
            self._warn_once(
                "behavior diagnostics probe.snapshot failed at finalize; using last latched state"
            )

    def _write_trace(self, trace: Any) -> str | None:
        if self._writer is None:
            return None
        try:
            return self._writer.write(self._episode_index, trace)
        except Exception:
            self._warn_once(
                "behavior diagnostics trace write failed; continuing without trace_path"
            )
            return None

    def _warn_once(self, message: str) -> None:
        if self._warned:
            return
        self._warned = True
        _LOG.warning(message, exc_info=True)


__all__ = [
    "DiagnosticsSession",
    "copy_vec",
]
