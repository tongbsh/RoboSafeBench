"""T_stop protocol on a host (paradigm §7; decisions D6, D8) and the protected Sigma entry point.

``run_t_stop_protocol``: in each of ``calibration.t_stop_repeats`` hazard-free control episodes the reference
policy drives the robot until its EEF speed reaches ``t_stop_protocol.trigger_speed_fraction`` x the nominal
approach speed (``TaskContext.provenance.eef_speed_median_mps``, the probe's median EEF speed); from the next
control step on the benchmark's native ``hold_action`` is issued for ``t_stop_protocol.record_s`` seconds. The
EEF speed after every control step goes to T02's ``estimate_t_stop`` (threshold
``calibration.t_stop_speed_threshold_mps``, ``calibration.t_stop_min_consecutive_samples`` samples in a row);
a repetition that never stops is censored. The k-th repetition uses the environment seed ``seed + k``.
Resolution: one control step.

With censoring (``n_censored > 0``) the reported T_stop is a **lower bound** (the median ignores the runs that
did not stop) and ``Sigma = tau / T_stop`` an **upper bound**. ``sigma_from_summary`` is the only place that
turns a T_stop summary into Sigma: it refuses censored summaries unless ``allow_censored`` is given, and then
says that the value is an upper bound. ``margin_sigma`` itself stays pure math (T02).
"""
import dataclasses
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import yaml

from .._numeric import non_negative_float, positive_float, positive_int
from .._settings import ValueReader, load_many
from ..config import ConfigError
from ..driving import EpisodeSession, describe_policy
from ..spec.task_context import TaskContext
from .stop import TStopSummary, aggregate_t_stop, estimate_t_stop, margin_sigma

CALIBRATION_VERSION = "1"


class CensoredTStopError(ValueError):
    """Sigma was requested from a censored T_stop summary without ``allow_censored``."""


def sigma_from_summary(tau_s: float, summary: TStopSummary, allow_censored: bool) -> Tuple[float, bool]:
    """``(Sigma, is_upper_bound)`` for ``tau_s`` and a T_stop summary (D6, D8)."""
    if not isinstance(summary, TStopSummary):
        raise TypeError("summary must be a TStopSummary, got %r" % (type(summary).__name__,))
    if not isinstance(allow_censored, bool):
        raise ValueError("allow_censored must be a bool, got %r" % (allow_censored,))
    if summary.median_s is None:
        raise ValueError("the T_stop summary has no valid repetition (%d of %d censored); Sigma is undefined"
                         % (summary.n_censored, summary.n_total))
    censored = summary.n_censored > 0
    if censored and not allow_censored:
        raise CensoredTStopError("T_stop is censored (%d of %d repetitions never stopped): it is a lower bound and "
                                 "Sigma would be an upper bound; pass allow_censored=True to accept that"
                                 % (summary.n_censored, summary.n_total))
    return margin_sigma(tau_s, summary.median_s), censored


def summary_to_dict(summary: TStopSummary) -> Dict[str, Any]:
    return dataclasses.asdict(summary)


def summary_from_dict(data: Mapping[str, Any]) -> TStopSummary:
    fields = [f.name for f in dataclasses.fields(TStopSummary)]
    if set(data) != set(fields):
        raise ValueError("a T_stop summary needs exactly %s, got %s" % (fields, sorted(data)))
    return TStopSummary(**{name: data[name] for name in fields})


@dataclasses.dataclass(frozen=True)
class TStopProtocolSettings:
    speed_threshold_mps: float
    min_consecutive_samples: int
    repeats: int
    trigger_speed_fraction: float
    record_s: float

    def __post_init__(self) -> None:
        for name in ("speed_threshold_mps", "trigger_speed_fraction", "record_s"):
            object.__setattr__(self, name, positive_float(name, getattr(self, name)))
        for name in ("min_consecutive_samples", "repeats"):
            object.__setattr__(self, name, positive_int(name, getattr(self, name)))

    @classmethod
    def from_config(cls, intensity: Mapping[str, Any]) -> "TStopProtocolSettings":
        reader = ValueReader("TStopProtocolSettings")
        values = {
            "speed_threshold_mps": reader.get("intensity", intensity, "calibration", "t_stop_speed_threshold_mps"),
            "min_consecutive_samples": reader.get("intensity", intensity, "calibration",
                                                  "t_stop_min_consecutive_samples"),
            "repeats": reader.get("intensity", intensity, "calibration", "t_stop_repeats"),
            "trigger_speed_fraction": reader.get("intensity", intensity, "t_stop_protocol", "trigger_speed_fraction"),
            "record_s": reader.get("intensity", intensity, "t_stop_protocol", "record_s"),
        }
        reader.done()
        try:
            return cls(**values)
        except ValueError as exc:
            raise ConfigError("TStopProtocolSettings: %s" % exc) from None

    @classmethod
    def load(cls, config_dir: Optional[Any] = None) -> "TStopProtocolSettings":
        return cls.from_config(load_many(["intensity"], config_dir)["intensity"])

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


class _HoldWhenFast:
    def __init__(self, trigger_mps: float, record_s: float) -> None:
        self.trigger_mps = trigger_mps
        self.record_s = record_s
        self.reset()

    def reset(self) -> None:
        self.hold_t = None  # type: Optional[float]
        self.hold_speed = None  # type: Optional[float]
        self.finished = False

    def decide(self, session: EpisodeSession, env: int, t_s: float, observation: Dict[str, Any], policy: Any,
               control_dt_s: float) -> Any:
        if self.hold_t is None:
            speed = float(np.linalg.norm(session.host.robot_state(env).eef_lin_vel_mps))
            if speed < self.trigger_mps:
                return policy.act(observation)
            self.hold_t, self.hold_speed = t_s, speed
        if t_s + control_dt_s >= self.hold_t + self.record_s:
            self.finished = True
        return session.host.hold_action(env)

    def summary(self) -> Dict[str, Any]:
        return {"controller": "t_stop_hold", "hold_t_s": self.hold_t, "hold_speed_mps": self.hold_speed}


@dataclasses.dataclass
class CalibrationResult:
    task_id: str
    layout_id: str
    task_context_hash: str
    bd_hash: str
    benchmark: str
    reference_policy: Dict[str, Any]
    nominal_speed_mps: float
    trigger_speed_mps: float
    settings: TStopProtocolSettings
    samples_s: List[Optional[float]]
    hold_speeds_mps: List[float]
    summary: TStopSummary

    @property
    def is_lower_bound(self) -> bool:
        return self.summary.n_censored > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calibration_version": CALIBRATION_VERSION, "benchmark": self.benchmark, "bd_hash": self.bd_hash,
            "task_id": self.task_id, "layout_id": self.layout_id, "task_context_hash": self.task_context_hash,
            "reference_policy": dict(self.reference_policy),
            "nominal_speed_mps": self.nominal_speed_mps, "nominal_speed_source": "TaskContext.provenance."
                                                                                 "eef_speed_median_mps",
            "trigger_speed_mps": self.trigger_speed_mps, "settings": self.settings.to_dict(),
            "samples_s": list(self.samples_s), "hold_speeds_mps": list(self.hold_speeds_mps),
            "t_stop": summary_to_dict(self.summary),
            "t_stop_is_lower_bound": self.is_lower_bound,
            "notes": ("T_stop is a lower bound: %d of %d repetitions never stopped; Sigma from it is an upper bound"
                      % (self.summary.n_censored, self.summary.n_total)) if self.is_lower_bound else "",
        }


def run_t_stop_protocol(session: EpisodeSession, policy: Any, task_context: TaskContext,
                        settings: TStopProtocolSettings, seed: int, env: int) -> CalibrationResult:
    nominal = task_context.provenance.get("eef_speed_median_mps")
    if nominal is None:
        raise ValueError("TaskContext %s/%s has no eef_speed_median_mps (no moving probe sample)"
                         % (task_context.task_id, task_context.layout_id))
    nominal = positive_float("eef_speed_median_mps", nominal)
    trigger = settings.trigger_speed_fraction * nominal
    horizon = float(task_context.phases[-1].t_end_s) + settings.record_s
    samples, hold_speeds = [], []  # type: List[Optional[float]], List[float]
    for k in range(settings.repeats):
        controller = _HoldWhenFast(trigger, settings.record_s)
        control_id = "t_stop:%s:%s:%d" % (task_context.task_id, task_context.layout_id, seed + k)
        result = session.run_episode(env, None, policy, task_context.task_id, task_context.layout_id, seed + k,
                                     horizon, controller=controller, control_id=control_id)
        if controller.hold_t is None:
            raise RuntimeError("T_stop repetition %d: the EEF never reached %r m/s (%r x nominal %r m/s) within %r s"
                               % (k, trigger, settings.trigger_speed_fraction, nominal, horizon))
        samples.append(estimate_t_stop(result.times_s, result.eef_speed_mps, settings.speed_threshold_mps,
                                       settings.min_consecutive_samples, controller.hold_t))
        hold_speeds.append(non_negative_float("hold speed", controller.hold_speed))
    return CalibrationResult(task_id=task_context.task_id, layout_id=task_context.layout_id,
                             task_context_hash=task_context.hash(), bd_hash=session.bd.hash(),
                             benchmark=session.bd.name, reference_policy=describe_policy(policy),
                             nominal_speed_mps=nominal, trigger_speed_mps=trigger, settings=settings,
                             samples_s=samples, hold_speeds_mps=hold_speeds, summary=aggregate_t_stop(samples))


def write_calibration(path: Any, result: CalibrationResult) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(result.to_dict(), handle, sort_keys=False)


def read_calibration_summary(path: Any) -> Tuple[TStopSummary, bool]:
    """``(summary, t_stop_is_lower_bound)`` from a ``calibration.yaml``."""
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    summary = summary_from_dict(data["t_stop"])
    flagged = data["t_stop_is_lower_bound"]
    if flagged != (summary.n_censored > 0):
        raise ValueError("%s: t_stop_is_lower_bound=%r contradicts n_censored=%r" % (path, flagged, summary.n_censored))
    return summary, flagged
