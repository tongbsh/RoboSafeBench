"""Stopping-time calibration math (paradigm 00 §7). Pure numpy, SI units, no defaults.

``T_stop`` is measured from the instant the benchmark's native hold action is issued to the first moment
the end-effector speed is below a threshold and stays there. ``Sigma = tau / T_stop`` is the embodiment-
normalised protection margin. ``staircase_update`` is one step of the simple up-down method used to
locate the tau at which contact probability is about 50% (the experiment itself is not part of T02).
"""
import dataclasses
from typing import Any, Optional, Sequence

import numpy as np

from .._numeric import finite_array, finite_float, non_negative_float, positive_float, positive_int


def estimate_t_stop(times_s: Any, speeds_mps: Any, v_threshold: float, k: int,
                    t_hold_s: float) -> Optional[float]:
    """Time from ``t_hold_s`` to the first sample of the first run of ``k`` consecutive samples whose
    speed is strictly below ``v_threshold``; ``None`` if there is no such run (never stopped within the
    record).

    Samples before ``t_hold_s`` are ignored. A NaN speed is a missing sample: it is not "below the
    threshold", so it breaks a run (a dropout can never make the arm look stopped earlier). The
    resolution is the sample spacing: the answer is a sample time, not an interpolated crossing.
    ``times_s`` must be finite and non-decreasing; speeds must be >= 0 or NaN.
    """
    times = finite_array("times_s", times_s)
    speeds = np.asarray(speeds_mps, dtype=np.float64)
    if times.ndim != 1 or speeds.shape != times.shape:
        raise ValueError("times_s and speeds_mps must be 1-D arrays of equal length, got %s and %s"
                         % (times.shape, speeds.shape))
    if times.size > 1 and not np.all(np.diff(times) >= 0):
        raise ValueError("times_s must be non-decreasing")
    if np.any(np.isinf(speeds)) or np.any(speeds[~np.isnan(speeds)] < 0):
        raise ValueError("speeds_mps must be finite and >= 0 (NaN marks a missing sample)")
    threshold = positive_float("v_threshold", v_threshold)
    run_length = positive_int("k", k)
    t_hold = finite_float("t_hold_s", t_hold_s)

    keep = times >= t_hold
    times = times[keep]
    below = speeds[keep] < threshold  # NaN < x is False: a missing sample breaks the run
    run = 0
    for index in range(times.size):
        run = run + 1 if below[index] else 0
        if run == run_length:
            return float(times[index - run_length + 1] - t_hold)
    return None


@dataclasses.dataclass(eq=False)
class TStopSummary:
    """Median and inter-quartile range of repeated ``T_stop`` measurements.

    ``n_censored`` counts repetitions that never stopped (``None``); they are excluded from the
    statistics, so with ``n_censored > 0`` the median is optimistic and callers must not treat the
    summary as complete. Quartiles use numpy's default linear interpolation. The statistics are
    ``None`` when there are no valid repetitions.
    """
    n_total: int
    n_valid: int
    n_censored: int
    median_s: Optional[float]
    q1_s: Optional[float]
    q3_s: Optional[float]
    iqr_s: Optional[float]


def aggregate_t_stop(samples: Sequence[Optional[float]]) -> TStopSummary:
    """Median / quartiles over ``samples`` (floats >= 0, or ``None`` for "never stopped")."""
    values = []
    censored = 0
    for index, sample in enumerate(samples):
        if sample is None:
            censored += 1
        else:
            values.append(non_negative_float("samples[%d]" % index, sample))
    if not values:
        return TStopSummary(len(samples), 0, censored, None, None, None, None)
    q1, median, q3 = (float(v) for v in np.percentile(np.array(values), [25, 50, 75]))
    return TStopSummary(len(samples), len(values), censored, median, q1, q3, q3 - q1)


def margin_sigma(tau_s: float, t_stop_s: float) -> float:
    """``Sigma = tau / T_stop``; below 1 the hazard is physically hard to avoid for this embodiment."""
    return non_negative_float("tau_s", tau_s) / positive_float("t_stop_s", t_stop_s)


def staircase_update(tau_s: float, contact: bool, step_s: float, tau_min_s: float,
                     tau_max_s: float) -> float:
    """One simple up-down step: after a contact make the next trial easier (``tau + step``), after an
    avoided hazard make it harder (``tau - step``); the result is clamped to ``[tau_min_s, tau_max_s]``.
    The track hovers around the tau with about 50% contact probability."""
    if not isinstance(contact, (bool, np.bool_)):
        raise ValueError("contact must be a bool, got %r" % (contact,))
    tau = finite_float("tau_s", tau_s)
    step = positive_float("step_s", step_s)
    lowest = finite_float("tau_min_s", tau_min_s)
    highest = finite_float("tau_max_s", tau_max_s)
    if not lowest < highest:
        raise ValueError("tau_min_s=%r must be below tau_max_s=%r" % (lowest, highest))
    if tau < lowest or tau > highest:
        raise ValueError("tau_s=%r is outside [%r, %r]" % (tau, lowest, highest))
    moved = tau + step if contact else tau - step
    return min(highest, max(lowest, moved))
