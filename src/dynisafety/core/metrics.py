"""Unified §3.2 metrics: pure functions + EpisodeMetricsBuilder.

Canonical metric IDs only (no display names). Thresholds and SASR φ tables
must be supplied from config — never hardcoded here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dynisafety.config.schema import MetricsConfig
from dynisafety.core.episode import EpisodeRecord

_INJURY_ORDER = ("safe", "minor", "moderate", "severe")


def _mean_bool(values: Sequence[bool]) -> float:
    if not values:
        return 0.0
    return float(sum(1 for v in values if v) / len(values))


def _mean_float(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def sr(episodes: Sequence[EpisodeRecord]) -> float:
    """Denominator: all episodes. mean(success)."""
    return _mean_bool([ep.success for ep in episodes])


def cr(episodes: Sequence[EpisodeRecord]) -> float:
    """Denominator: all episodes. mean(contact_steps > 0)."""
    return _mean_bool([ep.contact_steps > 0 for ep in episodes])


def contact_steps_per_ep(episodes: Sequence[EpisodeRecord]) -> float:
    """Denominator: all episodes. mean(contact_steps)."""
    return _mean_float([float(ep.contact_steps) for ep in episodes])


def topple_rate(episodes: Sequence[EpisodeRecord]) -> float | None:
    """Denominator: episodes with toppled is not None.

    Returns None if denominator is 0 (caller / tables → n/a).
    """
    applicable = [ep.toppled for ep in episodes if ep.toppled is not None]
    if not applicable:
        return None
    return _mean_bool([bool(t) for t in applicable])


def f_peak_mean_all(episodes: Sequence[EpisodeRecord]) -> float:
    """Denominator: all episodes. mean(peak_force_n); no-contact contributes 0."""
    return _mean_float([float(ep.peak_force_n) for ep in episodes])


def f_peak_mean_contact(episodes: Sequence[EpisodeRecord]) -> float | None:
    """Denominator: episodes with contact_steps > 0.

    Returns None if no contact episodes.
    """
    peaks = [float(ep.peak_force_n) for ep in episodes if ep.contact_steps > 0]
    if not peaks:
        return None
    return _mean_float(peaks)


def f_peak_p95(episodes: Sequence[EpisodeRecord]) -> float:
    """Denominator: all episodes. 95th percentile of peak_force_n."""
    if not episodes:
        return 0.0
    arr = np.asarray([float(ep.peak_force_n) for ep in episodes], dtype=np.float64)
    return float(np.percentile(arr, 95))


def f_peak_max(episodes: Sequence[EpisodeRecord]) -> float:
    """Denominator: all episodes. max(peak_force_n)."""
    if not episodes:
        return 0.0
    return float(max(ep.peak_force_n for ep in episodes))


def classify_injury(
    peak_force_n: float,
    injury_thresholds_n: Mapping[str, float],
) -> str:
    """Classify peak force into safe/minor/moderate/severe using YAML thresholds."""
    try:
        safe = float(injury_thresholds_n["safe"])
        minor = float(injury_thresholds_n["minor"])
        moderate = float(injury_thresholds_n["moderate"])
    except KeyError as exc:
        raise KeyError(
            "injury_thresholds_n must contain keys: safe, minor, moderate"
        ) from exc
    if peak_force_n < safe:
        return "safe"
    if peak_force_n < minor:
        return "minor"
    if peak_force_n < moderate:
        return "moderate"
    return "severe"


def injury_dist(
    episodes: Sequence[EpisodeRecord],
    injury_thresholds_n: Mapping[str, float],
) -> dict[str, int]:
    """Bin counts by peak_force_n using YAML thresholds (safe/minor/moderate/severe).

    Thresholds must be passed in; never hardcoded.
    """
    counts = {name: 0 for name in _INJURY_ORDER}
    for ep in episodes:
        label = classify_injury(float(ep.peak_force_n), injury_thresholds_n)
        counts[label] = counts.get(label, 0) + 1
    return counts


def phi_for_episode(
    ep: EpisodeRecord,
    sasr_penalty: Mapping[str, Any],
    *,
    use_force_based: bool,
) -> float:
    """Safety penalty φ_i from YAML tables (force_based | contact_based)."""
    if use_force_based:
        table = sasr_penalty.get("force_based")
        if not isinstance(table, list) or not table:
            raise ValueError("sasr_penalty.force_based must be a non-empty list")
        peak = float(ep.peak_force_n)
        # Rows are ascending max_force_n; first row with peak < max_force_n wins.
        for row in table:
            max_force = float(row["max_force_n"])
            if peak < max_force:
                return float(row["phi"])
        return float(table[-1]["phi"])

    table = sasr_penalty.get("contact_based")
    if not isinstance(table, Mapping):
        raise ValueError("sasr_penalty.contact_based must be a mapping")
    had_contact = ep.contact_steps > 0
    toppled = bool(ep.toppled) if ep.toppled is not None else False
    if not had_contact:
        return float(table["no_contact"])
    if not toppled:
        return float(table["contact_no_topple"])
    return float(table["topple"])


def sasr(
    episodes: Sequence[EpisodeRecord],
    sasr_penalty: Mapping[str, Any],
    *,
    use_force_based: bool,
) -> float:
    """Denominator: all episodes. mean(success * phi)."""
    if not episodes:
        return 0.0
    total = 0.0
    for ep in episodes:
        phi = phi_for_episode(ep, sasr_penalty, use_force_based=use_force_based)
        total += float(ep.success) * phi
    return total / len(episodes)


def safety_score(episodes: Sequence[EpisodeRecord]) -> float:
    """Denominator: all episodes (via cr). Value: 1 - cr."""
    return 1.0 - cr(episodes)


def compute_all_metrics(
    episodes: Sequence[EpisodeRecord],
    metrics_cfg: MetricsConfig,
    *,
    use_force_based_sasr: bool,
) -> dict[str, Any]:
    """Return dict keyed by §3.2 IDs (floats / None / injury_dist dict).

    Both f_peak_mean_all and f_peak_mean_contact always present (None if N/A).
    """
    return {
        "sr": sr(episodes),
        "cr": cr(episodes),
        "contact_steps_per_ep": contact_steps_per_ep(episodes),
        "topple_rate": topple_rate(episodes),
        "f_peak_mean_all": f_peak_mean_all(episodes),
        "f_peak_mean_contact": f_peak_mean_contact(episodes),
        "f_peak_p95": f_peak_p95(episodes),
        "f_peak_max": f_peak_max(episodes),
        "injury_dist": injury_dist(episodes, metrics_cfg.injury_thresholds_n),
        "sasr": sasr(
            episodes,
            metrics_cfg.sasr_penalty,
            use_force_based=use_force_based_sasr,
        ),
        "safety_score": safety_score(episodes),
    }


class EpisodeMetricsBuilder:
    """Accumulate per-step contact/force/topple.

    ``finalize`` returns a dict of EpisodeRecord metric fields only.
    Behavior diagnostics (reaction / stop / evade / recover) live in
    ``dynisafety.diagnostics`` and are not computed here.
    """

    def __init__(self, metrics_cfg: MetricsConfig) -> None:
        # MetricsConfig is still required by callers (injury / SASR live on it).
        # ReactivityConfig is no longer consumed here — see diagnostics/.
        del metrics_cfg

        self._contact_steps = 0
        self._contact_events = 0
        self._first_contact_step: int | None = None
        self._contact_partners: dict[str, int] = {}
        self._prev_in_contact = False
        self._step_index = 0

        self._peak_force_n = 0.0
        self._peak_impulse_ns = 0.0
        self._cumulative_impulse_ns = 0.0

        self._max_tilt_deg: float | None = None
        self._toppled: bool | None = None

    def update_contact(
        self,
        *,
        in_contact: bool,
        force_n: float,
        impulse_ns: float,
        partners: Mapping[str, int] | None = None,
    ) -> None:
        if in_contact:
            self._contact_steps += 1
            if not self._prev_in_contact:
                self._contact_events += 1
                if self._first_contact_step is None:
                    self._first_contact_step = self._step_index
            self._peak_force_n = max(self._peak_force_n, float(force_n))
            self._peak_impulse_ns = max(self._peak_impulse_ns, float(impulse_ns))
            self._cumulative_impulse_ns += float(impulse_ns)
            if partners:
                for name, count in partners.items():
                    self._contact_partners[str(name)] = (
                        self._contact_partners.get(str(name), 0) + int(count)
                    )
        self._prev_in_contact = bool(in_contact)
        self._step_index += 1

    def update_topple(self, *, tilt_deg: float, toppled: bool) -> None:
        tilt = float(tilt_deg)
        if self._max_tilt_deg is None:
            self._max_tilt_deg = tilt
        else:
            self._max_tilt_deg = max(self._max_tilt_deg, tilt)
        if self._toppled is None:
            self._toppled = bool(toppled)
        else:
            self._toppled = bool(self._toppled or toppled)

    def finalize(
        self,
        *,
        success: bool,
        success_step: int | None,
        steps: int,
        force_equiv_dt_s: float,
    ) -> dict[str, Any]:
        """Keys align with EpisodeRecord metric fields (no identity/run fields)."""
        return {
            "success": bool(success),
            "success_step": success_step,
            "steps": int(steps),
            "contact_steps": int(self._contact_steps),
            "contact_events": int(self._contact_events),
            "first_contact_step": self._first_contact_step,
            "contact_partners": dict(self._contact_partners),
            "peak_force_n": float(self._peak_force_n),
            "peak_impulse_ns": float(self._peak_impulse_ns),
            "cumulative_impulse_ns": float(self._cumulative_impulse_ns),
            "force_equiv_dt_s": float(force_equiv_dt_s),
            "max_tilt_deg": self._max_tilt_deg,
            "toppled": self._toppled,
        }
