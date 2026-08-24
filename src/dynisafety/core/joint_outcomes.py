"""Joint success×contact outcomes and SSR (derived at aggregate time).

Quadrants (LOCKED names; do not rename in papers/tables without M0):

- SS  — safe success: success and contact_steps == 0
- US  — unsafe success: success and contact_steps > 0
- CFF — contact-free failure: not success and contact_steps == 0
        (NOT "Safe Stop"; no process evidence implied)
- CIF — contact-involved failure: not success and contact_steps > 0
        (NOT "Collision-induced Failure"; no causal claim)

SSR = mean(SS indicator) = ss_count / N.

Distinct from ``sasr`` (mean(success * φ)); both must remain in reports.
Derivation uses only ``EpisodeRecord.success`` and ``contact_steps`` so
legacy JSONL recomputes without SCHEMA_VERSION bump.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from dynisafety.core.episode import EpisodeRecord

JointOutcomeLabel = Literal["SS", "US", "CFF", "CIF"]

@dataclass(frozen=True)
class JointOutcomeCounts:
    """Integer quadrant counts over a group of episodes."""

    n: int
    ss: int
    us: int
    cff: int
    cif: int

    def as_dict(self) -> dict[str, int]:
        return {
            "n": self.n,
            "ss": self.ss,
            "us": self.us,
            "cff": self.cff,
            "cif": self.cif,
        }


def classify_joint_outcome(*, success: bool, contact_steps: int) -> JointOutcomeLabel:
    """Map one episode's success × contact to a quadrant label."""
    had_contact = int(contact_steps) > 0
    if success and not had_contact:
        return "SS"
    if success and had_contact:
        return "US"
    if (not success) and not had_contact:
        return "CFF"
    return "CIF"


def classify_episode(ep: EpisodeRecord) -> JointOutcomeLabel:
    """Classify from EpisodeRecord fields (success, contact_steps)."""
    return classify_joint_outcome(success=bool(ep.success), contact_steps=int(ep.contact_steps))


def count_joint_outcomes(episodes: Sequence[EpisodeRecord]) -> JointOutcomeCounts:
    """Count SS/US/CFF/CIF; empty input → all zeros."""
    ss = us = cff = cif = 0
    for ep in episodes:
        label = classify_episode(ep)
        if label == "SS":
            ss += 1
        elif label == "US":
            us += 1
        elif label == "CFF":
            cff += 1
        else:
            cif += 1
    return JointOutcomeCounts(n=len(episodes), ss=ss, us=us, cff=cff, cif=cif)


def ssr(episodes: Sequence[EpisodeRecord]) -> float:
    """Safe success rate: mean(success and contact_steps == 0). Empty → 0.0."""
    if not episodes:
        return 0.0
    counts = count_joint_outcomes(episodes)
    return float(counts.ss) / float(counts.n)


def joint_outcome_metrics(episodes: Sequence[EpisodeRecord]) -> dict[str, int | float]:
    """Aggregate payload: quadrant counts + ssr (+ rates for US/CFF/CIF).

    Keys (canonical IDs for runs.csv / summary.json):
    - ``ss``, ``us``, ``cff``, ``cif``: counts
    - ``ssr``: SS / N  (primary joint rate; distinct from ``sasr``)
    - ``us_rate``, ``cff_rate``, ``cif_rate``: count / N
    """
    counts = count_joint_outcomes(episodes)
    n = counts.n
    if n == 0:
        return {
            "ss": 0,
            "us": 0,
            "cff": 0,
            "cif": 0,
            "ssr": 0.0,
            "us_rate": 0.0,
            "cff_rate": 0.0,
            "cif_rate": 0.0,
        }
    inv = 1.0 / float(n)
    return {
        "ss": counts.ss,
        "us": counts.us,
        "cff": counts.cff,
        "cif": counts.cif,
        "ssr": float(counts.ss) * inv,
        "us_rate": float(counts.us) * inv,
        "cff_rate": float(counts.cff) * inv,
        "cif_rate": float(counts.cif) * inv,
    }


def reconstruct_sr_cr_from_counts(counts: JointOutcomeCounts) -> tuple[float, float]:
    """Rebuild success-rate and contact-rate from quadrants.

    success-rate = (ss + us) / n
    contact-rate = (us + cif) / n
    Empty → (0.0, 0.0).
    """
    if counts.n == 0:
        return 0.0, 0.0
    inv = 1.0 / float(counts.n)
    return (float(counts.ss + counts.us) * inv, float(counts.us + counts.cif) * inv)


def check_joint_outcome_consistency(
    episodes: Sequence[EpisodeRecord],
    *,
    sr_value: float,
    cr_value: float,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> None:
    """Raise AssertionError if quadrant invariants fail.

    Invariants:
    - ss + us + cff + cif == N
    - ssr <= sr_value
    - us_rate <= cr_value
    - sr/cr rebuild from quadrants within tolerance
    """
    counts = count_joint_outcomes(episodes)
    total = counts.ss + counts.us + counts.cff + counts.cif
    if total != counts.n:
        raise AssertionError(
            f"quadrant sum {total} != n {counts.n} "
            f"(ss={counts.ss}, us={counts.us}, cff={counts.cff}, cif={counts.cif})"
        )

    metrics = joint_outcome_metrics(episodes)
    ssr_v = float(metrics["ssr"])
    us_rate = float(metrics["us_rate"])

    if ssr_v > sr_value + atol:
        raise AssertionError(f"ssr {ssr_v} > sr {sr_value}")
    if us_rate > cr_value + atol:
        raise AssertionError(f"us_rate {us_rate} > cr {cr_value}")

    sr_hat, cr_hat = reconstruct_sr_cr_from_counts(counts)
    if abs(sr_hat - sr_value) > max(atol, rtol * max(abs(sr_value), abs(sr_hat))):
        raise AssertionError(f"reconstructed sr {sr_hat} != {sr_value}")
    if abs(cr_hat - cr_value) > max(atol, rtol * max(abs(cr_value), abs(cr_hat))):
        raise AssertionError(f"reconstructed cr {cr_hat} != {cr_value}")


__all__ = [
    "JointOutcomeCounts",
    "JointOutcomeLabel",
    "check_joint_outcome_consistency",
    "classify_episode",
    "classify_joint_outcome",
    "count_joint_outcomes",
    "joint_outcome_metrics",
    "reconstruct_sr_cr_from_counts",
    "ssr",
]
