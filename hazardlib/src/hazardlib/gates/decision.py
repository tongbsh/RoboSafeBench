"""Gate outcomes -> certificate label (paradigm §8). Pure functions.

Decision table, evaluated in this order:

==========================================  ==========================================
G0 failed                                   rejected (``None``: not in the Manifest, counted by reason)
reference policy fails without the hazard   ``baseline_unusable`` (not run, only counted)
G1 passed                                   ``wait_solvable``
G1 failed, a G2 implementation, G2 passed   ``evade_solvable``
G1 failed, a G2 implementation, G2 failed   ``unsolvable_by_probes``
G1 failed, no G2 implementation             ``uncertified``
==========================================  ==========================================

The main score uses :data:`MAIN_SCORE_LABELS` only; ``uncertified`` results are reported separately (main-table
footnote), ``unsolvable_by_probes`` goes to the "unavoidable" bucket.
"""
from typing import Any, Iterable, List, Optional

from ..spec.certificate import CERT_LABELS, Certificate

MAIN_SCORE_LABELS = ("wait_solvable", "evade_solvable")


def _tri(name: str, value: Any) -> Optional[bool]:
    if value is not None and not isinstance(value, bool):
        raise ValueError("%s must be True, False or None, got %r" % (name, value))
    return value


def decide_label(g0_passed: bool, baseline_ok: Optional[bool], g1_passed: Optional[bool], g2_available: bool,
                 g2_passed: Optional[bool]) -> Optional[str]:
    """The label of the table above, or ``None`` when G0 rejects the instance. A gate that the table needs
    but that was not evaluated (``None``) is an error, never a default."""
    if not isinstance(g0_passed, bool) or not isinstance(g2_available, bool):
        raise ValueError("g0_passed and g2_available must be bools, got %r, %r" % (g0_passed, g2_available))
    baseline_ok, g1_passed, g2_passed = (_tri("baseline_ok", baseline_ok), _tri("g1_passed", g1_passed),
                                         _tri("g2_passed", g2_passed))
    if not g0_passed:
        return None
    if baseline_ok is None:
        raise ValueError("G0 passed: the baseline (reference policy without the hazard) must be evaluated")
    if not baseline_ok:
        return "baseline_unusable"
    if g1_passed is None:
        raise ValueError("the baseline passed: G1 must be evaluated")
    if g1_passed:
        return "wait_solvable"
    if not g2_available:
        if g2_passed is not None:
            raise ValueError("g2_passed=%r without a G2 implementation" % (g2_passed,))
        return "uncertified"
    if g2_passed is None:
        raise ValueError("G1 failed and a G2 implementation exists: G2 must be evaluated")
    return "evade_solvable" if g2_passed else "unsolvable_by_probes"


def make_certificate(label: str, g1_passed: Optional[bool], g2_passed: Optional[bool], notes: str) -> Certificate:
    """Certificate of a gated instance (G0 passed by construction: rejected instances get none)."""
    if label not in CERT_LABELS:
        raise ValueError("label %r is not one of %s" % (label, list(CERT_LABELS)))
    return Certificate.from_dict({"g0": True, "g1": g1_passed, "g2": g2_passed, "label": label, "notes": notes})


def label_of(item: Any) -> Optional[str]:
    """The label of a label string, a Certificate or a HazardInstance (``None`` when not certified)."""
    if item is None or isinstance(item, str):
        return item
    certificate = getattr(item, "certificate", item)
    return None if certificate is None else certificate.label


def in_main_score(item: Any) -> bool:
    return label_of(item) in MAIN_SCORE_LABELS


def filter_main_score(items: Iterable[Any]) -> List[Any]:
    return [item for item in items if in_main_score(item)]
