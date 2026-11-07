"""Solvability certificate (paradigm §8)."""
import dataclasses
from typing import Optional

from .._records import Record, choice, require

CERT_LABELS = ("wait_solvable", "evade_solvable", "unsolvable_by_probes", "uncertified",
               "baseline_unusable")


@dataclasses.dataclass
class Certificate(Record):
    """``g0``/``g1``/``g2`` are tri-state: True = passed, False = failed,
    None = not evaluated (e.g. G2 when no privileged controller exists, which is
    exactly the ``uncertified`` case). The label is *recorded* here; deriving it
    from gate outcomes is the gate task's job (T06). Only implications that
    follow directly from §8 are enforced.

    ``baseline_unusable`` (§8 G1 precondition): the reference policy does not
    succeed without the hazard, so G1/G2 cannot certify anything."""
    g0: Optional[bool]
    g1: Optional[bool]
    g2: Optional[bool]
    label: str = choice(CERT_LABELS)  # type: ignore[assignment]
    notes: str = ""

    def validate(self) -> None:
        if self.label == "wait_solvable":
            require(self.g0 is True and self.g1 is True, "wait_solvable requires g0 and g1 to be True")
        elif self.label == "evade_solvable":
            require(self.g0 is True and self.g2 is True, "evade_solvable requires g0 and g2 to be True")
        elif self.label == "unsolvable_by_probes":
            require(self.g1 is not True and self.g2 is not True,
                    "unsolvable_by_probes contradicts a passed g1/g2")
        elif self.label == "baseline_unusable":
            require(self.g1 is not True and self.g2 is not True,
                    "baseline_unusable contradicts a passed g1/g2")
