"""HazardInstance: one concrete, time-stamped hazard (paradigm §2, §5.2)."""
import dataclasses
import math
from typing import Any, Dict, List, Optional

from .._records import Record, check_nonempty_str, check_finite, check_vec, require
from .certificate import Certificate
from .hazard_spec import AgentSpec
from .pose_sequence import PoseSequence


@dataclasses.dataclass
class ReleaseVelocity(Record):
    """Velocities handed to ``release_to_physics`` (m/s, rad/s, world frame)."""
    lin_mps: List[float]
    ang_radps: List[float]

    def validate(self) -> None:
        check_vec("lin_mps", self.lin_mps, 3)
        check_vec("ang_radps", self.ang_radps, 3)


@dataclasses.dataclass
class HazardInstance(Record):
    """World-frame keyframes plus release data and the nominal hit.

    ``release_time_s`` and ``release_velocity`` are both set or both None (None
    for a hazard that is never handed to dynamics, e.g. a Tier-B kinematic
    hazard). ``onset_time_s <= nominal_hit_time_s`` because
    tau = t_hit - t_onset is a non-negative reaction time (paradigm §5.1).
    ``certificate`` is None until the gates have run.

    ``spec_id`` / ``family`` / ``agent`` are copied from the HazardSpec (D11 b):
    the runtime dispatches rollback on ``family`` (never inferred from the
    timing) and allocates the body from ``agent``.
    """
    instance_id: str
    spec_id: str
    family: str
    agent: AgentSpec
    keyframes: PoseSequence
    nominal_hit_point: List[float]
    nominal_hit_time_s: float
    onset_time_s: float
    release_time_s: Optional[float] = None
    release_velocity: Optional[ReleaseVelocity] = None
    certificate: Optional[Certificate] = None
    provenance: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def validate(self) -> None:
        check_nonempty_str("instance_id", self.instance_id)
        check_nonempty_str("spec_id", self.spec_id)
        check_nonempty_str("family", self.family)
        check_vec("nominal_hit_point", self.nominal_hit_point, 3)
        check_finite("nominal_hit_time_s", self.nominal_hit_time_s)
        check_finite("onset_time_s", self.onset_time_s)
        require(self.onset_time_s <= self.nominal_hit_time_s,
                "onset_time_s=%r is after nominal_hit_time_s=%r"
                % (self.onset_time_s, self.nominal_hit_time_s))
        require((self.release_time_s is None) == (self.release_velocity is None),
                "release_time_s and release_velocity must be both set or both None")
        if self.release_time_s is not None:
            check_finite("release_time_s", self.release_time_s)
