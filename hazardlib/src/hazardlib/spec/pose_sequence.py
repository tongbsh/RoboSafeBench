"""Time-parameterised pose sequence shared by TaskContext (nominal EEF path)
and HazardInstance (hazard keyframes)."""
import dataclasses
import math
from typing import List

from .._records import Record, check_vec, require


@dataclasses.dataclass
class PoseSequence(Record):
    """``time_s[i]`` -> ``pos[i]`` (m, world) and ``quat[i]`` (``wxyz``).

    Structural checks only: >= 1 sample, equal lengths, finite values,
    strictly increasing times, quaternions non-zero. Unit-norm tolerances and
    interpolation belong to later tasks (T02 / conformance).
    """
    time_s: List[float]
    pos: List[List[float]]
    quat: List[List[float]]

    def __len__(self) -> int:
        return len(self.time_s)

    def validate(self) -> None:
        n = len(self.time_s)
        require(n >= 1, "time_s must contain at least one sample")
        require(len(self.pos) == n and len(self.quat) == n,
                "time_s/pos/quat must have equal length, got %d/%d/%d"
                % (n, len(self.pos), len(self.quat)))
        for i, t in enumerate(self.time_s):
            require(math.isfinite(t), "time_s[%d] must be finite" % i)
            if i > 0:
                require(t > self.time_s[i - 1],
                        "time_s must be strictly increasing (time_s[%d]=%r <= time_s[%d]=%r)"
                        % (i, t, i - 1, self.time_s[i - 1]))
        for i in range(n):
            check_vec("pos[%d]" % i, self.pos[i], 3)
            check_vec("quat[%d]" % i, self.quat[i], 4)
            require(any(q != 0.0 for q in self.quat[i]), "quat[%d] must be non-zero" % i)
