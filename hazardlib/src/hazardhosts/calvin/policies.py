"""Reference policies on CALVIN (probe sources, R4): replay of the recorded play data.

The dataset's ``actions`` are absolute TCP targets (position, Euler angles, gripper). Sent as ``cartesian_abs``
they do not move ``Robot.target_pos`` (``robot.py:264-279`` at the pinned commit), so a later relative action
(``hold_action``: zero relative motion) would pull the robot back to the target of the reset. The replay therefore
sends each recorded target as the relative action that moves the robot's own integrated target onto it:
``rel = (target_k - target_{k-1}) / max_rel_pos`` (position) and the wrapped Euler difference over ``max_rel_orn``
(orientation), with ``target_{-1}`` the TCP pose of the reset. ``Robot.relative_to_absolute`` then integrates the
target back to the recorded one (up to rounding), and a hold in between leaves the integrated target where it is.

:class:`DemoReplay` is the ``hazardctl`` reference policy (``F(host, **args)``): it reads only the observation
(``robot_obs`` for the reset pose) and the dataset; ``act`` replays one recorded frame per call, so a controller
that holds the robot pauses the replay, which resumes from the same target. :func:`relative_actions` gives the
same action sequence for an uninterrupted replay (the conformance motion).
"""
from typing import Any, Dict, List

import numpy as np

from .dataset import CalvinDataset

DEMO_KIND = "demo_replay"


def _scales(env: Any) -> np.ndarray:
    robot = env.robot
    pos = float(robot.max_rel_pos) * float(robot.magic_scaling_factor_pos)
    orn = float(robot.max_rel_orn) * float(robot.magic_scaling_factor_orn)
    return np.array([pos, pos, pos, orn, orn, orn])


def _wrap(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def to_relative(target: np.ndarray, previous: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """The 7-D relative action moving the integrated target ``previous`` (6) onto the absolute ``target`` (7)."""
    delta = np.asarray(target[:6], dtype=np.float64) - np.asarray(previous[:6], dtype=np.float64)
    delta[3:] = _wrap(delta[3:])
    return np.concatenate([delta / scales, [float(target[6])]])


def relative_actions(dataset: CalvinDataset, split: str, start: int, count: int, scales: np.ndarray) -> List[np.ndarray]:
    """Relative actions replaying ``count`` recorded frames from ``start``; the reset target is ``robot_obs[:6]``
    of frame ``start``."""
    previous = dataset.frame(split, start)["robot_obs"][:6].astype(np.float64)
    out = []
    for k in range(count):
        target = dataset.frame(split, start + k)["actions"].astype(np.float64)
        out.append(to_relative(target, previous, scales))
        previous = target[:6]
    return out


class DemoReplay:
    """Replays the recorded play data from the reset frame of the layout (see the module docstring). Past the end
    of the recorded range it holds (zero relative motion, last gripper command)."""

    kind = DEMO_KIND
    under_test = False

    def __init__(self, host: Any, name: str = "calvin_play_data") -> None:
        self.name = name
        self._dataset = host.dataset
        self._scales = _scales(host.env)
        self._split = None  # type: Any
        self._frame = 0
        self._target = None  # type: Any
        self._gripper = 1.0

    def reset(self, task_id: str, layout_id: str, seed: int) -> None:
        layout = self._dataset.layout(layout_id)
        self._split, self._frame, self._target = layout.split, layout.frame, None

    def act(self, observation: Dict[str, Any]) -> np.ndarray:
        if self._target is None:
            obs = np.asarray(observation["robot_obs"], dtype=np.float64)
            self._target, self._gripper = obs[:6].copy(), float(obs[-1])
        if not self._dataset.split(self._split).contains(self._frame):
            return np.concatenate([np.zeros(6), [self._gripper]])
        target = self._dataset.frame(self._split, self._frame)["actions"].astype(np.float64)
        self._frame += 1
        action = to_relative(target, self._target, self._scales)
        self._target, self._gripper = target[:6].copy(), float(target[6])
        return action
