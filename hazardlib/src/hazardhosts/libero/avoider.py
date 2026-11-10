"""G2 privileged avoider for original LIBERO (paradigm §8): nominal-path re-timing with bounded detours.

The reference of the gates is an open-loop replay of demonstration actions. A reaction that leaves the
recorded states cannot return to it by replaying deltas, so the avoider tracks the **absolute OSC goals**
of the hazard-free replay instead (:class:`NominalPath`): the action for nominal state ``s`` is the delta
that reproduces the recorded goal from wherever the robot is now. Without a hazard this is the recorded
action itself (same goal, same state), so the hazard-free episode is the reference replay.

With the ground-truth hazard (``runtime.hazard_state`` and the armed instance's keyframes) it decides,
once per control step, which nominal state to track next:

* **moving hazard**: the keep-out is the hazard's bounding sphere swept along its predicted path
  (keyframes while kinematic, ballistic after release). Advance only while the next
  ``stop_lookahead_steps`` nominal configurations stay outside; hold if the current one is outside;
  otherwise retreat along the path already travelled.
* **resting hazard** (debris): shift the goals by the smallest offset of a fixed candidate set that
  clears the upcoming configurations, ramped per step; hold when none does.

All of this is a heuristic that proposes motion. Whether an instance is ``evade_solvable`` is decided
by the physical G2 episode alone (no official contact and task success), never by this module.
"""
import math

import numpy as np

from hazardlib.kinematics.path import SampledPath
from hazardlib.runtime import KINEMATIC

from .geometry import group_points, mat_to_quat, quat_to_mat
from .policies import DemoReplay


class GoalRecorder(DemoReplay):
    """Demonstration replay that also records, before every action, the robot bodies and the OSC goal
    the previous action set. Reading them does not touch the simulation."""

    def reset(self, task_id, layout_id, seed):
        super().reset(task_id, layout_id, seed)
        self.states, self.goals, self.taken = [], [], []

    def capture(self):
        host = self.host
        robot = host.robot_state(0)
        controller = host.env.env.robots[0].controller
        bodies = {name: (np.array(pos, dtype=float), quat_to_mat(quat))
                  for name, pos, quat in zip(robot.collision_body_names, robot.collision_body_pos,
                                             robot.collision_body_quat)}
        self.states.append({"eef_pos": np.array(robot.eef_pose.pos, dtype=float),
                            "eef_rot": quat_to_mat(robot.eef_pose.quat), "bodies": bodies,
                            "held": robot.held_object})
        if self.taken:
            self.goals.append((np.array(controller.goal_pos, dtype=float),
                               np.array(controller.goal_ori, dtype=float)))

    def act(self, observation):
        self.capture()
        action = super().act(observation)
        self.taken.append(np.array(action, dtype=float))
        return action


class NominalPath(object):
    """States ``0..n`` and goals ``0..n-1`` of one hazard-free replay (goal ``k`` moves state ``k`` to
    ``k+1``), plus the surface samples of the robot bodies and of every object held on the way."""

    def __init__(self, recorder, points, stride):
        if len(recorder.goals) != len(recorder.taken) or len(recorder.states) != len(recorder.taken) + 1:
            raise ValueError("incomplete goal recording: %d states, %d goals, %d actions"
                             % (len(recorder.states), len(recorder.goals), len(recorder.taken)))
        self.states = recorder.states
        self.goal_pos = np.array([g[0] for g in recorder.goals])
        self.goal_rot = np.array([g[1] for g in recorder.goals])
        self.grip = np.array([a[-1] for a in recorder.taken])
        self.n = len(self.goal_pos)
        self.points = {name: np.asarray(p, dtype=float)[::stride] for name, p in points.items()}
        self.bounds = {name: ((p.min(axis=0) + p.max(axis=0)) / 2, (p.max(axis=0) - p.min(axis=0)) / 2)
                       for name, p in self.points.items()}

    @classmethod
    def record(cls, session, task_id, layout_id, seed, horizon_s, stride):
        host = session.host
        recorder = GoalRecorder(host)
        result = session.run_episode(0, None, recorder, task_id, layout_id, seed, horizon_s,
                                     control_id="g2-nominal:%s:%s:%d" % (task_id, layout_id, seed),
                                     stop_on_success=True)
        if not result.success:
            raise ValueError("hazard-free replay of %s / %s did not succeed" % (task_id, layout_id))
        recorder.capture()
        points = host.robot_collision_points(0)
        spacing = float(host.describe().robot_point_spacing_m)
        for name in sorted(set(s["held"] for s in recorder.states if s["held"] is not None)):
            group = next(g for g, label in host.label.items() if label == name
                         and host.groups.frame_body[g] == host.objects[name])
            points[name] = group_points(host.m, host.groups, group, spacing)
        return cls(recorder, points, stride)

    def hits(self, index, offset, centres, radius):
        """Whether any surface sample of nominal state ``index``, shifted by ``offset``, is inside one of
        the spheres ``centres`` / ``radius``. Body bounding boxes prune the samples (exact for the kept
        samples; conservative rounding is not needed for a proposal heuristic)."""
        local = np.asarray(centres, dtype=float) - np.asarray(offset, dtype=float)
        for name, (pos, rot) in self.states[index]["bodies"].items():
            if name not in self.points:
                continue
            mid, half = self.bounds[name]
            gap = np.maximum(np.abs(local - (rot.dot(mid) + pos)) - np.abs(rot).dot(half), 0.0)
            near = local[np.linalg.norm(gap, axis=1) < radius]
            if not len(near):
                continue
            world = self.points[name].dot(rot.T) + pos
            for centre in near:
                if np.any(np.sum((world - centre) ** 2, axis=1) < radius * radius):
                    return True
        return False


def offset_candidates(settings):
    """Zero first, then by length; at equal length the upward offset first."""
    out = [(0.0, 0, np.zeros(3))]
    for up in settings["offset_up_m"]:
        out.append((float(up), 1, np.array([0.0, 0.0, float(up)])))
    for side in settings["offset_side_m"]:
        for lift in settings["offset_side_lift_m"]:
            for i in range(8):
                angle = math.pi * i / 4
                vector = np.array([side * math.cos(angle), side * math.sin(angle), lift])
                out.append((float(np.linalg.norm(vector)), 2 + i, vector))
    out.sort(key=lambda item: (item[0], item[1]))
    return [vector for _, _, vector in out]


class NominalPathAvoider(object):
    """``PrivilegedAvoider`` (``filter_action``) over one :class:`NominalPath`. ``phase`` returns the
    runtime phase of the hazard (``session.phase(env)``); ``arm`` gives the instance of the coming
    episode (privileged: the kinematic keyframes); ``reset`` starts an episode."""

    def __init__(self, host, nominal, settings, phase):
        self.host, self.nominal, self.settings, self.phase = host, nominal, settings, phase
        self.candidates = offset_candidates(settings)
        self.instance = None
        self.reset()

    def arm(self, instance):
        self.instance = instance
        self.path = SampledPath.from_pose_sequence(instance.keyframes)
        self.radius = float(np.linalg.norm(instance.agent.half_extents_m)) + self.settings["clearance_margin_m"]

    def reset(self):
        self.s, self.steps, self.grip_index, self.slow = 0, 0, -1, 0
        self.offset = np.zeros(3)
        self.detour = None
        self.stats = {"held_steps": 0, "retreat_steps": 0, "max_offset_m": 0.0, "blocked_steps": 0}

    # ------------------------------------------------------------------------------------ prediction

    def _keep_out(self, t, hazard):
        """``(centres, moving)`` of the hazard from now on."""
        cfg, inst = self.settings, self.instance
        horizon = np.arange(0.0, cfg["prediction_horizon_s"] + cfg["prediction_dt_s"] / 2, cfg["prediction_dt_s"])
        gravity = np.array([0.0, 0.0, -abs(float(self.host.m.opt.gravity[2]))])
        pos, vel = np.asarray(hazard.pos, dtype=float), np.asarray(hazard.lin_vel_mps, dtype=float)
        if self.phase() == KINEMATIC:
            self.slow = 0
            release = inst.release_time_s
            before = horizon[t + horizon <= release]
            centres = [pos[None, :], self.path.pos(t + before)]
            after = horizon[t + horizon > release] + t - release
            if len(after):
                start, v0 = self.path.pos(release), np.asarray(inst.release_velocity.lin_mps, dtype=float)
                centres.append(start + after[:, None] * v0 + 0.5 * after[:, None] ** 2 * gravity)
            return np.concatenate(centres), True
        if float(np.linalg.norm(vel)) < cfg["rest_speed_mps"]:
            self.slow += 1
        else:
            self.slow = 0
        if self.slow >= cfg["settle_steps"]:
            return pos[None, :], False
        return pos + horizon[:, None] * vel + 0.5 * horizon[:, None] ** 2 * gravity, True

    # -------------------------------------------------------------------------------------- decision

    def _clear(self, index, offset, centres):
        return not self.nominal.hits(index, offset, centres, self.radius)

    def _window(self, first, count, offset, centres):
        return all(self._clear(j, offset, centres) for j in range(first, min(first + count - 1, self.nominal.n) + 1))

    def _plan(self, centres, moving):
        cfg, s, n = self.settings, self.s, self.nominal.n
        if moving:
            if s < n and self._window(s + 1, cfg["stop_lookahead_steps"], self.offset, centres):
                return s + 1
            if self._clear(s, self.offset, centres):
                self.stats["held_steps"] += 1
                return s
            self.stats["retreat_steps"] += 1
            return max(s - 1, 0)
        if s >= n:
            return n
        order = [np.zeros(3)] + ([] if self.detour is None else [self.detour]) + self.candidates
        chosen = next((o for o in order if self._window(s + 1, cfg["detour_lookahead_steps"], o, centres)), None)
        if chosen is None:
            self.stats["blocked_steps"] += 1
            return s if self._clear(s, self.offset, centres) else max(s - 1, 0)
        self.detour = chosen if np.any(chosen) else None
        change = chosen - self.offset
        length = float(np.linalg.norm(change))
        stepped = chosen if length <= cfg["max_offset_step_m"] else self.offset + change * cfg["max_offset_step_m"] / length
        for offset in (stepped, self.offset):
            if self._clear(s + 1, offset, centres):
                self.offset = offset
                return s + 1
        if self._clear(s, stepped, centres):
            self.offset = stepped
        self.stats["held_steps"] += 1
        return s

    def filter_action(self, action, robot_state, hazard_state):
        t = self.steps * self.host.control_dt
        self.steps += 1
        if hazard_state is None:
            target = min(self.s + 1, self.nominal.n)
        else:
            target = self._plan(*self._keep_out(t, hazard_state))
        self.s = target
        self.stats["max_offset_m"] = max(self.stats["max_offset_m"], float(np.linalg.norm(self.offset)))
        return self._action(target)

    # --------------------------------------------------------------------------------------- control

    def _action(self, target):
        """Delta OSC action whose goal is the recorded goal of ``target`` (shifted by the offset). The
        gripper command never goes back to an earlier one, so a retreat keeps a held object."""
        nominal = self.nominal
        if target >= 1:
            goal_pos, goal_rot = nominal.goal_pos[target - 1], nominal.goal_rot[target - 1]
            self.grip_index = max(self.grip_index, target - 1)
        else:
            goal_pos, goal_rot = nominal.states[0]["eef_pos"], nominal.states[0]["eef_rot"]
        grip = self.host._gripper_cmd if self.grip_index < 0 else nominal.grip[self.grip_index]
        controller = self.host.env.env.robots[0].controller
        controller.update(force=True)
        scale = np.asarray(controller.output_max, dtype=float)
        delta = (goal_pos + self.offset - np.asarray(controller.ee_pos, dtype=float)) / scale[:3]
        error = goal_rot.dot(np.asarray(controller.ee_ori_mat, dtype=float).T)
        quat = mat_to_quat(error)
        angle = 2.0 * math.atan2(float(np.linalg.norm(quat[1:])), float(quat[0]))
        if angle > math.pi:
            angle -= 2.0 * math.pi
        norm = float(np.linalg.norm(quat[1:]))
        rotation = np.zeros(3) if norm < np.finfo(float).eps else quat[1:] / norm * angle / scale[3:]
        return np.concatenate([np.clip(delta, -1.0, 1.0), np.clip(rotation, -1.0, 1.0), [grip]])

    def summary(self):
        return dict(self.stats, final_state=self.s, n_states=self.nominal.n)
