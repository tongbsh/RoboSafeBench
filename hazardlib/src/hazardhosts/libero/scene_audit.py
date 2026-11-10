"""Robot-absent physical audit of a hazard (D19 c): does it disturb the scene on its own?

One episode per candidate with the robot held by the Oracle (stop-wait-resume, the same controller and
arguments as G1) while the hazard is active: during that first hold everything that is touched or moves is
caused by the hazard alone. The audit records

* the scene bodies the hazard overlaps while it is kinematic (collisions are off then, but the host still
  reports overlaps: this is visual penetration),
* the scene bodies it touches after a release (debris contact),
* how far each task object moved during the **first** hold only (later pauses follow robot motion),
* where the hazard rests and what it rests on, whether a follow-through hazard finished its exit path
  (``exit_t_s``) and whether and how it was released.

Support bodies (the table) are not scene bodies. A hazard that hits the frozen robot is not rejected for
that: the robot-hazard contact is the benchmark's event (robot links and the held object are robot-side).
The object the robot holds when the first hold starts is excluded from the displacement test for the same
reason; its displacement is recorded under ``held_object_moved_m``.

The horizon is the G1 one (nominal success time + ``g1.horizon_extra_s``). The Oracle holds until a
follow-through hazard has exited, so the audit can end before the task is done; ``success`` and
``hold_complete`` are recorded, and an audit whose first hold did not finish is rejected.
"""
import numpy as np

from hazardlib.gates.episodes import OracleController, _horizon
from hazardlib.host import PartnerClass
from hazardlib.runtime import KINEMATIC, RELEASED


class AuditController(OracleController):
    """The Oracle, plus bookkeeping of the first hold (see the module docstring)."""

    def __init__(self, host, session, taps, *args):
        self._host, self._session, self._taps = host, session, taps
        OracleController.__init__(self, *args)

    def reset(self):
        OracleController.reset(self)
        self.reference, self.moved, self.rest, self.rest_on, self.held = None, {}, None, None, None
        self.first_hold_over = False

    def decide(self, session, env, t_s, observation, policy, control_dt_s):
        action = OracleController.decide(self, session, env, t_s, observation, policy, control_dt_s)
        if not self._holding and self.reference is not None:
            self.first_hold_over = True  # later pauses follow robot motion: no longer hazard-only
        if self._holding and not self.first_hold_over:
            poses = {n: np.array(p.pos) for n, p in self._host.object_poses(0).items()}
            if self.reference is None:
                self.reference, self.held = poses, self._host.robot_state(0).held_object
            for name, pos in poses.items():
                if name in self.reference:
                    self.moved[name] = max(self.moved.get(name, 0.0), float(np.linalg.norm(pos - self.reference[name])))
            self.rest = np.array(session.runtime.hazard_state(env).pos).tolist()
            self.rest_on = list(self._taps["last"])
        return action


def audit_verdict(record, moved_tolerance_m):
    """Rejection reasons (empty = admitted) of one audit record."""
    reasons = []
    if not record["hold_complete"]:
        reasons.append("hold_incomplete")
    reasons += ["kinematic_overlap:" + p for p in sorted(record["kinematic_overlap"])]
    reasons += ["released_contact:" + p for p in sorted(record["released_contact"])]
    reasons += ["moved:" + n for n, d in sorted(record["objects_moved_m"].items()) if d > moved_tolerance_m]
    return reasons


def rejection_reason(reasons):
    return "scene_audit:" + ";".join(reasons)


def run_audit(session, instance, reference, task_context, gate_settings, seed, supports, moved_tolerance_m):
    """One physical audit episode; returns the record (``admitted``, ``reasons`` included)."""
    host = session.host
    taps = {"kinematic": {}, "released": {}, "last": []}
    original = host.hazard_contacts

    def tapped(handle):
        contacts = original(handle)
        phase = session.phase(0)
        if phase in (KINEMATIC, RELEASED):
            taps["last"] = sorted(set(c.partner_name for c in contacts))
            for c in contacts:
                if c.partner_class is PartnerClass.SCENE and c.partner_name not in supports:
                    slot = taps[phase].setdefault(c.partner_name, [c.time_s, c.time_s])
                    slot[1] = c.time_s
        return contacts

    ws = task_context.workspace_bounds
    margin = gate_settings.g0.workspace_margin_m
    controller = AuditController(host, session, taps, gate_settings.resume_hazard_speed_mps,
                                 gate_settings.resume_settle_s, np.array(ws.min_m) - margin,
                                 np.array(ws.max_m) + margin)
    host.hazard_contacts = tapped
    try:
        result = session.run_episode(0, instance, reference, task_context.task_id, task_context.layout_id, seed,
                                     _horizon(task_context, instance, gate_settings.g1.horizon_extra_s),
                                     controller=controller)
    finally:
        del host.hazard_contacts
    info = session.runtime.episode_info(0)
    release = info["release"]
    held = controller.held
    record = {"t_hit_s": instance.nominal_hit_time_s, "held_at_hold_start": held,
              "kinematic_overlap": taps["kinematic"], "released_contact": taps["released"],
              "objects_moved_m": {k: round(v, 6) for k, v in sorted(controller.moved.items()) if k != held},
              "held_object_moved_m": None if held is None else round(controller.moved.get(held, 0.0), 6),
              "rest_pos": controller.rest, "rest_on": controller.rest_on,
              "hold_complete": controller.first_hold_over, "paused_s": controller.paused_s,
              "released": release is not None, "release_kind": None if release is None else release["kind"],
              "exit_t_s": info["exit_t_s"], "exited": info["exit_t_s"] is not None, "final_phase": info["phase"],
              "success": result.success, "official_contacts": result.official_contact_count,
              "first_official_contact_t_s": result.log.first_official_contact_t_s,
              "horizon_s": _horizon(task_context, instance, gate_settings.g1.horizon_extra_s),
              "moved_tolerance_m": moved_tolerance_m}
    record["reasons"] = audit_verdict(record, moved_tolerance_m)
    record["admitted"] = not record["reasons"]
    return record
