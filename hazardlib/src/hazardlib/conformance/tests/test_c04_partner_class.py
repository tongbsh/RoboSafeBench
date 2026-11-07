"""C4: contact detection and partner classification.

A hazard released onto a robot link gives an official contact with partner ``robot_link``; onto a held object
(when the scenario can hold one) ``held_object``; dropped onto a support surface only ``scene`` contacts and no
official contact. Every contact names the struck body as in ``RobotState.collision_body_names`` (01 §3.3 item 9),
has its normal pointing from the partner to the hazard (D9) and carries an absolute time inside the physics step
whose ``hazard_contacts`` call returned it (items 4 and 6: same clock, "take" semantics, no late delivery).
"""
import math

import numpy as np

from hazardlib.conformance.common import angle_between, contact_clock_violations, strike_probe
from hazardlib.conformance.harness import drop_instance, support_radius
from hazardlib.conformance.suite import Check
from hazardlib.host import PartnerClass
from hazardlib.semantics import EventKind

SECTION = "c4_partner_class"
ROBOT_SIDE = (PartnerClass.ROBOT_LINK, PartnerClass.HELD_OBJECT)


def _strike(ctx, probe, strike, wanted, label, tol):
    speed = ctx.number("probe", "strike_speed_mps")
    run, geometry = strike_probe(ctx, probe, 0, strike, speed, "c4-%s" % label, ctx.number("probe", "tail_s"))
    ctx.require(geometry["expected_partner"] is wanted,
                "scenario: the BD's robot_contact_bodies classify %r as %s, not %s"
                % (strike.body, geometry["expected_partner"].value, wanted.value))
    log = run.log
    robot_events = [e for e in run.events if e.partner_class in ROBOT_SIDE]
    first = robot_events[0] if robot_events else None
    names = ctx.host.robot_state(0).collision_body_names or []
    normal_error = None
    if first is not None and first.normal is not None:
        normal_error = angle_between(first.normal, -np.array(strike.direction))
    violations = contact_clock_violations(probe, run, ctx.physics_dt, tol["clock"])
    ctx.record(label, {"partner_class": log.partner_class, "first_official_contact_t_s": log.first_official_contact_t_s,
                       "expected_hit_time_s": geometry["expected_hit_time_s"],
                       "partner_name": None if first is None else first.partner_name,
                       "event_kind": None if first is None else first.kind, "normal_error_rad": normal_error,
                       "clock_violations": len(violations)})
    ctx.expect(log.partner_class is wanted, "%s: a hazard released onto %r gives official partner %r, expected %s"
               % (label, strike.body, log.partner_class, wanted.value))
    if first is not None:
        ctx.expect(first.kind is EventKind.DYNAMIC, "%s: the first robot-side event is %s, expected dynamic"
                   % (label, first.kind.value))
        ctx.expect(first.partner_name == strike.body, "%s: the contact names %r, the struck body is %r"
                   % (label, first.partner_name, strike.body))
        ctx.expect(first.partner_name in names, "%s: contact partner %r is not in RobotState.collision_body_names"
                   % (label, first.partner_name))
        ctx.expect(normal_error is not None, "%s: the contact carries no normal" % label)
        if normal_error is not None:
            ctx.expect(normal_error <= tol["normal"], "%s: the contact normal is %.3g rad from 'partner -> hazard' "
                                                      "(> %r)" % (label, normal_error, tol["normal"]))
    ctx.expect(not violations, "%s: %s" % (label, "; ".join(violations[:3])))


def _drop(ctx, probe, tol):
    scenario, agent = ctx.scenario, ctx.scenario.probe_agent
    up = np.array([0.0, 0.0, 1.0])
    height = scenario.drop_point_m[2] - scenario.drop_surface_height_m - support_radius(agent, up)
    fall_time = math.sqrt(2.0 * height / ctx.bd.gravity_mps2)
    onset = ctx.on_grid(ctx.number("probe", "onset_s"))
    instance = drop_instance("c4-drop", agent, scenario.drop_point_m, onset, fall_time)
    run = probe.run(0, instance, scenario.still, onset + fall_time + ctx.number("probe", "tail_s"), probe.hold(0))
    log = run.log
    scene = [e for e in run.events if e.partner_class is PartnerClass.SCENE]
    robot = [e for e in run.events if e.partner_class in ROBOT_SIDE]
    normal_error = angle_between(scene[0].normal, up) if scene and scene[0].normal is not None else None
    violations = contact_clock_violations(probe, run, ctx.physics_dt, tol["clock"])
    ctx.record("drop", {"first_scene_contact_t_s": log.first_scene_contact_t_s,
                        "first_official_contact_t_s": log.first_official_contact_t_s,
                        "scene_events": len(scene), "robot_side_events": len(robot),
                        "first_scene_partner": scene[0].partner_name if scene else None,
                        "normal_error_rad": normal_error, "clock_violations": len(violations)})
    ctx.expect(scene, "a hazard dropped onto the support surface produced no scene contact")
    ctx.expect(log.first_scene_contact_t_s is not None, "first_scene_contact_t_s is not set after landing")
    ctx.expect(not robot and log.first_official_contact_t_s is None and log.partner_class is None,
               "landing on the support surface counted as a robot contact (%d robot-side event(s))" % len(robot))
    if scene:
        ctx.expect(normal_error is not None and normal_error <= tol["normal"],
                   "the support contact normal is %r rad from +z (tolerance %r)" % (normal_error, tol["normal"]))
    ctx.expect(not violations, "drop: %s" % "; ".join(violations[:3]))


def run(ctx):
    tol = {"clock": ctx.number(SECTION, "contact_time_slack_physics_steps") * ctx.physics_dt,
           "normal": ctx.number(SECTION, "normal_angle_tol_rad")}
    probe = ctx.probe()
    _strike(ctx, probe, ctx.scenario.strike, PartnerClass.ROBOT_LINK, "robot_link", tol)
    if ctx.scenario.held_strike is None:
        ctx.record("held_object", "not constructible: the scenario holds no object in its still episode")
        ctx.note("held_object classification not tested: scenario.held_strike is None")
    else:
        _strike(ctx, probe, ctx.scenario.held_strike, PartnerClass.HELD_OBJECT, "held_object", tol)
    _drop(ctx, probe, tol)


CHECK = Check("C4", "contact detection and partner classification",
              "robot link -> robot_link, held object -> held_object, support surface -> scene only; partner names, "
              "normals (partner -> hazard) and contact timestamps follow the contract", run)


def test_c04_partner_class(conformance):
    conformance.execute(CHECK)
