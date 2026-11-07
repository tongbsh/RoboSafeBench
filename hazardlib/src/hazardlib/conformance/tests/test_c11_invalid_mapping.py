"""C11: invalid-sample mapping (R11, D9 d, D12 g).

* A slide placed overlapping a robot body at its first pose cannot be placed: ``setup_invalid``, re-parked (no
  contact or render afterwards), excluded from the metric denominator.
* An engine fault forced by the adapter (``ConformanceTarget.inject_engine_fault``) must reach the runtime through
  ``report_instability`` (01 §3.3 item 12): in a hazard-free control, and in a hazard episode before the onset, it is
  ``sim_unstable`` (excluded); in the hazard body during the exposure it is ``hazard_induced_unstable`` (kept).

A host that cannot force a fault fails this check: the mapping would otherwise be unproven.
"""
import numpy as np

from hazardlib.conformance.common import still_snapshot
from hazardlib.conformance.harness import body_position, held_instance, support_radius
from hazardlib.conformance.suite import Check
from hazardlib.semantics import InvalidClass, metric_denominator


def _faulted(ctx, probe, instance, inject_at_s, hazard_body, label, duration):
    scenario, env = ctx.scenario, 0
    injected = []

    def inject(run_, k):
        if not injected and run_.samples[-1].t_s >= inject_at_s:
            ctx.target.inject_engine_fault(env, run_.handle if hazard_body else None)
            injected.append(run_.samples[-1].t_s)

    control_id = None if instance is not None else "c11-%s" % label
    run_ = probe.run(env, instance, scenario.still, duration, probe.hold(env), control_id=control_id, on_step=inject)
    ctx.require(injected, "%s: the fault was never injected" % label)
    return run_


def run(ctx):
    scenario, strike, agent = ctx.scenario, ctx.scenario.strike, ctx.scenario.probe_agent
    exposure = ctx.number("probe", "exposure_s")
    tail = ctx.number("probe", "tail_s")
    onset = ctx.on_grid(ctx.number("probe", "onset_s"))
    probe = ctx.probe()
    logs = {}

    # forced unplaceable: the first pose overlaps the strike body
    robot, _ = still_snapshot(ctx, probe, 0)
    body = body_position(robot, strike.body)
    direction = np.array(strike.direction)
    inside = body - direction * 0.5 * (strike.radius_m + support_radius(agent, direction))
    instance = held_instance("c11-unplaceable", agent, inside, onset, onset + exposure)
    run_ = probe.run(0, instance, scenario.still, onset + exposure + tail, probe.hold(0))
    after = [r for r in probe.records(probe.rec.contacts, run_) if r.parked and r.t_s > run_.t_start + onset]
    renders = [r for r in probe.rec.parked_render if r[1] == run_.handle.key and r[3]]
    logs["setup_invalid"] = run_.log
    ctx.record("unplaceable", {"setup_failed": run_.info["setup_failed"], "setup_reason": run_.info["setup_reason"],
                               "invalid_class": run_.log.invalid_class, "phase": run_.info["phase"],
                               "contacts_after_reparking": sum(len(r.contacts) for r in after),
                               "render_check_true_after_reparking": len(renders)})
    ctx.expect(run_.info["setup_failed"] and run_.log.invalid_class is InvalidClass.SETUP_INVALID,
               "a hazard whose first pose overlaps %r is %r, expected setup_invalid"
               % (strike.body, run_.log.invalid_class))
    ctx.expect(not any(r.contacts for r in after), "the re-parked hazard still reports contacts")
    ctx.expect(not renders, "the re-parked hazard is still in a policy camera (render_check)")

    if ctx.target.inject_engine_fault is None:
        ctx.expect(False, "the adapter offers no way to force an engine fault (ConformanceTarget.inject_engine_fault),"
                          " so the sim_unstable / hazard_induced_unstable mapping is unproven")
    else:
        duration = onset + exposure
        floating = lambda name: held_instance(name, agent, scenario.free_point_m, onset, duration + exposure)  # noqa
        cases = [("control_fault", None, onset, False, InvalidClass.SIM_UNSTABLE),
                 ("pre_onset_fault", floating("c11-pre-onset"), 0.5 * onset, False, InvalidClass.SIM_UNSTABLE),
                 ("hazard_fault", floating("c11-hazard-fault"), onset + 0.5 * exposure, True,
                  InvalidClass.HAZARD_INDUCED_UNSTABLE)]
        for label, inst, at, hazard_body, wanted in cases:
            run_ = _faulted(ctx, probe, inst, at, hazard_body, label, duration)
            report = run_.info["instability"]
            logs[label] = run_.log
            ctx.record(label, {"instability": report, "invalid_class": run_.log.invalid_class,
                               "attribution_ambiguous": run_.log.attribution_ambiguous})
            ctx.expect(report is not None, "%s: the forced fault never reached report_instability" % label)
            ctx.expect(run_.log.invalid_class is wanted, "%s: classified %r, expected %s"
                       % (label, run_.log.invalid_class, wanted.value))
    summary = metric_denominator(list(logs.values()))
    ctx.record("denominator", {"n_total": summary.n_total, "n_setup_invalid": summary.n_setup_invalid,
                               "n_sim_unstable": summary.n_sim_unstable,
                               "n_hazard_induced_unstable": summary.n_hazard_induced_unstable,
                               "n_in_denominator": summary.n_in_denominator})
    kept = sum(1 for log in logs.values() if log.invalid_class in (InvalidClass.NONE,
                                                                 InvalidClass.HAZARD_INDUCED_UNSTABLE))
    ctx.expect(summary.n_in_denominator == kept, "the denominator counts %d records, expected %d"
               % (summary.n_in_denominator, kept))


CHECK = Check("C11", "invalid-sample mapping",
              "an unplaceable instance is setup_invalid, re-parked and excluded; forced engine faults reach "
              "report_instability and map to sim_unstable (excluded) or hazard_induced_unstable (kept)", run)


def test_c11_invalid_mapping(conformance):
    conformance.execute(CHECK)
