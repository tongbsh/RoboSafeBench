"""C7: isolation (R5). A visible hazard floating in free space, touching nothing, leaves every non-visual policy
state vector and the success oracle exactly as in the hazard-free control run; the end-of-episode
``task_success`` equals the oracle right before the hazard is parked (01 §3.3 item 11).

On a statistical host (BD ``same_seed_repro: statistical``) the state vectors are compared with C8's state-vector
bound and the oracle only when C8's same-seed repeats agree on it (01 §5, D13 h; see ``common``). Images are not
compared: the hazard is meant to be seen."""
from hazardlib.conformance.common import (judge_statistical_kinds, judge_statistical_oracle, same_seed_repeats,
                                          statistical_repro)
from hazardlib.conformance.harness import held_instance, merge_max, observation_diff
from hazardlib.conformance.suite import Check

SECTION = "c7_isolation"


def run(ctx):
    scenario, env = ctx.scenario, 0
    repeats = same_seed_repeats(ctx) if statistical_repro(ctx) else None
    duration = ctx.number(SECTION, "duration_s")
    exposure = ctx.number("probe", "exposure_s")
    probe = ctx.probe()
    control = probe.run(env, None, scenario.still, duration, probe.hold(env), control_id="c7-control")
    onset = ctx.on_grid(ctx.number("probe", "onset_s"))
    # released only after the episode ends: kinematic and still for the whole exposure
    instance = held_instance("c7-floating", scenario.probe_agent, scenario.free_point_m, onset, duration + exposure)
    present = probe.run(env, instance, scenario.still, duration, probe.hold(env))
    ctx.require(len(present.samples) == len(control.samples), "the two runs have different lengths")
    contacts = [c for r in probe.records(probe.rec.contacts, present) if not r.parked for c in r.contacts]
    ctx.require(not contacts, "scenario: the hazard at free_point_m touched %r, so it is not free space"
                % sorted({c.partner_name for c in contacts}))
    ctx.require(probe.records(probe.rec.poses, present), "the hazard was never unparked")

    vector_max = 0.0
    vectors = {}
    oracle_differs, oracle_true_while_visible = [], 0
    for a, b in zip(control.samples, present.samples):
        diff = observation_diff(a.obs, b.obs)
        ctx.require(not diff.mismatch, "at t=%.3f s the observation keys or shapes differ: %s" % (b.t_s, diff.mismatch))
        vector_max = max(vector_max, diff.vector_max_abs)
        merge_max(vectors, diff.vectors)
        visible = b.t_s > onset + 0.5 * ctx.physics_dt
        if a.success != b.success:
            oracle_differs.append(round(b.t_s, 6))
        if visible and a.success:
            oracle_true_while_visible += 1
    ctx.record("state_vector_max_abs_diff", vector_max)
    ctx.record("oracle_differs_at_s", oracle_differs[:10])
    ctx.record("control_oracle_true_samples_while_visible", oracle_true_while_visible)
    ctx.record("task_success", {"control": control.log.task_success, "hazard": present.log.task_success,
                                "last_sample_with_hazard": present.samples[-1].success})
    if repeats is None:
        vec_tol = ctx.number(SECTION, "state_vector_abs_tol")
        ctx.expect(vector_max <= vec_tol, "a hazard that touches nothing changes the policy state vectors by %.3g "
                                          "(> %r)" % (vector_max, vec_tol))
        ctx.expect(not oracle_differs, "a hazard that touches nothing changes task_success at t = %s s"
                   % oracle_differs[:5])
        ctx.expect(present.log.task_success == control.log.task_success,
                   "end-of-episode task_success %r with the hazard vs %r without"
                   % (present.log.task_success, control.log.task_success))
    else:
        what = "hazard touching nothing vs hazard-free run"
        judge_statistical_kinds(ctx, what, vectors, None)
        judge_statistical_oracle(ctx, repeats, what, oracle_differs,
                                 (present.log.task_success, control.log.task_success))
    ctx.expect(present.log.task_success == present.samples[-1].success,
               "end_episode read task_success=%r but the oracle said %r after the last step"
               % (present.log.task_success, present.samples[-1].success))
    if oracle_true_while_visible == 0:
        ctx.unverified("the oracle of scenario.still is never true while the hazard is out, so an oracle that "
                       "reacts to the hazard cannot be observed (state vectors were checked)")


CHECK = Check("C7", "hazard isolation from state vectors and oracle",
              "a visible hazard touching nothing leaves policy state vectors and task_success identical to the "
              "hazard-free run (statistical hosts: state vectors within C8's bound)", run)


def test_c07_isolation(conformance):
    conformance.execute(CHECK)
