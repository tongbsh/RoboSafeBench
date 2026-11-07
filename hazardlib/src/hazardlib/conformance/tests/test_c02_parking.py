"""C2: a parked hazard is invisible and inert: the policy observations and the oracle equal those of a hazard-free
run, it reports no contact, it is not in any camera and it does not move (01 §3.1, §3.3 item 6, R6).

On a statistical host (BD ``same_seed_repro: statistical``) the observations are compared per kind with C8's bounds
and the oracle only when C8's same-seed repeats agree on it (01 §5, D13 h; see ``common``)."""
import numpy as np

from hazardlib.conformance.common import (judge_statistical_kinds, judge_statistical_oracle, same_seed_repeats,
                                          statistical_repro)
from hazardlib.conformance.harness import held_instance, merge_max, observation_diff
from hazardlib.conformance.suite import Check

SECTION = "c2_parking"


def run(ctx):
    host, env, scenario = ctx.host, 0, ctx.scenario
    repeats = same_seed_repeats(ctx) if statistical_repro(ctx) else None
    pose_tol = ctx.number(SECTION, "parked_pose_abs_tol_m")
    duration = ctx.number(SECTION, "duration_s")
    exposure = ctx.number("probe", "exposure_s")
    probe = ctx.probe()
    control = probe.run(env, None, scenario.still, duration, probe.hold(env), control_id="c2-control")
    # onset after the end of the episode: the hazard is allocated and parked throughout
    onset = ctx.on_grid(duration + exposure)
    instance = held_instance("c2-parked", scenario.probe_agent, scenario.free_point_m, onset, onset + exposure)
    renders = []
    render_check = getattr(host, "render_check", None)

    def look(run_, k):
        if callable(render_check) and run_.handle is not None:
            renders.append(bool(render_check(run_.handle)))

    parked = probe.run(env, instance, scenario.still, duration, probe.hold(env), on_step=look)
    ctx.require(parked.handle is not None, "begin_episode allocated no hazard")
    ctx.require(len(parked.samples) == len(control.samples), "the two runs have different lengths")

    vector_max = image_max = 0.0
    pixels = 0
    vectors, images = {}, {}
    for a, b in zip(control.samples, parked.samples):
        diff = observation_diff(a.obs, b.obs)
        ctx.require(not diff.mismatch, "at t=%.3f s the observation keys or shapes differ: %s" % (b.t_s, diff.mismatch))
        vector_max = max(vector_max, diff.vector_max_abs)
        image_max = max(image_max, diff.image_max_abs)
        pixels = max(pixels, diff.image_pixels)
        merge_max(vectors, diff.vectors)
        merge_max(images, diff.images)
    success_differs = [b.t_s for a, b in zip(control.samples, parked.samples) if a.success != b.success]
    ctx.record("state_vector_max_abs_diff", vector_max)
    ctx.record("image_max_abs_diff", image_max)
    ctx.record("image_max_differing_pixels", pixels)
    if repeats is None:
        vec_tol = ctx.number(SECTION, "state_vector_abs_tol")
        img_tol = ctx.number(SECTION, "image_abs_tol")
        ctx.expect(vector_max <= vec_tol, "a parked hazard changes the policy state vectors by %.3g (> %r)"
                   % (vector_max, vec_tol))
        ctx.expect(image_max <= img_tol, "a parked hazard changes the policy images by up to %r grey levels in %d "
                                         "pixels (> %r)" % (image_max, pixels, img_tol))
        ctx.expect(not success_differs, "a parked hazard changes task_success at t = %s s" % success_differs[:5])
        ctx.expect(parked.log.task_success == control.log.task_success,
                   "end-of-episode task_success %r with a parked hazard vs %r without"
                   % (parked.log.task_success, control.log.task_success))
    else:
        what = "parked hazard vs hazard-free run"
        judge_statistical_kinds(ctx, what, vectors, images)
        judge_statistical_oracle(ctx, repeats, what, [round(t, 6) for t in success_differs],
                                 (parked.log.task_success, control.log.task_success))

    contacts = [c for r in probe.records(probe.rec.contacts, parked) if r.parked for c in r.contacts]
    ctx.record("parked_contacts", len(contacts))
    ctx.expect(not contacts, "the parked hazard reported %d contact(s), first with %r"
               % (len(contacts), contacts[0].partner_name if contacts else None))
    parked_renders = [r for r in probe.rec.parked_render if r[0] == env and r[1] == parked.handle.key]
    render_hits = sum(renders) + sum(1 for r in parked_renders if r[3])
    ctx.record("render_check_available", callable(render_check))
    ctx.record("render_check_true_while_parked", render_hits)
    ctx.expect(render_hits == 0, "render_check says the parked hazard is in a policy camera (%d time(s))" % render_hits)
    states = [s.state.pos for s in probe.records(probe.rec.parked_states, parked)]
    drift = float(np.max(np.abs(np.array(states) - states[0]))) if states else 0.0
    ctx.record("parked_pose_drift_m", drift)
    ctx.expect(states, "hazard_contacts was never called while parked (the runtime drains it every post_step)")
    ctx.expect(drift <= pose_tol, "the parked hazard moved by %.3g m (> %r): it is not held out of the physics"
               % (drift, pose_tol))
    ctx.expect(not parked.events, "the runtime recorded %d event(s) for a parked hazard" % len(parked.events or []))


CHECK = Check("C2", "parked hazard is invisible and inert",
              "with the hazard parked for the whole episode, state vectors, images and task_success equal a "
              "hazard-free run (statistical hosts: within C8's per-kind bounds); no contacts, no render_check hit, "
              "no motion while parked", run)


def test_c02_parking(conformance):
    conformance.execute(CHECK)
