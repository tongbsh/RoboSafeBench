"""C9: visibility (R16). A hazard at the free point (inside a policy camera's view) changes the policy images
relative to the hazard-free run while it is out, and changes nothing while parked; ``render_check`` (when the
host has it) agrees."""
from hazardlib.conformance.harness import held_instance, observation_diff, split_observation
from hazardlib.conformance.suite import Check

SECTION = "c9_visibility"


def run(ctx):
    host, scenario, env = ctx.host, ctx.scenario, 0
    min_pixels = ctx.integer(SECTION, "min_pixel_diff")
    parked_max = ctx.integer(SECTION, "parked_pixel_diff_max")
    duration = ctx.number(SECTION, "duration_s")
    exposure = ctx.number("probe", "exposure_s")
    probe = ctx.probe()
    control = probe.run(env, None, scenario.still, duration, probe.hold(env), control_id="c9-control")
    images, _ = split_observation(control.samples[0].obs)
    ctx.require(images, "policy_observation has no image, so the hazard cannot be seen by the policy (R16)")
    onset = ctx.on_grid(ctx.number("probe", "onset_s"))
    instance = held_instance("c9-visible", scenario.probe_agent, scenario.free_point_m, onset, duration + exposure)
    render_check = getattr(host, "render_check", None)
    renders = []

    def look(run_, k):
        if callable(render_check) and run_.handle is not None:
            renders.append((run_.samples[-1].t_s, bool(render_check(run_.handle))))

    present = probe.run(env, instance, scenario.still, duration, probe.hold(env), on_step=look)
    ctx.require(len(present.samples) == len(control.samples), "the two runs have different lengths")
    boundary = onset + 0.5 * ctx.physics_dt
    parked_pixels, visible_pixels = [], []
    for a, b in zip(control.samples, present.samples):
        diff = observation_diff(a.obs, b.obs)
        ctx.require(not diff.mismatch, "at t=%.3f s the observation keys or shapes differ: %s" % (b.t_s, diff.mismatch))
        (visible_pixels if b.t_s > boundary else parked_pixels).append(diff.image_pixels)
    ctx.require(visible_pixels and parked_pixels, "the probe has no parked or no visible samples")
    ctx.record("image_keys", sorted(images))
    ctx.record("parked_max_differing_pixels", max(parked_pixels))
    ctx.record("visible_min_differing_pixels", min(visible_pixels))
    ctx.expect(max(parked_pixels) <= parked_max, "while parked the hazard changes up to %d pixels (> %d)"
               % (max(parked_pixels), parked_max))
    ctx.expect(min(visible_pixels) >= min_pixels, "while out at the free point the hazard changes only %d pixels in "
                                                  "the policy images (< %d)" % (min(visible_pixels), min_pixels))
    if callable(render_check):
        wrong = [round(t, 6) for t, seen in renders if seen != (t > boundary)]
        ctx.record("render_check_disagreements", len(wrong))
        ctx.expect(not wrong, "render_check disagrees with the hazard state at t = %s s" % wrong[:5])


CHECK = Check("C9", "hazard visibility in the policy cameras",
              "out at the free point the hazard changes the policy images (>= min_pixel_diff pixels), parked it "
              "changes none; render_check agrees", run)


def test_c09_visibility(conformance):
    conformance.execute(CHECK)
