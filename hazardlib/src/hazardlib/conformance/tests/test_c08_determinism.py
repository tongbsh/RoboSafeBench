"""C8: determinism level. Two resets with the same seed and the same scripted motion are compared observation by
observation; the difference must agree with the BD's ``same_seed_repro``: ``exact`` means no difference at all,
``statistical`` means a measured magnitude within the configured bound of each observation kind (state vectors;
images, every camera on its own, in grey levels; D13 h). The magnitudes are reported either way; a kind whose bound
is still PLACEHOLDER is measured but not judged (``unverified``). ``unknown`` is not a declaration. With ``exact``
and ``snapshot_restore`` the snapshot must also replay exactly (R12: paired controls are taken from snapshots).

The repeats are shared with C2 / C7 / C12, which on statistical hosts judge ``task_success`` only when it agreed
between them (``common.same_seed_repeats``)."""
from hazardlib.conformance.common import judge_statistical_kinds, oracle_note, same_seed_repeats
from hazardlib.conformance.harness import observation_diff
from hazardlib.conformance.suite import Check

SECTION = "c8_determinism"


def _difference(ctx, runs_a, runs_b):
    vector = image = 0.0
    success = []
    for a, b in zip(runs_a, runs_b):
        diff = observation_diff(a.obs, b.obs)
        ctx.require(not diff.mismatch, "observation keys or shapes differ between repeats: %s" % diff.mismatch)
        vector = max(vector, diff.vector_max_abs)
        image = max(image, diff.image_max_abs)
        if a.success != b.success:
            success.append(round(b.t_s, 6))
    return vector, image, success


def _snapshot_replay(ctx, probe, steps):
    host, scenario, env = ctx.host, ctx.scenario, 0
    run = probe.start(env, None, scenario.motion, control_id="c8-snapshot")
    try:
        total = max(1, int(round(scenario.motion_duration_s / run.control_dt)))
        middle = total // 2
        for k in range(middle):
            probe.step(env, ctx.target.motion_action(k * run.control_dt))
        snap = host.snapshot(env)
        first = len(run.samples)
        for k in range(middle, middle + steps):
            probe.step(env, ctx.target.motion_action(k * run.control_dt))
        host.restore(env, snap)
        second = len(run.samples)
        for k in range(middle, middle + steps):
            probe.step(env, ctx.target.motion_action(k * run.control_dt))
    finally:
        probe.finish(env)
    return run.samples[first:second], run.samples[second:]


def run(ctx):
    declared = ctx.bd.determinism.same_seed_repro
    probe = ctx.probe()
    repeats = same_seed_repeats(ctx, probe)
    vector = max(repeats.vectors.values(), default=0.0)
    image = max(repeats.images.values(), default=0.0)
    success = repeats.oracle_differs_at_s
    measured = max(vector, image)
    ctx.record("declared_same_seed_repro", declared)
    ctx.record("repeat_max_abs_diff", {"vectors": vector, "images": image, "oracle_differs_at_s": success[:10]})
    if declared == "exact":
        bound = ctx.number(SECTION, "exact_max_abs_diff")
        ctx.expect(measured <= bound and not success,
                   "BD declares exact repro but two same-seed resets differ by %.3g (bound %r)%s"
                   % (measured, bound, oracle_note(success)))
        if ctx.bd.snapshot_restore is True:
            steps = ctx.integer(SECTION, "snapshot_replay_steps")
            ctx.require(callable(getattr(ctx.host, "snapshot", None)) and callable(getattr(ctx.host, "restore", None)),
                        "BD declares snapshot_restore but the host has no snapshot / restore")
            first, second = _snapshot_replay(ctx, probe, steps)
            s_vector, s_image, s_success = _difference(ctx, first, second)
            times_match = all(a.t_s == b.t_s for a, b in zip(first, second))
            ctx.record("snapshot_replay_max_abs_diff", {"vectors": s_vector, "images": s_image,
                                                        "oracle_differs_at_s": s_success[:10],
                                                        "clock_restored": times_match})
            ctx.expect(max(s_vector, s_image) <= bound and not s_success,
                       "restore(snapshot) does not replay exactly: difference %.3g (bound %r)%s"
                       % (max(s_vector, s_image), bound, oracle_note(s_success)))
            ctx.expect(times_match, "restore(snapshot) does not restore the simulation clock")
    elif declared == "statistical":
        if measured == 0.0 and not success:
            ctx.note("declared statistical but the repeats were identical; the BD could declare exact")
        judge_statistical_kinds(ctx, "same-seed resets", repeats.vectors, repeats.images)
    else:
        ctx.expect(False, "BD same_seed_repro is %r; it must be 'exact' or 'statistical' (measured difference %.3g)"
                   % (declared, measured))


CHECK = Check("C8", "determinism level",
              "same-seed resets with the same scripted motion differ by a magnitude consistent with the BD's "
              "same_seed_repro (exact: none, also after restore(snapshot); statistical: within the bound of each "
              "observation kind)", run)


def test_c08_determinism(conformance):
    conformance.execute(CHECK)
