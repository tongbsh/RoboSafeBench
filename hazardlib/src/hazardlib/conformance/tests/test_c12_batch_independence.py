"""C12: batch independence. While a hazard is released onto the robot in environment 0, the last environment runs
a hazard-free control; its policy observations and oracle must equal those of the same control run alone.

Runs on every host with more than one environment, batched or not: environments served by one host object can
share state either way. Skipped only on single-environment hosts, where there is nothing to compare (01 §5:
batched benchmarks only)."""
import numpy as np

from hazardlib._records import is_unknown
from hazardlib.conformance.common import flight_start, oracle_note, still_snapshot
from hazardlib.conformance.harness import body_position, observation_diff, strike_instance
from hazardlib.conformance.suite import Check

SECTION = "c12_batch_independence"


def _skip(session):
    n = session.bd.parallel.num_envs
    if not is_unknown(n) and n == 1:
        return "single-environment host (parallel.num_envs = 1): no other environment to observe"
    return None


def run(ctx):
    host, scenario, strike, agent = ctx.host, ctx.scenario, ctx.scenario.strike, ctx.scenario.probe_agent
    n_envs = ctx.bd.parallel.num_envs
    ctx.require(not is_unknown(n_envs), "BD parallel.num_envs is unknown")
    tol = ctx.number(SECTION, "observation_abs_tol")
    duration = ctx.number(SECTION, "duration_s")
    speed = ctx.number("probe", "strike_speed_mps")
    gap_steps = ctx.number("probe", "strike_gap_physics_steps")
    exposure = ctx.number("probe", "exposure_s")
    hazard_env, observer = 0, n_envs - 1
    probe = ctx.probe()
    alone = probe.run(observer, None, scenario.still, duration, probe.hold(observer), control_id="c12-alone")

    robot, _ = still_snapshot(ctx, probe, hazard_env)
    body = body_position(robot, strike.body)
    gap = speed * ctx.physics_dt * gap_steps
    onset = ctx.on_grid(ctx.number("probe", "onset_s"))
    release = ctx.on_grid(onset + exposure)
    instance = strike_instance("c12-strike", agent, flight_start(body, strike, agent, gap),
                               np.array(strike.direction) * speed, onset, release, release, body)
    try:
        loud = probe.start(hazard_env, instance, scenario.still)
        quiet = probe.start(observer, None, scenario.still, control_id="c12-with-neighbour")
        for _ in range(len(alone.samples) - 1):
            probe.step(hazard_env, host.hold_action(hazard_env))
            probe.step(observer, host.hold_action(observer))
        probe.finish(hazard_env)
        probe.finish(observer)
    except BaseException:
        probe.abort()
        raise
    ctx.require(loud.log.first_official_contact_t_s is not None,
                "the hazard in env %d never hit the robot, so the probe exercised nothing" % hazard_env)
    vector = image = 0.0
    success = []
    for a, b in zip(alone.samples, quiet.samples):
        diff = observation_diff(a.obs, b.obs)
        ctx.require(not diff.mismatch, "env %d observation keys or shapes changed: %s" % (observer, diff.mismatch))
        vector = max(vector, diff.vector_max_abs)
        image = max(image, diff.image_max_abs)
        if a.success != b.success:
            success.append(round(b.t_s, 6))
    ctx.record("environments", {"hazard": hazard_env, "observer": observer, "num_envs": n_envs})
    ctx.record("observer_max_abs_diff", {"vectors": vector, "images": image, "oracle_differs_at_s": success[:10]})
    ctx.expect(max(vector, image) <= tol and not success,
               "a hazard in env %d changes the observations of env %d by %.3g (tol %r)%s"
               % (hazard_env, observer, max(vector, image), tol, oracle_note(success)))


CHECK = Check("C12", "batch independence",
              "a hazard striking the robot in env 0 leaves the observations and oracle of the last env identical to "
              "the same run alone", run, skip_when=_skip)


def test_c12_batch_independence(conformance):
    conformance.execute(CHECK)
