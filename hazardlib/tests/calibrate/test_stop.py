import numpy as np
import pytest

from hazardlib.calibrate import (TStopSummary, aggregate_t_stop, estimate_t_stop, margin_sigma,
                                 staircase_update)


def decay_record(v0, time_constant, dt, n, t_hold):
    times = t_hold + dt * np.arange(n)
    return times, v0 * np.exp(-(times - t_hold) / time_constant)


# ------------------------------------------------------------------ estimate_t_stop

def test_exponential_decay_matches_the_analytic_crossing():
    for v0, tc, thr, dt in ((0.5, 0.15, 0.01, 0.001), (1.2, 0.4, 0.05, 0.002), (0.3, 0.08, 0.005, 0.0005)):
        analytic = tc * np.log(v0 / thr)
        for t_hold in (0.0, 2.5):
            times, speeds = decay_record(v0, tc, dt, int(3 * analytic / dt), t_hold)
            for k in (1, 3, 8):
                estimate = estimate_t_stop(times, speeds, thr, k, t_hold)
                # the first sample below the threshold: within one sample spacing after the analytic time
                assert analytic <= estimate + 1e-12
                assert estimate - analytic < dt + 1e-12


def test_estimate_is_relative_to_the_hold_time_not_the_first_sample():
    times, speeds = decay_record(0.5, 0.15, 0.001, 2000, 7.0)
    assert estimate_t_stop(times, speeds, 0.01, 1, 7.0) == pytest.approx(0.15 * np.log(50), abs=0.0011)
    assert estimate_t_stop(times, speeds, 0.01, 1, 7.0) == pytest.approx(
        estimate_t_stop(times - 7.0, speeds, 0.01, 1, 0.0), abs=1e-12)


def test_samples_before_the_hold_are_ignored():
    times = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    speeds = np.array([0.0, 0.0, 0.0, 0.9, 0.5, 0.0])          # at rest before the hold at t = 0.3
    assert estimate_t_stop(times, speeds, 0.1, 1, 0.3) == pytest.approx(0.2)
    assert estimate_t_stop(times, speeds, 0.1, 1, 0.0) == 0.0


def test_a_single_sample_glitch_does_not_trigger_a_stop():
    times, speeds = decay_record(0.5, 0.15, 0.001, 2000, 0.0)
    clean = estimate_t_stop(times, speeds, 0.01, 3, 0.0)
    glitched = speeds.copy()
    glitched[40] = 0.0                                           # one-sample dip at t = 0.04 s
    assert estimate_t_stop(times, glitched, 0.01, 3, 0.0) == clean
    assert estimate_t_stop(times, glitched, 0.01, 1, 0.0) == pytest.approx(0.04)   # k = 1 is fooled
    glitched[41] = 0.0
    assert estimate_t_stop(times, glitched, 0.01, 3, 0.0) == clean              # two samples: still no
    glitched[42] = 0.0
    assert estimate_t_stop(times, glitched, 0.01, 3, 0.0) == pytest.approx(0.04)   # three: a real run


def test_many_isolated_dropouts_do_not_change_the_estimate():
    times, speeds = decay_record(0.5, 0.15, 0.001, 2000, 0.0)
    clean = estimate_t_stop(times, speeds, 0.01, 4, 0.0)
    rng = np.random.RandomState(7)
    noisy = speeds.copy()
    candidates = np.arange(5, int(clean / 0.001) - 5, 7)          # isolated, all before the crossing
    noisy[candidates] = rng.uniform(0.0, 0.005, candidates.size)  # dips below the threshold
    assert estimate_t_stop(times, noisy, 0.01, 4, 0.0) == clean
    assert estimate_t_stop(times, noisy, 0.01, 1, 0.0) < clean   # without the rule it would fire early


def test_missing_samples_break_a_run():
    times = np.arange(7) * 0.1
    nan = float("nan")
    speeds = np.array([0.9, 0.0, 0.0, nan, 0.0, 0.0, 0.0])
    assert estimate_t_stop(times, speeds, 0.1, 2, 0.0) == pytest.approx(0.1)
    assert estimate_t_stop(times, speeds, 0.1, 3, 0.0) == pytest.approx(0.4)    # the run restarts after nan
    assert estimate_t_stop(times, np.array([0.9, 0.0, 0.0, nan, 0.9, 0.0, 0.0]), 0.1, 3, 0.0) is None


def test_threshold_is_strict_and_never_stopping_gives_none():
    times = np.array([0.0, 0.1, 0.2])
    assert estimate_t_stop(times, np.array([0.5, 0.2, 0.2]), 0.2, 1, 0.0) is None   # equal is not below
    assert estimate_t_stop(times, np.array([0.5, 0.4, 0.3]), 0.2, 1, 0.0) is None
    assert estimate_t_stop(times, np.array([0.1, 0.1, 0.1]), 0.2, 3, 0.0) == 0.0     # at rest from the start
    assert estimate_t_stop(times, np.array([0.1, 0.1, 0.1]), 0.2, 4, 0.0) is None    # record too short for k
    assert estimate_t_stop(times, np.array([0.1, 0.1, 0.1]), 0.2, 1, 5.0) is None    # nothing after the hold


@pytest.mark.parametrize("kwargs,match", [
    (dict(times_s=[0.0, 0.1], speeds_mps=[0.1]), "equal length"),
    (dict(times_s=[[0.0, 0.1]], speeds_mps=[[0.1, 0.1]]), "1-D"),
    (dict(times_s=[0.2, 0.1], speeds_mps=[0.1, 0.1]), "non-decreasing"),
    (dict(times_s=[0.0, float("nan")], speeds_mps=[0.1, 0.1]), "times_s"),
    (dict(times_s=[0.0, 0.1], speeds_mps=[0.1, -0.1]), "speeds_mps"),
    (dict(times_s=[0.0, 0.1], speeds_mps=[0.1, float("inf")]), "speeds_mps"),
    (dict(v_threshold=0.0), "v_threshold"),
    (dict(v_threshold=-1.0), "v_threshold"),
    (dict(k=0), "k"),
    (dict(k=1.5), "k"),
    (dict(k=True), "k"),
    (dict(t_hold_s=float("nan")), "t_hold_s"),
])
def test_estimate_t_stop_rejects_invalid_inputs(kwargs, match):
    args = dict(times_s=[0.0, 0.1], speeds_mps=[0.1, 0.1], v_threshold=0.2, k=1, t_hold_s=0.0)
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        estimate_t_stop(**args)


# ------------------------------------------------------------------- aggregate_t_stop

def test_median_and_iqr_use_linear_interpolation_quartiles():
    summary = aggregate_t_stop([5.0, 1.0, 3.0, 2.0, 4.0])
    assert isinstance(summary, TStopSummary)
    assert (summary.n_total, summary.n_valid, summary.n_censored) == (5, 5, 0)
    assert (summary.median_s, summary.q1_s, summary.q3_s, summary.iqr_s) == (3.0, 2.0, 4.0, 2.0)
    even = aggregate_t_stop([1.0, 2.0, 3.0, 4.0])
    assert (even.median_s, even.q1_s, even.q3_s) == (2.5, 1.75, 3.25)
    assert even.iqr_s == pytest.approx(1.5)


def test_censored_repetitions_are_counted_and_excluded():
    summary = aggregate_t_stop([0.2, None, 0.4, None, 0.3])
    assert (summary.n_total, summary.n_valid, summary.n_censored) == (5, 3, 2)
    assert summary.median_s == pytest.approx(0.3)
    nothing = aggregate_t_stop([None, None])
    assert (nothing.n_valid, nothing.n_censored) == (0, 2)
    assert nothing.median_s is None and nothing.iqr_s is None and nothing.q1_s is None
    empty = aggregate_t_stop([])
    assert empty.n_total == 0 and empty.median_s is None


def test_aggregate_accepts_numpy_floats_and_rejects_bad_values():
    assert aggregate_t_stop([np.float64(0.5)]).median_s == 0.5
    single = aggregate_t_stop([0.5])
    assert single.median_s == single.q1_s == single.q3_s == 0.5 and single.iqr_s == 0.0
    for bad in (-0.1, float("nan"), float("inf"), True, "0.5"):
        with pytest.raises(ValueError):
            aggregate_t_stop([0.1, bad])


def test_identical_repetitions_have_zero_spread():
    summary = aggregate_t_stop([0.25] * 9)
    assert summary.iqr_s == 0.0 and summary.median_s == 0.25


# ----------------------------------------------------------------------- margin_sigma

def test_margin_sigma_is_tau_over_t_stop():
    assert margin_sigma(0.6, 0.3) == pytest.approx(2.0)
    assert margin_sigma(0.15, 0.3) == pytest.approx(0.5)    # < 1: hard to avoid
    assert margin_sigma(0.0, 0.3) == 0.0
    for bad in ((-0.1, 0.3), (0.5, 0.0), (0.5, -0.3), (float("nan"), 0.3), (0.5, float("inf"))):
        with pytest.raises(ValueError):
            margin_sigma(*bad)


# -------------------------------------------------------------------- staircase_update

def test_one_step_moves_up_after_contact_and_down_after_avoidance():
    assert staircase_update(1.0, True, 0.1, 0.0, 2.0) == pytest.approx(1.1)
    assert staircase_update(1.0, False, 0.1, 0.0, 2.0) == pytest.approx(0.9)
    assert staircase_update(1.0, np.bool_(True), 0.1, 0.0, 2.0) == pytest.approx(1.1)


def test_staircase_is_clamped_to_its_bounds():
    assert staircase_update(1.95, True, 0.1, 0.0, 2.0) == 2.0
    assert staircase_update(0.05, False, 0.1, 0.0, 2.0) == 0.0
    assert staircase_update(2.0, True, 0.1, 0.0, 2.0) == 2.0
    assert staircase_update(0.0, False, 0.1, 0.0, 2.0) == 0.0


@pytest.mark.parametrize("args", [
    (1.0, 1, 0.1, 0.0, 2.0), (1.0, "yes", 0.1, 0.0, 2.0), (1.0, None, 0.1, 0.0, 2.0),
    (1.0, True, 0.0, 0.0, 2.0), (1.0, True, -0.1, 0.0, 2.0), (1.0, True, 0.1, 2.0, 2.0),
    (1.0, True, 0.1, 3.0, 2.0), (3.0, True, 0.1, 0.0, 2.0), (-1.0, True, 0.1, 0.0, 2.0),
    (float("nan"), True, 0.1, 0.0, 2.0),
])
def test_staircase_rejects_invalid_arguments(args):
    with pytest.raises(ValueError):
        staircase_update(*args)


def run_staircase(observer, tau0, step, lo, hi, trials):
    tau, track = tau0, []
    for _ in range(trials):
        track.append(tau)
        tau = staircase_update(tau, observer(tau), step, lo, hi)
    return np.array(track)


def test_staircase_converges_to_a_deterministic_threshold_from_either_side():
    tau50, step = 0.73, 0.05
    for tau0 in (0.1, 1.9):
        track = run_staircase(lambda tau: tau < tau50, tau0, step, 0.0, 2.0, 200)
        tail = track[-50:]
        assert np.all(np.abs(tail - tau50) <= step)           # hovers within one step of the threshold
        assert abs(np.mean(tail) - tau50) <= step / 2 + 1e-9


def test_staircase_finds_the_fifty_percent_point_of_a_noisy_observer():
    rng = np.random.RandomState(11)
    tau50, scale, step = 0.8, 0.1, 0.05

    def observer(tau):          # contact probability 50% at tau50 (logistic psychometric curve)
        return bool(rng.uniform() < 1.0 / (1.0 + np.exp((tau - tau50) / scale)))

    track = run_staircase(observer, 1.5, step, 0.0, 3.0, 3000)
    assert abs(np.mean(track[500:]) - tau50) < 0.05


def test_staircase_stays_at_the_bound_when_the_observer_never_contacts_or_always_contacts():
    track = run_staircase(lambda tau: False, 1.0, 0.1, 0.2, 2.0, 30)
    assert track[-1] == 0.2 and np.all(track >= 0.2)
    track = run_staircase(lambda tau: True, 1.0, 0.1, 0.2, 2.0, 30)
    assert track[-1] == 2.0 and np.all(track <= 2.0)
