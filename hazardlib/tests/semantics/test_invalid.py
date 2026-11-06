import pytest

from hazardlib.semantics import (HazardEventLog, InvalidAttribution, InvalidClass, classify_invalid,
                                 metric_denominator)

MAX_D = 0.5          # test data (m), not a project value
ONSET = 2.0


def classify(setup_failed=False, engine_fault=False, t=None, onset=ONSET, distance=None, max_d=MAX_D):
    return classify_invalid(setup_failed, engine_fault, t, onset, distance, max_d)


@pytest.mark.parametrize("kwargs,expected_class,ambiguous", [
    # setup problems are decided first
    (dict(setup_failed=True), InvalidClass.SETUP_INVALID, False),
    (dict(setup_failed=True, engine_fault=True, t=3.0, distance=0.1), InvalidClass.SETUP_INVALID, False),
    # nothing went wrong
    (dict(), InvalidClass.NONE, False),
    # the attribution table of D9 d
    (dict(engine_fault=True, t=1.0, distance=0.1), InvalidClass.SIM_UNSTABLE, False),                # before onset
    (dict(engine_fault=True, t=3.0, distance=0.1), InvalidClass.HAZARD_INDUCED_UNSTABLE, False),      # after, near
    (dict(engine_fault=True, t=3.0, distance=0.9), InvalidClass.SIM_UNSTABLE, False),                # after, far
    (dict(engine_fault=True, t=3.0, distance=None), InvalidClass.HAZARD_INDUCED_UNSTABLE, True),     # unknown: keep
    # boundaries are inclusive on both conditions
    (dict(engine_fault=True, t=ONSET, distance=0.1), InvalidClass.HAZARD_INDUCED_UNSTABLE, False),
    (dict(engine_fault=True, t=3.0, distance=MAX_D), InvalidClass.HAZARD_INDUCED_UNSTABLE, False),
    (dict(engine_fault=True, t=3.0, distance=MAX_D * 1.0000001), InvalidClass.SIM_UNSTABLE, False),
    (dict(engine_fault=True, t=ONSET - 1e-9, distance=0.1), InvalidClass.SIM_UNSTABLE, False),
    # an unknown distance before the onset cannot be hazard induced, and is not "ambiguous"
    (dict(engine_fault=True, t=1.0, distance=None), InvalidClass.SIM_UNSTABLE, False),
    (dict(engine_fault=True, t=3.0, distance=0.0), InvalidClass.HAZARD_INDUCED_UNSTABLE, False),
])
def test_attribution_rule_table(kwargs, expected_class, ambiguous):
    result = classify(**kwargs)
    assert isinstance(result, InvalidAttribution)
    assert result.invalid_class is expected_class
    assert result.attribution_ambiguous is ambiguous


@pytest.mark.parametrize("kwargs,match", [
    (dict(setup_failed=1), "setup_failed"),
    (dict(engine_fault="yes"), "engine_fault"),
    (dict(engine_fault=True, t=None), "instability_time_s is None"),
    (dict(engine_fault=True, t=3.0, onset=None), "onset_time_s is None"),
    (dict(engine_fault=False, t=3.0), "engine_fault is false"),
    (dict(engine_fault=True, t=float("nan")), "instability_time_s"),
    (dict(engine_fault=True, t=3.0, distance=-0.1), "distance_m"),
    (dict(engine_fault=True, t=3.0, distance=float("inf")), "distance_m"),
    (dict(max_d=-1.0), "max_distance_m"),
    (dict(max_d=None), "max_distance_m"),
])
def test_invalid_arguments_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        classify(**kwargs)


def records():
    """A set with every class: 4 none, 2 setup_invalid, 3 sim_unstable, 3 hazard_induced (1 ambiguous)."""
    attr = lambda c, a=False: InvalidAttribution(c, a)
    return ([attr(InvalidClass.NONE)] * 4 + [attr(InvalidClass.SETUP_INVALID)] * 2
            + [attr(InvalidClass.SIM_UNSTABLE)] * 3
            + [attr(InvalidClass.HAZARD_INDUCED_UNSTABLE), attr(InvalidClass.HAZARD_INDUCED_UNSTABLE),
               attr(InvalidClass.HAZARD_INDUCED_UNSTABLE, True)])


def test_denominator_drops_the_first_two_classes_and_keeps_the_third():
    summary = metric_denominator(records())
    assert (summary.n_total, summary.n_none, summary.n_setup_invalid, summary.n_sim_unstable,
            summary.n_hazard_induced_unstable, summary.n_attribution_ambiguous) == (12, 4, 2, 3, 3, 1)
    assert summary.n_in_denominator == 7                 # 4 none + 3 hazard_induced; setup and sim leave
    assert summary.n_in_denominator == summary.n_total - summary.n_setup_invalid - summary.n_sim_unstable


def test_denominator_accepts_event_logs_and_string_classes():
    logs = [HazardEventLog.new("a", invalid_class="hazard_induced_unstable", attribution_ambiguous=True),
            HazardEventLog.new("b", invalid_class=InvalidClass.NONE),
            HazardEventLog.new("c", invalid_class="sim_unstable", attribution_ambiguous=False)]
    summary = metric_denominator(logs)
    assert (summary.n_hazard_induced_unstable, summary.n_none, summary.n_sim_unstable) == (1, 1, 1)
    assert summary.n_attribution_ambiguous == 1 and summary.n_in_denominator == 2


def test_denominator_of_nothing_and_of_a_generator():
    empty = metric_denominator([])
    assert (empty.n_total, empty.n_in_denominator, empty.n_attribution_ambiguous) == (0, 0, 0)
    assert metric_denominator(r for r in records()).n_total == 12


def test_a_record_that_was_never_classified_is_an_error_not_a_skip():
    with pytest.raises(ValueError, match="record 1 has no invalid_class"):
        metric_denominator([InvalidAttribution(InvalidClass.NONE, False), HazardEventLog.new("x")])
    with pytest.raises(ValueError, match="not one of"):
        metric_denominator([InvalidAttribution("garbage", False)])


def test_hazard_induced_cases_are_never_dropped_even_when_ambiguous():
    only = [classify(engine_fault=True, t=3.0, distance=None)]            # ambiguous attribution
    summary = metric_denominator(only)
    assert summary.n_in_denominator == 1 and summary.n_attribution_ambiguous == 1
