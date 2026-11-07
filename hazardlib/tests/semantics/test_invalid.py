import pytest

from hazardlib.semantics import (HazardEventLog, InvalidAttribution, InvalidClass, classify_invalid,
                                 metric_denominator)

MAX_D = 0.5          # test data (m), not a project value
START = 2.0          # window start (the onset), test data
END = 5.0            # window end (nominal hit + attribution window), test data


def classify(setup_failed=False, engine_fault=False, t=None, start=START, end=END, distance=None, max_d=MAX_D,
             is_hazard=False):
    return classify_invalid(setup_failed, engine_fault, t, start, end, distance, max_d, is_hazard)


@pytest.mark.parametrize("kwargs,expected_class,ambiguous", [
    # setup problems are decided first
    (dict(setup_failed=True), InvalidClass.SETUP_INVALID, False),
    (dict(setup_failed=True, engine_fault=True, t=3.0, distance=0.1), InvalidClass.SETUP_INVALID, False),
    (dict(setup_failed=True, engine_fault=True, t=3.0, is_hazard=True), InvalidClass.SETUP_INVALID, False),
    # nothing went wrong
    (dict(), InvalidClass.NONE, False),
    (dict(is_hazard=True), InvalidClass.NONE, False),
    # the attribution table of D9 d / D10 b
    (dict(engine_fault=True, t=1.0, distance=0.1), InvalidClass.SIM_UNSTABLE, False),                # before window
    (dict(engine_fault=True, t=3.0, distance=0.1), InvalidClass.HAZARD_INDUCED_UNSTABLE, False),      # inside, near
    (dict(engine_fault=True, t=3.0, distance=0.9), InvalidClass.SIM_UNSTABLE, False),                # inside, far
    (dict(engine_fault=True, t=3.0, distance=None), InvalidClass.HAZARD_INDUCED_UNSTABLE, True),     # unknown: keep
    # D10 b: long after the window it is not the hazard's fault, however close the entity is
    (dict(engine_fault=True, t=60.0, distance=0.0), InvalidClass.SIM_UNSTABLE, False),
    (dict(engine_fault=True, t=60.0, distance=None), InvalidClass.SIM_UNSTABLE, False),
    # D10 b: the hazard itself becoming unstable is always hazard induced once the window has opened
    (dict(engine_fault=True, t=60.0, is_hazard=True), InvalidClass.HAZARD_INDUCED_UNSTABLE, False),
    (dict(engine_fault=True, t=3.0, distance=0.9, is_hazard=True), InvalidClass.HAZARD_INDUCED_UNSTABLE, False),
    (dict(engine_fault=True, t=3.0, distance=None, is_hazard=True), InvalidClass.HAZARD_INDUCED_UNSTABLE, False),
    (dict(engine_fault=True, t=START, is_hazard=True), InvalidClass.HAZARD_INDUCED_UNSTABLE, False),
    (dict(engine_fault=True, t=1.0, is_hazard=True), InvalidClass.SIM_UNSTABLE, False),             # before onset
    # both window ends and the distance bound are inclusive
    (dict(engine_fault=True, t=START, distance=0.1), InvalidClass.HAZARD_INDUCED_UNSTABLE, False),
    (dict(engine_fault=True, t=END, distance=0.1), InvalidClass.HAZARD_INDUCED_UNSTABLE, False),
    (dict(engine_fault=True, t=START - 1e-9, distance=0.1), InvalidClass.SIM_UNSTABLE, False),
    (dict(engine_fault=True, t=END + 1e-9, distance=0.1), InvalidClass.SIM_UNSTABLE, False),
    (dict(engine_fault=True, t=3.0, distance=MAX_D), InvalidClass.HAZARD_INDUCED_UNSTABLE, False),
    (dict(engine_fault=True, t=3.0, distance=MAX_D * 1.0000001), InvalidClass.SIM_UNSTABLE, False),
    # a zero-length window still holds its single instant
    (dict(engine_fault=True, t=START, end=START, distance=0.1), InvalidClass.HAZARD_INDUCED_UNSTABLE, False),
    # an unknown distance outside the window cannot be hazard induced, and is not "ambiguous"
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
    (dict(engine_fault=True, t=3.0, start=None), "window is incomplete"),
    (dict(engine_fault=True, t=3.0, end=None), "window is incomplete"),
    (dict(engine_fault=True, t=3.0, start=5.0, end=2.0), "window_end_s"),
    (dict(start=5.0, end=2.0), "window_end_s"),                       # checked even without a fault
    (dict(engine_fault=True, t=3.0, start=float("nan")), "window_start_s"),
    (dict(engine_fault=True, t=3.0, end=float("inf")), "window_end_s"),
    (dict(engine_fault=False, t=3.0), "engine_fault is false"),
    (dict(engine_fault=True, t=float("nan")), "instability_time_s"),
    (dict(engine_fault=True, t=3.0, distance=-0.1), "distance_m"),
    (dict(engine_fault=True, t=3.0, distance=float("inf")), "distance_m"),
    (dict(max_d=-1.0), "max_distance_m"),
    (dict(max_d=None), "max_distance_m"),
    (dict(is_hazard=None), "unstable_entity_is_hazard"),
    (dict(is_hazard=1), "unstable_entity_is_hazard"),
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


def test_a_late_instability_leaves_the_denominator_but_an_unstable_hazard_stays():
    late = classify(engine_fault=True, t=60.0, distance=0.0)
    hazard = classify(engine_fault=True, t=60.0, is_hazard=True)
    summary = metric_denominator([late, hazard])
    assert (summary.n_sim_unstable, summary.n_hazard_induced_unstable, summary.n_in_denominator) == (1, 1, 1)
