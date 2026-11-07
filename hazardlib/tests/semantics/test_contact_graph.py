import numpy as np
import pytest

from hazardlib.host import ContactObs
from hazardlib.semantics import (ConstraintState, ContactStep, continuous_runs, summarize_contacts)

T_Q = 0.5            # test data, not a project value
TOL = 0.1            # rad, test data
DT = 0.01


def obs(name, partner_class, normal, t=0.0):
    return ContactObs(partner_name=name, partner_class=partner_class, position=[0, 0, 0], normal=normal, time_s=t)


# normals point from the partner to the hazard: a hand above pushes down, the table below pushes up
HAND_DOWN = [0.0, 0.0, -1.0]
TABLE_UP = [0.0, 0.0, 1.0]


def pinched(t):
    return [obs("finger", "robot_link", HAND_DOWN, t), obs("table", "scene", TABLE_UP, t)]


def sequence(contact_for, start, stop, end=1.2, dt=DT):
    """Steps every dt from 0 to end; ``contact_for(t)`` gives the contacts for start <= t <= stop, else none."""
    steps = []
    for i in range(int(round(end / dt)) + 1):
        t = round(i * dt, 10)
        steps.append(ContactStep(t, contact_for(t) if start - 1e-9 <= t <= stop + 1e-9 else []))
    return steps


def summary(steps, t_q=T_Q, tol=TOL, gap=0.0):
    return summarize_contacts(steps, t_q, tol, gap)


# -------------------------------------------------------------------- clamp and t_q

def test_clamp_needs_the_pinch_to_last_at_least_t_q():
    short = summary(sequence(pinched, 0.1, 0.4))          # 0.3 s between the robot and the table
    long_ = summary(sequence(pinched, 0.1, 0.9))          # 0.8 s
    assert short.clamp_duration_s == pytest.approx(0.3) and short.constraint_state is ConstraintState.CONSTRAINED
    assert long_.clamp_duration_s == pytest.approx(0.8) and long_.constraint_state is ConstraintState.CLAMP
    assert long_.contact_duration_s == pytest.approx(0.8) and long_.first_contact_t_s == pytest.approx(0.1)


def test_t_q_is_inclusive():
    exactly = summary(sequence(pinched, 0.1, 0.6))        # 0.5 s
    assert exactly.clamp_duration_s == pytest.approx(0.5)
    assert summary(sequence(pinched, 0.1, 0.6), t_q=0.5).constraint_state is ConstraintState.CLAMP
    assert summary(sequence(pinched, 0.1, 0.6), t_q=0.5000001).constraint_state is ConstraintState.CONSTRAINED


def test_one_sided_contact_is_unconstrained():
    one_side = summary(sequence(lambda t: [obs("finger", "robot_link", HAND_DOWN, t)], 0.1, 0.9))
    assert one_side.constraint_state is ConstraintState.UNCONSTRAINED
    assert one_side.clamp_duration_s is None and one_side.contact_duration_s == pytest.approx(0.8)


def test_no_contact_and_scene_only_contact_give_no_state():
    for steps in (sequence(lambda t: [], 0.0, 0.0),
                  sequence(lambda t: [obs("table", "scene", TABLE_UP, t)], 0.1, 0.9)):
        result = summary(steps)
        assert result.constraint_state is None and result.segments == []
        assert result.first_contact_t_s is None and result.contact_duration_s is None
        assert result.longest_contact_s is None and result.clamp_duration_s is None
    assert summary([]).constraint_state is None


def test_normals_that_are_not_roughly_opposite_are_constrained_not_clamped():
    sideways = lambda t: [obs("finger", "robot_link", [1.0, 0.0, 0.0], t), obs("table", "scene", TABLE_UP, t)]
    result = summary(sequence(sideways, 0.1, 0.9))
    assert result.constraint_state is ConstraintState.CONSTRAINED and result.clamp_duration_s is None


def test_the_opposing_normals_tolerance_is_strict_and_symmetric():
    def tilted(angle_off):   # the hand normal turned by angle_off from exactly opposite to the table normal
        n = [np.sin(angle_off), 0.0, -np.cos(angle_off)]
        return lambda t: [obs("finger", "robot_link", n, t), obs("table", "scene", TABLE_UP, t)]
    assert summary(sequence(tilted(0.05), 0.1, 0.9), tol=0.1).constraint_state is ConstraintState.CLAMP
    assert summary(sequence(tilted(-0.05), 0.1, 0.9), tol=0.1).constraint_state is ConstraintState.CLAMP
    assert summary(sequence(tilted(0.05), 0.1, 0.9), tol=0.04).constraint_state is ConstraintState.CONSTRAINED
    assert summary(sequence(tilted(0.2), 0.1, 0.9), tol=0.1).constraint_state is ConstraintState.CONSTRAINED


def test_normals_need_not_be_unit_length():
    scaled = lambda t: [obs("finger", "robot_link", [0, 0, -7.0], t), obs("table", "scene", [0, 0, 0.2], t)]
    assert summary(sequence(scaled, 0.1, 0.9)).constraint_state is ConstraintState.CLAMP


# --------------------------------------------------------------- which contacts count

def test_a_pinch_between_two_robot_parts_or_a_held_object_and_the_table_is_a_clamp():
    two_links = lambda t: [obs("finger_l", "robot_link", [1.0, 0, 0], t), obs("finger_r", "robot_link", [-1.0, 0, 0], t)]
    held = lambda t: [obs("mug", "held_object", HAND_DOWN, t), obs("table", "scene", TABLE_UP, t)]
    assert summary(sequence(two_links, 0.1, 0.9)).constraint_state is ConstraintState.CLAMP
    assert summary(sequence(held, 0.1, 0.9)).constraint_state is ConstraintState.CLAMP


def test_two_scene_bodies_and_other_hazards_never_make_a_robot_contact_or_a_clamp():
    squeezed = lambda t: [obs("shelf", "scene", HAND_DOWN, t), obs("table", "scene", TABLE_UP, t)]
    assert summary(sequence(squeezed, 0.1, 0.9)).constraint_state is None
    with_other_hazard = lambda t: [obs("finger", "robot_link", HAND_DOWN, t), obs("hazard_1", "hazard", TABLE_UP, t)]
    assert summary(sequence(with_other_hazard, 0.1, 0.9)).constraint_state is ConstraintState.UNCONSTRAINED


def test_one_partner_reporting_opposing_normals_is_not_two_partners():
    same = lambda t: [obs("finger", "robot_link", HAND_DOWN, t), obs("finger", "robot_link", TABLE_UP, t)]
    assert summary(sequence(same, 0.1, 0.9)).constraint_state is ConstraintState.UNCONSTRAINED


def test_a_zero_normal_is_rejected():
    steps = sequence(lambda t: [obs("finger", "robot_link", [0, 0, 0], t)], 0.1, 0.2)
    with pytest.raises(ValueError, match="non-zero"):
        summary(steps)


# ----------------------------------------------------------------------- gaps

def with_hole(first_missing, last_missing):
    """A 0.8 s pinch (0.1..0.9) with no contact for the samples first_missing..last_missing."""
    def contacts(t):
        return [] if first_missing - 1e-9 <= t <= last_missing + 1e-9 else pinched(t)
    return sequence(contacts, 0.1, 0.9)


def test_a_short_gap_is_bridged_only_when_the_tolerance_covers_it():
    steps = with_hole(0.45, 0.49)            # contact up to 0.44, again from 0.50: a gap of 0.06 s
    assert summary(steps, gap=0.10).constraint_state is ConstraintState.CLAMP
    assert summary(steps, gap=0.10).clamp_duration_s == pytest.approx(0.8)
    assert summary(steps, gap=0.10).contact_duration_s == pytest.approx(0.8)
    broken = summary(steps, gap=0.02)        # 0.34 s and 0.40 s pieces, each shorter than t_q
    assert broken.constraint_state is ConstraintState.CONSTRAINED
    assert [round(s.duration_s, 6) for s in broken.segments] == [0.34, 0.4]
    assert broken.contact_duration_s == pytest.approx(0.34) and broken.longest_contact_s == pytest.approx(0.4)
    assert summary(steps, gap=0.0).constraint_state is ConstraintState.CONSTRAINED      # no tolerance at all


def test_gap_tolerance_boundary_is_inclusive():
    dt = 0.125                                # exactly representable, so the gap is exactly 0.25 s
    def steps(gap):
        times = [i * dt for i in range(0, 17)]
        return [ContactStep(t, [] if 0.5 < t < 0.75 else pinched(t)) for t in times]
    assert summary(steps(None), t_q=2.0, tol=TOL, gap=0.25).clamp_duration_s == pytest.approx(2.0)
    assert summary(steps(None), t_q=2.0, tol=TOL, gap=0.2499).clamp_duration_s == pytest.approx(1.25)


def test_a_gap_only_counts_when_a_contact_free_step_was_recorded():
    far_apart = [ContactStep(0.0, pinched(0.0)), ContactStep(5.0, pinched(5.0))]    # nothing recorded between
    result = summary(far_apart, gap=0.0)
    assert result.clamp_duration_s == pytest.approx(5.0)
    assert len(result.segments) == 1


def test_clamp_is_broken_by_a_step_with_only_one_side_unless_the_gap_is_tolerated():
    def contacts(t):
        return [obs("finger", "robot_link", HAND_DOWN, t)] if 0.45 <= t <= 0.49 else pinched(t)
    steps = sequence(contacts, 0.1, 0.9)
    assert summary(steps, gap=0.0).constraint_state is ConstraintState.CONSTRAINED       # robot contact never lost
    assert len(summary(steps, gap=0.0).segments) == 1                                      # ... so one contact segment
    assert summary(steps, gap=0.1).constraint_state is ConstraintState.CLAMP


def test_continuous_runs_helper():
    # true at t = 0, 1, 3, 6. The gap is (next true time) - (last true time): 3 - 1 = 2, then 6 - 3 = 3.
    t = [0, 1, 2, 3, 4, 5, 6]
    flags = [1, 1, 0, 1, 0, 0, 1]
    assert continuous_runs(t, flags, 1.0) == [(0, 1), (3, 3), (6, 6)]      # neither gap is covered
    assert continuous_runs(t, flags, 1.99) == [(0, 1), (3, 3), (6, 6)]
    assert continuous_runs(t, flags, 2.0) == [(0, 3), (6, 6)]              # the gap of 2 is covered (inclusive)
    assert continuous_runs(t, flags, 2.99) == [(0, 3), (6, 6)]
    assert continuous_runs(t, flags, 3.0) == [(0, 6)]                      # both covered
    assert continuous_runs(t, flags, 0.0) == [(0, 1), (3, 3), (6, 6)]
    assert continuous_runs(t, [0] * 7, 1.0) == []
    assert continuous_runs(t, [1] * 7, 0.0) == [(0, 6)]                    # no false sample, no gap
    assert continuous_runs([0, 1, 2], [1, 0, 0], 5.0) == [(0, 0)]          # a trailing hole ends the run
    assert continuous_runs([], [], 0.0) == []
    with pytest.raises(ValueError, match="equal length"):
        continuous_runs([0, 1], [True], 0.0)


# ------------------------------------------------------------------ validation

@pytest.mark.parametrize("kwargs,match", [
    (dict(t_q=0.0), "t_q_s"), (dict(t_q=-1.0), "t_q_s"), (dict(tol=0.0), "opposing_normals_tol_rad"),
    (dict(tol=3.2), "<= pi"), (dict(gap=-0.01), "gap_tolerance_s"), (dict(t_q=float("nan")), "t_q_s"),
])
def test_parameters_are_validated(kwargs, match):
    with pytest.raises(ValueError, match=match):
        summary(sequence(pinched, 0.1, 0.9), **kwargs)


def test_steps_must_be_contact_steps_with_increasing_times():
    with pytest.raises(ValueError, match="strictly increasing"):
        summary([ContactStep(0.1, []), ContactStep(0.1, [])])
    with pytest.raises(ValueError, match="strictly increasing"):
        summary([ContactStep(0.2, []), ContactStep(0.1, [])])
    with pytest.raises(ValueError, match="ContactStep"):
        summary([(0.0, [])])
    with pytest.raises(ValueError, match="ContactObs"):
        ContactStep(0.0, ["finger"])
    with pytest.raises(ValueError, match="time_s"):
        ContactStep(float("nan"), [])
