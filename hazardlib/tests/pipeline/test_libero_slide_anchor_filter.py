"""The slide anchor filter of the generator is the builder's own, public support rule (L2, D20)."""
import numpy as np
import pytest

import pipeline_world as W
import test_pipeline_geometry as G
from hazardlib.instantiate import InstanceRejected, slide_support_rule
from hazardlib.kinematics import SampledPath


def modes():
    return [("scalar", W.make_host(), W.task_context("low")), ("geometric", G.geo_host(), G.geo_tc("low"))]


@pytest.mark.parametrize("name", ["scalar", "geometric"])
def test_rule_agrees_with_the_builder_on_a_synthetic_case(name):
    host, tc = {n: (h, t) for n, h, t in modes()}[name]
    nominal = SampledPath.from_pose_sequence(tc.nominal_ee_path)
    phase = next(p for p in tc.phases if p.name == "approach")
    rule = slide_support_rule(tc, host.describe(), W.slide_spec().agent)
    outcomes = set()
    for frac in (0.15, 0.5, 0.85):
        t_star = phase.t_start_s + frac * (phase.t_end_s - phase.t_start_s)
        for dz in (-0.02, 0.0, 0.03, 0.05, 0.07, 0.085, 0.12, 0.3, 0.6):
            offset = (0.0, 0.0, dz)
            supported = bool(rule(nominal.pos(t_star) + np.array(offset), t_star))
            try:
                hit = W.build(W.slide_spec(frac=frac, offset=offset), "low", host=host, tc=tc).nominal_hit_time_s
                anchored, no_support = abs(hit - t_star) < 1e-9, False
            except InstanceRejected as exc:
                anchored, no_support = False, exc.reason == "no_support_surface"
            # A supported anchor is never refused for lack of support, and the builder hits exactly there when
            # it builds; an unsupported anchor is never used as the hit instant.
            assert not (supported and no_support), (name, frac, dz)
            assert not (anchored and not supported), (name, frac, dz)
            outcomes.add(supported)
    assert outcomes == {True, False}


def test_rule_reads_the_bd_radius_and_the_agent_height():
    tc = W.task_context("low")
    host = W.make_host()
    agent = W.slide_spec().agent
    top = tc.support_surfaces[0].max_m[2]
    rule = slide_support_rule(tc, host.describe(), agent)
    reach = 2 * W.HAZARD_R + host.describe().eef_contact_radius_m
    assert rule(np.array([0., 0., top + reach - 1e-3]), 1.0)
    assert not rule(np.array([0., 0., top + reach + 1e-3]), 1.0)
    assert not rule(np.array([0., 0., top - 0.05]), 1.0)          # below the top
