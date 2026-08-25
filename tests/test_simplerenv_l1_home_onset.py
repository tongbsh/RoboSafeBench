"""SimplerEnv L1 home-onset latch (no simulator)."""

from __future__ import annotations

import pytest

from dynisafety.backends.simplerenv.disturbance.l1 import eef_xy_should_drop
from dynisafety.config.loader import LoadRequest, load_config


def test_off_circle_unchanged_when_home_onset_on() -> None:
    assert eef_xy_should_drop(
        xy_dist=0.19,
        trigger_dist=0.20,
        init_xy_dist=0.35,
        home_inside=False,
        home_onset=True,
        approach_eps=0.02,
    )
    assert not eef_xy_should_drop(
        xy_dist=0.21,
        trigger_dist=0.20,
        init_xy_dist=0.35,
        home_inside=False,
        home_onset=True,
        approach_eps=0.02,
    )


def test_home_onset_off_matches_legacy_xy() -> None:
    assert eef_xy_should_drop(
        xy_dist=0.10,
        trigger_dist=0.20,
        init_xy_dist=0.10,
        home_inside=True,
        home_onset=False,
        approach_eps=0.02,
    )


def test_home_inside_does_not_drop_at_bind_distance() -> None:
    assert not eef_xy_should_drop(
        xy_dist=0.10,
        trigger_dist=0.20,
        init_xy_dist=0.10,
        home_inside=True,
        home_onset=True,
        approach_eps=0.02,
    )


def test_home_inside_drops_only_after_closer_than_eps() -> None:
    assert not eef_xy_should_drop(
        xy_dist=0.09,
        trigger_dist=0.20,
        init_xy_dist=0.10,
        home_inside=True,
        home_onset=True,
        approach_eps=0.02,
    )
    assert eef_xy_should_drop(
        xy_dist=0.07,
        trigger_dist=0.20,
        init_xy_dist=0.10,
        home_inside=True,
        home_onset=True,
        approach_eps=0.02,
    )


def test_resolved_simplerenv_l1_enables_home_onset() -> None:
    resolved = load_config(
        LoadRequest(backend="simplerenv", level=1, suite="bridge", model="random")
    )
    l1 = resolved.data["extra"]["simplerenv"]["l1"]
    assert l1["home_onset"] is True
    assert float(l1["home_approach_eps"]) == pytest.approx(0.02)
    assert resolved.provenance["extra.simplerenv.l1.home_onset"].endswith(
        "backends/simplerenv/levels/l1.yaml"
    )
