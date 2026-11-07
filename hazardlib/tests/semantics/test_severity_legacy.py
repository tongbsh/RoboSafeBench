"""Equivalence with the legacy ``src/dynisafety/core/newton_force.py`` (loaded by file path, never imported
through ``dynisafety``). The legacy normal points from the hand (the hazard) toward the partner; the new
convention is the opposite, so the normal is negated before it is fed to the legacy functions."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from hazardlib.semantics import analytic_impulse_ns, closing_normal_speed

LEGACY_DIR = Path(__file__).resolve().parents[3] / "src" / "dynisafety" / "core"
STUBBED = ("dynisafety", "dynisafety.core", "dynisafety.core.contact")


def load_by_path(module_name, path):
    """Load a file as a module under a private name. It is registered in ``sys.modules`` while it executes
    because ``@dataclass`` under ``from __future__ import annotations`` looks its own module up there."""
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


@pytest.fixture(scope="module")
def legacy():
    contact = load_by_path("t03_legacy_contact", LEGACY_DIR / "contact.py")      # numpy-only, loads as is
    saved = {name: sys.modules.get(name) for name in STUBBED}
    package, core, stub = (types.ModuleType(n) for n in STUBBED)
    stub.CONTACT_IMPULSE_EPS = contact.CONTACT_IMPULSE_EPS                         # the same value, not a guess
    package.core, core.contact = core, stub
    sys.modules.update(dict(zip(STUBBED, (package, core, stub))))
    try:
        newton = load_by_path("t03_legacy_newton_force", LEGACY_DIR / "newton_force.py")
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    return types.SimpleNamespace(newton=newton, contact=contact)


def test_the_stub_packages_are_removed_again(legacy):
    for name in STUBBED:
        assert name not in sys.modules      # dynisafety is not installed in the hazardlib test environments


def random_case(rng):
    mass = rng.uniform(0.01, 5.0)
    v_hazard = rng.normal(scale=1.5, size=3)
    v_partner = rng.normal(scale=1.5, size=3) * rng.randint(0, 2)       # half of the partners are at rest
    normal = rng.normal(size=3) * rng.uniform(0.1, 10.0)                  # partner -> hazard, not unit length
    return mass, v_hazard, v_partner, normal


def test_analytic_impulse_matches_the_legacy_implementation_on_200_random_inputs(legacy):
    rng = np.random.RandomState(2026)
    positive = 0
    for _ in range(200):
        mass, v_hazard, v_partner, normal = random_case(rng)
        e = [0.0, 0.3, 1.0][rng.randint(3)]
        new = analytic_impulse_ns(mass, closing_normal_speed(v_hazard, v_partner, normal), e)
        old = legacy.newton.l3_newton_normal_impulse_ns(mass, v_hazard, v_partner, -normal, restitution=e)
        assert new == pytest.approx(old, rel=1e-9, abs=0.0), (mass, v_hazard, v_partner, normal, e)
        positive += int(old > 0)
    assert 40 < positive < 160            # the sample really mixes approaching and separating contacts


def test_closing_speed_matches_the_legacy_implementation(legacy):
    rng = np.random.RandomState(7)
    for _ in range(200):
        _, v_hazard, v_partner, normal = random_case(rng)
        new = closing_normal_speed(v_hazard, v_partner, normal)
        old = legacy.newton.l3_closing_speed_mps(v_hazard, v_partner, -normal)
        assert new == pytest.approx(old, rel=1e-9, abs=1e-15)


def test_zero_normal_is_where_the_new_and_the_legacy_behaviour_differ(legacy):
    """Legacy: a zero normal silently falls back to the direction of the hazard's own velocity, which turns
    a missing normal into a plausible closing speed. New (D9 a): it is an error."""
    v_hazard = np.array([0.0, 0.0, -1.5])
    old = legacy.newton.l3_closing_speed_mps(v_hazard, np.zeros(3), np.zeros(3))
    assert old == pytest.approx(1.5)                                 # fallback to the velocity direction
    with pytest.raises(ValueError, match="non-zero"):
        closing_normal_speed(v_hazard, np.zeros(3), np.zeros(3))
    # the legacy impulse inherits the fallback; the new code refuses before computing one
    assert legacy.newton.l3_newton_normal_impulse_ns(0.2, v_hazard, np.zeros(3), np.zeros(3)) > 0


def test_both_implementations_reject_a_negative_restitution(legacy):
    with pytest.raises(ValueError):
        legacy.newton.l3_newton_normal_impulse_ns(0.2, [0, 0, -1.0], np.zeros(3), [0, 0, 1.0], restitution=-0.1)
    with pytest.raises(ValueError):
        analytic_impulse_ns(0.2, 1.0, -0.1)


def test_the_new_normal_convention_is_the_opposite_of_the_legacy_one(legacy):
    v_hazard, n_new = np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 1.0])
    assert closing_normal_speed(v_hazard, np.zeros(3), n_new) == pytest.approx(1.0)
    assert legacy.newton.l3_closing_speed_mps(v_hazard, np.zeros(3), -n_new) == pytest.approx(1.0)
    assert legacy.newton.l3_closing_speed_mps(v_hazard, np.zeros(3), n_new) == 0.0     # un-negated: separating
