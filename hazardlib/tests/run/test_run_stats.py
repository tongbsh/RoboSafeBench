"""Interval estimates and quartiles of ``hazardctl report`` (R1 item 5)."""
import math

import pytest

from hazardlib.run.stats import normal_quantile, proportion, proportion_interval, quartiles


def _tail_ge(k, n, p):
    return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1))


@pytest.mark.parametrize("k, n, lo, hi", [
    (3, 10, 0.06673951, 0.65245285),     # R: binom.test(3, 10)$conf.int
    (0, 5, 0.0, 0.52182375),             # R: binom.test(0, 5)$conf.int
    (5, 5, 0.47817625, 1.0),             # R: binom.test(5, 5)$conf.int
])
def test_clopper_pearson_matches_reference_values(k, n, lo, hi):
    got = proportion_interval(k, n, "clopper_pearson", 0.95)
    assert got[0] == pytest.approx(lo, abs=1e-7) and got[1] == pytest.approx(hi, abs=1e-7)


@pytest.mark.parametrize("k, n, confidence", [(1, 4, 0.9), (7, 20, 0.95), (12, 13, 0.99), (40, 100, 0.8)])
def test_clopper_pearson_bounds_solve_the_binomial_tails(k, n, confidence):
    alpha = 1 - confidence
    lo, hi = proportion_interval(k, n, "clopper_pearson", confidence)
    assert _tail_ge(k, n, lo) == pytest.approx(alpha / 2, abs=1e-9)
    assert 1 - _tail_ge(k + 1, n, hi) == pytest.approx(alpha / 2, abs=1e-9)


def test_wilson_matches_the_closed_form():
    z = 1.959963984540054
    assert normal_quantile(0.975) == pytest.approx(z, abs=1e-12)
    lo, hi = proportion_interval(3, 10, "wilson", 0.95)
    p, n = 0.3, 10
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z / (1 + z * z / n) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    assert (lo, hi) == pytest.approx((center - half, center + half), abs=1e-12)
    assert (lo, hi) == pytest.approx((0.1077913, 0.6032219), abs=1e-7)   # R: prop.test(3, 10, correct=FALSE)


def test_proportion_and_quartiles_edge_cases():
    assert proportion(0, 0, "wilson", 0.95) == {"k": 0, "n": 0, "rate": None, "ci": None}
    assert proportion(2, 4, "wilson", 0.95)["rate"] == 0.5
    assert quartiles([None, None]) == {"n": 0, "median": None, "q1": None, "q3": None}
    assert quartiles([4.0, None, 1.0, 3.0, 2.0]) == {"n": 4, "median": 2.5, "q1": 1.75, "q3": 3.25}
    with pytest.raises(ValueError):
        proportion_interval(3, 2, "wilson", 0.95)
    with pytest.raises(ValueError):
        proportion_interval(1, 2, "exact", 0.95)
