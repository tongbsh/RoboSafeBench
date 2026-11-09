"""Interval estimates of proportions and robust summaries for ``hazardctl report`` (numpy + math only).

* ``clopper_pearson``: the exact interval, from the binomial tails: the lower bound is the p with
  ``P(X >= k | n, p) = alpha/2`` (0 when k = 0), the upper bound the p with ``P(X <= k | n, p) = alpha/2``
  (1 when k = n). Both are found by bisection on the binomial tail, summed in log space with ``math.lgamma``
  (equivalent to the Beta quantiles; no scipy).
* ``wilson``: the score interval with ``z`` the two-sided normal quantile of ``confidence``, found by bisection on
  ``math.erf``.

Bisection runs until the bracket stops shrinking in floating point (an algorithmic limit, not a tolerance).
The method and the confidence level come from ``run.yaml: report.interval`` (no defaults here).
"""
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

INTERVAL_METHODS = ("clopper_pearson", "wilson")


def _bisect(f: Callable[[float], float], lo: float, hi: float) -> float:
    """The root of ``f`` on ``[lo, hi]`` for ``f`` increasing with ``f(lo) <= 0 <= f(hi)``."""
    while True:
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            return mid
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid


def _log_pmf(j: int, n: int, p: float) -> float:
    return (math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
            + j * math.log(p) + (n - j) * math.log1p(-p))


def binom_cdf(k: int, n: int, p: float) -> float:
    """``P(X <= k)`` for ``X ~ Binomial(n, p)``, ``0 < p < 1``."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    logs = [_log_pmf(j, n, p) for j in range(k + 1)]
    top = max(logs)
    return min(1.0, math.exp(top) * sum(math.exp(v - top) for v in logs))


def normal_quantile(q: float) -> float:
    """``Phi^-1(q)`` for ``0 < q < 1``."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie in (0, 1), got %r" % (q,))
    cdf = lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))  # noqa: E731
    lo, hi = -1.0, 1.0
    while cdf(lo) > q:
        lo *= 2.0
    while cdf(hi) < q:
        hi *= 2.0
    return _bisect(lambda x: cdf(x) - q, lo, hi)


def proportion_interval(k: int, n: int, method: str, confidence: float) -> Tuple[float, float]:
    if method not in INTERVAL_METHODS:
        raise ValueError("interval method %r is not one of %s" % (method, list(INTERVAL_METHODS)))
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1), got %r" % (confidence,))
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n and n > 0, got k=%r n=%r" % (k, n))
    alpha = 1.0 - confidence
    if method == "wilson":
        z = normal_quantile(1.0 - alpha / 2.0)
        p = k / float(n)
        denom = 1.0 + z * z / n
        center = (p + z * z / (2.0 * n)) / denom
        half = z / denom * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
        return max(0.0, center - half), min(1.0, center + half)
    lower = 0.0 if k == 0 else _bisect(lambda p: (1.0 - binom_cdf(k - 1, n, p)) - alpha / 2.0, 0.0, 1.0)
    upper = 1.0 if k == n else _bisect(lambda p: alpha / 2.0 - binom_cdf(k, n, p), 0.0, 1.0)
    return lower, upper


def proportion(k: int, n: int, method: str, confidence: float) -> Dict[str, Any]:
    """``{"k", "n", "rate", "ci"}``; ``rate`` and ``ci`` are ``None`` when ``n == 0``."""
    if n == 0:
        return {"k": 0, "n": 0, "rate": None, "ci": None}
    lo, hi = proportion_interval(k, n, method, confidence)
    return {"k": int(k), "n": int(n), "rate": k / float(n), "ci": [lo, hi]}


def quartiles(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    """``{"n", "median", "q1", "q3"}`` of the non-``None`` values (numpy's linear interpolation)."""
    kept = [float(v) for v in values if v is not None]  # type: List[float]
    if not kept:
        return {"n": 0, "median": None, "q1": None, "q3": None}
    q1, median, q3 = (float(v) for v in np.percentile(np.array(kept), [25.0, 50.0, 75.0]))
    return {"n": len(kept), "median": median, "q1": q1, "q3": q3}
