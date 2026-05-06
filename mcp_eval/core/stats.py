"""Statistical helpers for the eval suite.

Two routines, both used directly in user-facing reports:

* :func:`wilson_ci` produces a Wilson score confidence interval for a binomial
  proportion. Wilson is the standard in ML evals because it stays well-behaved
  near 0 and 1, where the normal approximation explodes.
* :func:`mcnemar` runs McNemar's exact test on a 2x2 paired contingency table
  and returns a p-value. Used in Phase 3 to compare two model versions on the
  same eval suite.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

# Two-sided z-score for a 95% confidence interval (standard normal).
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class WilsonInterval:
    successes: int
    trials: int
    proportion: float
    lower: float
    upper: float
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _z_for_confidence(confidence: float) -> float:
    if confidence <= 0.0 or confidence >= 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if math.isclose(confidence, 0.95):
        return Z_95
    # Pull scipy lazily so importing mcp_eval.core.stats stays cheap.
    from scipy.stats import norm

    return float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))


def wilson_ci(successes: int, trials: int, confidence: float = 0.95) -> WilsonInterval:
    """Wilson score interval for a binomial proportion at the given confidence level.

    Reference: Wilson, E. B. (1927) "Probable inference, the law of succession, and
    statistical inference". Journal of the American Statistical Association
    22(158), pp. 209-212.
    """
    if trials < 0:
        raise ValueError("trials must be non-negative")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be in [0, trials]")

    if trials == 0:
        return WilsonInterval(0, 0, 0.0, 0.0, 1.0, confidence)

    z = _z_for_confidence(confidence)
    p_hat = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    center = (p_hat + z2 / (2.0 * trials)) / denom
    half = z * math.sqrt(p_hat * (1.0 - p_hat) / trials + z2 / (4.0 * trials * trials)) / denom
    # Clamp the boundary cases exactly so callers do not see floating-point dust
    # on the [0, 1] edges where the formula is mathematically tight.
    lower = 0.0 if successes == 0 else max(0.0, center - half)
    upper = 1.0 if successes == trials else min(1.0, center + half)
    return WilsonInterval(successes, trials, p_hat, lower, upper, confidence)


@dataclass(frozen=True)
class McNemarResult:
    """Outcome of McNemar's test on a 2x2 paired contingency table.

    The table layout (matching the regression eval semantics):

                       B correct    B wrong
        A correct       n00          n01
        A wrong         n10          n11

    Only the off-diagonal cells (b = n01, c = n10) drive the test.
    """

    b: int
    c: int
    statistic: float
    p_value: float
    method: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _binomial_two_sided_p(b: int, c: int) -> float:
    """Two-sided exact binomial test on (b, b+c) under H0: p=0.5."""
    n = b + c
    if n == 0:
        return 1.0
    # P(X = k) under Binomial(n, 0.5) = C(n, k) / 2**n.
    log2 = math.log(2.0)
    log_pmf = [
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * log2
        for k in range(n + 1)
    ]
    target = log_pmf[b]
    p = 0.0
    for k in range(n + 1):
        if log_pmf[k] <= target + 1e-12:
            p += math.exp(log_pmf[k])
    return min(1.0, p)


def mcnemar(b: int, c: int, *, exact: bool | None = None) -> McNemarResult:
    """McNemar's test on the off-diagonal counts of a paired 2x2 table.

    ``b`` is the count where model A was correct and model B was wrong;
    ``c`` is the count where model A was wrong and model B was correct.
    With ``exact=True`` the function uses the exact two-sided binomial test
    (recommended when ``b + c < 25``). With ``exact=False`` it uses the
    chi-squared approximation with continuity correction. The default picks
    automatically based on sample size.
    """
    if b < 0 or c < 0:
        raise ValueError("b and c must be non-negative")

    n = b + c
    if exact is None:
        exact = n < 25

    if exact:
        p = _binomial_two_sided_p(min(b, c), max(b, c))
        statistic = float(min(b, c))
        return McNemarResult(b=b, c=c, statistic=statistic, p_value=p, method="binomial_exact")

    if n == 0:
        return McNemarResult(b=0, c=0, statistic=0.0, p_value=1.0, method="chi2_continuity")

    chi2 = (abs(b - c) - 1) ** 2 / n
    # scipy.stats.chi2.sf with df=1 == erfc(sqrt(chi2/2)).
    p = math.erfc(math.sqrt(max(chi2, 0.0) / 2.0))
    return McNemarResult(b=b, c=c, statistic=chi2, p_value=p, method="chi2_continuity")
