"""Tests for the Wilson CI and McNemar's test helpers."""

from __future__ import annotations

import math

import pytest
from scipy.stats import binomtest

from mcp_eval.core.stats import mcnemar, wilson_ci


def test_wilson_8_of_10_textbook_reference() -> None:
    # Classic textbook example: 8/10 at 95% confidence yields a Wilson interval of about [0.49, 0.94].
    ci = wilson_ci(8, 10)
    assert ci.successes == 8
    assert ci.trials == 10
    assert math.isclose(ci.proportion, 0.8)
    assert math.isclose(ci.lower, 0.49, abs_tol=0.01)
    assert math.isclose(ci.upper, 0.94, abs_tol=0.01)


def test_wilson_zero_trials_returns_full_interval() -> None:
    ci = wilson_ci(0, 0)
    assert ci.proportion == 0.0
    assert ci.lower == 0.0
    assert ci.upper == 1.0


def test_wilson_all_correct_upper_bound_is_one() -> None:
    ci = wilson_ci(20, 20)
    assert ci.proportion == 1.0
    assert ci.upper == 1.0
    assert ci.lower < 1.0


def test_wilson_zero_correct_lower_bound_is_zero() -> None:
    ci = wilson_ci(0, 50)
    assert ci.proportion == 0.0
    assert ci.lower == 0.0
    assert ci.upper > 0.0


@pytest.mark.parametrize("k,n", [(3, 17), (50, 100), (1, 5), (99, 100), (12, 24), (7, 23)])
def test_wilson_matches_scipy(k: int, n: int) -> None:
    ours = wilson_ci(k, n)
    ref = binomtest(k, n).proportion_ci(method="wilson", confidence_level=0.95)
    assert math.isclose(ours.lower, ref.low, abs_tol=1e-6)
    assert math.isclose(ours.upper, ref.high, abs_tol=1e-6)


def test_wilson_99_percent_confidence_widens_interval() -> None:
    narrow = wilson_ci(8, 10, confidence=0.95)
    wide = wilson_ci(8, 10, confidence=0.99)
    assert wide.lower < narrow.lower
    assert wide.upper > narrow.upper


def test_wilson_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        wilson_ci(-1, 5)
    with pytest.raises(ValueError):
        wilson_ci(6, 5)
    with pytest.raises(ValueError):
        wilson_ci(2, -1)
    with pytest.raises(ValueError):
        wilson_ci(1, 1, confidence=0.0)
    with pytest.raises(ValueError):
        wilson_ci(1, 1, confidence=1.0)


def test_mcnemar_exact_matches_scipy_binomtest_when_small() -> None:
    # When b + c < 25, the exact binomial test is the right reference.
    for b, c in [(3, 7), (1, 9), (5, 5), (0, 4), (10, 14)]:
        ours = mcnemar(b, c, exact=True)
        ref = binomtest(min(b, c), b + c, p=0.5, alternative="two-sided").pvalue
        assert math.isclose(ours.p_value, ref, abs_tol=1e-9)


def test_mcnemar_chi2_matches_scipy_chi2() -> None:
    from scipy.stats import chi2 as scipy_chi2

    for b, c in [(40, 60), (100, 50), (20, 80)]:
        ours = mcnemar(b, c, exact=False)
        chi = (abs(b - c) - 1) ** 2 / (b + c)
        expected_p = scipy_chi2.sf(chi, df=1)
        assert math.isclose(ours.statistic, chi, abs_tol=1e-9)
        assert math.isclose(ours.p_value, expected_p, abs_tol=1e-6)


def test_mcnemar_no_disagreement_returns_one() -> None:
    result = mcnemar(0, 0)
    assert result.p_value == 1.0


def test_mcnemar_default_picks_exact_for_small_samples() -> None:
    small = mcnemar(2, 3)
    large = mcnemar(40, 60)
    assert small.method == "binomial_exact"
    assert large.method == "chi2_continuity"


def test_mcnemar_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        mcnemar(-1, 3)
    with pytest.raises(ValueError):
        mcnemar(3, -1)
