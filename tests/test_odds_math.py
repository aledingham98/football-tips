from __future__ import annotations

import pytest

from pricing.odds_math import (
    acca_total_margin,
    assess_value,
    clv_pct,
    decimal_odds,
    implied_prob,
    kelly_fraction,
    no_vig_two_way,
    overround,
    remove_overround_equal_margin,
    remove_overround_proportional,
)


def test_prob_odds_round_trip():
    assert implied_prob(decimal_odds(0.25)) == pytest.approx(0.25)
    assert decimal_odds(implied_prob(3.5)) == pytest.approx(3.5)


def test_overround_fair_and_juiced():
    assert overround([2.0, 2.0]) == pytest.approx(0.0)
    assert overround([1.9, 1.9]) > 0.05


def test_margin_removal_sums_to_one():
    for fn in (remove_overround_proportional, remove_overround_equal_margin):
        probs = fn([2.5, 3.4, 3.1])
        assert sum(probs) == pytest.approx(1.0)
        assert all(p > 0 for p in probs)


def test_equal_margin_shifts_value_toward_longshots_vs_proportional():
    odds = [1.5, 8.0]
    prop = remove_overround_proportional(odds)
    eq = remove_overround_equal_margin(odds)
    # equal-margin gives the longshot a higher fair probability
    assert eq[1] > prop[1]


def test_no_vig_two_way_midpoint():
    # back 2.0 (0.5) and lay 2.1 (~0.476) -> ~0.488
    assert no_vig_two_way(2.0, 2.1) == pytest.approx(0.5 * (0.5 + 1 / 2.1))


def test_kelly_positive_only_for_positive_ev():
    assert kelly_fraction(0.60, 2.0, fraction=1.0) == pytest.approx(0.20)  # (1*0.6-0.4)/1
    assert kelly_fraction(0.40, 2.0, fraction=1.0) == 0.0
    assert kelly_fraction(0.55, 2.0, fraction=0.25) == pytest.approx(0.025)


def test_assess_value_verdicts():
    good = assess_value(0.55, 2.10)  # ev = 0.155
    assert good.verdict == "VALUE" and good.ev_pct > 3
    flat = assess_value(0.50, 2.00)  # ev = 0
    assert flat.verdict in {"marginal", "no value"}
    bad = assess_value(0.30, 2.00)  # ev = -0.4
    assert bad.verdict == "bad price"


def test_acca_margin_compounds():
    # [1.90, 1.90] is a ~5.3% single-market margin.
    one_leg = acca_total_margin([[1.90, 1.90]])
    five_legs = acca_total_margin([[1.90, 1.90]] * 5)
    assert one_leg == pytest.approx(2 / 1.9 - 1, rel=1e-6)
    assert five_legs > 4 * one_leg  # compounding, not additive
    assert 0.25 < five_legs < 0.34  # ~29% for 5 legs at ~5.3% each


def test_clv():
    assert clv_pct(2.10, 2.00) == pytest.approx(0.05)
    assert clv_pct(1.90, 2.00) < 0
