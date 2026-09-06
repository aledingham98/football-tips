from __future__ import annotations

import pytest

from pricing.acca import AccaLeg, drawdown_simulation, price_acca


def _legs(probs, book=None):
    return [
        AccaLeg(fixture=f"F{i}", label=f"leg {i}", model_prob=p, book_odds=book)
        for i, p in enumerate(probs)
    ]


def test_compound_probability_multiplies():
    q = price_acca(_legs([0.5, 0.5, 0.5]))
    assert q.compound_prob == pytest.approx(0.125)
    assert q.fair_odds == pytest.approx(8.0)
    assert q.n_legs == 3


def test_total_margin_vs_price_is_positive_when_book_shortens():
    # fair 8.0; book pays 6.0 -> implied 0.1667 vs model 0.125 -> ~33% margin
    q = price_acca(_legs([0.5, 0.5, 0.5]), book_odds=6.0)
    assert q.total_margin_vs_price == pytest.approx((1 / 6) / 0.125 - 1, rel=1e-6)
    assert q.total_margin_vs_price > 0.3
    assert q.ev < 0 and q.verdict in {"no value", "bad price"}


def test_compounded_leg_margins_beat_single_leg():
    q1 = price_acca(_legs([0.55], book=1.8))
    q5 = price_acca(_legs([0.55] * 5, book=1.8))
    assert q1.total_margin_from_legs is not None
    assert q5.total_margin_from_legs > 4 * q1.total_margin_from_legs


def test_value_verdict_when_price_beats_fair():
    q = price_acca(_legs([0.6, 0.6]), book_odds=1 / (0.36) * 1.2)  # 20% over fair
    assert q.ev > 0.1
    assert q.verdict == "VALUE"


def test_drawdown_simulation_shapes_and_monotone_risk():
    lo = drawdown_simulation(0.55, 2.0, stake_fraction=0.05, n_bets=150, n_paths=2000, seed=1)
    hi = drawdown_simulation(0.55, 2.0, stake_fraction=0.25, n_bets=150, n_paths=2000, seed=1)
    assert 0 < lo.median_max_drawdown < hi.median_max_drawdown  # bigger stake -> deeper DD
    assert hi.prob_bust_50 >= lo.prob_bust_50
    assert lo.sample_paths.shape[1] == 151  # start + n_bets
    assert lo.median_final > 1.0  # +EV bet grows on median


def test_full_kelly_is_wilder_than_quarter():
    q = drawdown_simulation(0.58, 1.9, stake_fraction=1.0, n_bets=200, n_paths=3000, seed=2)
    qk = drawdown_simulation(0.58, 1.9, stake_fraction=0.25, n_bets=200, n_paths=3000, seed=2)
    assert q.p05_final < qk.p05_final  # full Kelly's bad case is worse
    assert q.worst_max_drawdown > qk.worst_max_drawdown
