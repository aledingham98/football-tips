from __future__ import annotations

import pytest

from pricing.builder import quote_builder
from simulation.markets import Leg


def test_quote_builder_breaks_down_legs(big_sim_result):
    res = big_sim_result
    legs = [
        Leg("result", {"outcome": "home"}),
        Leg("btts", {"yes": True}),
        Leg("total_goals", {"line": 2.5, "side": "over"}),
    ]
    q = quote_builder(res, legs)
    assert len(q.legs) == 3
    assert all(0 < lb.standalone_prob < 1 for lb in q.legs)
    # joint sits at or below the least-likely leg
    assert q.price.joint_prob <= min(lb.standalone_prob for lb in q.legs) + 1e-12
    assert q.joint_fair_odds == pytest.approx(1 / q.price.joint_prob, rel=1e-9)


def test_positive_correlation_gives_shorter_than_naive(big_sim_result):
    res = big_sim_result
    legs = [Leg("total_goals", {"line": 3.5, "side": "over"}), Leg("btts", {"yes": True})]
    q = quote_builder(res, legs)
    assert q.joint_fair_odds < q.naive_fair_odds
    assert q.correlation_factor > 1.10
    assert "positively correlated" in q.correlation_note.lower()


def test_value_assessment_populated_with_book_price(big_sim_result):
    res = big_sim_result
    legs = [Leg("result", {"outcome": "home"}), Leg("total_goals", {"line": 1.5, "side": "over"})]
    q = quote_builder(res, legs)
    fair = q.joint_fair_odds
    # price generously above fair -> should read as value
    good = quote_builder(res, legs, book_odds=fair * 1.25)
    assert good.value is not None
    assert good.value.ev > 0
    assert good.value.verdict in {"VALUE", "marginal"}
    # price well below fair -> bad
    bad = quote_builder(res, legs, book_odds=fair * 0.7)
    assert bad.value.ev < 0
    assert any("Verdict" in ln for ln in bad.summary_lines())


def test_negative_correlation_warning_flows_through(big_sim_result):
    res = big_sim_result
    top = max(res.inputs.home.player_names, key=lambda n: res.home.goals[:, res.home.col(n)].mean())
    legs = [
        Leg("total_goals", {"line": 2.5, "side": "under"}),
        Leg("player_goals", {"team": "home", "player": top, "n": 2}),
    ]
    q = quote_builder(res, legs)
    assert q.correlation_factor < 0.9
    assert q.warnings
    assert q.joint_fair_odds > q.naive_fair_odds
