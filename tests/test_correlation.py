"""The correlation contract: for a same-match builder the joint probability is
counted over shared simulations, and for correlated legs it must diverge
meaningfully from the naive product of leg probabilities.
"""

from __future__ import annotations

from simulation.markets import Leg, price_builder, probability


def test_negatively_correlated_pair_joint_far_below_product(big_sim_result):
    res = big_sim_result
    # "Under 2.5 goals" AND "home team to score 3+" is near mutually exclusive:
    # a team scoring 3 forces the match total to at least 3.
    under = Leg("total_goals", {"line": 2.5, "side": "under"})
    home_hattrick = Leg("team_goals", {"team": "home", "line": 2.5, "side": "over"})

    bp = price_builder(res, [under, home_hattrick])
    assert bp.naive_prob > 0.01  # each leg is individually plausible
    assert bp.joint_prob < 0.2 * bp.naive_prob  # but together, almost never
    assert bp.correlation_factor < 0.85
    assert any("Negatively correlated" in w for w in bp.warnings)


def test_striker_two_plus_with_under_2_5_flags_warning(big_sim_result):
    res = big_sim_result
    top_striker = max(
        res.inputs.home.player_names,
        key=lambda nm: res.home.goals[:, res.home.col(nm)].mean(),
    )
    legs = [
        Leg("total_goals", {"line": 2.5, "side": "under"}),
        Leg("player_goals", {"team": "home", "player": top_striker, "n": 2}),
    ]
    bp = price_builder(res, legs)
    assert bp.joint_prob < bp.naive_prob
    assert bp.correlation_factor < 0.85
    assert bp.warnings


def test_positively_correlated_pair_joint_above_product(big_sim_result):
    res = big_sim_result
    over_35 = Leg("total_goals", {"line": 3.5, "side": "over"})
    btts = Leg("btts", {"yes": True})
    bp = price_builder(res, [over_35, btts])
    assert bp.joint_prob > 1.15 * bp.naive_prob
    assert bp.correlation_factor > 1.15
    assert any("Positively correlated" in w for w in bp.warnings)


def test_near_independent_pair_has_no_warning(big_sim_result):
    res = big_sim_result
    # A home-corners line and an away player's card chance share little structure.
    away_player = res.inputs.away.player_names[3]
    legs = [
        Leg("team_corners", {"team": "home", "line": 4.5, "side": "over"}),
        Leg("player_carded", {"team": "away", "player": away_player}),
    ]
    bp = price_builder(res, legs)
    assert 0.88 < bp.correlation_factor < 1.12
    assert not bp.warnings


def test_single_leg_builder_equals_leg_probability(big_sim_result):
    res = big_sim_result
    leg = Leg("result", {"outcome": "home"})
    bp = price_builder(res, [leg])
    assert bp.joint_prob == probability(res, leg)
    assert bp.correlation_factor == 1.0


def test_joint_never_exceeds_min_leg(big_sim_result):
    res = big_sim_result
    legs = [
        Leg("result", {"outcome": "home"}),
        Leg("btts", {"yes": True}),
        Leg("total_goals", {"line": 2.5, "side": "over"}),
    ]
    bp = price_builder(res, legs)
    assert bp.joint_prob <= min(bp.leg_probs) + 1e-12
