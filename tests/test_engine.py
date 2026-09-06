"""Engine sanity: shapes, minutes model, and the goals-reconciliation contract
(per-player goals sum exactly to the Dixon-Coles team score in every sim).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from simulation.engine import simulate
from simulation.markets import Leg, probability
from simulation.synthetic import synthetic_match_inputs


def test_output_shapes(sim_result):
    res = sim_result
    n = res.n_sims
    assert res.home_goals.shape == (n,)
    assert res.away_goals.shape == (n,)
    assert res.home.shots.shape == (n, res.inputs.home.n_players)
    assert res.away.sot.shape == (n, res.inputs.away.n_players)
    assert res.home_goal_minutes.shape[0] == n


def test_per_player_goals_sum_to_team_score(sim_result):
    res = sim_result
    assert np.array_equal(res.home.goals.sum(axis=1), res.home_goals)
    assert np.array_equal(res.away.goals.sum(axis=1), res.away_goals)


def test_sot_never_exceeds_shots_per_player(sim_result):
    res = sim_result
    assert (res.home.sot <= res.home.shots).all()
    assert (res.away.sot <= res.away.shots).all()


def test_goals_never_exceed_sot_plus_slack(sim_result):
    # Scorers are allocated from SoT weight; a player with goals should almost
    # always have >=1 SoT. Allow a tiny fraction from the minutes-only fallback.
    res = sim_result
    scored = res.home.goals >= 1
    have_sot = res.home.sot >= 1
    frac_ok = (have_sot | ~scored).mean()
    assert frac_ok > 0.98


def test_minutes_within_bounds_and_starters_play_more(sim_result):
    res = sim_result
    m = res.home.minutes
    assert (m >= 0).all() and (m <= 90).all()
    starters = res.inputs.home.start_prob > 0.5
    assert m[:, starters].mean() > m[:, ~starters].mean() + 30


def test_half_time_goals_do_not_exceed_full_time(sim_result):
    res = sim_result
    assert (res.home_ht_goals <= res.home_goals).all()
    assert (res.away_ht_goals <= res.away_goals).all()


def test_corners_and_shots_are_positive_on_average(sim_result):
    res = sim_result
    assert 8 < res.home_shots.mean() < 20
    assert 3 < res.home_corners.mean() < 9


def test_anytime_scorer_ordering_by_position(sim_result):
    res = sim_result
    home = res.inputs.home
    names_pos = list(zip(home.player_names, home.position, strict=True))
    fwd = [n for n, p in names_pos if p == 3 and home.start_prob[home.player_id(n)] > 0.5]
    dfn = [n for n, p in names_pos if p == 1 and home.start_prob[home.player_id(n)] > 0.5]

    def anytime(name: str) -> float:
        return probability(res, Leg("player_goals", {"team": "home", "player": name, "n": 1}))

    assert np.mean([anytime(n) for n in fwd]) > np.mean([anytime(n) for n in dfn])


def test_reproducible_with_seed():
    mi = synthetic_match_inputs(seed=3)
    a = simulate(mi, n_sims=5_000, seed=42)
    b = simulate(mi, n_sims=5_000, seed=42)
    assert np.array_equal(a.home_goals, b.home_goals)
    assert np.array_equal(a.home.shots, b.home.shots)


@pytest.mark.perf
def test_fifty_thousand_sims_under_one_second():
    mi = synthetic_match_inputs(seed=5)
    simulate(mi, n_sims=2_000, seed=1)  # warm import / JIT-free but fair
    t0 = time.perf_counter()
    simulate(mi, n_sims=50_000, seed=1)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"50k sims took {elapsed:.3f}s (budget 1.0s)"
