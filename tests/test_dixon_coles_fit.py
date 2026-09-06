"""The Dixon-Coles fit must recover known team parameters from simulated matches,
and produce sensible fixture probabilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.dixon_coles import DixonColesConfig, analytic_markets, fit


def _simulate_league(n_teams=18, n_matches=460, seed=0):
    rng = np.random.default_rng(seed)
    atk = rng.normal(0, 0.33, n_teams)
    dfn = rng.normal(0, 0.28, n_teams)
    mu, gamma = np.log(1.35), 0.25
    teams = [f"T{i:02d}" for i in range(n_teams)]
    base = pd.Timestamp("2024-08-01")
    rows = []
    for k in range(n_matches):
        h, a = rng.choice(n_teams, 2, replace=False)
        # model convention: higher dfn == better defence, so it subtracts
        lh = np.exp(mu + gamma + atk[h] - dfn[a])
        la = np.exp(mu + atk[a] - dfn[h])
        rows.append(
            {
                "date": base + pd.Timedelta(days=k // 4),
                "home_team": teams[h],
                "away_team": teams[a],
                "fthg": rng.poisson(lh),
                "ftag": rng.poisson(la),
                "tier": 1,
            }
        )
    return pd.DataFrame(rows), atk, dfn, teams


def test_fit_converges_cleanly_and_quickly():
    import time

    df, *_ = _simulate_league(n_matches=900, seed=7)
    t0 = time.perf_counter()
    ratings = fit(df, DixonColesConfig(time_decay_half_life_days=1e6))
    assert ratings.meta["converged"]
    assert ratings.meta["opt_status"] == 0
    assert time.perf_counter() - t0 < 3.0  # analytic gradient keeps refits cheap


def test_fit_recovers_team_strength_ordering():
    df, true_atk, true_dfn, teams = _simulate_league(seed=1)
    ratings = fit(df, DixonColesConfig(time_decay_half_life_days=1e6, ratings_l2=0.02))
    assert ratings.meta["converged"]

    est = ratings.table.loc[teams]
    true_strength = (true_atk - true_atk.mean()) + (true_dfn - true_dfn.mean())
    est_strength = (est["atk"] - est["atk"].mean()) + (est["dfn"] - est["dfn"].mean())
    corr = np.corrcoef(true_strength, est_strength)[0, 1]
    assert corr > 0.85, f"strength recovery corr {corr:.2f}"


def test_fit_recovers_home_advantage_and_intercept():
    df, *_ = _simulate_league(seed=2)
    ratings = fit(df, DixonColesConfig(time_decay_half_life_days=1e6))
    assert ratings.gamma == pytest.approx(0.25, abs=0.12)
    assert np.exp(ratings.mu) == pytest.approx(1.35, abs=0.35)
    assert -0.2 <= ratings.rho <= 0.2


def test_stronger_team_has_higher_home_win_probability():
    df, true_atk, true_dfn, teams = _simulate_league(seed=3)
    ratings = fit(df, DixonColesConfig(time_decay_half_life_days=1e6))
    strength = true_atk + true_dfn
    strong, weak = teams[int(np.argmax(strength))], teams[int(np.argmin(strength))]

    p_strong_home = analytic_markets(ratings.score_matrix(strong, weak))["result_home"]
    p_weak_home = analytic_markets(ratings.score_matrix(weak, strong))["result_home"]
    assert p_strong_home > 0.55
    assert p_strong_home > p_weak_home + 0.25


def test_unknown_team_falls_back_to_tier_mean():
    df, *_ = _simulate_league(seed=4)
    ratings = fit(df, DixonColesConfig())
    lh, la = ratings.lambdas("T00", "Some New Team", away_tier=1)
    assert 0.3 < lh < 4.0
    assert 0.3 < la < 4.0


def test_roundtrip_parquet(tmp_path):
    df, *_ = _simulate_league(seed=5)
    ratings = fit(df, DixonColesConfig())
    p = tmp_path / "ratings.parquet"
    ratings.to_parquet(p)

    from models.dixon_coles import TeamRatings

    loaded = TeamRatings.from_parquet(p)
    assert loaded.mu == pytest.approx(ratings.mu)
    assert loaded.rho == pytest.approx(ratings.rho)
    lh1, la1 = ratings.lambdas(ratings.table.index[0], ratings.table.index[1])
    lh2, la2 = loaded.lambdas(loaded.table.index[0], loaded.table.index[1])
    assert (lh1, la1) == pytest.approx((lh2, la2))
