"""Referee card rates + real team shot/corner/card rates from match history."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ingest.teams import canonical_key, canonical_name
from models.referee import fit_referee_rates
from models.team_rates import fit_team_rates


def _history(seed=0, n=600):
    rng = np.random.default_rng(seed)
    teams = [f"Team {i}" for i in range(12)]
    refs = ["A Strict", "B Lenient", "C Average"]
    ref_rate = {"A Strict": 5.5, "B Lenient": 2.5, "C Average": 4.0}
    base = pd.Timestamp("2024-08-01")
    rows = []
    for k in range(n):
        h, a = rng.choice(teams, 2, replace=False)
        ref = rng.choice(refs)
        # team 0 is a shot machine, team 11 gets battered
        hs = rng.poisson(18 if h == "Team 0" else 12)
        as_ = rng.poisson(18 if a == "Team 0" else 12)
        rows.append(
            {
                "date": base + pd.Timedelta(days=k // 3),
                "league": "EPL",
                "home_team": h,
                "away_team": a,
                "fthg": rng.poisson(1.4),
                "ftag": rng.poisson(1.1),
                "referee": ref,
                "home_shots": hs,
                "away_shots": as_,
                "home_sot": rng.binomial(hs, 0.35),
                "away_sot": rng.binomial(as_, 0.35),
                "home_corners": rng.poisson(5),
                "away_corners": rng.poisson(5),
                "home_fouls": rng.poisson(11),
                "away_fouls": rng.poisson(11),
                "home_yellows": rng.poisson(ref_rate[ref] / 2),
                "away_yellows": rng.poisson(ref_rate[ref] / 2),
                "home_reds": rng.poisson(0.05),
                "away_reds": rng.poisson(0.05),
            }
        )
    return pd.DataFrame(rows)


def test_referee_rates_recover_strict_vs_lenient():
    rr = fit_referee_rates(_history(), pseudo_matches=5)
    strict_y, _ = rr.for_referee("A Strict")
    lenient_y, _ = rr.for_referee("B Lenient")
    assert strict_y > lenient_y + 0.6  # per team per match
    assert rr.for_referee("Unknown Ref")[0] == pytest.approx(rr.league_yellows_pm)


def test_referee_pseudocount_shrinks_thin_samples():
    df = _history(n=600)
    # a ref with 3 matches should sit close to the league mean
    extra = df.head(3).copy()
    extra["referee"] = "D Rookie"
    rr = fit_referee_rates(pd.concat([df, extra]), pseudo_matches=20)
    y, _ = rr.for_referee("D Rookie")
    assert abs(y - rr.league_yellows_pm) < 0.3


def test_team_rates_expected_reflects_strength():
    tr = fit_team_rates(_history())
    assert tr.has("Team 0") and tr.has("Team 5")
    hi, _ = tr.expected("Team 0", "Team 5", "shots")  # shot machine at home
    lo, _ = tr.expected("Team 5", "Team 0", "shots")
    assert hi > lo
    assert 8 < lo < 20 and 12 < hi < 30


def test_team_rates_unknown_team_is_gap_free():
    tr = fit_team_rates(_history())
    h, a = tr.expected("Team 0", "Nonexistent FC", "corners")
    assert h > 0 and a > 0  # falls back to league mean, no crash


def test_canonical_names():
    assert canonical_key("Man City") == canonical_key("Manchester City FC") == "manchester city"
    assert canonical_key("Nott'm Forest") == canonical_key("Nottingham Forest")
    assert canonical_name("Wolves") == "Wolverhampton Wanderers"
    assert canonical_name("Spurs") == "Tottenham Hotspur"
    assert canonical_name("brighton") == "Brighton & Hove Albion"
