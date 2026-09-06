"""Shared cached fixture simulation + a standard market panel for the ranking pages."""

from __future__ import annotations

import streamlit as st

import data_access
from models.fixture import build_match_inputs, default_n_sims
from simulation.engine import simulate
from simulation.markets import Leg, probability

PRICE_SEED = 20260906  # fixed -> a stable quote across reruns


@st.cache_resource(show_spinner="Simulating 50,000 matches…")
def simulate_fixture(home: str, away: str, league: str, tier: int, n_sims: int, _mtime: float):
    R = data_access.ratings()
    mi = build_match_inputs(
        R, home, away, league=league, tier=tier, model_cfg=data_access.get_model_config()
    )
    return simulate(mi, n_sims=n_sims, seed=PRICE_SEED), mi


def sim_for(home: str, away: str, league: str, tier: int):
    return simulate_fixture(
        home,
        away,
        league,
        tier,
        default_n_sims(data_access.get_model_config()),
        data_access._mtime(data_access.RATINGS_PATH),
    )


# a compact, fixed set of markets for Value / High-Confidence ranking
STANDARD_MARKETS: list[tuple[str, Leg]] = [
    ("Home win", Leg("result", {"outcome": "home"})),
    ("Draw", Leg("result", {"outcome": "draw"})),
    ("Away win", Leg("result", {"outcome": "away"})),
    ("Over 1.5", Leg("total_goals", {"line": 1.5, "side": "over"})),
    ("Over 2.5", Leg("total_goals", {"line": 2.5, "side": "over"})),
    ("Under 2.5", Leg("total_goals", {"line": 2.5, "side": "under"})),
    ("Over 3.5", Leg("total_goals", {"line": 3.5, "side": "over"})),
    ("BTTS", Leg("btts", {"yes": True})),
    ("BTTS No", Leg("btts", {"yes": False})),
    ("Home or draw", Leg("double_chance", {"outcome": "home_draw"})),
    ("Draw or away", Leg("double_chance", {"outcome": "draw_away"})),
    ("Home -1 (win by 2+)", Leg("team_goals", {"team": "home", "line": 1.5, "side": "over"})),
]


def _calib_key(leg: Leg) -> str | None:
    """Map a Leg to the calibration-map family key, or None if it isn't covered."""
    p = leg.params
    if leg.market == "result":
        return f"result_{p['outcome']}"
    if leg.market == "btts" and p.get("yes", True):
        return "btts_yes"
    if leg.market == "total_goals" and p.get("side") == "over":
        return f"over_{p['line']}"
    return None


@st.cache_data(show_spinner="Pricing the upcoming slate…")
def slate(_ratings_mtime: float, _fixtures_mtime: float, n_sims: int = 20_000):
    """Model probabilities for every standard market across all upcoming fixtures."""
    import pandas as pd

    fx = data_access.fixtures(_fixtures_mtime)
    R = data_access.ratings()
    if fx is None or len(fx) == 0 or R is None:
        return pd.DataFrame()
    mcfg = data_access.get_model_config()
    rows = []
    for f in fx.itertuples(index=False):
        home, away, league, tier = f.home_team, f.away_team, f.league, int(f.tier)
        if home not in R.table.index or away not in R.table.index:
            continue
        mi = build_match_inputs(R, home, away, league=league, tier=tier, model_cfg=mcfg)
        res = simulate(mi, n_sims=n_sims, seed=PRICE_SEED)
        # raw model probabilities only - a per-market calibration map would make
        # complementary lines (Over/Under) incoherent in a scanner view. The map
        # is applied per-selection where it's validated (Phase 4 value ranking).
        for r in market_row(res, mi, cmap=None, league=league):
            rows.append(
                {
                    "kickoff": f.kickoff,
                    "fixture": f"{home} v {away}",
                    "league": league,
                    "market": r["label"],
                    "model_prob": r["prob"],
                    "cal_prob": r["prob_cal"],
                    "fair_odds": r["fair"],
                }
            )
    return pd.DataFrame(rows)


def load_slate():
    return slate(
        data_access._mtime(data_access.RATINGS_PATH),
        data_access._mtime(data_access.FIXTURES_PATH),
    )


def market_row(res, mi, cmap=None, league: str | None = None) -> list[dict]:
    """One dict per standard market for a fixture: label, model probability, and
    the calibration-map-adjusted probability where the map covers that market."""
    out = []
    for label, leg in STANDARD_MARKETS:
        p = probability(res, leg)
        key = _calib_key(leg)
        p_cal = cmap.apply(key, p, league) if (cmap is not None and key) else p
        out.append({"label": label, "prob": p, "prob_cal": p_cal, "fair": 1.0 / max(p_cal, 1e-9)})
    return out
