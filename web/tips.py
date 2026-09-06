"""Build the tips payload the static site renders.

Pure functions over the committed parquet/JSON: fitted ratings, upcoming
fixtures, match history (for form) and the latest odds snapshot. Produces one
dict -> ``public/data.json``.

Two kinds of tip, never merged (per the brief):

* **Confidence** - the model's most likely outcomes, ranked by model
  probability, with the price shown so a bad price is obvious. Labelled LIKELY,
  not VALUE.
* **Value** - selections where the model has a real edge over the no-vig market
  consensus AND a book prices them generously. Expected to be short or empty.

Accas are assembled from the confidence pool (one leg per fixture); a value acca
only appears if there are enough value tips. Builders combine *positively
correlated* same-game legs, priced jointly by the Monte-Carlo engine, so the
correlation shortens the price - shown as the selling point.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from config.loader import league_table, model_config
from ingest.matches import normalize_team_name
from models.dixon_coles import TeamRatings, analytic_markets
from models.fixture import build_match_inputs
from models.referee import RefereeRates, fit_referee_rates
from models.team_rates import TeamRates, fit_team_rates
from pricing.odds_math import kelly_fraction
from simulation.engine import simulate
from simulation.markets import Leg, price_builder

DATA = Path("data")
RATINGS = DATA / "model" / "ratings.parquet"
MATCHES = DATA / "processed" / "matches.parquet"
FIXTURES = DATA / "processed" / "fixtures.parquet"
ODDS = DATA / "processed" / "odds_latest.parquet"
CALIB = DATA / "processed" / "calibration_metrics.parquet"

EV_BAR = 0.03
AGREE = 0.025  # value tip: model within this of the no-vig consensus
# The model under-spreads team strength, so for the *tips* layer it's anchored to
# the no-vig market: blended = MODEL_WEIGHT*model + (1-MODEL_WEIGHT)*consensus.
# Value detection stays on the raw model-vs-market gap.
MODEL_WEIGHT = 0.35

# label -> key in analytic_markets(score_matrix)
MARKETS = {
    "Home win": "result_home",
    "Draw": "result_draw",
    "Away win": "result_away",
    "Home or draw": "home_dc",
    "Draw or away": "away_dc",
    "Over 1.5": "over_1.5",
    "Over 2.5": "over_2.5",
    "Under 2.5": "under_2.5",
    "Over 3.5": "over_3.5",
    "BTTS": "btts_yes",
    "BTTS No": "btts_no",
}
# Confidence tips are restricted to markets the odds feed actually carries, so
# every tip has a real price. Prefer an outright result call; fall back to a
# totals lean. Double chance is offered on the fixture card as the safer option,
# never as the headline tip.
CONF_RESULT = ["Home win", "Away win"]
CONF_TOTALS = ["Over 2.5", "Under 2.5"]
CONF_RESULT_MIN = 0.58  # blended probability floor for a headline result tip
CONF_TOTALS_MIN = 0.60
CONF_SANITY_GAP = 0.20  # skip if the raw model is this far from the market (something's off)
# The model's draw estimate isn't sharp enough to bet the edge on - excluded.
VALUE_MARKETS = ["Home win", "Away win", "Over 2.5", "Under 2.5"]

_ODDS_TO_LABEL = {
    ("result", "home"): "Home win",
    ("result", "draw"): "Draw",
    ("result", "away"): "Away win",
    ("total_goals", "over 1.5"): "Over 1.5",
    ("total_goals", "over 2.5"): "Over 2.5",
    ("total_goals", "under 2.5"): "Under 2.5",
    ("total_goals", "over 3.5"): "Over 3.5",
}
_DEVIG = {"1x2": ["Home win", "Draw", "Away win"], "ou25": ["Over 2.5", "Under 2.5"]}


# --------------------------------------------------------------------------- #
# form
# --------------------------------------------------------------------------- #
def team_form(matches: pd.DataFrame, team: str, *, before: pd.Timestamp, n: int = 5) -> dict:
    m = matches[
        ((matches.home_team == team) | (matches.away_team == team)) & (matches.date < before)
    ].sort_values("date")
    if m.empty:
        return {"last5": "", "gf": None, "ga": None, "pts": None, "played": 0}
    m = m.tail(n)
    home = m.home_team == team
    gf = np.where(home, m.fthg, m.ftag).astype(float)
    ga = np.where(home, m.ftag, m.fthg).astype(float)
    res = np.where(gf > ga, "W", np.where(gf < ga, "L", "D"))
    pts = int(np.sum(np.where(res == "W", 3, np.where(res == "D", 1, 0))))
    return {
        "last5": "".join(res[::-1]),
        "gf": round(float(gf.mean()), 2),
        "ga": round(float(ga.mean()), 2),
        "pts": pts,
        "played": int(len(m)),
        "unbeaten": bool(np.all(res != "L")),
    }


# --------------------------------------------------------------------------- #
# odds join
# --------------------------------------------------------------------------- #
def _odds_for_fixture(odds: pd.DataFrame, hk: str, ak: str) -> tuple[dict, dict]:
    """(best_price {label: (odds, book)}, consensus {label: novig_prob})."""
    if odds.empty:
        return {}, {}
    g = odds[
        (odds.home_team.map(normalize_team_name) == hk)
        & (odds.away_team.map(normalize_team_name) == ak)
    ]
    if g.empty:
        return {}, {}
    g = g.assign(
        label=[
            _ODDS_TO_LABEL.get((m, str(s).lower()))
            for m, s in zip(g.market, g.selection, strict=True)
        ]
    ).dropna(subset=["label"])
    best = {}
    for lbl, sub in g.groupby("label"):
        i = sub.decimal_odds.idxmax()
        best[lbl] = (float(sub.decimal_odds.max()), str(sub.loc[i, "provider"]))
    med = g.groupby("label").decimal_odds.median()
    consensus: dict[str, float] = {}
    for grp in _DEVIG.values():
        have = [x for x in grp if x in med.index]
        if len(have) == len(grp):
            inv = {x: 1.0 / med[x] for x in have}
            tot = sum(inv.values())
            consensus.update({x: inv[x] / tot for x in have})
    return best, consensus


# --------------------------------------------------------------------------- #
# per-fixture rows
# --------------------------------------------------------------------------- #
def build_fixtures(
    ratings: TeamRatings,
    fixtures: pd.DataFrame,
    matches: pd.DataFrame,
    odds: pd.DataFrame,
    team_rates: TeamRates,
    referee_rates: RefereeRates,
) -> list[dict]:
    lt = league_table()
    now = pd.Timestamp.now(tz="UTC")
    fx = fixtures[pd.to_datetime(fixtures.kickoff, utc=True) > now].sort_values("kickoff")
    min_games = model_config().get("promotion_shrinkage", {}).get("min_confident_games", 10)
    low = {
        normalize_team_name(t)
        for t, r in ratings.table.iterrows()
        if r.get("current_tier_games", 999) < min_games
    }
    out = []
    for f in fx.itertuples(index=False):
        if f.home_team not in ratings.table.index or f.away_team not in ratings.table.index:
            continue
        tier = int(f.tier)
        sm = ratings.score_matrix(f.home_team, f.away_team, home_tier=tier, away_tier=tier)
        am = analytic_markets(sm)
        lh, la = ratings.lambdas(f.home_team, f.away_team, home_tier=tier, away_tier=tier)
        hk, ak = normalize_team_name(f.home_team), normalize_team_name(f.away_team)
        best, consensus = _odds_for_fixture(odds, hk, ak)
        have_rates = team_rates.has(f.home_team) and team_rates.has(f.away_team)
        exp_shots = team_rates.expected(f.home_team, f.away_team, "shots") if have_rates else None
        exp_corners = (
            team_rates.expected(f.home_team, f.away_team, "corners") if have_rates else None
        )
        ref_y, ref_r = referee_rates.for_referee(None)
        ko = pd.to_datetime(f.kickoff, utc=True)
        markets = {}
        for label, key in MARKETS.items():
            mp = am[key]
            price, book = best.get(label, (None, None))
            cons = consensus.get(label)
            blended = MODEL_WEIGHT * mp + (1 - MODEL_WEIGHT) * cons if cons is not None else mp
            markets[label] = {
                "model": round(mp, 4),
                "consensus": round(cons, 4) if cons is not None else None,
                "blended": round(blended, 4),
                "price": price,
                "book": book,
                "fair": round(1.0 / max(blended, 1e-9), 2),
                "ev": round(mp * price - 1.0, 4) if price else None,
            }
        out.append(
            {
                "id": f"{hk}__{ak}",
                "home": f.home_team,
                "away": f.away_team,
                "league": f.league,
                "league_name": lt[f.league]["name"],
                "tier": tier,
                "kickoff": ko.isoformat(),
                "kickoff_short": ko.strftime("%a %d %b, %H:%M"),
                "lambda": [round(lh, 2), round(la, 2)],
                "markets": markets,
                "form": {
                    "home": team_form(matches, f.home_team, before=ko.tz_localize(None)),
                    "away": team_form(matches, f.away_team, before=ko.tz_localize(None)),
                },
                "expected": {
                    "shots": exp_shots,
                    "corners": exp_corners,
                    "ref_cards_pm": round(ref_y * 2 + ref_r, 1),
                    "has_team_stats": have_rates,
                },
                "low_data": hk in low or ak in low,
                "has_odds": bool(best),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# tips
# --------------------------------------------------------------------------- #
def _reasons(fx: dict, label: str, kind: str) -> list[str]:
    r: list[str] = []
    fh, fa = fx["form"]["home"], fx["form"]["away"]
    m = fx["markets"][label]
    home_side = label in ("Home win", "Home or draw")
    away_side = label in ("Away win", "Draw or away")
    if home_side and fh["last5"]:
        if fh["unbeaten"] and fh["played"] >= 4:
            r.append(f"{fx['home']} unbeaten in {fh['played']} ({'-'.join(fh['last5'])})")
        elif fh["pts"] and fh["pts"] >= 9:
            r.append(f"{fx['home']} {fh['pts']}/15 pts, form {'-'.join(fh['last5'])}")
    if away_side and fa["last5"]:
        if fa["unbeaten"] and fa["played"] >= 4:
            r.append(f"{fx['away']} unbeaten in {fa['played']} ({'-'.join(fa['last5'])})")
        elif fa["pts"] and fa["pts"] >= 9:
            r.append(f"{fx['away']} {fa['pts']}/15 pts on the road-ish")
    gpg = ((fh["gf"] or 0) + (fh["ga"] or 0) + (fa["gf"] or 0) + (fa["ga"] or 0)) / 2
    if (label.startswith("Over") or label == "BTTS") and gpg >= 2.9:
        r.append(f"both sides average ~{gpg:.1f} goals/game recently")
    if (label.startswith("Under") or label == "BTTS No") and gpg and gpg <= 2.4:
        r.append(f"tight recent form (~{gpg:.1f} goals/game)")
    if kind == "value" and m["price"] and m["consensus"]:
        r.append(f"{m['price']:.2f} available vs {1 / m['consensus']:.2f} fair")
    if kind == "confidence":
        r.append(
            f"model {m['model']:.0%}, market {m['consensus']:.0%}"
            if m["consensus"]
            else f"model {m['model']:.0%}"
        )
    return r[:3]


def _tip_row(fx: dict, label: str, kind: str) -> dict:
    m = fx["markets"][label]
    return {
        "fixture_id": fx["id"],
        "fixture": f"{fx['home']} v {fx['away']}",
        "home": fx["home"],
        "away": fx["away"],
        "league": fx["league"],
        "kickoff": fx["kickoff"],
        "kickoff_short": fx["kickoff_short"],
        "market": label,
        "prob": m["blended"],  # headline: model anchored to market
        "model_prob": m["model"],
        "consensus": m["consensus"],
        "best_odds": m["price"],
        "book": m["book"],
        "fair_odds": m["fair"],
        "ev": m["ev"],
        "edge": round(m["model"] - m["consensus"], 4) if m["consensus"] is not None else None,
        "reasons": _reasons(fx, label, kind),
    }


def _conf_pick(fx: dict, labels: list[str], floor: float) -> str | None:
    """Best label whose blended (model-anchored-to-market) probability clears
    `floor`, has a price, and where the raw model isn't wildly off the market."""
    best, best_p = None, 0.0
    for label in labels:
        m = fx["markets"][label]
        if m["price"] is None or m["consensus"] is None:
            continue
        if abs(m["model"] - m["consensus"]) > CONF_SANITY_GAP:
            continue
        if m["blended"] >= floor and m["blended"] > best_p:
            best, best_p = label, m["blended"]
    return best


def rank_confidence(fixtures: list[dict]) -> list[dict]:
    """At most one tip per fixture: a favourite (blended prob) or a totals lean,
    ranked by blended probability. Prefer a result call."""
    tips = []
    for fx in fixtures:
        if fx["low_data"] or not fx["has_odds"]:
            continue
        label = _conf_pick(fx, CONF_RESULT, CONF_RESULT_MIN) or _conf_pick(
            fx, CONF_TOTALS, CONF_TOTALS_MIN
        )
        if label is None:
            continue
        row = _tip_row(fx, label, "confidence")
        row["blended"] = fx["markets"][label]["blended"]
        if label in CONF_RESULT:
            dc = "Home or draw" if label == "Home win" else "Draw or away"
            row["safer"] = {"market": dc, "prob": round(fx["markets"][dc]["blended"], 3)}
        tips.append(row)
    return sorted(tips, key=lambda t: -t["blended"])


def rank_value(fixtures: list[dict]) -> list[dict]:
    """Selections with a real edge over the market consensus."""
    tips = []
    for fx in fixtures:
        if fx["low_data"] or not fx["has_odds"]:
            continue
        for label in VALUE_MARKETS:
            m = fx["markets"][label]
            if m["price"] is None or m["consensus"] is None or m["ev"] is None:
                continue
            div = m["model"] - m["consensus"]
            if m["ev"] > EV_BAR and abs(div) <= AGREE and div >= -0.01:
                tips.append(_tip_row(fx, label, "value"))
    return sorted(tips, key=lambda t: -t["ev"])


# --------------------------------------------------------------------------- #
# accas
# --------------------------------------------------------------------------- #
def _acca(legs: list[dict], name: str, blurb: str) -> dict | None:
    legs = _one_per_fixture(legs)
    if len(legs) < 2:
        return None
    prob = float(np.prod([leg["prob"] for leg in legs]))
    best = (
        float(np.prod([leg["best_odds"] for leg in legs if leg["best_odds"]]))
        if all(leg["best_odds"] for leg in legs)
        else None
    )
    fair = 1.0 / max(prob, 1e-12)
    return {
        "name": name,
        "blurb": blurb,
        "legs": legs,
        "n_legs": len(legs),
        "model_prob": round(prob, 4),
        "fair_odds": round(fair, 2),
        "best_odds": round(best, 2) if best else None,
        "margin": round(fair / best - 1.0, 3)
        if best
        else None,  # how much the price is shaded vs fair
        "ev": round(prob * best - 1.0, 3) if best else None,
        "stake_pct": round(kelly_fraction(prob, best, 0.25) * 100, 2) if best else None,
    }


def _one_per_fixture(pool: list[dict]) -> list[dict]:
    seen, keep = set(), []
    for t in pool:
        if t["fixture_id"] in seen:
            continue
        seen.add(t["fixture_id"])
        keep.append(t)
    return keep


def build_accas(confidence: list[dict], value: list[dict]) -> list[dict]:
    ranked = sorted(confidence, key=lambda t: -t["prob"])
    accas = [
        _acca(ranked[:3], "Banker", "The three most likely tips, one per match."),
        _acca(
            [t for t in ranked if 0.50 <= t["prob"] < 0.70][:4],
            "Punt",
            "Four less-certain picks for a bigger return — small stakes.",
        ),
    ]
    if len(_one_per_fixture(value)) >= 3:
        accas.append(
            _acca(value[:4], "Value", "Legs where the model beats the market price. Rare.")
        )
    return [a for a in accas if a]


# --------------------------------------------------------------------------- #
# same-game builders (Monte-Carlo, correlation-aware)
# --------------------------------------------------------------------------- #
def _builder_legs(favourite: str) -> list[tuple[str, Leg]]:
    """Legs that move together when `favourite` (home/away) is the stronger side."""
    if favourite == "home":
        return [
            ("Home win", Leg("result", {"outcome": "home"})),
            ("Over 1.5", Leg("total_goals", {"line": 1.5, "side": "over"})),
            ("Home 2+ goals", Leg("team_goals", {"team": "home", "line": 1.5, "side": "over"})),
        ]
    return [
        ("Away win", Leg("result", {"outcome": "away"})),
        ("Over 1.5", Leg("total_goals", {"line": 1.5, "side": "over"})),
        ("Away 2+ goals", Leg("team_goals", {"team": "away", "line": 1.5, "side": "over"})),
    ]


def build_builders(
    ratings: TeamRatings,
    fixtures: list[dict],
    team_rates: TeamRates,
    referee_rates: RefereeRates,
    *,
    n_fixtures: int = 5,
) -> list[dict]:
    mcfg = model_config()
    ranked = sorted(
        (f for f in fixtures if not f["low_data"]),
        key=lambda f: abs(f["markets"]["Home win"]["model"] - f["markets"]["Away win"]["model"]),
        reverse=True,
    )[:n_fixtures]
    out = []
    for f in ranked:
        fav = (
            "home"
            if f["markets"]["Home win"]["model"] >= f["markets"]["Away win"]["model"]
            else "away"
        )
        legs = _builder_legs(fav)
        mi = build_match_inputs(
            ratings,
            f["home"],
            f["away"],
            league=f["league"],
            tier=f["tier"],
            model_cfg=mcfg,
            team_rates=team_rates,
            referee_rates=referee_rates,
        )
        res = simulate(mi, n_sims=40_000, seed=20260906)
        bp = price_builder(res, [leg for _, leg in legs])
        out.append(
            {
                "fixture_id": f["id"],
                "fixture": f"{f['home']} v {f['away']}",
                "league": f["league"],
                "kickoff_short": f["kickoff_short"],
                "legs": [
                    {"label": lbl, "prob": round(p, 3)}
                    for (lbl, _), p in zip(legs, bp.leg_probs, strict=True)
                ],
                "joint_prob": round(bp.joint_prob, 4),
                "fair_odds": round(bp.fair_odds, 2),
                "naive_odds": round(bp.naive_fair_odds, 2),
                "correlation": round(bp.correlation_factor, 2),
                "shorter_by": round((bp.naive_fair_odds - bp.fair_odds), 2),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# assemble
# --------------------------------------------------------------------------- #
def assemble() -> dict:
    ratings = TeamRatings.from_parquet(RATINGS)
    matches = pd.read_parquet(MATCHES)
    matches["date"] = pd.to_datetime(matches["date"])
    fixtures_df = pd.read_parquet(FIXTURES) if FIXTURES.exists() else pd.DataFrame()
    odds = (
        pd.read_parquet(ODDS)
        if ODDS.exists()
        else pd.DataFrame(
            columns=["home_team", "away_team", "market", "selection", "decimal_odds", "provider"]
        )
    )

    team_rates = fit_team_rates(matches)
    referee_rates = fit_referee_rates(matches)

    fixtures = (
        build_fixtures(ratings, fixtures_df, matches, odds, team_rates, referee_rates)
        if len(fixtures_df)
        else []
    )
    confidence = rank_confidence(fixtures)
    value = rank_value(fixtures)
    accas = build_accas(confidence, value)
    builders = build_builders(ratings, fixtures, team_rates, referee_rates) if fixtures else []

    priced = [
        m
        for f in fixtures
        for m in f["markets"].values()
        if m["price"] and m["consensus"] is not None
    ]
    mad = float(np.mean([abs(m["model"] - m["consensus"]) for m in priced])) if priced else None

    calib = None
    if CALIB.exists():
        c = pd.read_parquet(CALIB).reset_index()
        calib = {
            row["market"]: {
                "n": int(row["n"]),
                "model": round(row["mean_pred"], 3),
                "actual": round(row["base_rate"], 3),
                "ece": round(row["ece"], 3),
            }
            for _, row in c.iterrows()
        }

    return {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "model": {
            "fitted_through": ratings.meta.get("date_max"),
            "n_matches": ratings.meta.get("n_matches"),
            "n_teams": ratings.meta.get("n_teams"),
            "home_edge": round(ratings.gamma, 3),
            "vs_market_mad": round(mad, 4) if mad is not None else None,
            "calibration": calib,
        },
        "counts": {
            "fixtures": len(fixtures),
            "confidence": len(confidence),
            "value": len(value),
            "accas": len(accas),
            "builders": len(builders),
        },
        "confidence": confidence,
        "value": value,
        "accas": accas,
        "builders": builders,
        "fixtures": fixtures,
    }
