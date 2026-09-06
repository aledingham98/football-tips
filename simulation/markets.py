"""Count any market - single or joint - over a :class:`~simulation.state.SimResult`.

Every market is a boolean reduction over the simulation rows. A bet-builder price
is the mean of the elementwise AND of its legs' boolean arrays, so same-match leg
correlation is captured automatically. Leg probabilities are never multiplied for
a same-match builder; :func:`price_builder` reports the naive product only so the
correlation effect is visible and can be warned on.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from simulation.state import SimResult

BoolArray = np.ndarray
Evaluator = Callable[[SimResult, Mapping[str, Any]], BoolArray]

_MARKETS: dict[str, Evaluator] = {}


def market(key: str) -> Callable[[Evaluator], Evaluator]:
    def deco(fn: Evaluator) -> Evaluator:
        _MARKETS[key] = fn
        return fn

    return deco


@dataclass
class Leg:
    """One selection in a builder or acca."""

    market: str
    params: dict[str, Any] = field(default_factory=dict)
    label: str | None = None

    def describe(self) -> str:
        if self.label:
            return self.label
        p = self.params
        if self.market == "result":
            return {"home": "Home win", "draw": "Draw", "away": "Away win"}[p["outcome"]]
        if self.market == "btts":
            return "BTTS" if p.get("yes", True) else "BTTS - No"
        if self.market in ("total_goals", "total_corners", "total_cards", "total_goals_first_half"):
            noun = self.market.replace("total_", "").replace("_", " ")
            return f"{p.get('side', 'over').title()} {p['line']} {noun}"
        if self.market in ("team_goals", "team_corners", "team_cards"):
            noun = self.market.replace("team_", "")
            return f"{p['team'].title()} {p.get('side', 'over')} {p['line']} {noun}"
        if self.market.startswith("player_"):
            what = self.market.replace("player_", "").replace("_", " ")
            n = p.get("n", 1)
            return (
                f"{p['player']} {n}+ {what}"
                if "n" in p or self.market != "player_goal_or_assist"
                else f"{p['player']} {what}"
            )
        return f"{self.market} {p}"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _totals(res: SimResult) -> BoolArray:
    return res.home_goals + res.away_goals


def _line(values: BoolArray, line: float, side: str) -> BoolArray:
    return values > line if side == "over" else values < line


def _team_arrays(res: SimResult, team: str):
    return res.team(team)


# --------------------------------------------------------------------------- #
# match markets
# --------------------------------------------------------------------------- #
@market("result")
def _result(res: SimResult, p: Mapping[str, Any]) -> BoolArray:
    hg, ag = res.home_goals, res.away_goals
    return {"home": hg > ag, "draw": hg == ag, "away": hg < ag}[p["outcome"]]


@market("double_chance")
def _double_chance(res: SimResult, p: Mapping[str, Any]) -> BoolArray:
    hg, ag = res.home_goals, res.away_goals
    return {
        "home_draw": hg >= ag,
        "home_away": hg != ag,
        "draw_away": hg <= ag,
    }[p["outcome"]]


@market("btts")
def _btts(res: SimResult, p: Mapping[str, Any]) -> BoolArray:
    both = (res.home_goals >= 1) & (res.away_goals >= 1)
    return both if p.get("yes", True) else ~both


@market("result_and_btts")
def _result_and_btts(res: SimResult, p: Mapping[str, Any]) -> BoolArray:
    hg, ag = res.home_goals, res.away_goals
    btts = (hg >= 1) & (ag >= 1)
    table = {
        "home_yes": (hg > ag) & btts,
        "home_no": (hg > ag) & ~btts,
        "draw_yes": (hg == ag) & btts,
        "away_yes": (hg < ag) & btts,
    }
    return table[p["outcome"]]


@market("correct_score")
def _correct_score(res: SimResult, p: Mapping[str, Any]) -> BoolArray:
    return (res.home_goals == p["home"]) & (res.away_goals == p["away"])


@market("total_goals")
def _total_goals(res: SimResult, p: Mapping[str, Any]) -> BoolArray:
    return _line(_totals(res), p["line"], p.get("side", "over"))


@market("team_goals")
def _team_goals(res: SimResult, p: Mapping[str, Any]) -> BoolArray:
    v = res.home_goals if p["team"] == "home" else res.away_goals
    return _line(v, p["line"], p.get("side", "over"))


@market("clean_sheet")
def _clean_sheet(res: SimResult, p: Mapping[str, Any]) -> BoolArray:
    return (res.away_goals == 0) if p["team"] == "home" else (res.home_goals == 0)


@market("win_to_nil")
def _win_to_nil(res: SimResult, p: Mapping[str, Any]) -> BoolArray:
    hg, ag = res.home_goals, res.away_goals
    return (hg > ag) & (ag == 0) if p["team"] == "home" else (ag > hg) & (hg == 0)


@market("total_goals_first_half")
def _fh_goals(res: SimResult, p: Mapping[str, Any]) -> BoolArray:
    return _line(res.home_ht_goals + res.away_ht_goals, p["line"], p.get("side", "over"))


@market("half_time_result")
def _ht_result(res: SimResult, p: Mapping[str, Any]) -> BoolArray:
    hg, ag = res.home_ht_goals, res.away_ht_goals
    return {"home": hg > ag, "draw": hg == ag, "away": hg < ag}[p["outcome"]]


@market("goal_both_halves")
def _goal_both_halves(res: SimResult, p: Mapping[str, Any]) -> BoolArray:
    fh = res.home_ht_goals + res.away_ht_goals
    sh = (res.home_goals + res.away_goals) - fh
    return (fh >= 1) & (sh >= 1)


# --------------------------------------------------------------------------- #
# player markets (shot mechanism)
# --------------------------------------------------------------------------- #
def _player_col(res: SimResult, p: Mapping[str, Any]):
    team = _team_arrays(res, p["team"])
    return team, team.col(p["player"])


@market("player_shots")
def _player_shots(res: SimResult, p: Mapping[str, Any]) -> BoolArray:
    team, c = _player_col(res, p)
    return team.shots[:, c] >= p.get("n", 1)


@market("player_sot")
def _player_sot(res: SimResult, p: Mapping[str, Any]) -> BoolArray:
    team, c = _player_col(res, p)
    return team.sot[:, c] >= p.get("n", 1)


@market("player_goals")
def _player_goals(res: SimResult, p: Mapping[str, Any]) -> BoolArray:
    team, c = _player_col(res, p)
    return team.goals[:, c] >= p.get("n", 1)


@market("player_assists")
def _player_assists(res: SimResult, p: Mapping[str, Any]) -> BoolArray:
    team, c = _player_col(res, p)
    return team.assists[:, c] >= p.get("n", 1)


@market("player_goal_or_assist")
def _player_ga(res: SimResult, p: Mapping[str, Any]) -> BoolArray:
    team, c = _player_col(res, p)
    return (team.goals[:, c] + team.assists[:, c]) >= p.get("n", 1)


@market("player_carded")
def _player_carded(res: SimResult, p: Mapping[str, Any]) -> BoolArray:
    team, c = _player_col(res, p)
    return (team.yellows[:, c] + team.reds[:, c]) >= 1


# --------------------------------------------------------------------------- #
# cards / corners
# --------------------------------------------------------------------------- #
def _match_cards(res: SimResult) -> BoolArray:
    return (
        res.home.yellows.sum(1)
        + res.home.reds.sum(1)
        + res.away.yellows.sum(1)
        + res.away.reds.sum(1)
    )


@market("total_cards")
def _total_cards(res: SimResult, p: Mapping[str, Any]) -> BoolArray:
    return _line(_match_cards(res), p["line"], p.get("side", "over"))


@market("team_cards")
def _team_cards(res: SimResult, p: Mapping[str, Any]) -> BoolArray:
    t = _team_arrays(res, p["team"])
    return _line(t.yellows.sum(1) + t.reds.sum(1), p["line"], p.get("side", "over"))


@market("red_card")
def _red_card(res: SimResult, p: Mapping[str, Any]) -> BoolArray:
    total_reds = res.home.reds.sum(1) + res.away.reds.sum(1)
    return (total_reds >= 1) if p.get("present", True) else (total_reds == 0)


@market("total_corners")
def _total_corners(res: SimResult, p: Mapping[str, Any]) -> BoolArray:
    return _line(res.home_corners + res.away_corners, p["line"], p.get("side", "over"))


@market("team_corners")
def _team_corners(res: SimResult, p: Mapping[str, Any]) -> BoolArray:
    v = res.home_corners if p["team"] == "home" else res.away_corners
    return _line(v, p["line"], p.get("side", "over"))


# --------------------------------------------------------------------------- #
# evaluation + builder pricing
# --------------------------------------------------------------------------- #
def evaluate(res: SimResult, leg: Leg) -> BoolArray:
    """Boolean array, one entry per simulation, for a single leg."""
    try:
        fn = _MARKETS[leg.market]
    except KeyError as exc:
        raise KeyError(f"unknown market {leg.market!r}; known: {sorted(_MARKETS)}") from exc
    out = np.asarray(fn(res, leg.params), dtype=bool)
    if out.shape != (res.n_sims,):
        raise ValueError(f"evaluator for {leg.market!r} returned shape {out.shape}")
    return out


def probability(res: SimResult, leg: Leg) -> float:
    return float(evaluate(res, leg).mean())


@dataclass
class BuilderPrice:
    legs: list[Leg]
    leg_probs: list[float]
    joint_prob: float
    fair_odds: float
    naive_prob: float  # product of leg probs - WRONG for a builder, shown for contrast
    naive_fair_odds: float
    correlation_factor: float  # joint / naive; >1 positive dependence, <1 negative
    mc_std_error: float  # Monte-Carlo s.e. on joint_prob
    n_sims: int
    warnings: list[str]

    @property
    def fair_odds_ci(self) -> tuple[float, float]:
        lo = max(self.joint_prob - 1.96 * self.mc_std_error, 1e-9)
        hi = min(self.joint_prob + 1.96 * self.mc_std_error, 1.0)
        return 1.0 / hi, 1.0 / lo


def price_builder(
    res: SimResult,
    legs: list[Leg],
    *,
    neg_corr_ratio: float = 0.85,
    pos_corr_ratio: float = 1.15,
    longshot_prob: float = 0.005,
) -> BuilderPrice:
    """Joint price for a same-match multi by counting over shared simulations."""
    if not legs:
        raise ValueError("a builder needs at least one leg")
    masks = [evaluate(res, leg) for leg in legs]
    leg_probs = [float(m.mean()) for m in masks]
    joint_mask = np.logical_and.reduce(masks)
    joint = float(joint_mask.mean())
    naive = float(np.prod(leg_probs))

    corr = joint / naive if naive > 0 else math.inf
    se = math.sqrt(max(joint * (1.0 - joint), 0.0) / res.n_sims)

    warnings: list[str] = []
    if len(legs) > 1 and naive > 0:
        if corr <= neg_corr_ratio:
            warnings.append(
                f"Negatively correlated legs: joint probability is {corr:.2f}x the naive "
                f"product ({joint:.1%} vs {naive:.1%}). The true price is LONGER than "
                f"multiplying legs suggests."
            )
        elif corr >= pos_corr_ratio:
            warnings.append(
                f"Positively correlated legs: joint probability is {corr:.2f}x the naive "
                f"product ({joint:.1%} vs {naive:.1%})."
            )
    if 0.0 < joint < longshot_prob:
        warnings.append(
            f"Extreme longshot: model probability {joint:.2%}. Monte-Carlo noise alone is "
            f"+/-{se:.2%}; treat the price as indicative only."
        )
    if joint == 0.0:
        warnings.append(
            f"Zero hits in {res.n_sims:,} simulations - probability is below "
            f"~{3.0 / res.n_sims:.2%}. Increase n_sims for a real estimate."
        )

    return BuilderPrice(
        legs=legs,
        leg_probs=leg_probs,
        joint_prob=joint,
        fair_odds=(1.0 / joint) if joint > 0 else math.inf,
        naive_prob=naive,
        naive_fair_odds=(1.0 / naive) if naive > 0 else math.inf,
        correlation_factor=corr,
        mc_std_error=se,
        n_sims=res.n_sims,
        warnings=warnings,
    )


def known_markets() -> list[str]:
    return sorted(_MARKETS)
