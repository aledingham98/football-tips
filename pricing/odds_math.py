"""Decimal-odds / probability helpers, margin removal, Kelly, CLV.

No simulation logic here - just the arithmetic that turns model probabilities and
bookmaker prices into EV, edge, stakes and closing-line value.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def implied_prob(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        raise ValueError(f"decimal odds must be > 1.0, got {decimal_odds}")
    return 1.0 / decimal_odds


def decimal_odds(prob: float) -> float:
    if not 0.0 < prob <= 1.0:
        raise ValueError(f"probability must be in (0, 1], got {prob}")
    return 1.0 / prob


def overround(decimal_odds_list: Sequence[float]) -> float:
    """Book sum minus 1. e.g. 0.05 == a 5% margin over a complete market."""
    return sum(1.0 / o for o in decimal_odds_list) - 1.0


def remove_overround_proportional(decimal_odds_list: Sequence[float]) -> list[float]:
    """Fair probabilities by proportional (multiplicative) margin removal.

    The workhorse for book prices from The Odds API. The Betfair no-vig midpoint
    (:func:`no_vig_two_way`) is the primary fair-price benchmark; this is used
    where only book prices are available.
    """
    raw = [1.0 / o for o in decimal_odds_list]
    s = sum(raw)
    return [r / s for r in raw]


def remove_overround_equal_margin(decimal_odds_list: Sequence[float]) -> list[float]:
    """Fair probabilities by subtracting an equal margin per outcome (additive).

    Compared with proportional removal this shifts relative value toward
    longshots; quoting both is a cheap sensitivity check on a value edge.
    """
    raw = [1.0 / o for o in decimal_odds_list]
    m = (sum(raw) - 1.0) / len(raw)
    adj = [max(r - m, 1e-9) for r in raw]
    s = sum(adj)
    return [a / s for a in adj]


def no_vig_two_way(back: float, lay: float) -> float:
    """Fair probability of the outcome from an exchange back/lay pair.

    Midpoint of the back-implied and lay-implied probabilities. Returns the
    single-outcome probability; the complement is ``1 - result``.
    """
    p_back = 1.0 / back
    p_lay = 1.0 / lay
    return 0.5 * (p_back + p_lay)


@dataclass
class ValueAssessment:
    prob: float
    book_odds: float
    fair_odds: float
    ev: float  # expected profit per 1 staked, e.g. 0.04 == +4%
    edge: float  # prob - implied_prob(book_odds)
    kelly_stake: float  # fraction of bankroll at the given Kelly fraction
    verdict: str

    @property
    def ev_pct(self) -> float:
        return 100.0 * self.ev


def kelly_fraction(prob: float, book_odds: float, fraction: float = 0.25) -> float:
    """Fractional-Kelly stake as a share of bankroll. Never returns < 0."""
    b = book_odds - 1.0
    q = 1.0 - prob
    f_star = (b * prob - q) / b
    return max(0.0, f_star * fraction)


def assess_value(
    prob: float, book_odds: float, kelly: float = 0.25, ev_threshold: float = 0.03
) -> ValueAssessment:
    fair = decimal_odds(prob) if prob > 0 else math.inf
    ev = prob * book_odds - 1.0
    edge = prob - implied_prob(book_odds)
    stake = kelly_fraction(prob, book_odds, kelly)
    if ev >= ev_threshold:
        verdict = "VALUE"
    elif ev >= 0:
        verdict = "marginal"
    elif ev >= -0.05:
        verdict = "no value"
    else:
        verdict = "bad price"
    return ValueAssessment(prob, book_odds, fair, ev, edge, stake, verdict)


def acca_total_margin(leg_odds_markets: Sequence[Sequence[float]]) -> float:
    """Compounded book margin across acca legs.

    ``leg_odds_markets`` is one complete price set per leg (all outcomes of that
    market). Five legs at ~5% each compound to ~23%.
    """
    kept = 1.0
    for market in leg_odds_markets:
        kept *= 1.0 / (1.0 + overround(market))
    return 1.0 / kept - 1.0


def clv_pct(price_taken: float, closing_price: float) -> float:
    """Closing-line value. Positive == you beat the close."""
    return price_taken / closing_price - 1.0
