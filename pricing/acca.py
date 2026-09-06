"""Accumulator pricing - legs in SEPARATE matches, so probabilities multiply.

The compounding bookmaker margin is surfaced explicitly (five legs at ~5% each
is ~23% total), and a Monte-Carlo bankroll path shows the drawdown a given
fractional-Kelly stake exposes you to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pricing.odds_math import implied_prob, kelly_fraction


@dataclass
class AccaLeg:
    fixture: str
    label: str
    model_prob: float
    book_odds: float | None = None  # optional per-leg price, for a per-leg margin view


@dataclass
class AccaQuote:
    legs: list[AccaLeg]
    compound_prob: float
    fair_odds: float
    book_odds: float | None = None
    ev: float | None = None
    edge: float | None = None
    kelly_stake: float | None = None
    total_margin_vs_price: float | None = None  # book overround implied by your acca price
    total_margin_from_legs: float | None = None  # compounded per-leg margins, if legs priced
    verdict: str | None = None

    @property
    def n_legs(self) -> int:
        return len(self.legs)


def price_acca(
    legs: list[AccaLeg],
    *,
    book_odds: float | None = None,
    kelly: float = 0.25,
    ev_threshold: float = 0.0,
) -> AccaQuote:
    if not legs:
        raise ValueError("an acca needs at least one leg")
    probs = np.array([leg.model_prob for leg in legs], dtype=float)
    compound = float(np.prod(probs))
    q = AccaQuote(
        legs=legs,
        compound_prob=compound,
        fair_odds=(1.0 / compound) if compound > 0 else float("inf"),
    )

    priced = [leg.book_odds for leg in legs if leg.book_odds]
    if len(priced) == len(legs) and compound > 0:
        # compounded margin = (product of per-leg book-implied probs) / model - 1
        comp_book_implied = float(np.prod([implied_prob(o) for o in priced]))
        q.total_margin_from_legs = comp_book_implied / compound - 1.0

    if book_odds:
        q.book_odds = book_odds
        q.ev = compound * book_odds - 1.0
        q.edge = compound - implied_prob(book_odds)
        q.total_margin_vs_price = implied_prob(book_odds) / compound - 1.0 if compound > 0 else None
        q.kelly_stake = kelly_fraction(compound, book_odds, kelly)
        if q.ev >= max(ev_threshold, 0.03):
            q.verdict = "VALUE"
        elif q.ev >= 0:
            q.verdict = "marginal"
        elif q.ev >= -0.10:
            q.verdict = "no value"
        else:
            q.verdict = "bad price"
    return q


@dataclass
class DrawdownResult:
    n_bets: int
    n_paths: int
    stake_fraction: float
    edge_per_bet: float
    median_final: float  # multiple of starting bankroll
    p05_final: float
    p95_final: float
    prob_bust_50: float  # P(bankroll ever below 50% of start)
    prob_down_at_end: float
    median_max_drawdown: float  # fraction
    worst_max_drawdown: float
    sample_paths: np.ndarray = field(repr=False, default=None)


def drawdown_simulation(
    win_prob: float,
    book_odds: float,
    stake_fraction: float,
    *,
    n_bets: int = 200,
    n_paths: int = 5000,
    seed: int | None = 0,
) -> DrawdownResult:
    """Compound a fractional-Kelly stake over ``n_bets`` independent bets.

    Each bet stakes ``stake_fraction`` of the *current* bankroll; a win multiplies
    it by ``1 + f*(odds-1)``, a loss by ``1 - f``.
    """
    rng = np.random.default_rng(seed)
    b = book_odds - 1.0
    win_mult = 1.0 + stake_fraction * b
    lose_mult = 1.0 - stake_fraction
    wins = rng.random((n_paths, n_bets)) < win_prob
    step = np.where(wins, win_mult, lose_mult)
    paths = np.cumprod(step, axis=1)
    paths = np.concatenate([np.ones((n_paths, 1)), paths], axis=1)  # start at 1.0

    running_peak = np.maximum.accumulate(paths, axis=1)
    drawdown = 1.0 - paths / running_peak
    max_dd = drawdown.max(axis=1)
    final = paths[:, -1]
    ever_below_50 = (paths < 0.5).any(axis=1)

    idx = rng.choice(n_paths, size=min(40, n_paths), replace=False)
    return DrawdownResult(
        n_bets=n_bets,
        n_paths=n_paths,
        stake_fraction=stake_fraction,
        edge_per_bet=win_prob * book_odds - 1.0,
        median_final=float(np.median(final)),
        p05_final=float(np.percentile(final, 5)),
        p95_final=float(np.percentile(final, 95)),
        prob_bust_50=float(ever_below_50.mean()),
        prob_down_at_end=float((final < 1.0).mean()),
        median_max_drawdown=float(np.median(max_dd)),
        worst_max_drawdown=float(max_dd.max()),
        sample_paths=paths[idx],
    )
