"""Bet-builder pricing: joint model price -> value assessment against a typed-in
bookmaker price.

Wraps :func:`simulation.markets.price_builder` (the joint count over shared sims)
and adds the EV / edge / verdict / stake layer plus a per-leg breakdown so the
correlation effect is visible. The bookmaker price for the whole builder is
entered by the user (Sky Bet has no free API); nothing here calls a network.
"""

from __future__ import annotations

from dataclasses import dataclass

from pricing.odds_math import ValueAssessment, assess_value, implied_prob
from simulation.markets import BuilderPrice, Leg, price_builder
from simulation.state import SimResult


@dataclass
class LegBreakdown:
    leg: Leg
    label: str
    standalone_prob: float
    standalone_fair_odds: float


@dataclass
class BuilderQuote:
    price: BuilderPrice
    legs: list[LegBreakdown]
    naive_fair_odds: float  # from multiplying legs - what a naive builder assumes
    joint_fair_odds: float  # the correct, correlation-aware price
    correlation_factor: float
    warnings: list[str]
    # populated once the user enters the bookmaker's builder price:
    book_odds: float | None = None
    value: ValueAssessment | None = None

    @property
    def correlation_note(self) -> str:
        cf = self.correlation_factor
        if cf >= 1.10:
            return (
                f"Legs are positively correlated (joint {cf:.2f}x the naive product): "
                f"the fair price is SHORTER than multiplying legs — {self.joint_fair_odds:.2f} "
                f"vs {self.naive_fair_odds:.2f}."
            )
        if cf <= 0.90:
            return (
                f"Legs are negatively correlated (joint {cf:.2f}x the naive product): "
                f"the fair price is LONGER than multiplying legs — {self.joint_fair_odds:.2f} "
                f"vs {self.naive_fair_odds:.2f}."
            )
        return "Legs are close to independent; joint price ~ the product of legs."

    def with_book_price(self, book_odds: float, *, kelly: float = 0.25) -> BuilderQuote:
        self.book_odds = book_odds
        self.value = assess_value(self.price.joint_prob, book_odds, kelly=kelly)
        return self

    def summary_lines(self) -> list[str]:
        out = [
            f"Model joint probability: {self.price.joint_prob:.1%}"
            f"  (±{1.96 * self.price.mc_std_error:.1%} MC)",
            f"Model fair odds:         {self.joint_fair_odds:.2f}",
            f"Naive (multiply legs):   {self.naive_fair_odds:.2f}   [{self.correlation_factor:.2f}x]",
        ]
        if self.value is not None:
            v = self.value
            out += [
                f"Bookmaker price:         {self.book_odds:.2f}  (implied {implied_prob(self.book_odds):.1%})",
                f"EV:                      {v.ev_pct:+.1f}%   edge {v.edge:+.1%}",
                f"Verdict:                 {v.verdict.upper()}",
                f"Quarter-Kelly stake:     {v.kelly_stake:.1%} of bankroll",
            ]
        return out


def quote_builder(
    res: SimResult,
    legs: list[Leg],
    *,
    book_odds: float | None = None,
    kelly: float = 0.25,
) -> BuilderQuote:
    bp: BuilderPrice = price_builder(res, legs)
    breakdown = [
        LegBreakdown(
            leg=leg,
            label=leg.describe(),
            standalone_prob=p,
            standalone_fair_odds=(1.0 / p) if p > 0 else float("inf"),
        )
        for leg, p in zip(legs, bp.leg_probs, strict=True)
    ]
    q = BuilderQuote(
        price=bp,
        legs=breakdown,
        naive_fair_odds=bp.naive_fair_odds,
        joint_fair_odds=bp.fair_odds,
        correlation_factor=bp.correlation_factor,
        warnings=list(bp.warnings),
    )
    if book_odds is not None:
        q.with_book_price(book_odds, kelly=kelly)
    return q
