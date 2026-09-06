"""``OddsProvider`` abstract base class and the normalised quote record.

Three concrete providers sit behind this: :class:`~ingest.odds.the_odds_api.TheOddsApiProvider`
(free Starter tier, credit-budgeted), ``BetfairDelayedProvider`` (free delayed
app key, no-vig midpoints) and ``ManualOddsProvider`` (Sky Bet prices typed in by
hand, the only source for bet builders and player props).

The Streamlit app never instantiates a network-backed provider; it reads the
parquet snapshots these write during the scheduled jobs.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PriceQuote:
    """One decimal price for one selection of one market on one fixture."""

    provider: str
    league: str
    fixture_id: str
    home_team: str
    away_team: str
    kickoff_iso: str
    market: str  # normalised key, e.g. "result", "total_goals", "btts"
    selection: str  # e.g. "home", "over 2.5", "yes", "Bukayo Saka 2+ sot"
    decimal_odds: float
    fetched_iso: str = field(default_factory=lambda: dt.datetime.now(dt.UTC).isoformat())
    # exchange extras (Betfair): back/lay and whether liquidity is known
    back: float | None = None
    lay: float | None = None
    liquidity_known: bool = True

    def implied_prob(self) -> float:
        return 1.0 / self.decimal_odds


class OddsProvider(ABC):
    """Fetch normalised :class:`PriceQuote` records for a set of fixtures.

    Implementations must be safe to call from a scheduled job only. Rate limits,
    credit budgets and auth are the implementation's responsibility.
    """

    name: str = "base"
    #: markets this provider can supply, as normalised keys
    supported_markets: Sequence[str] = ()

    @abstractmethod
    def fetch(
        self,
        leagues: Iterable[str],
        *,
        markets: Sequence[str] | None = None,
        date_from: dt.date | None = None,
        date_to: dt.date | None = None,
    ) -> list[PriceQuote]: ...

    def health(self) -> dict[str, object]:
        """Cheap status dict for the nightly job log (quota left, auth ok, ...)."""
        return {"provider": self.name, "ok": True}
