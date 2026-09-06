"""The Odds API provider - free Starter tier, hard credit budget.

Usage rules baked in:
* region is always ``uk``; markets default to ``h2h,totals,btts`` (3 credits/league/call)
* every call is gated by :class:`~ingest.odds.budget.CreditBudgeter` *before* any
  HTTP request, and the ``x-requests-remaining`` header is persisted afterwards
* responses are returned as normalised :class:`PriceQuote` records for the caller
  to cache to parquet; the app never calls this class
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence

import requests

from ingest.odds.base import OddsProvider, PriceQuote
from ingest.odds.budget import CreditBudgeter, call_cost

API_ROOT = "https://api.the-odds-api.com/v4"
REGION = "uk"
DEFAULT_MARKETS = ("h2h", "totals", "btts")
# The Odds API sport-key -> internal league code (only the covered two).
SPORT_KEYS = {"soccer_epl": "EPL", "soccer_efl_champ": "ECH"}
_MARKET_MAP = {"h2h": "result", "totals": "total_goals", "btts": "btts"}


class TheOddsApiProvider(OddsProvider):
    name = "the_odds_api"
    supported_markets = ("result", "total_goals", "btts")

    def __init__(
        self,
        api_key: str,
        budgeter: CreditBudgeter,
        *,
        session: requests.Session | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.api_key = api_key
        self.budgeter = budgeter
        self.session = session or requests.Session()
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    def fetch(
        self,
        leagues: Iterable[str],
        *,
        markets: Sequence[str] | None = None,
        date_from: dt.date | None = None,
        date_to: dt.date | None = None,
    ) -> list[PriceQuote]:
        api_markets = tuple(markets) if markets else DEFAULT_MARKETS
        want = {v for k, v in SPORT_KEYS.items()}
        targets = [lg for lg in leagues if lg in want]
        cost_each = call_cost(len(api_markets), n_regions=1)

        quotes: list[PriceQuote] = []
        for sport_key, league_code in SPORT_KEYS.items():
            if league_code not in targets:
                continue
            # BUDGET GATE - raises QuotaExceeded before any network call
            self.budgeter.check(cost_each)
            resp = self.session.get(
                f"{API_ROOT}/sports/{sport_key}/odds",
                params={
                    "apiKey": self.api_key,
                    "regions": REGION,
                    "markets": ",".join(api_markets),
                    "oddsFormat": "decimal",
                    "dateFormat": "iso",
                },
                timeout=self.timeout,
            )
            self._absorb_quota(resp)
            resp.raise_for_status()
            quotes.extend(self._parse(resp.json(), league_code))
        return quotes

    # ------------------------------------------------------------------ #
    def _absorb_quota(self, resp: requests.Response) -> None:
        header = resp.headers.get("x-requests-remaining")
        if header is not None:
            self.budgeter.update_from_header(header)
        else:
            self.budgeter.record_spend(call_cost(len(DEFAULT_MARKETS)))

    @staticmethod
    def _parse(payload: list[dict], league_code: str) -> list[PriceQuote]:
        out: list[PriceQuote] = []
        fetched = dt.datetime.now(dt.UTC).isoformat()
        for event in payload:
            home, away = event.get("home_team", ""), event.get("away_team", "")
            fid = event.get("id", f"{home}-{away}")
            ko = event.get("commence_time", "")
            best: dict[tuple[str, str], float] = {}
            for book in event.get("bookmakers", []):
                for mk in book.get("markets", []):
                    market = _MARKET_MAP.get(mk.get("key", ""), mk.get("key", ""))
                    for oc in mk.get("outcomes", []):
                        sel = _selection_label(market, oc, home, away)
                        price = float(oc.get("price", 0) or 0)
                        if price <= 1.0:
                            continue
                        key = (market, sel)
                        if price > best.get(key, 0.0):
                            best[key] = price
            for (market, sel), price in best.items():
                out.append(
                    PriceQuote(
                        provider="the_odds_api",
                        league=league_code,
                        fixture_id=fid,
                        home_team=home,
                        away_team=away,
                        kickoff_iso=ko,
                        market=market,
                        selection=sel,
                        decimal_odds=price,
                        fetched_iso=fetched,
                    )
                )
        return out

    def health(self) -> dict[str, object]:
        return {
            "provider": self.name,
            "credits_remaining": self.budgeter.remaining,
            "floor": self.budgeter.floor,
            "warning": self.budgeter.warn_if_low(),
        }


def _selection_label(market: str, outcome: dict, home: str, away: str) -> str:
    name = str(outcome.get("name", ""))
    point = outcome.get("point")
    if market == "result":
        if name == home:
            return "home"
        if name == away:
            return "away"
        return "draw"
    if market == "total_goals":
        return f"{name.lower()} {point}"
    if market == "btts":
        return name.lower()
    return name.lower()
