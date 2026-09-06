"""Odds refresh - run by .github/workflows/odds-refresh.yml, twice daily.

The Odds API only (Betfair deferred). region=uk, markets=h2h,totals,btts ->
3 credits per league per call, 2 leagues = 6 credits. The CreditBudgeter
hard-stops before any call that would take the remaining balance below the floor.

Writes:
  data/raw/odds/odds_<timestamp>.parquet   (snapshot, gitignored history is fine)
  data/processed/odds_latest.parquet       (best price per fixture/market/selection)
  data/raw/odds/quota.json                 (remaining credits, committed)
"""

from __future__ import annotations

import datetime as dt
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from ingest._net import get_secret, make_session
from ingest.odds.budget import CreditBudgeter, QuotaExceeded
from ingest.odds.the_odds_api import TheOddsApiProvider

QUOTA_PATH = Path("data/raw/odds/quota.json")
LATEST_PATH = Path("data/processed/odds_latest.parquet")
LEAGUES = ["EPL", "ECH"]


def main() -> int:
    key = get_secret("the_odds_api", "api_key", env="THE_ODDS_API_KEY")
    if not key:
        print("no The Odds API key (THE_ODDS_API_KEY / [the_odds_api].api_key) - skipping")
        return 0
    floor = int(
        get_secret("the_odds_api", "quota_floor", env="THE_ODDS_API_QUOTA_FLOOR", default="50")
    )

    budgeter = CreditBudgeter(QUOTA_PATH, floor=floor)
    warn = budgeter.warn_if_low()
    if warn:
        print(f"::warning::{warn}")

    provider = TheOddsApiProvider(key, budgeter, session=make_session())
    try:
        quotes = provider.fetch(LEAGUES)
    except QuotaExceeded as exc:
        print(f"::warning::{exc}")
        return 0  # expected backstop, not a job failure

    if not quotes:
        print("no quotes returned (no fixtures in window?)")
        return 0

    df = pd.DataFrame([asdict(q) for q in quotes])
    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M")
    QUOTA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(QUOTA_PATH.parent / f"odds_{ts}.parquet")

    # keep the best (longest) price per fixture/market/selection
    best = (
        df.sort_values("decimal_odds")
        .groupby(["league", "home_team", "away_team", "market", "selection"], as_index=False)
        .last()
    )
    LATEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    best.to_parquet(LATEST_PATH)

    print(
        f"{len(quotes)} quotes across {best['home_team'].nunique()} fixtures; "
        f"quota remaining {budgeter.remaining} (floor {floor})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
