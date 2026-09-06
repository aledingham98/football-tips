"""Upcoming fixtures -> data/processed/fixtures.parquet (committed).

The app's Value / High-Confidence / Acca pages read this; they never hit an API.
Premier League + Championship come from football-data.org; League One / Two fall
back to unplayed rows in the FBref schedule.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.loader import league_table

OUT_PATH = Path("data/processed/fixtures.parquet")
SCHEMA = ["kickoff", "league", "tier", "home_team", "away_team", "matchday", "status", "fixture_id"]


def build_fixtures(
    leagues: list[str] | None = None,
    *,
    days_ahead: int = 10,
    out_path: str | Path = OUT_PATH,
    write: bool = True,
) -> pd.DataFrame:
    lt = league_table()
    leagues = leagues or list(lt)
    frames: list[pd.DataFrame] = []

    org_leagues = [lg for lg in leagues if lt[lg].get("football_data_org")]
    if org_leagues:
        try:
            from ingest.football_data_org import FootballDataOrg

            fx = FootballDataOrg().upcoming_fixtures(org_leagues, days_ahead=days_ahead)
            if len(fx):
                fx["status"] = fx.get("status", "SCHEDULED")
                frames.append(fx)
        except Exception as exc:  # noqa: BLE001
            print(f"  football-data.org fixtures unavailable: {exc}")

    covered = set(pd.concat(frames)["league"].unique()) if frames else set()
    missing = [lg for lg in leagues if lg not in covered]
    if missing:
        try:
            from ingest.fbref import load_upcoming_schedule

            fb = load_upcoming_schedule(missing)
            if len(fb):
                frames.append(fb)
        except Exception as exc:  # noqa: BLE001
            print(f"  FBref upcoming-fixture fallback unavailable: {exc}")

    if not frames:
        print("  no fixture source available - writing empty fixtures parquet")
        df = pd.DataFrame(columns=SCHEMA)
    else:
        from ingest.teams import canonical_name

        df = pd.concat(frames, ignore_index=True)
        df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True, errors="coerce")
        df["home_team"] = df["home_team"].map(canonical_name)
        df["away_team"] = df["away_team"].map(canonical_name)
        df["tier"] = df["league"].map(lambda lg: lt[lg]["tier"])
        df["matchday"] = df.get("matchday")
        df["fixture_id"] = df.get("fixture_id", df["home_team"] + " v " + df["away_team"])
        df = (
            df[SCHEMA]
            .dropna(subset=["kickoff", "home_team", "away_team"])
            .sort_values("kickoff")
            .reset_index(drop=True)
        )

    if write:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path)
        print(f"  wrote {len(df)} fixtures -> {out_path}")
    return df


if __name__ == "__main__":
    build_fixtures()
