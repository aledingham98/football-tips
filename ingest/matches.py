"""Build the committed match-history parquet from the free result sources.

Priority: football-data.co.uk (4 tiers, 5+ seasons, closing odds + referee) when
reachable; football-data.org (PL + Championship, ~3 seasons, no odds) fills gaps
and is the only source that works from a proxied dev machine.

Output: ``data/processed/matches.parquet`` - one row per completed match, with a
``tier`` column, consumed by the Dixon-Coles fit and the Phase 2 backtest.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.loader import league_table
from ingest.football_data_couk import FootballDataCoUk
from ingest.football_data_org import FootballDataOrg
from ingest.teams import canonical_key, canonical_name

OUT_PATH = Path("data/processed/matches.parquet")


def normalize_team_name(name: str) -> str:
    """Canonical lowercase key for a team, aliases resolved (Man City ->
    manchester city, Spurs -> tottenham hotspur, ...)."""
    return canonical_key(name)


def _dedupe_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["date"].dt.strftime("%Y-%m-%d")
        + "|"
        + df["home_team"].map(normalize_team_name)
        + "|"
        + df["away_team"].map(normalize_team_name)
    )


def build_match_history(
    leagues: list[str] | None = None,
    seasons: list[str] | None = None,
    *,
    out_path: str | Path = OUT_PATH,
    write: bool = True,
) -> pd.DataFrame:
    lt = league_table()
    leagues = leagues or list(lt)
    if seasons is None:
        from config.loader import leagues_config

        seasons = leagues_config()["seasons"]["history"]

    frames: list[pd.DataFrame] = []

    try:
        couk = FootballDataCoUk().load_matches(leagues, seasons)
        couk["source"] = "football_data_couk"
        frames.append(couk)
        print(f"  football-data.co.uk: {len(couk)} matches")
    except Exception as exc:  # noqa: BLE001
        print(f"  football-data.co.uk unavailable: {exc}")

    org_leagues = [lg for lg in leagues if lt[lg].get("football_data_org")]
    if org_leagues:
        try:
            org = FootballDataOrg().load_matches(org_leagues, seasons)
            org = org[org["status"].eq("FINISHED")].copy()
            frames.append(org)
            print(f"  football-data.org: {len(org)} matches")
        except Exception as exc:  # noqa: BLE001
            print(f"  football-data.org unavailable: {exc}")

    covered = set(pd.concat(frames)["league"].unique()) if frames else set()
    missing = [lg for lg in leagues if lg not in covered]
    if missing:
        # FBref schedules: the brief's fallback for League One / Two, and it also
        # carries xG + referee for the top two tiers.
        try:
            from ingest.fbref import load_schedules

            fb = load_schedules(missing, seasons)
            if len(fb):
                frames.append(fb)
                print(f"  FBref schedules: {len(fb)} matches for {sorted(fb['league'].unique())}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FBref schedule fallback unavailable: {exc}")

    if not frames:
        raise RuntimeError("no match sources available")

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    # prefer the richer source: keep first occurrence after ordering by source rank
    src_rank = {"football_data_couk": 0, "football_data_org": 1, "fbref_schedule": 2}
    combined["_rank"] = combined["source"].map(src_rank).fillna(9)
    combined = combined.sort_values(["_rank", "date"])
    combined = combined[~_dedupe_key(combined).duplicated()].drop(columns="_rank")

    # one consistent display name everywhere downstream
    combined["home_team"] = combined["home_team"].map(canonical_name)
    combined["away_team"] = combined["away_team"].map(canonical_name)

    combined["tier"] = combined["league"].map(lambda lg: lt[lg]["tier"])
    combined = combined.dropna(subset=["fthg", "ftag"]).sort_values("date").reset_index(drop=True)
    combined["fthg"] = combined["fthg"].astype(int)
    combined["ftag"] = combined["ftag"].astype(int)

    if write:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(out_path)
        print(f"  wrote {len(combined)} matches -> {out_path}")
    return combined


if __name__ == "__main__":
    build_match_history()
