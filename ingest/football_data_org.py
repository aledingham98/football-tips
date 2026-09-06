"""football-data.org v4 - fixtures, results and standings.

Free tier: key required, 10 requests/minute, covers Premier League ("PL") and
Championship ("ELC"); historical data limited to the three most recent seasons.
League One / Two fixtures fall back to FBref.

Raw JSON is cached under ``data/raw/football_data_org/``; tidy results are merged
into ``data/processed/matches.parquet`` alongside the football-data.co.uk history.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ingest._net import get_secret, make_session

API_ROOT = "https://api.football-data.org/v4"
LEAGUE_COMP = {"EPL": "PL", "ECH": "ELC"}
_MIN_INTERVAL = 6.5  # seconds between calls -> under 10/min

MATCH_SCHEMA = [
    "date",
    "league",
    "season",
    "home_team",
    "away_team",
    "fthg",
    "ftag",
    "hthg",
    "htag",
    "status",
    "matchday",
    "source",
]


@dataclass
class FootballDataOrg:
    api_key: str | None = None
    cache_dir: Path = Path("data/raw/football_data_org")
    _last_call: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        self.api_key = self.api_key or get_secret(
            "football_data_org", "api_key", env="FOOTBALL_DATA_ORG_API_KEY"
        )
        if not self.api_key:
            raise RuntimeError(
                "football-data.org API key not found (env FOOTBALL_DATA_ORG_API_KEY "
                "or [football_data_org].api_key in .streamlit/secrets.toml)"
            )
        self._session = make_session({"X-Auth-Token": self.api_key})

    # ------------------------------------------------------------------ #
    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        wait = _MIN_INTERVAL - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        resp = self._session.get(f"{API_ROOT}/{endpoint}", params=params or {}, timeout=30)
        self._last_call = time.monotonic()
        if resp.status_code == 429:
            time.sleep(60)
            return self._get(endpoint, params)
        resp.raise_for_status()
        return resp.json()

    def _matches_payload(self, league: str, season_start_year: int, *, refresh: bool) -> dict:
        comp = LEAGUE_COMP[league]
        path = self.cache_dir / f"{comp}_{season_start_year}.json"
        if path.exists() and not refresh:
            return json.loads(path.read_text())
        payload = self._get(f"competitions/{comp}/matches", {"season": season_start_year})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        return payload

    # ------------------------------------------------------------------ #
    def load_matches(
        self, leagues: list[str], seasons: list[str], *, refresh: bool = False
    ) -> pd.DataFrame:
        """One row per match, schema-compatible with football-data.co.uk output.

        ``seasons`` are ``"YYYY-YYYY"`` labels; football-data.org keys a season by
        its start year.
        """
        rows: list[dict] = []
        for league in leagues:
            if league not in LEAGUE_COMP:
                continue
            for season in seasons:
                start_year = int(season.split("-")[0])
                try:
                    payload = self._matches_payload(league, start_year, refresh=refresh)
                except Exception as exc:  # noqa: BLE001
                    print(f"  football-data.org skip {league} {season}: {exc}")
                    continue
                for m in payload.get("matches", []):
                    ft = m.get("score", {}).get("fullTime", {})
                    ht = m.get("score", {}).get("halfTime", {})
                    rows.append(
                        {
                            "date": pd.to_datetime(m.get("utcDate"), errors="coerce", utc=True),
                            "league": league,
                            "season": season,
                            "home_team": (m.get("homeTeam") or {}).get("name"),
                            "away_team": (m.get("awayTeam") or {}).get("name"),
                            "fthg": ft.get("home"),
                            "ftag": ft.get("away"),
                            "hthg": ht.get("home"),
                            "htag": ht.get("away"),
                            "status": m.get("status"),
                            "matchday": m.get("matchday"),
                            "source": "football_data_org",
                        }
                    )

        df = pd.DataFrame(rows, columns=MATCH_SCHEMA)
        if df.empty:
            return df
        df["date"] = df["date"].dt.tz_localize(None)
        finished = df["status"].eq("FINISHED") & df["fthg"].notna() & df["ftag"].notna()
        df.loc[finished, ["fthg", "ftag"]] = df.loc[finished, ["fthg", "ftag"]].astype(int)
        return df.sort_values("date").reset_index(drop=True)

    def upcoming_fixtures(self, leagues: list[str], *, days_ahead: int = 8) -> pd.DataFrame:
        """Scheduled matches in the next ``days_ahead`` days, for the pricer."""
        date_to = (pd.Timestamp.utcnow() + pd.Timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        date_from = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        rows: list[dict] = []
        for league in leagues:
            if league not in LEAGUE_COMP:
                continue
            payload = self._get(
                f"competitions/{LEAGUE_COMP[league]}/matches",
                {"dateFrom": date_from, "dateTo": date_to},
            )
            for m in payload.get("matches", []):
                rows.append(
                    {
                        "kickoff": pd.to_datetime(m.get("utcDate"), utc=True),
                        "league": league,
                        "home_team": (m.get("homeTeam") or {}).get("name"),
                        "away_team": (m.get("awayTeam") or {}).get("name"),
                        "matchday": m.get("matchday"),
                        "status": m.get("status"),
                        "fixture_id": str(m.get("id")),
                    }
                )
        return (
            pd.DataFrame(rows).sort_values("kickoff").reset_index(drop=True)
            if rows
            else pd.DataFrame()
        )
