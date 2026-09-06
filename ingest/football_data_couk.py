"""football-data.co.uk historical CSVs.

Free, stable, covers all four English tiers back many seasons with full-time and
half-time scores, referee (recent seasons) and multi-bookmaker odds *including
closing prices*. Used to fit the Dixon-Coles model and, in Phase 2, as the
held-out backtest set with the closing line as the calibration benchmark.

Raw CSVs are cached under ``data/raw/football_data_couk/`` (gitignored); the tidy
output is written to ``data/processed/matches.parquet`` and committed.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281"

# internal league code -> football-data.co.uk division code
DIV_CODE = {"EPL": "E0", "ECH": "E1", "EL1": "E2", "EL2": "E3"}


# "2024-2025" -> "2425"
def season_code(season: str) -> str:
    start, end = season.split("-")
    return start[2:] + end[2:]


# closing-odds columns we keep when present (they vary by season / division)
_ODDS_COLS = {
    "psch": "close_home_odds",  # Pinnacle closing - sharpest benchmark
    "pscd": "close_draw_odds",
    "psca": "close_away_odds",
    "avgc>2.5": "close_over25_odds",
    "avgc<2.5": "close_under25_odds",
    "b365ch": "b365_close_home_odds",
    "b365cd": "b365_close_draw_odds",
    "b365ca": "b365_close_away_odds",
}
_KEEP_BASE = ["date", "league", "season", "home_team", "away_team", "fthg", "ftag", "hthg", "htag"]


@dataclass
class FootballDataCoUk:
    cache_dir: Path = Path("data/raw/football_data_couk")
    session: requests.Session | None = None
    timeout: float = 30.0

    def _session(self) -> requests.Session:
        if self.session is None:
            self.session = requests.Session()
            self.session.headers.update({"User-Agent": "football-tips/0.1 (personal use)"})
        return self.session

    def _fetch_csv(self, div: str, season: str, *, refresh: bool = False) -> pd.DataFrame:
        sc = season_code(season)
        path = self.cache_dir / sc / f"{div}.csv"
        if path.exists() and not refresh:
            raw = path.read_bytes()
        else:
            url = f"{BASE_URL}/{sc}/{div}.csv"
            resp = self._session().get(url, timeout=self.timeout)
            resp.raise_for_status()
            raw = resp.content
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        return pd.read_csv(io.BytesIO(raw), encoding="latin-1", on_bad_lines="skip", dtype=str)

    def load_matches(
        self,
        leagues: list[str],
        seasons: list[str],
        *,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Tidy one row per match across the requested leagues and seasons."""
        frames: list[pd.DataFrame] = []
        for lg in leagues:
            div = DIV_CODE[lg]
            for season in seasons:
                try:
                    raw = self._fetch_csv(div, season, refresh=refresh)
                except (requests.HTTPError, requests.ConnectionError) as exc:  # noqa: PERF203
                    print(f"  skip {lg} {season}: {exc}")
                    continue
                raw.columns = [c.strip().lower() for c in raw.columns]
                if "hometeam" not in raw.columns or "fthg" not in raw.columns:
                    continue
                df = pd.DataFrame(
                    {
                        "date": pd.to_datetime(
                            raw["date"], dayfirst=True, errors="coerce", format="mixed"
                        ),
                        "league": lg,
                        "season": season,
                        "home_team": raw["hometeam"].str.strip(),
                        "away_team": raw["awayteam"].str.strip(),
                        "fthg": pd.to_numeric(raw["fthg"], errors="coerce"),
                        "ftag": pd.to_numeric(raw["ftag"], errors="coerce"),
                        "hthg": pd.to_numeric(raw.get("hthg"), errors="coerce"),
                        "htag": pd.to_numeric(raw.get("htag"), errors="coerce"),
                        "referee": raw.get(
                            "referee", pd.Series(index=raw.index, dtype=str)
                        ).str.strip(),
                    }
                )
                for src, dst in _ODDS_COLS.items():
                    df[dst] = (
                        pd.to_numeric(raw[src], errors="coerce") if src in raw.columns else pd.NA
                    )
                frames.append(df)

        if not frames:
            raise RuntimeError("no football-data.co.uk data loaded - check connectivity / seasons")

        out = pd.concat(frames, ignore_index=True)
        out = out.dropna(subset=["date", "home_team", "away_team", "fthg", "ftag"])
        out = out[(out["fthg"] >= 0) & (out["ftag"] >= 0)].reset_index(drop=True)
        out["fthg"] = out["fthg"].astype(int)
        out["ftag"] = out["ftag"].astype(int)
        return out.sort_values("date").reset_index(drop=True)


def load_matches(
    leagues: list[str], seasons: list[str], *, cache_dir: str | Path = "data/raw/football_data_couk"
) -> pd.DataFrame:
    return FootballDataCoUk(cache_dir=Path(cache_dir)).load_matches(leagues, seasons)
