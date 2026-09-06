"""football-data.co.uk historical CSVs.

Free, stable, covers all four English tiers back many seasons with:
* full-time / half-time scores
* **referee name**
* **per-match team stats** - shots, shots on target, fouls, corners, cards
* multi-bookmaker odds including closing prices

These per-match stats + referee are what power the referee card model and the
real team shot / corner / card rates (instead of deriving them from expected
goals). Raw CSVs cached under ``data/raw/football_data_couk/`` (gitignored); the
tidy output is merged into ``data/processed/matches.parquet``.
"""

from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

from ingest._net import make_session

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _parse_dates(col: pd.Series) -> pd.Series:
    """football-data.co.uk direct uses DD/MM/YYYY; the GitHub mirror uses ISO."""
    sample = col.dropna().astype(str).head(20)
    if len(sample) and (sample.str.match(_ISO_DATE).mean() > 0.5):
        return pd.to_datetime(col, errors="coerce")
    return pd.to_datetime(col, dayfirst=True, errors="coerce", format="mixed")


BASE_URL = "https://www.football-data.co.uk/mmz4281"
# football-data.co.uk intermittently 503s (and rate-limits cloud IPs); retry a
# couple of times with short backoff before giving up on a division-season.
_RETRIES = 3
_BACKOFF = 2.0
# Corporate proxies (Zscaler etc.) return a block page instead of the CSV -
# detect and bail immediately rather than retrying pointlessly.
_PROXY_MARKERS = ("zscaler", "<!doctype html", "<html")

# internal league code -> football-data.co.uk division code
DIV_CODE = {"EPL": "E0", "ECH": "E1", "EL1": "E2", "EL2": "E3"}

# GitHub mirror of the same CSVs (core cols + referee + team stats, no odds).
# Only covers the top-5 leagues, so EPL only for our tiers - the direct source
# stays primary and is the only route for the EFL + closing odds.
_MIRROR = {
    "EPL": "https://raw.githubusercontent.com/datasets/football-datasets/main/datasets/premier-league/season-{sc}.csv"
}


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
# per-match team stats (football-data.co.uk short codes)
_STAT_COLS = {
    "hs": "home_shots",
    "as": "away_shots",
    "hst": "home_sot",
    "ast": "away_sot",
    "hf": "home_fouls",
    "af": "away_fouls",
    "hc": "home_corners",
    "ac": "away_corners",
    "hy": "home_yellows",
    "ay": "away_yellows",
    "hr": "home_reds",
    "ar": "away_reds",
}


@dataclass
class FootballDataCoUk:
    cache_dir: Path = Path("data/raw/football_data_couk")
    session: requests.Session | None = None
    timeout: float = 30.0
    _sess: requests.Session | None = field(default=None, repr=False)

    def _session(self) -> requests.Session:
        if self.session is not None:
            return self.session
        if self._sess is None:
            self._sess = make_session()
        return self._sess

    def _get(self, url: str) -> bytes | None:
        last_exc: Exception | None = None
        for attempt in range(_RETRIES):
            try:
                resp = self._session().get(url, timeout=self.timeout)
                head = resp.content[:200].lower()
                if resp.headers.get("Server", "").lower().startswith("zscaler") or any(
                    m.encode() in head for m in _PROXY_MARKERS
                ):
                    return None  # proxy block page, not the CSV - don't retry
                resp.raise_for_status()
                return resp.content
            except (requests.HTTPError, requests.Timeout, requests.ConnectionError) as exc:
                last_exc = exc
                if attempt < _RETRIES - 1:
                    time.sleep(_BACKOFF * (2**attempt))
        if last_exc:
            raise last_exc
        return None

    def _fetch_csv(
        self, league: str, div: str, season: str, *, refresh: bool = False
    ) -> pd.DataFrame:
        sc = season_code(season)
        path = self.cache_dir / sc / f"{div}.csv"
        if path.exists() and not refresh:
            return pd.read_csv(
                io.BytesIO(path.read_bytes()), encoding="latin-1", on_bad_lines="skip", dtype=str
            )

        raw = self._get(f"{BASE_URL}/{sc}/{div}.csv")
        if raw is None and league in _MIRROR:  # direct blocked/failed -> GitHub mirror
            raw = self._get(_MIRROR[league].format(sc=sc))
        if raw is None:
            raise requests.ConnectionError(f"{league} {season}: no source reachable")

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
                    raw = self._fetch_csv(lg, div, season, refresh=refresh)
                except (requests.HTTPError, requests.ConnectionError) as exc:  # noqa: PERF203
                    print(f"  skip {lg} {season}: {exc}")
                    continue
                raw.columns = [c.strip().lower() for c in raw.columns]
                if "hometeam" not in raw.columns or "fthg" not in raw.columns:
                    continue
                df = pd.DataFrame(
                    {
                        "date": _parse_dates(raw["date"]),
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
                for src, dst in {**_ODDS_COLS, **_STAT_COLS}.items():
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
