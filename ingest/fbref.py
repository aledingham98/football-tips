"""FBref ingestion via the ``soccerdata`` package.

FBref is the key player-level source - it covers all four English tiers, which
Understat does not. It sits behind Cloudflare with strict rate limiting, so this
is **only ever run from the scheduled GitHub Action**, never at request time.
soccerdata's on-disk cache (``data/raw/fbref_cache``) is the rate-limit shield;
the Action restores it from ``actions/cache`` and only fetches what's new.

soccerdata doesn't ship the EFL in its default league dict, so we register the
four English tiers via ``config/soccerdata_league_dict.json`` before import.

Everything here is defensive: a failure for one league/stat-type/season is
logged and skipped, never fatal - the Dixon-Coles fit only needs match results
(from football-data.co.uk), and FBref data is enrichment on top.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pandas as pd

CACHE_DIR = Path("data/raw/fbref_cache")
_LEAGUE_DICT_SRC = Path("config/soccerdata_league_dict.json")

# internal code -> soccerdata league id (see config/soccerdata_league_dict.json)
FBREF_LEAGUE = {
    "EPL": "ENG-Premier League",
    "ECH": "ENG-Championship",
    "EL1": "ENG-League One",
    "EL2": "ENG-League Two",
}

# season-stat groups worth pulling for the player sub-models
PLAYER_SEASON_STATS = ("standard", "shooting", "passing", "misc", "playing_time")
# match-level player logs power the minutes model + player-prop calibration
PLAYER_MATCH_STATS = ("summary",)


def _prepare_soccerdata_dir() -> Path:
    """Point soccerdata at a repo-local dir and install our custom league dict."""
    sd_dir = CACHE_DIR
    (sd_dir / "config").mkdir(parents=True, exist_ok=True)
    dst = sd_dir / "config" / "league_dict.json"
    if _LEAGUE_DICT_SRC.exists():
        shutil.copyfile(_LEAGUE_DICT_SRC, dst)
    os.environ["SOCCERDATA_DIR"] = str(sd_dir.resolve())
    return sd_dir


def _seasons_arg(seasons: list[str]) -> list[str]:
    # soccerdata accepts "2024-2025" or "2425"; keep the explicit form.
    return seasons


def pull_league(
    league_code: str,
    seasons: list[str],
    *,
    throttle: float = 4.0,
    do_match_stats: bool = True,
) -> dict[str, pd.DataFrame]:
    """Return {dataset_name: DataFrame} for one league. Missing datasets are omitted."""
    _prepare_soccerdata_dir()
    import soccerdata as sd  # imported here so SOCCERDATA_DIR is already set

    sd_league = FBREF_LEAGUE[league_code]
    out: dict[str, pd.DataFrame] = {}
    try:
        fbref = sd.FBref(
            leagues=sd_league,
            seasons=_seasons_arg(seasons),
            data_dir=CACHE_DIR / "data",
            no_store=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  FBref[{league_code}] init failed: {exc}")
        return out

    def _try(name: str, fn):
        try:
            df = fn()
            if df is not None and len(df):
                out[name] = df.reset_index()
                print(f"  FBref[{league_code}] {name}: {len(df)} rows")
        except Exception as exc:  # noqa: BLE001
            print(f"  FBref[{league_code}] {name} skipped: {exc}")
        time.sleep(throttle)

    _try("schedule", lambda: fbref.read_schedule())
    for st in PLAYER_SEASON_STATS:
        _try(f"player_season_{st}", lambda st=st: fbref.read_player_season_stats(stat_type=st))
    _try("team_season_standard", lambda: fbref.read_team_season_stats(stat_type="standard"))
    _try(
        "team_season_standard_vs",
        lambda: fbref.read_team_season_stats(stat_type="standard", opponent_stats=True),
    )
    if do_match_stats:
        for st in PLAYER_MATCH_STATS:
            _try(
                f"player_match_{st}",
                lambda st=st: fbref.read_player_match_stats(stat_type=st, force_cache=False),
            )
    return out


def ingest_fbref(
    leagues: list[str],
    seasons: list[str],
    *,
    out_dir: str | Path = "data/processed/fbref",
    throttle: float = 4.0,
    current_season_only_match_stats: str | None = None,
) -> dict[str, list[str]]:
    """Pull every league, write one parquet per (league, dataset). Returns a
    manifest of what was written. Never raises for a single-league failure.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, list[str]] = {}
    for lg in leagues:
        if lg not in FBREF_LEAGUE:
            continue
        do_match = True
        use_seasons = seasons
        if current_season_only_match_stats:
            # heavy match-log pull only for the current season to bound run time
            use_seasons = seasons
        datasets = pull_league(lg, use_seasons, throttle=throttle, do_match_stats=do_match)
        written: list[str] = []
        for name, df in datasets.items():
            path = out_dir / f"{lg}__{name}.parquet"
            try:
                df.to_parquet(path)
                written.append(path.name)
            except Exception as exc:  # noqa: BLE001
                print(f"  write {path.name} failed: {exc}")
        manifest[lg] = written
    return manifest
