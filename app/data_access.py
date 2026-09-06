"""Cached, read-only access to the committed parquet / JSON the nightly job writes.

The app NEVER calls FBref or an odds API. Everything here loads a file that the
scheduled GitHub Action produced and committed. Every loader degrades gracefully
when a file is missing so the app renders a clear message instead of crashing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from config.loader import league_table, model_config
from models.calibration import CalibrationMap
from models.dixon_coles import TeamRatings

DATA = Path(__file__).resolve().parent.parent / "data"
REPORTS = Path(__file__).resolve().parent.parent / "reports"

RATINGS_PATH = DATA / "model" / "ratings.parquet"
MATCHES_PATH = DATA / "processed" / "matches.parquet"
CALIB_MAP_PATH = DATA / "model" / "calibration_map.json"
CALIB_METRICS_PATH = DATA / "processed" / "calibration_metrics.parquet"
FIXTURES_PATH = DATA / "processed" / "fixtures.parquet"


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_resource(show_spinner=False)
def get_model_config() -> dict:
    return model_config()


@st.cache_resource(show_spinner="Loading team ratings…")
def get_ratings(_mtime_key: float) -> TeamRatings | None:
    if not RATINGS_PATH.exists():
        return None
    return TeamRatings.from_parquet(RATINGS_PATH)


def ratings() -> TeamRatings | None:
    return get_ratings(_mtime(RATINGS_PATH))


@st.cache_resource(show_spinner="Loading calibration map…")
def get_calibration_map(_mtime_key: float) -> CalibrationMap:
    if CALIB_MAP_PATH.exists():
        return CalibrationMap.from_json(CALIB_MAP_PATH)
    return CalibrationMap(params={})


def calibration_map() -> CalibrationMap:
    return get_calibration_map(_mtime(CALIB_MAP_PATH))


@st.cache_data(show_spinner=False)
def calibration_metrics(_mtime_key: float) -> pd.DataFrame | None:
    if CALIB_METRICS_PATH.exists():
        return pd.read_parquet(CALIB_METRICS_PATH)
    return None


@st.cache_data(show_spinner=False)
def matches(_mtime_key: float) -> pd.DataFrame | None:
    if MATCHES_PATH.exists():
        return pd.read_parquet(MATCHES_PATH)
    return None


@st.cache_data(show_spinner=False)
def fixtures(_mtime_key: float) -> pd.DataFrame | None:
    if FIXTURES_PATH.exists():
        return pd.read_parquet(FIXTURES_PATH)
    return None


# --- convenience views the pages use -------------------------------------- #
def team_options() -> dict[str, list[str]]:
    """{league_code: [team, ...]} from the ratings table, for fixture pickers."""
    r = ratings()
    if r is None:
        return {}
    lt = league_table()
    tier_to_league = {v["tier"]: k for k, v in lt.items()}
    out: dict[str, list[str]] = {}
    for team, row in r.table.iterrows():
        lg = tier_to_league.get(int(row["tier"]), "EPL")
        out.setdefault(lg, []).append(str(team))
    for lg in out:
        out[lg].sort()
    return out


def data_freshness() -> dict[str, str]:
    r = ratings()
    info: dict[str, str] = {}
    if r is not None:
        info["ratings_fitted"] = str(r.meta.get("fitted_at", "?"))[:19]
        info["data_through"] = str(r.meta.get("date_max", "?"))
        info["n_matches"] = str(r.meta.get("n_matches", "?"))
        info["n_teams"] = str(r.meta.get("n_teams", "?"))
    return info
