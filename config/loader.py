"""Load and cache the YAML config files."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml

_DIR = Path(__file__).parent


@cache
def _load(name: str) -> dict:
    return yaml.safe_load((_DIR / name).read_text())


def leagues_config() -> dict:
    return _load("leagues.yaml")


def model_config() -> dict:
    return _load("model.yaml")


def markets_config() -> dict:
    return _load("markets.yaml")


def league_table() -> dict[str, dict]:
    """``{"EPL": {name, tier, fbref, football_data_org, ...}, ...}``."""
    return leagues_config()["leagues"]


def tier_of(league_code: str) -> int:
    return int(league_table()[league_code]["tier"])
