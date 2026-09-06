"""Canonical team-name keys.

The four data sources name teams differently - "Man City" / "Manchester City FC"
/ "Manchester City", "Nott'm Forest" / "Nottingham Forest FC", "Spurs" /
"Tottenham Hotspur FC". :func:`canonical_key` maps any of them to one lowercase
key so match dedup, odds joins and ratings lookups line up.
"""

from __future__ import annotations

import re

_SUFFIX = re.compile(r"\b(fc|afc|cf|club)\b", re.I)


def _norm(name: str) -> str:
    n = _SUFFIX.sub("", str(name).lower())
    n = n.replace("&", " and ").replace("'", "")
    n = re.sub(r"[^a-z0-9 ]", " ", n)
    return re.sub(r"\s+", " ", n).strip()


# short / irregular form  ->  canonical key (already _norm-ed on both sides)
_ALIASES: dict[str, str] = {
    "man city": "manchester city",
    "man utd": "manchester united",
    "man united": "manchester united",
    "spurs": "tottenham hotspur",
    "tottenham": "tottenham hotspur",
    "wolves": "wolverhampton wanderers",
    "nottm forest": "nottingham forest",
    "notts forest": "nottingham forest",
    "sheffield weds": "sheffield wednesday",
    "sheff wed": "sheffield wednesday",
    "sheff utd": "sheffield united",
    "west brom": "west bromwich albion",
    "wba": "west bromwich albion",
    "brighton": "brighton and hove albion",
    "brighton hove albion": "brighton and hove albion",
    "qpr": "queens park rangers",
    "newcastle": "newcastle united",
    "leeds": "leeds united",
    "leicester": "leicester city",
    "norwich": "norwich city",
    "hull": "hull city",
    "stoke": "stoke city",
    "cardiff": "cardiff city",
    "swansea": "swansea city",
    "birmingham": "birmingham city",
    "coventry": "coventry city",
    "bristol city": "bristol city",
    "bristol rvs": "bristol rovers",
    "preston": "preston north end",
    "blackburn": "blackburn rovers",
    "bolton": "bolton wanderers",
    "wycombe": "wycombe wanderers",
    "peterboro": "peterborough united",
    "peterborough": "peterborough united",
    "oxford": "oxford united",
    "oxford utd": "oxford united",
    "milton keynes dons": "mk dons",
    "afc wimbledon": "wimbledon",
    "accrington": "accrington stanley",
    "crawley": "crawley town",
    "leyton orient": "leyton orient",
    "burton": "burton albion",
    "cambridge": "cambridge united",
    "colchester": "colchester united",
    "exeter": "exeter city",
    "forest green": "forest green rovers",
    "grimsby": "grimsby town",
    "harrogate": "harrogate town",
    "mansfield": "mansfield town",
    "newport": "newport county",
    "notts co": "notts county",
    "salford": "salford city",
    "tranmere": "tranmere rovers",
    "walsall": "walsall",
    "bradford": "bradford city",
    "carlisle": "carlisle united",
    "cheltenham": "cheltenham town",
    "crewe": "crewe alexandra",
    "doncaster": "doncaster rovers",
    "gillingham": "gillingham",
    "port vale": "port vale",
    "shrewsbury": "shrewsbury town",
    "stevenage": "stevenage",
    "sutton": "sutton united",
    "swindon": "swindon town",
    "rotherham": "rotherham united",
    "wrexham": "wrexham",
    "plymouth": "plymouth argyle",
    "huddersfield": "huddersfield town",
    "luton": "luton town",
    "ipswich": "ipswich town",
    "sunderland": "sunderland",
    "middlesbrough": "middlesbrough",
    "boro": "middlesbrough",
    "west ham": "west ham united",
    "bournemouth": "bournemouth",
}


def canonical_key(name: str) -> str:
    n = _norm(name)
    return _ALIASES.get(n, n)


# canonical key -> clean display name (only where title-casing the key is wrong)
_DISPLAY: dict[str, str] = {
    "brighton and hove albion": "Brighton & Hove Albion",
    "west bromwich albion": "West Bromwich Albion",
    "queens park rangers": "Queens Park Rangers",
    "afc wimbledon": "AFC Wimbledon",
    "wimbledon": "AFC Wimbledon",
    "mk dons": "MK Dons",
    "manchester city": "Manchester City",
    "manchester united": "Manchester United",
    "tottenham hotspur": "Tottenham Hotspur",
    "wolverhampton wanderers": "Wolverhampton Wanderers",
    "nottingham forest": "Nottingham Forest",
}

_TITLE_KEEP_LOWER = {"and", "of", "the"}


def canonical_name(name: str) -> str:
    """A clean, consistent display name for a team, used everywhere so match
    history, fixtures, odds and ratings all join on the same string."""
    key = canonical_key(name)
    if key in _DISPLAY:
        return _DISPLAY[key]
    return " ".join(
        w if w in _TITLE_KEEP_LOWER else w.capitalize() for w in key.split()
    ).strip() or str(name)
