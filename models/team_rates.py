"""Real per-team shot / corner / card rates from the match history.

Where the history carries per-match team stats (football-data.co.uk), this gives
each team a time-weighted rate *for* and *against* for shots, shots on target,
corners, fouls and yellows. A fixture's expected value for a stat is then

    home_expected = league_avg * (home_for / league_avg) * (away_against / league_avg)

i.e. the classic attack-strength x opponent-weakness form, same as the goals
model. Teams with no stat history (currently the EFL) fall back to the league
average, and :func:`models.fixture.build_match_inputs` blends toward its
expected-goals heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_STATS = ["shots", "sot", "corners", "fouls", "yellows"]


@dataclass
class TeamRates:
    table: pd.DataFrame  # index = team; cols like shots_for, shots_against, ...
    league: dict[str, float]  # league mean per match for each stat
    meta: dict = field(default_factory=dict)

    def expected(self, home: str, away: str, stat: str) -> tuple[float, float]:
        lg = self.league.get(stat)
        if lg is None or lg <= 0:
            return 0.0, 0.0
        hf = self._get(home, f"{stat}_for", lg)
        ha = self._get(home, f"{stat}_against", lg)
        af = self._get(away, f"{stat}_for", lg)
        aa = self._get(away, f"{stat}_against", lg)
        home_exp = lg * (hf / lg) * (aa / lg)
        away_exp = lg * (af / lg) * (ha / lg)
        return round(home_exp, 2), round(away_exp, 2)

    def has(self, team: str) -> bool:
        return team in self.table.index

    def _get(self, team: str, col: str, default: float) -> float:
        if team in self.table.index and pd.notna(self.table.at[team, col]):
            return float(self.table.at[team, col])
        return default


def fit_team_rates(matches: pd.DataFrame, *, half_life_days: float = 240.0) -> TeamRates:
    have = [f"home_{s}" for s in _STATS if f"home_{s}" in matches.columns]
    if not have:
        return TeamRates(table=pd.DataFrame(), league={}, meta={"note": "no team stats in history"})

    m = matches.dropna(subset=have).copy()
    m["date"] = pd.to_datetime(m["date"])
    if m.empty:
        return TeamRates(table=pd.DataFrame(), league={}, meta={"note": "no team stats"})
    w = np.exp(-np.log(2.0) * (m["date"].max() - m["date"]).dt.days / half_life_days)
    m["w"] = w

    league = {}
    for s in _STATS:
        if f"home_{s}" in m.columns:
            vals = pd.concat([m[f"home_{s}"], m[f"away_{s}"]])
            wts = pd.concat([m["w"], m["w"]])
            league[s] = float(np.average(vals, weights=wts))

    # long form: one row per team-appearance
    rows = []
    for r in m.itertuples(index=False):
        for side, opp in (("home", "away"), ("away", "home")):
            row = {"team": getattr(r, f"{side}_team"), "w": r.w}
            for s in _STATS:
                if f"{side}_{s}" in m.columns:
                    row[f"{s}_for"] = getattr(r, f"{side}_{s}")
                    row[f"{s}_against"] = getattr(r, f"{opp}_{s}")
            rows.append(row)
    long = pd.DataFrame(rows)

    def wmean(d: pd.DataFrame) -> pd.Series:
        out = {"matches": len(d)}
        for c in d.columns:
            if c in ("team", "w", "matches"):
                continue
            out[c] = np.average(d[c], weights=d["w"])
        return pd.Series(out)

    table = long.groupby("team").apply(wmean, include_groups=False)
    table["matches"] = table["matches"].astype(int)
    return TeamRates(
        table=table.round(2),
        league={k: round(v, 2) for k, v in league.items()},
        meta={"n_teams": int(table.shape[0]), "half_life_days": half_life_days},
    )
