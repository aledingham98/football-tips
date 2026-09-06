"""Per-referee card rates.

Referee identity is one of the more predictable inputs in football. From the
football-data.co.uk match history (which carries the referee name + per-match
yellow/red counts) this builds, per referee, an expected cards-per-match rate
shrunk toward the league mean by a pseudo-count, and the same for reds.

Used by :func:`models.fixture.build_match_inputs` when the referee for a fixture
is known; otherwise the league mean is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RefereeRates:
    table: pd.DataFrame  # index = referee, cols: matches, yellows_pm, reds_pm, cards_pm
    league_yellows_pm: float  # per team per match
    league_reds_pm: float

    def for_referee(self, name: str | None) -> tuple[float, float]:
        """(expected yellows per team per match, expected reds per match) for a
        named referee, or the league mean when unknown."""
        if name and name in self.table.index:
            row = self.table.loc[name]
            return float(row["yellows_pt"]), float(row["reds_pm"])
        return self.league_yellows_pm, self.league_reds_pm

    def top(self, n: int = 15) -> pd.DataFrame:
        return self.table.sort_values("cards_pm", ascending=False).head(n)


def fit_referee_rates(
    matches: pd.DataFrame, *, pseudo_matches: float = 20.0, half_life_days: float = 540.0
) -> RefereeRates:
    """``matches`` needs ``referee``, ``home_yellows/away_yellows`` and
    ``home_reds/away_reds`` (football-data.co.uk). Rows without them are ignored."""
    need = ["referee", "home_yellows", "away_yellows", "home_reds", "away_reds"]
    m = matches.dropna(subset=need).copy()
    m = m[m["referee"].astype(str).str.strip() != ""]
    if m.empty:
        return RefereeRates(
            table=pd.DataFrame(
                columns=["matches", "yellows_pm", "yellows_pt", "reds_pm", "cards_pm"]
            ),
            league_yellows_pm=1.9,
            league_reds_pm=0.11,
        )

    m["y_total"] = m["home_yellows"] + m["away_yellows"]
    m["r_total"] = m["home_reds"] + m["away_reds"]
    ref_dt = pd.to_datetime(m["date"])
    w = np.exp(-np.log(2.0) * (ref_dt.max() - ref_dt).dt.days / half_life_days)
    m["w"] = w

    league_y_pm = float(np.average(m["y_total"], weights=m["w"]))  # per match (both teams)
    league_r_pm = float(np.average(m["r_total"], weights=m["w"]))

    g = m.groupby("referee")
    agg = g.apply(
        lambda d: pd.Series(
            {
                "matches": len(d),
                "w": d["w"].sum(),
                "y_pm": np.average(d["y_total"], weights=d["w"]),
                "r_pm": np.average(d["r_total"], weights=d["w"]),
            }
        ),
        include_groups=False,
    )
    k = pseudo_matches
    agg["yellows_pm"] = (agg["w"] * agg["y_pm"] + k * league_y_pm) / (agg["w"] + k)
    agg["reds_pm"] = (agg["w"] * agg["r_pm"] + k * league_r_pm) / (agg["w"] + k)
    agg["yellows_pt"] = agg["yellows_pm"] / 2.0  # per team per match
    agg["cards_pm"] = agg["yellows_pm"] + agg["reds_pm"]
    table = agg[["matches", "yellows_pm", "yellows_pt", "reds_pm", "cards_pm"]].round(3)
    table["matches"] = table["matches"].astype(int)

    return RefereeRates(
        table=table.sort_index(),
        league_yellows_pm=league_y_pm / 2.0,
        league_reds_pm=league_r_pm,
    )
