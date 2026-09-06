"""High Confidence - upcoming markets ranked by raw model probability.

These are LIKELY, not VALUE. EV is shown for any price you enter so a badly
priced 'likely' outcome is obvious. Never merged with the Value Bets list.
"""

from __future__ import annotations

import streamlit as st
from _sim import load_slate

import data_access

st.set_page_config(page_title="High Confidence", page_icon="✅", layout="centered")
st.title("✅ High Confidence")
st.caption(
    "Most likely outcomes across the upcoming slate. **Likely ≠ value** — a 90% shot at 1.05 is still a bad bet."
)

fx = data_access.fixtures(data_access._mtime(data_access.FIXTURES_PATH))
if fx is None or len(fx) == 0:
    st.warning(
        "No upcoming fixtures committed yet. The nightly job writes "
        "`data/processed/fixtures.parquet` (football-data.org for PL + Championship)."
    )
    st.stop()

df = load_slate()
if df.empty:
    st.warning(
        "Slate is empty — fixtures exist but none of their teams are in the ratings table yet."
    )
    st.stop()

min_p = st.slider("Minimum model probability", 0.50, 0.95, 0.70, 0.01)
leagues = st.multiselect(
    "Leagues", sorted(df["league"].unique()), default=sorted(df["league"].unique())
)
view = df[(df["cal_prob"] >= min_p) & (df["league"].isin(leagues))].sort_values(
    "cal_prob", ascending=False
)

st.write(f"**{len(view)}** selections at ≥ {min_p:.0%}")
st.caption(
    "Raw model probabilities. 1X2 is well calibrated; Championship goals lines read "
    "~4pp low, so treat high Over/BTTS numbers there as a floor."
)
for r in view.itertuples(index=False):
    with st.container(border=True):
        st.write(f"**{r.market}** — {r.fixture}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Model", f"{r.cal_prob:.0%}")
        c2.metric("Fair odds", f"{r.fair_odds:.2f}")
        price = c3.number_input(
            "Your price", 0.0, step=0.01, key=f"hc_{r.fixture}_{r.market}", format="%.2f"
        )
        if price:
            ev = r.cal_prob * price - 1
            (st.success if ev >= 0.03 else st.warning if ev >= 0 else st.error)(
                f"EV {ev * 100:+.1f}%  ·  {'take it' if ev >= 0.03 else 'thin' if ev >= 0 else 'no value'}"
            )
        st.caption(str(r.kickoff)[:16])
