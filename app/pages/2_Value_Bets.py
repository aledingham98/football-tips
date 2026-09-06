"""Value Bets - sorted by EV% = model_prob x best_available_odds - 1, filtered to
EV > 3%. Expect this list to be short or empty most days; that is correct.

Until the Phase 4 odds feed (The Odds API + Betfair) is committed, this is a
manual scanner: the slate shows model prices, you enter the best price you can
find per line, and the EV column fills in.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from _sim import load_slate

import data_access

st.set_page_config(page_title="Value Bets", page_icon="💰", layout="centered")
st.title("💰 Value Bets")

metrics = data_access.calibration_metrics(data_access._mtime(data_access.CALIB_METRICS_PATH))
st.caption(
    "EV = model probability × best available odds − 1. Filtered to EV > 3% to cut noise. "
    "A short or empty list is the expected outcome — bet builders and most markets carry "
    "more margin than the model can beat."
)

fx = data_access.fixtures(data_access._mtime(data_access.FIXTURES_PATH))
if fx is None or len(fx) == 0:
    st.warning(
        "No committed fixtures yet — the nightly job writes `data/processed/fixtures.parquet`."
    )
    st.stop()

df = load_slate()
if df.empty:
    st.warning("Slate empty (fixture teams not in the ratings table yet).")
    st.stop()

st.info(
    "**No automated odds feed yet.** Phase 4 wires The Odds API (UK, h2h/totals/btts, "
    "budget-gated) and Betfair no-vig midpoints, and this page auto-ranks. For now, "
    "enter prices you've seen below.",
    icon="🛠️",
)

ev_key = "value_prices"
st.session_state.setdefault(ev_key, {})

leagues = st.multiselect(
    "Leagues", sorted(df["league"].unique()), default=sorted(df["league"].unique())
)
markets = st.multiselect(
    "Markets",
    sorted(df["market"].unique()),
    default=["Home win", "Away win", "Over 2.5", "Under 2.5", "BTTS"],
)
work = df[df["league"].isin(leagues) & df["market"].isin(markets)].copy()

rows = []
for r in work.itertuples(index=False):
    k = f"{r.fixture}|{r.market}"
    with st.expander(
        f"{r.fixture} — {r.market}  ·  model {r.cal_prob:.0%} (fair {r.fair_odds:.2f})"
    ):
        price = st.number_input(
            "Best price you've seen", 0.0, step=0.01, key=f"vb_{k}", format="%.2f"
        )
        if price:
            ev = r.cal_prob * price - 1
            rows.append(
                {
                    "fixture": r.fixture,
                    "market": r.market,
                    "model": r.cal_prob,
                    "fair": r.fair_odds,
                    "price": price,
                    "EV%": 100 * ev,
                }
            )

if rows:
    out = pd.DataFrame(rows).sort_values("EV%", ascending=False)
    st.subheader("Ranked by EV")
    value = out[out["EV%"] > 3.0]
    st.dataframe(
        value.round(2) if len(value) else out.round(2), hide_index=True, use_container_width=True
    )
    st.caption(
        f"{len(value)} selection(s) over the 3% EV bar out of {len(out)} priced. "
        "If this list is ever long, treat it as evidence the model is broken."
    )
else:
    st.caption("Enter a price on any line above to rank it.")
