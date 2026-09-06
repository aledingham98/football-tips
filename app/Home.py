"""football-tips - personal EPL/EFL bet-builder & accumulator value pricer.

Mobile-first Streamlit app. Reads only committed parquet/JSON; never calls a
network API. Deployed on Streamlit Community Cloud.
"""

from __future__ import annotations

import streamlit as st

import data_access

st.set_page_config(page_title="football-tips", page_icon="⚽", layout="centered")

st.title("⚽ football-tips")
st.caption(
    "Personal value pricer for Premier League, Championship, League One & Two. "
    "One Monte-Carlo match simulator; every market is a count over simulations."
)

ratings = data_access.ratings()
if ratings is None:
    st.error(
        "No fitted ratings found. The nightly GitHub Action writes "
        "`data/model/ratings.parquet`; run it (Actions → nightly-data → Run workflow) "
        "or `uv run python -m scripts.fit_ratings` locally."
    )
    st.stop()

fresh = data_access.data_freshness()
c1, c2 = st.columns(2)
c1.metric("Data through", fresh.get("data_through", "?"))
c2.metric("Teams rated", fresh.get("n_teams", "?"))
st.caption(
    f"Ratings fitted {fresh.get('ratings_fitted', '?')} UTC on "
    f"{fresh.get('n_matches', '?')} matches · rho={ratings.rho:+.3f} · "
    f"home edge={ratings.gamma:+.3f} · tempo var={ratings.tempo_var:.4f}"
)

st.divider()
st.subheader("Pages")
st.page_link(
    "pages/1_Bet_Builder.py", label="🎯 Bet Builder — price a same-game multi by joint simulation"
)
st.page_link(
    "pages/2_Value_Bets.py", label="💰 Value Bets — EV > 3%, ranked (manual odds until Phase 4)"
)
st.page_link(
    "pages/3_High_Confidence.py", label="✅ High Confidence — most likely outcomes (likely ≠ value)"
)
st.page_link(
    "pages/4_Stats_Browser.py", label="📊 Stats Browser — team ratings, form, home/away splits"
)
st.page_link(
    "pages/5_Acca_Builder.py", label="🧮 Acca Builder — compound price, margin, Kelly, drawdown"
)
st.page_link(
    "pages/6_Bet_Log.py", label="📒 Bet Log — record bets, closing-line value, calibration"
)
st.caption(
    "Player props use a synthetic squad until FBref player data is committed; "
    "Value Bets auto-ranks once the Phase 4 odds feed lands."
)

st.divider()
metrics = data_access.calibration_metrics(data_access._mtime(data_access.CALIB_METRICS_PATH))
if metrics is not None and len(metrics):
    st.subheader("Held-out calibration (last complete season)")
    show = metrics.reset_index()[["market", "n", "base_rate", "mean_pred", "brier", "ece"]]
    show = show.rename(columns={"base_rate": "actual", "mean_pred": "model"})
    st.dataframe(show.round(3), hide_index=True, use_container_width=True)
    st.caption(
        "1X2 markets are well calibrated (ECE ≲ 4%). Goals markets read a few points "
        "low on Championship — a data-recency artifact the nightly job narrows as "
        "history and xG accumulate. Not a tipster: an empty value list is normal."
    )
