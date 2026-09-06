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
st.page_link("pages/1_Bet_Builder.py", label="🎯 Bet Builder pricer", icon=None)
st.write(
    "- **Bet Builder** — price a same-game multi by joint simulation, see each "
    "leg's standalone probability next to the correlation-aware joint price, and "
    "check a bookmaker's price for value."
)
st.caption(
    "Value Bets, High Confidence, Stats Browser, Acca Builder and Bet Log arrive in later phases."
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
