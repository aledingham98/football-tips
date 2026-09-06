"""Value Bets - EV% = model_prob x best_available_odds - 1, filtered to EV > 3%.

Auto-ranks against the committed The Odds API snapshot (UK, h2h/totals/btts for
PL + Championship). Expect this list to be short or empty most days - that is
correct. If it is ever long, treat it as evidence the model is broken.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from _sim import load_slate, value_table

import data_access

st.set_page_config(page_title="Value Bets", page_icon="💰", layout="centered")
st.title("💰 Value Bets")
st.caption(
    "EV = model probability × best available odds − 1. Filtered to EV > 3% to cut noise. "
    "A short or empty list is the expected outcome."
)

fx = data_access.fixtures(data_access._mtime(data_access.FIXTURES_PATH))
if fx is None or len(fx) == 0:
    st.warning(
        "No committed fixtures yet — the nightly job writes `data/processed/fixtures.parquet`."
    )
    st.stop()

slate = load_slate()
if slate.empty:
    st.warning("Slate empty (fixture teams not in the ratings table yet).")
    st.stop()

odds = data_access.odds_latest(data_access._mtime(data_access.ODDS_LATEST_PATH))
quota = data_access.odds_quota()
vt = value_table(slate, odds)

if odds is None or len(odds) == 0:
    st.info(
        "**No odds committed yet.** The `odds-refresh` GitHub Action fetches The Odds "
        "API twice daily (budget-gated). Run it once (Actions → odds-refresh → Run "
        "workflow), or enter prices manually below.",
        icon="🛠️",
    )
else:
    rem = quota.get("remaining", "?")
    st.caption(f"Odds: {len(odds)} lines · The Odds API credits remaining **{rem}** / 500")

priced = vt.dropna(subset=["best_odds"]).copy()

# --- auto-ranked value ---
if len(priced):
    priced = priced.sort_values("ev", ascending=False)
    value = priced[priced["ev"] > 0.03]
    st.subheader(f"Value (EV > 3%) — {len(value)}")
    if len(value):
        show = value[["fixture", "market", "model_prob", "fair_odds", "best_odds", "book", "ev"]]
        show = show.rename(columns={"model_prob": "model", "best_odds": "price", "ev": "EV"})
        show["model"] = (show["model"] * 100).round(0).astype(int).astype(str) + "%"
        show["EV"] = (show["EV"] * 100).round(1).astype(str) + "%"
        st.dataframe(show.round(2), hide_index=True, use_container_width=True)
    else:
        st.success("Nothing over the 3% bar. That's the normal result.")

    with st.expander(f"All {len(priced)} priced lines"):
        allp = priced[["fixture", "market", "model_prob", "best_odds", "ev"]].copy()
        allp["model_prob"] = (allp["model_prob"] * 100).round(0).astype(int)
        allp["ev"] = (allp["ev"] * 100).round(1)
        st.dataframe(allp, hide_index=True, use_container_width=True)

# --- manual entry for anything the feed didn't cover ---
st.divider()
st.subheader("Manual check")
st.caption("For player props, builders, or lines the feed doesn't carry.")
unpriced = vt[vt["best_odds"].isna()]
lg_sel = st.multiselect(
    "Leagues", sorted(unpriced["league"].unique()), default=sorted(unpriced["league"].unique())
)
rows = []
for r in unpriced[unpriced["league"].isin(lg_sel)].itertuples(index=False):
    with st.expander(
        f"{r.fixture} — {r.market}  ·  model {r.model_prob:.0%} (fair {r.fair_odds:.2f})"
    ):
        p = st.number_input(
            "Price", 0.0, step=0.01, key=f"vb_{r.fixture}_{r.market}", format="%.2f"
        )
        if p:
            rows.append(
                {
                    "fixture": r.fixture,
                    "market": r.market,
                    "model": r.model_prob,
                    "price": p,
                    "EV%": round((r.model_prob * p - 1) * 100, 1),
                }
            )
if rows:
    man = pd.DataFrame(rows).sort_values("EV%", ascending=False)
    st.dataframe(man, hide_index=True, use_container_width=True)
