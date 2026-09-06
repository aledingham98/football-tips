"""Value Bets - EV% = model_prob x best_available_odds - 1.

Auto-ranks against the committed The Odds API snapshot (UK h2h + totals, PL +
Championship). A bet only qualifies as *value* when the model essentially agrees
with the no-vig market consensus and a book is pricing it generously - a
mediocre goals-only model finding a genuine edge looks like "model matches the
market, one bookmaker lags", not "model thinks this is far more likely". Sharp
model/market disagreements are surfaced as probable model error (per the brief),
and fixtures with a just-promoted / just-relegated team are held out.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from _sim import load_slate, value_table

import data_access

st.set_page_config(page_title="Value Bets", page_icon="💰", layout="centered")
st.title("💰 Value Bets")

EV_BAR = 0.03
AGREE = 0.03  # model must be within this of the no-vig consensus
MAX_PLAUSIBLE = 6  # more "value" than this = the model is mispriced

fx = data_access.fixtures(data_access._mtime(data_access.FIXTURES_PATH))
if fx is None or len(fx) == 0:
    st.warning(
        "No committed fixtures yet — the nightly job writes `data/processed/fixtures.parquet`."
    )
    st.stop()

slate = load_slate()
if slate.empty:
    st.warning("No future fixtures in the slate (all kicked off, or teams not in the ratings).")
    st.stop()

odds = data_access.odds_latest(data_access._mtime(data_access.ODDS_LATEST_PATH))
if odds is None or len(odds) == 0:
    st.info("**No odds committed yet.** Run the `odds-refresh` Action, then reload.", icon="🛠️")
    st.stop()

quota = data_access.odds_quota()
vt = value_table(slate, odds)
priced = vt.dropna(subset=["best_odds"]).copy()
cons = priced.dropna(subset=["mkt_novig"])
mad = float(cons["divergence"].abs().mean()) if len(cons) else float("nan")

c1, c2, c3 = st.columns(3)
c1.metric("Odds lines", len(priced))
c2.metric("Credits left", f"{quota.get('remaining', '?')}/500")
c3.metric(
    "Model vs market",
    f"{mad:.1%}" if mad == mad else "—",
    help="mean |model − no-vig consensus|. Under ~5% is as good as a goals-only model gets.",
)

priced = priced.sort_values("ev", ascending=False)
q = priced["ev"] > EV_BAR
value = priced[
    q
    & (priced["divergence"].abs() <= AGREE)
    & (priced["divergence"] >= -0.01)
    & ~priced["low_data"].fillna(False)
]
disagree = priced[q & (priced["divergence"].abs() > AGREE) & ~priced["low_data"].fillna(False)]
held = priced[q & priced["low_data"].fillna(False)]


def _tbl(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["fixture", "market", "model_prob", "mkt_novig", "best_odds", "book", "ev"]].copy()
    out.columns = ["fixture", "market", "model", "consensus", "price", "book", "EV"]
    out["model"] = (out["model"] * 100).round(0).astype("Int64").astype(str) + "%"
    out["consensus"] = (out["consensus"] * 100).round(0).astype("Int64").astype(str) + "%"
    out["EV"] = (out["EV"] * 100).round(1).astype(str) + "%"
    return out.round(2)


st.subheader(f"Value — {len(value)}")
if len(value) > MAX_PLAUSIBLE:
    st.error(
        f"{len(value)} qualifying bets is more than a real edge produces. Treat the model as "
        "mispriced and do not bet off this until it's investigated."
    )
if len(value):
    st.dataframe(_tbl(value), hide_index=True, use_container_width=True)
    st.caption(
        "EV = model probability × best price − 1. These are marginal by nature — size small."
    )
else:
    st.success("Nothing clears the bar that also agrees with the market. That's the normal result.")

if len(disagree):
    with st.expander(
        f"⚠️ Model disagrees with the market — {len(disagree)} (probable model error, not value)"
    ):
        st.dataframe(_tbl(disagree), hide_index=True, use_container_width=True)
        st.caption(
            "Model is > 3pp from the no-vig consensus. The brief's rule: surface as a model problem, not a bet."
        )

if len(held):
    with st.expander(f"Held out — {len(held)} lines with a just-promoted / just-relegated team"):
        st.dataframe(_tbl(held), hide_index=True, use_container_width=True)
        st.caption("The model has too few current-division games for these sides to be trusted.")

with st.expander(f"All {len(priced)} priced lines"):
    allp = priced[
        ["fixture", "market", "model_prob", "mkt_novig", "best_odds", "ev", "low_data"]
    ].copy()
    for col in ("model_prob", "mkt_novig"):
        allp[col] = (allp[col] * 100).round(0)
    allp["ev"] = (allp["ev"] * 100).round(1)
    st.dataframe(allp, hide_index=True, use_container_width=True)

# --- manual entry for props / uncovered lines ---
st.divider()
st.subheader("Manual check")
st.caption("Player props, builders, or lines the feed doesn't carry.")
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
                    "model": round(r.model_prob, 3),
                    "price": p,
                    "EV%": round((r.model_prob * p - 1) * 100, 1),
                }
            )
if rows:
    st.dataframe(
        pd.DataFrame(rows).sort_values("EV%", ascending=False),
        hide_index=True,
        use_container_width=True,
    )
