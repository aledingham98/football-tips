"""Acca Builder - legs across separate fixtures, compound price, margin, Kelly,
bankroll drawdown simulation.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from _sim import STANDARD_MARKETS, sim_for

import data_access
from config.loader import league_table
from pricing.acca import AccaLeg, drawdown_simulation, price_acca
from simulation.markets import probability

st.set_page_config(page_title="Acca Builder", page_icon="🧮", layout="centered")
st.title("🧮 Acca Builder")

R = data_access.ratings()
if R is None:
    st.error("No fitted ratings yet.")
    st.stop()
LT = league_table()
TEAM_OPTS = data_access.team_options()
MARKET_LABELS = [lbl for lbl, _ in STANDARD_MARKETS]
LEG_BY_LABEL = dict(STANDARD_MARKETS)

st.session_state.setdefault("acca", [])

st.subheader("Add a leg")
lg = st.selectbox(
    "League", [c for c in TEAM_OPTS if TEAM_OPTS[c]], format_func=lambda c: LT[c]["name"]
)
teams = TEAM_OPTS[lg]
tier = LT[lg]["tier"]
c1, c2 = st.columns(2)
h = c1.selectbox("Home", teams, key="acca_h")
a = c2.selectbox("Away", teams, index=min(1, len(teams) - 1), key="acca_a")
mkt = st.selectbox("Market", MARKET_LABELS, key="acca_mkt")
if h != a and st.button("Add leg", use_container_width=True):
    res, _ = sim_for(h, a, lg, tier)
    p = probability(res, LEG_BY_LABEL[mkt])
    st.session_state["acca"].append({"fixture": f"{h} v {a}", "label": mkt, "prob": float(p)})
    st.rerun()

if not st.session_state["acca"]:
    st.info("Add at least two legs in different fixtures.")
    st.stop()

st.subheader("Legs")
book_legs: list[AccaLeg] = []
for i, d in enumerate(st.session_state["acca"]):
    cc1, cc2, cc3 = st.columns([5, 2, 1])
    cc1.write(f"**{i + 1}.** {d['fixture']} — {d['label']}  ·  {d['prob']:.1%}")
    leg_book = cc2.number_input("book", 0.0, step=0.05, key=f"lb_{i}", label_visibility="collapsed")
    if cc3.button("✕", key=f"acc_rm_{i}"):
        st.session_state["acca"].pop(i)
        st.rerun()
    book_legs.append(AccaLeg(d["fixture"], d["label"], d["prob"], leg_book or None))

acca_price = st.number_input(
    "Bookmaker's acca price (decimal, 0 to skip)", 0.0, step=0.1, format="%.2f"
)
q = price_acca(book_legs, book_odds=acca_price or None)

st.subheader("Price")
m1, m2 = st.columns(2)
m1.metric("Model probability", f"{q.compound_prob:.2%}")
m2.metric("Fair odds", f"{q.fair_odds:.2f}")
if q.total_margin_vs_price is not None:
    st.metric(
        "Bookmaker margin in your price",
        f"{q.total_margin_vs_price:+.1%}",
        help="How much the acca price is shaded vs the model's fair odds.",
    )
if q.total_margin_from_legs is not None:
    st.caption(
        f"Compounded from the per-leg book prices you entered: {q.total_margin_from_legs:+.1%}"
    )
if q.verdict:
    style = {
        "VALUE": st.success,
        "marginal": st.info,
        "no value": st.warning,
        "bad price": st.error,
    }[q.verdict]
    style(f"**{q.verdict.upper()}** — EV {q.ev * 100:+.1f}%, edge {q.edge:+.1%}")
st.caption(
    f"{q.n_legs} legs. Accumulator legs sit in separate matches, so probabilities "
    f"multiply — and so does the margin. Five ~5%-margin legs compound to ~23%."
)

# --------------------------------------------------------------------------- #
st.subheader("Bankroll drawdown")
if not acca_price:
    st.caption("Enter an acca price above to simulate staking it repeatedly.")
else:
    cc1, cc2, cc3 = st.columns(3)
    frac = cc1.select_slider("Kelly fraction", [0.1, 0.25, 0.5, 1.0], value=0.25)
    nbets = cc2.select_slider("Bets", [50, 100, 200, 500], value=200)
    kelly_full = max(0.0, (q.compound_prob * acca_price - 1) / (acca_price - 1))
    stake = kelly_full * frac
    cc3.metric("Stake / bet", f"{stake:.1%}")
    if stake <= 0:
        st.warning("Full Kelly is ≤ 0 here (no edge) — nothing to simulate.")
    else:
        dd = drawdown_simulation(
            q.compound_prob, acca_price, stake, n_bets=int(nbets), n_paths=4000
        )
        d1, d2, d3 = st.columns(3)
        d1.metric("Median bankroll", f"×{dd.median_final:.2f}")
        d2.metric("5th percentile", f"×{dd.p05_final:.2f}")
        d3.metric("P(down 50%+ ever)", f"{dd.prob_bust_50:.0%}")
        d4, d5 = st.columns(2)
        d4.metric("Median max drawdown", f"{dd.median_max_drawdown:.0%}")
        d5.metric("Worst path drawdown", f"{dd.worst_max_drawdown:.0%}")
        chart = pd.DataFrame(dd.sample_paths.T)
        chart.index.name = "bet"
        st.line_chart(chart, height=240)
        st.caption(
            f"{chart.shape[1]} sample bankroll paths. Full Kelly on multiples is a bankroll-killer — quarter-Kelly default."
        )
