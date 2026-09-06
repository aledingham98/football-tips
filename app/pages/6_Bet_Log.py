"""Bet Log - record bets, track closing-line value, calibration by market.

SQLite storage. Streamlit Cloud's filesystem is ephemeral, so use Export / Import
to carry the log between sessions until Supabase is wired (Phase 5).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from storage.bet_log import BetLog, calibration_table, summarise

st.set_page_config(page_title="Bet Log", page_icon="📒", layout="centered")
st.title("📒 Bet Log")

log = BetLog()
df = log.all()

st.warning(
    "Streamlit Cloud resets its disk on redeploy. **Export** the log after a session "
    "and **Import** it next time — or wire Supabase (Phase 5) for real persistence.",
    icon="💾",
)

tab_add, tab_list, tab_analysis, tab_io = st.tabs(
    ["Record", "Open bets", "Analysis", "Export / Import"]
)

with tab_add, st.form("add_bet", clear_on_submit=True):
    fixture = st.text_input("Fixture", placeholder="Arsenal FC v Everton FC")
    market = st.text_input("Market / builder", placeholder="H1 & BTTS & Saka 2+ SoT")
    selection = st.text_input("Selection / notes", "")
    c1, c2 = st.columns(2)
    stake = c1.number_input("Stake (£)", 0.0, step=1.0)
    price = c2.number_input("Price taken", 1.01, step=0.05, format="%.2f")
    model_prob = st.slider("Model probability", 0.0, 1.0, 0.0, 0.005)
    if st.form_submit_button("Add bet", use_container_width=True) and fixture and stake > 0:
        log.add(
            placed_at=dt.datetime.now().isoformat(timespec="minutes"),
            fixture=fixture,
            market=market,
            selection=selection,
            stake=stake,
            price_taken=price,
            model_prob=model_prob or None,
        )
        st.success("Logged.")
        st.rerun()

with tab_list:
    open_bets = df[df["result"] == "pending"]
    if open_bets.empty:
        st.caption("No open bets.")
    for r in open_bets.itertuples(index=False):
        with st.container(border=True):
            st.write(f"**{r.fixture}** — {r.market}")
            st.caption(
                f"£{r.stake:.2f} @ {r.price_taken:.2f} · model {r.model_prob or 0:.0%} · {r.placed_at}"
            )
            c1, c2, c3 = st.columns(3)
            close = c1.number_input(
                "Closing price", 0.0, step=0.05, key=f"cp_{r.id}", format="%.2f"
            )
            outcome = c2.selectbox("Result", ["pending", "won", "lost", "void"], key=f"res_{r.id}")
            if c3.button("Save", key=f"sv_{r.id}"):
                log.update(r.id, closing_price=close or None, result=outcome)
                st.rerun()

with tab_analysis:
    s = summarise(df)
    if s["n"] == 0:
        st.caption("No settled bets yet.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Settled bets", s["n"])
        c2.metric("P/L", f"£{s['profit']:+.2f}")
        c3.metric("ROI", f"{s['roi']:+.1%}")
        st.caption(f"Staked £{s['staked']:.2f} · win rate {s['win_rate']:.0%}")
        st.divider()
        st.write(
            "**Closing-line value** — beating the close is the real edge signal; P/L over a few hundred bets is mostly noise."
        )
        if s["n_clv"]:
            c1, c2 = st.columns(2)
            c1.metric("Mean CLV", f"{s['clv_mean']:+.1%}", help="price taken / closing price − 1")
            c2.metric(
                "Beat the close",
                f"{s['beat_close_rate']:.0%}",
                help=f"of {s['n_clv']} bets with a closing price",
            )
        else:
            st.caption("Add closing prices to open bets to track CLV.")
        st.divider()
        st.write("**Calibration by model probability**")
        cal = calibration_table(df)
        if len(cal):
            plot = cal.set_index("bucket")[["model", "actual"]]
            st.bar_chart(plot)
            st.caption(
                "Model vs realised hit-rate per probability band. Bands should sit near the diagonal once you have enough bets."
            )
        else:
            st.caption("Not enough settled bets with a model probability yet.")

with tab_io:
    st.download_button(
        "⬇️ Export log (CSV)",
        df.to_csv(index=False).encode(),
        file_name=f"bet_log_{dt.date.today()}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    up = st.file_uploader("⬆️ Import a CSV (appends)", type="csv")
    if up is not None and st.button("Import", use_container_width=True):
        n = log.import_csv(pd.read_csv(up))
        st.success(f"Imported {n} rows.")
        st.rerun()
