"""Stats Browser - team pages from the fitted ratings + match history.

Player pages arrive with the FBref nightly data; team pages work on what's
committed now.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import data_access
from config.loader import league_table

st.set_page_config(page_title="Stats Browser", page_icon="📊", layout="centered")
st.title("📊 Stats Browser")

R = data_access.ratings()
M = data_access.matches(data_access._mtime(data_access.MATCHES_PATH))
if R is None:
    st.error("No fitted ratings yet.")
    st.stop()

LT = league_table()
tier_to_league = {v["tier"]: k for k, v in LT.items()}

tab_team, tab_league, tab_player = st.tabs(["Team", "League table", "Players"])

# --------------------------------------------------------------------------- #
with tab_team:
    team = st.selectbox("Team", sorted(R.table.index))
    row = R.table.loc[team]
    tier = int(row["tier"])
    league = tier_to_league.get(tier, "EPL")
    st.caption(
        f"{LT[league]['name']} · fitted on {R.meta.get('n_matches', '?')} matches "
        f"to {R.meta.get('data_through', R.meta.get('date_max', '?'))}"
    )

    within = R.table[R.table["tier"] == tier]
    strength = (within["atk"] + within["dfn"]).sort_values(ascending=False)
    rank = list(strength.index).index(team) + 1

    c1, c2, c3 = st.columns(3)
    c1.metric("League rank (model)", f"{rank} / {len(within)}")
    c2.metric("Attack", f"{row['atk']:+.2f}", help="log goal-rate deviation; higher = scores more")
    c3.metric("Defence", f"{row['dfn']:+.2f}", help="higher = concedes fewer")
    c4, c5 = st.columns(2)
    c4.metric("Home tilt (attack)", f"{row['atk_ha']:+.2f}")
    c5.metric("Home tilt (defence)", f"{row['dfn_ha']:+.2f}")

    # implied goal expectation vs an average team in the same tier, at home / away
    avg_atk, avg_dfn = R.tier_atk.get(tier, 0.0), R.tier_dfn.get(tier, 0.0)
    lh = np.exp(R.mu + R.gamma + (row["atk"] + row["atk_ha"]) - (avg_dfn))
    la_home = np.exp(R.mu + (avg_atk) - (row["dfn"] + row["dfn_ha"]))
    lh_away = np.exp(R.mu + (row["atk"] - row["atk_ha"]) - (avg_dfn))
    la_away = np.exp(R.mu + R.gamma + (avg_atk) - (row["dfn"] - row["dfn_ha"]))
    st.write("**Vs an average team in this division**")
    st.write(
        f"- Home: expect **{lh:.2f}** – {la_home:.2f}\n"
        f"- Away: expect {lh_away:.2f} – **{la_away:.2f}**"
    )

    if M is not None:
        played = M[(M["home_team"] == team) | (M["away_team"] == team)].sort_values("date")
        if len(played):
            gf = np.where(played["home_team"] == team, played["fthg"], played["ftag"])
            ga = np.where(played["home_team"] == team, played["ftag"], played["fthg"])
            res = np.where(gf > ga, "W", np.where(gf < ga, "L", "D"))
            last10 = res[-10:]
            st.write("**Recent form (last 10):** " + " ".join(last10[::-1]))
            f1, f2, f3 = st.columns(3)
            f1.metric(
                "Last-10 pts/game",
                f"{np.mean(np.where(last10 == 'W', 3, np.where(last10 == 'D', 1, 0))):.2f}",
            )
            f2.metric("Goals for /game", f"{gf[-10:].mean():.2f}")
            f3.metric("Goals against /game", f"{ga[-10:].mean():.2f}")

            home_g = played[played["home_team"] == team]
            away_g = played[played["away_team"] == team]
            split = pd.DataFrame(
                {
                    "": ["Home", "Away"],
                    "P": [len(home_g), len(away_g)],
                    "GF/g": [home_g["fthg"].mean(), away_g["ftag"].mean()],
                    "GA/g": [home_g["ftag"].mean(), away_g["fthg"].mean()],
                }
            )
            st.write("**Home / away split (all committed history)**")
            st.dataframe(split.round(2), hide_index=True, use_container_width=True)

# --------------------------------------------------------------------------- #
with tab_league:
    lg = st.selectbox("Division", list(LT), format_func=lambda c: LT[c]["name"], key="lg_tbl")
    tier = LT[lg]["tier"]
    sub = R.table[R.table["tier"] == tier].copy()
    sub["strength"] = sub["atk"] + sub["dfn"]
    sub = sub.sort_values("strength", ascending=False)
    show = sub.reset_index()[["team", "atk", "dfn", "atk_ha", "strength"]].round(3)
    show.index = np.arange(1, len(show) + 1)
    st.dataframe(show, use_container_width=True)
    st.caption(
        "Model strength = attack + defence (opponent-adjusted, time-decayed). Not a form table."
    )

# --------------------------------------------------------------------------- #
with tab_player:
    st.info(
        "Player pages need the FBref nightly pull (per-90 shots, SoT, goals, "
        "assists, cards, minutes with sample-size warnings). Until that data is "
        "committed, player-prop legs in the Bet Builder use a synthetic squad "
        "scaled by team strength and are indicative only."
    )
