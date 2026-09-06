"""Bet Builder pricer - the priority page.

Pick a fixture, add legs, see the correlation-aware joint price next to each
leg's standalone probability, then check a bookmaker's price for value.
"""

from __future__ import annotations

import streamlit as st

import data_access
from config.loader import league_table, markets_config
from models.fixture import build_match_inputs, default_n_sims
from pricing.builder import quote_builder
from simulation.engine import simulate
from simulation.markets import Leg

st.set_page_config(page_title="Bet Builder", page_icon="🎯", layout="centered")
st.title("🎯 Bet Builder pricer")

R = data_access.ratings()
if R is None:
    st.error("No fitted ratings. Run the nightly job or `scripts.fit_ratings`.")
    st.stop()

MCFG = data_access.get_model_config()
MARKET_LIST = markets_config()["markets"]
TEAM_OPTS = data_access.team_options()
LEAGUES = league_table()
PRICE_SEED = 20260906  # fixed so the quote is stable across reruns

# --------------------------------------------------------------------------- #
# 1. fixture
# --------------------------------------------------------------------------- #
st.subheader("1 · Fixture")
leagues = [lg for lg in TEAM_OPTS if TEAM_OPTS[lg]]
league = st.selectbox("League", leagues, format_func=lambda c: LEAGUES[c]["name"])
teams = TEAM_OPTS[league]
tier = LEAGUES[league]["tier"]

col_h, col_a = st.columns(2)
home = col_h.selectbox("Home", teams, index=0)
away = col_a.selectbox("Away", teams, index=min(1, len(teams) - 1))
if home == away:
    st.warning("Pick two different teams.")
    st.stop()


@st.cache_resource(show_spinner="Simulating 50,000 matches…")
def run_sim(home: str, away: str, league: str, tier: int, n_sims: int, _mtime: float):
    mi = build_match_inputs(R, home, away, league=league, tier=tier, model_cfg=MCFG)
    return simulate(mi, n_sims=n_sims, seed=PRICE_SEED), mi


res, mi = run_sim(
    home, away, league, tier, default_n_sims(MCFG), data_access._mtime(data_access.RATINGS_PATH)
)

p_h = max(float((res.home_goals > res.away_goals).mean()), 1e-9)
p_d = max(float((res.home_goals == res.away_goals).mean()), 1e-9)
p_a = max(float((res.home_goals < res.away_goals).mean()), 1e-9)
m1, m2, m3 = st.columns(3)
m1.metric(f"{home[:12]} win", f"{p_h:.0%}", f"fair {1 / p_h:.2f}", delta_color="off")
m2.metric("Draw", f"{p_d:.0%}", f"fair {1 / p_d:.2f}", delta_color="off")
m3.metric(f"{away[:12]} win", f"{p_a:.0%}", f"fair {1 / p_a:.2f}", delta_color="off")
st.caption(
    f"Model goal expectation: {mi.lambda_home:.2f} – {mi.lambda_away:.2f}. "
    f"Squads are synthetic (no committed FBref player data yet) — player-prop legs "
    f"are indicative only."
)

# --------------------------------------------------------------------------- #
# 2. legs
# --------------------------------------------------------------------------- #
st.subheader("2 · Legs")
st.session_state.setdefault("legs", [])
fixture_key = f"{league}:{home}:{away}"
if st.session_state.get("fixture_key") != fixture_key:
    st.session_state["legs"] = []
    st.session_state["fixture_key"] = fixture_key


def _leg_inputs(spec: dict) -> Leg | None:
    key, kind = spec["key"], spec["kind"]
    ent = spec["entity"]
    params: dict = {}
    team = None
    if ent == "team":
        team = st.radio("Team", ["home", "away"], horizontal=True, key=f"t_{key}")
        params["team"] = team
    if ent == "player":
        team = st.radio("Team", ["home", "away"], horizontal=True, key=f"pt_{key}")
        squad = mi.home.player_names if team == "home" else mi.away.player_names
        params["team"] = team
        params["player"] = st.selectbox("Player", squad, key=f"pl_{key}")

    if kind == "enum":
        choice = st.selectbox("Selection", spec["options"], key=f"e_{key}")
        params["outcome"] = choice
    elif kind == "boolean":
        if key in ("btts",):
            params["yes"] = (
                st.radio("Selection", ["yes", "no"], horizontal=True, key=f"b_{key}") == "yes"
            )
    elif kind == "line":
        params["line"] = st.selectbox(
            "Line", spec["options"], index=spec["options"].index(spec["default"]), key=f"l_{key}"
        )
        params["side"] = st.radio("Side", ["over", "under"], horizontal=True, key=f"s_{key}")
    elif kind == "threshold":
        params["n"] = st.selectbox("Threshold (N+)", spec["options"], key=f"n_{key}")

    if st.button("Add leg", key=f"add_{key}", use_container_width=True):
        return Leg(market=key, params=params)
    return None


spec_by_label = {m["label"]: m for m in MARKET_LIST}
picked = st.selectbox("Market", list(spec_by_label), key="market_pick")
with st.container(border=True):
    new_leg = _leg_inputs(spec_by_label[picked])
if new_leg is not None:
    st.session_state["legs"].append({"market": new_leg.market, "params": new_leg.params})
    st.rerun()

legs = [Leg(market=d["market"], params=d["params"]) for d in st.session_state["legs"]]
if not legs:
    st.info("Add at least one leg to price the builder.")
    st.stop()

for i, leg in enumerate(legs):
    c1, c2 = st.columns([5, 1])
    c1.write(f"**{i + 1}.** {leg.describe()}")
    if c2.button("✕", key=f"rm_{i}"):
        st.session_state["legs"].pop(i)
        st.rerun()

# --------------------------------------------------------------------------- #
# 3. price
# --------------------------------------------------------------------------- #
st.subheader("3 · Model price")
book_odds = st.number_input(
    "Bookmaker's price for the whole builder (decimal, 0 to skip)",
    min_value=0.0,
    value=0.0,
    step=0.05,
    format="%.2f",
)
q = quote_builder(res, legs, book_odds=book_odds or None)

pcol, fcol = st.columns(2)
pcol.metric(
    "Model probability",
    f"{q.price.joint_prob:.1%}",
    f"±{1.96 * q.price.mc_std_error:.1%} MC",
    delta_color="off",
)
fcol.metric(
    "Fair odds", f"{q.joint_fair_odds:.2f}", f"naive {q.naive_fair_odds:.2f}", delta_color="off"
)

if q.correlation_factor <= 0.90:
    st.warning(q.correlation_note)
elif q.correlation_factor >= 1.10:
    st.info(q.correlation_note)
else:
    st.caption(q.correlation_note)
for w in q.warnings:
    st.warning(w)

with st.expander("Per-leg standalone vs joint", expanded=True):
    for lb in q.legs:
        st.write(
            f"- {lb.label} — **{lb.standalone_prob:.1%}** (fair {lb.standalone_fair_odds:.2f})"
        )
    st.caption(
        f"Product of legs = {q.price.naive_prob:.1%}; joint (this builder) = "
        f"{q.price.joint_prob:.1%}. Multiplying legs is {'too generous' if q.correlation_factor < 1 else 'too harsh'} here."
    )

if q.value is not None:
    st.subheader("4 · Value")
    v = q.value
    verdict_style = {
        "VALUE": st.success,
        "marginal": st.info,
        "no value": st.warning,
        "bad price": st.error,
    }.get(v.verdict, st.info)
    verdict_style(f"**{v.verdict.upper()}** — EV {v.ev_pct:+.1f}%, edge {v.edge:+.1%}")
    s1, s2 = st.columns(2)
    s1.metric("EV", f"{v.ev_pct:+.1f}%")
    s2.metric("¼-Kelly stake", f"{v.kelly_stake:.1%}")
    st.caption(
        "EV = model probability × your price − 1. Bet builders carry 15–30% margin; "
        "most priced builders will not clear the bar, and that is the point."
    )
