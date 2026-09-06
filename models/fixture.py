"""Assemble a :class:`~simulation.state.MatchInputs` for a real fixture.

Bridges the fitted :class:`~models.dixon_coles.TeamRatings` and (when available)
FBref squad data into the flat input the engine consumes. Until FBref player data
is committed, squads come from :mod:`simulation.synthetic`, scaled by each team's
rating so player props are at least keyed to real team strength - the brief's
model-only fallback for props.
"""

from __future__ import annotations

import numpy as np

from models.dixon_coles import TeamRatings
from models.referee import RefereeRates
from models.team_rates import TeamRates
from simulation.state import MatchInputs, TeamSim
from simulation.synthetic import synthetic_team

# league-average conversion ratios (rough, from public aggregates) - the fallback
# for turning expected goals into shot / corner volume when no real team-stat
# history exists for a side (currently the EFL).
_SHOTS_PER_GOAL = 9.0
_CORNERS_PER_GOAL = 3.6


def _strength(ratings: TeamRatings, team: str, tier: int | None) -> float:
    atk, dfn, *_ = ratings._team_row(team, tier)
    return atk + dfn


def build_match_inputs(
    ratings: TeamRatings,
    home_team: str,
    away_team: str,
    *,
    league: str,
    tier: int,
    model_cfg: dict,
    home_squad: TeamSim | None = None,
    away_squad: TeamSim | None = None,
    team_rates: TeamRates | None = None,
    referee_rates: RefereeRates | None = None,
    referee: str | None = None,
    derby: bool = False,
    kickoff_iso: str | None = None,
) -> MatchInputs:
    lam_h, lam_a = ratings.lambdas(home_team, away_team, home_tier=tier, away_tier=tier)

    shots_cfg = model_cfg.get("shots", {})
    cards_cfg = model_cfg.get("cards", {})
    corners_cfg = model_cfg.get("corners", {})
    timing_cfg = model_cfg.get("goal_timing", {})

    # squads: real if supplied, else synthetic scaled by team strength
    s_home = 1.0 + 0.25 * np.tanh(_strength(ratings, home_team, tier))
    s_away = 1.0 + 0.25 * np.tanh(_strength(ratings, away_team, tier))
    home = home_squad or synthetic_team(home_team, strength=float(s_home))
    away = away_squad or synthetic_team(away_team, strength=float(s_away))

    # --- shots / corners: real team rates when both sides have a stat history ---
    have_rates = team_rates is not None and team_rates.has(home_team) and team_rates.has(away_team)
    if have_rates:
        exp_shots_h, exp_shots_a = team_rates.expected(home_team, away_team, "shots")
        exp_corners_h, exp_corners_a = team_rates.expected(home_team, away_team, "corners")
    else:
        exp_shots_h, exp_shots_a = lam_h * _SHOTS_PER_GOAL, lam_a * _SHOTS_PER_GOAL
        exp_corners_h = lam_h * _CORNERS_PER_GOAL + 2.0
        exp_corners_a = lam_a * _CORNERS_PER_GOAL + 2.0

    # --- cards: referee base rate x each team's discipline tendency x derby ---
    derby_mult = cards_cfg.get("derby_multiplier", 1.15) if derby else 1.0
    if referee_rates is not None:
        ref_yellow_pt, red_pm = referee_rates.for_referee(referee)
    else:
        ref_yellow_pt = cards_cfg.get("ref_yellow_rate_default", 1.9)
        red_pm = cards_cfg.get("red_rate_default", 0.11) * 2.0
    if have_rates and "yellows" in team_rates.league:
        lg_y = team_rates.league["yellows"]
        h_tend = team_rates.table.at[home_team, "yellows_for"] / lg_y if lg_y else 1.0
        a_tend = team_rates.table.at[away_team, "yellows_for"] / lg_y if lg_y else 1.0
    else:
        h_tend = a_tend = 1.0

    return MatchInputs(
        home_team=home_team,
        away_team=away_team,
        league=league,
        lambda_home=lam_h,
        lambda_away=lam_a,
        rho=ratings.rho,
        tempo_var=ratings.tempo_var,
        home=home,
        away=away,
        exp_shots_home=exp_shots_h,
        exp_shots_away=exp_shots_a,
        shots_dispersion_k=shots_cfg.get("dispersion_k", 12.0),
        shots_game_state_beta=shots_cfg.get("game_state_beta", 0.08),
        yellow_rate_home=ref_yellow_pt * float(np.clip(h_tend, 0.6, 1.6)) * derby_mult,
        yellow_rate_away=ref_yellow_pt * float(np.clip(a_tend, 0.6, 1.6)) * derby_mult,
        red_rate_match=red_pm,
        second_yellow_prob=cards_cfg.get("second_yellow_prob", 0.06),
        exp_corners_home=exp_corners_h,
        exp_corners_away=exp_corners_a,
        corners_dispersion_k=corners_cfg.get("dispersion_k", 14.0),
        corners_game_state_beta=corners_cfg.get("game_state_beta", 0.05),
        assist_fraction=shots_cfg.get("assist_fraction", 0.78),
        goal_bucket_weights=np.array(
            timing_cfg.get("bucket_weights", [0.78, 0.95, 1.15, 1.0, 1.10, 1.30])
        ),
        first_half_stoppage_mean=timing_cfg.get("first_half_stoppage_mean", 2.0),
        second_half_stoppage_mean=timing_cfg.get("second_half_stoppage_mean", 5.0),
        kickoff_iso=kickoff_iso,
    )


def default_n_sims(model_cfg: dict) -> int:
    return int(model_cfg.get("simulation", {}).get("n_sims", 50_000))
