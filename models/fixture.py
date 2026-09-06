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
from simulation.state import MatchInputs, TeamSim
from simulation.synthetic import synthetic_team

# league-average conversion ratios (rough, from public aggregates) - used to turn
# an expected-goals rate into shot / corner volume when FBref team data is absent.
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

    derby_mult = cards_cfg.get("derby_multiplier", 1.15) if derby else 1.0
    ref_yellow = cards_cfg.get("ref_yellow_rate_default", 1.9)

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
        exp_shots_home=lam_h * _SHOTS_PER_GOAL,
        exp_shots_away=lam_a * _SHOTS_PER_GOAL,
        shots_dispersion_k=shots_cfg.get("dispersion_k", 12.0),
        shots_game_state_beta=shots_cfg.get("game_state_beta", 0.08),
        yellow_rate_home=ref_yellow * derby_mult,
        yellow_rate_away=ref_yellow * derby_mult,
        red_rate_match=cards_cfg.get("red_rate_default", 0.11) * 2.0,
        second_yellow_prob=cards_cfg.get("second_yellow_prob", 0.06),
        exp_corners_home=lam_h * _CORNERS_PER_GOAL + 2.0,
        exp_corners_away=lam_a * _CORNERS_PER_GOAL + 2.0,
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
