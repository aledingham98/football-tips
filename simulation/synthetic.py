"""Plausible synthetic :class:`MatchInputs` for tests and for the UI demo mode
before real fitted data is committed. Not used in production pricing.
"""

from __future__ import annotations

import numpy as np

from simulation.state import MatchInputs, Pos, TeamSim

_FORMATION = [Pos.GK] + [Pos.DEF] * 4 + [Pos.MID] * 3 + [Pos.FWD] * 3
_BENCH = [Pos.GK, Pos.DEF, Pos.MID, Pos.FWD, Pos.FWD]

# per-position priors: (shot_share_weight, sot_rate, finish_rate, assist_weight, foul_weight)
_POS_PRIOR = {
    Pos.GK: (0.001, 0.20, 0.02, 0.2, 0.4),
    Pos.DEF: (0.6, 0.28, 0.05, 0.7, 1.3),
    Pos.MID: (1.1, 0.32, 0.08, 1.4, 1.1),
    Pos.FWD: (2.4, 0.37, 0.13, 1.1, 0.8),
}


def synthetic_team(
    name: str, strength: float = 1.0, rng: np.random.Generator | None = None
) -> TeamSim:
    rng = rng or np.random.default_rng(abs(hash(name)) % (2**32))
    positions = _FORMATION + _BENCH
    n = len(positions)
    names = [f"{name} {p.name}{i}" for i, p in enumerate(positions)]

    start_prob = np.array([0.92 if i < 11 else 0.22 for i in range(n)])
    play_full = np.array([0.55 if i < 11 else 0.05 for i in range(n)])
    sub_off_mean = np.full(n, 68.0)
    sub_off_sd = np.full(n, 16.0)
    sub_on_prob = np.array([0.10 if i < 11 else 0.42 for i in range(n)])
    sub_on_mean = np.full(n, 66.0)
    sub_on_sd = np.full(n, 14.0)

    shot_w = np.array([_POS_PRIOR[p][0] for p in positions]) * rng.uniform(0.7, 1.3, n) * strength
    shot_share = shot_w / shot_w.sum()
    sot_rate = np.clip([_POS_PRIOR[p][1] for p in positions] + rng.normal(0, 0.03, n), 0.1, 0.6)
    finish_rate = np.clip(
        [_POS_PRIOR[p][2] for p in positions] + rng.normal(0, 0.015, n), 0.02, 0.30
    )
    assist_weight = np.array([_POS_PRIOR[p][3] for p in positions]) * rng.uniform(0.7, 1.3, n)
    foul_weight = np.array([_POS_PRIOR[p][4] for p in positions]) * rng.uniform(0.7, 1.3, n)

    return TeamSim(
        player_names=names,
        position=np.array([int(p) for p in positions]),
        start_prob=start_prob,
        play_full_prob=play_full,
        sub_off_mean=sub_off_mean,
        sub_off_sd=sub_off_sd,
        sub_on_prob=sub_on_prob,
        sub_on_mean=sub_on_mean,
        sub_on_sd=sub_on_sd,
        shot_share=shot_share,
        sot_rate=sot_rate,
        finish_rate=finish_rate,
        assist_weight=assist_weight,
        foul_weight=foul_weight,
    )


def synthetic_match_inputs(
    lambda_home: float = 1.55,
    lambda_away: float = 1.15,
    rho: float = -0.10,
    *,
    home_team: str = "Home FC",
    away_team: str = "Away FC",
    league: str = "EPL",
    seed: int | None = 7,
) -> MatchInputs:
    rng = np.random.default_rng(seed)
    return MatchInputs(
        home_team=home_team,
        away_team=away_team,
        league=league,
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        rho=rho,
        home=synthetic_team(home_team, strength=1.05, rng=rng),
        away=synthetic_team(away_team, strength=0.95, rng=rng),
        exp_shots_home=13.5,
        exp_shots_away=10.8,
        shots_dispersion_k=12.0,
        shots_game_state_beta=0.08,
        yellow_rate_home=1.9,
        yellow_rate_away=2.0,
        red_rate_match=0.11,
        second_yellow_prob=0.06,
        exp_corners_home=5.9,
        exp_corners_away=4.6,
        corners_dispersion_k=14.0,
        corners_game_state_beta=0.05,
        assist_fraction=0.78,
    )
