"""Vectorised Monte-Carlo match simulator.

One call to :func:`simulate` produces ``n_sims`` complete synthetic matches as a
:class:`~simulation.state.SimResult`. Every market is then a count over those
rows (see :mod:`simulation.markets`) - joint markets included, which is how
same-match multi correlation is captured without ever multiplying leg
probabilities.

Design rules:
* No Python loop over simulations. The only loops are over *event slots*
  (<= ~40 iterations), each body a vectorised NumPy op over all ``n_sims`` rows.
* Target: 50_000 sims of one fixture in well under 1 second.
"""

from __future__ import annotations

import numpy as np

from models.dixon_coles import score_matrix
from simulation.state import MatchInputs, SimResult, TeamSim, TeamSimResult

NDArray = np.ndarray

# Array-width caps (see config/model.yaml -> simulation).
MAX_SHOTS = 40
# Must be > the Dixon-Coles score-grid max (15) so every drawn team score can be
# fully attributed to players; a lower cap would silently drop goals.
MAX_GOALS_ALLOC = 16
MAX_CARDS_ALLOC = 8

# Dixon-Coles score grid runs 0..SCORE_GRID_MAX per team.
SCORE_GRID_MAX = 15
assert MAX_GOALS_ALLOC > SCORE_GRID_MAX


# --------------------------------------------------------------------------- #
# low-level vectorised primitives
# --------------------------------------------------------------------------- #
def draw_negbin(mean: NDArray, k: float, rng: np.random.Generator) -> NDArray:
    """Negative-binomial counts with the given per-row ``mean`` and dispersion ``k``.

    Over-dispersed relative to Poisson (matches real shot / corner data);
    ``k -> inf`` recovers Poisson.
    """
    mean = np.maximum(np.asarray(mean, dtype=np.float64), 1e-6)
    p = k / (k + mean)
    return rng.negative_binomial(k, p)


def allocate_slots(
    totals: NDArray, weights: NDArray, cap: int, rng: np.random.Generator
) -> tuple[NDArray, NDArray]:
    """Assign ``totals[i]`` discrete events in row ``i`` to columns of ``weights``.

    ``weights`` is ``(N, P)`` (per-sim) or ``(P,)`` (shared). Returns
    ``(picks, valid)`` each ``(N, cap)``: ``picks[i, j]`` is the column the
    j-th event in row i landed on, ``valid[i, j]`` is ``j < totals[i]``.

    Sampling is inverse-CDF via a broadcast compare. The loop is over ``cap``
    (<= 40), not over ``N``.
    """
    n = totals.shape[0]
    if weights.ndim == 1:
        weights = np.broadcast_to(weights, (n, weights.shape[0]))
    p = weights.shape[1]
    wsum = weights.sum(axis=1, keepdims=True)
    wsum = np.where(wsum > 0, wsum, 1.0)
    cdf = np.cumsum(weights / wsum, axis=1)
    cdf[:, -1] = 1.0  # guard against fp drift so searchsorted never overflows

    picks = np.zeros((n, cap), dtype=np.int32)
    max_needed = int(totals.max()) if n else 0
    for j in range(min(cap, max_needed)):
        u = rng.random((n, 1))
        picks[:, j] = np.clip((u > cdf).sum(axis=1), 0, p - 1)
    valid = np.arange(cap)[None, :] < totals[:, None]
    return picks, valid


def counts_from_slots(picks: NDArray, valid: NDArray, n_players: int) -> NDArray:
    """Collapse ``(N, cap)`` slot picks into ``(N, n_players)`` per-player counts."""
    n = picks.shape[0]
    flat = np.arange(n)[:, None] * n_players + picks
    return np.bincount(flat[valid], minlength=n * n_players).reshape(n, n_players).astype(np.int32)


def _minute_from_buckets(
    bucket_weights: NDArray, shape: tuple[int, ...], rng: np.random.Generator
) -> NDArray:
    """Draw goal minutes in ``[0, 90]`` from six 15-minute bucket weights."""
    w = bucket_weights / bucket_weights.sum()
    bucket = rng.choice(6, size=shape, p=w)
    within = rng.random(shape)
    return bucket * 15.0 + within * 15.0


# --------------------------------------------------------------------------- #
# per-team pieces
# --------------------------------------------------------------------------- #
def simulate_minutes(team: TeamSim, n: int, rng: np.random.Generator) -> NDArray:
    """``(n, P)`` minutes played. Starters may be subbed off; non-starters may
    come on. Substitution *timing* is drawn, not a point estimate."""
    p = team.n_players
    starts = rng.random((n, p)) < team.start_prob[None, :]
    full = rng.random((n, p)) < team.play_full_prob[None, :]
    off_time = np.clip(
        rng.normal(team.sub_off_mean[None, :], team.sub_off_sd[None, :], (n, p)), 1.0, 90.0
    )
    start_minutes = np.where(full, 90.0, off_time)

    came_on = (~starts) & (rng.random((n, p)) < team.sub_on_prob[None, :])
    on_time = np.clip(
        rng.normal(team.sub_on_mean[None, :], team.sub_on_sd[None, :], (n, p)), 1.0, 89.0
    )
    cameo = 90.0 - on_time

    return np.where(starts, start_minutes, np.where(came_on, cameo, 0.0))


def simulate_team_shots(
    exp_shots: float,
    own_goals: NDArray,
    opp_goals: NDArray,
    k: float,
    game_state_beta: float,
    rng: np.random.Generator,
) -> NDArray:
    """Team shot count. A mild game-state term lifts the shot rate for the team
    that ended up chasing the game (drawn margin used as the proxy)."""
    gs = np.exp(game_state_beta * (opp_goals - own_goals))
    gs = np.clip(gs, 0.75, 1.4)
    return draw_negbin(exp_shots * gs, k, rng)


def _team_result(
    team: TeamSim,
    minutes: NDArray,
    team_goals: NDArray,
    team_shots: NDArray,
    *,
    yellow_rate: float,
    red_rate_team: float,
    second_yellow_prob: float,
    assist_fraction: float,
    rng: np.random.Generator,
) -> tuple[TeamSimResult, tuple[NDArray, NDArray]]:
    """Shots -> SoT -> (scorers within the DC score) -> assists -> cards for one team.

    Returns the per-player result plus the ``(n, MAX_GOALS_ALLOC)`` goal-slot
    scorer picks + validity (needed for goal-minute attribution).
    """
    n, p = minutes.shape
    mins_frac = minutes / 90.0

    # --- shots to players: share weighted by time on pitch ---
    shot_w = team.shot_share[None, :] * mins_frac
    shot_w = np.where(shot_w.sum(axis=1, keepdims=True) > 0, shot_w, mins_frac + 1e-9)
    shot_picks, shot_valid = allocate_slots(team_shots, shot_w, MAX_SHOTS, rng)
    player_shots = counts_from_slots(shot_picks, shot_valid, p)

    # --- shots on target: per-shot Bernoulli with the shooter's SoT rate ---
    sot_hit = (rng.random((n, MAX_SHOTS)) < team.sot_rate[shot_picks]) & shot_valid
    player_sot = counts_from_slots(shot_picks, sot_hit, p)

    # --- goals: allocate exactly the DC-drawn team total to players ---
    goal_w = player_sot * team.finish_rate[None, :] + 1e-6 * mins_frac * (
        team.shot_share[None, :] + 1e-9
    )
    goal_w = np.where(goal_w.sum(axis=1, keepdims=True) > 0, goal_w, mins_frac + 1e-9)
    goal_picks, goal_valid = allocate_slots(team_goals, goal_w, MAX_GOALS_ALLOC, rng)
    player_goals = counts_from_slots(goal_picks, goal_valid, p)

    # --- assists: a fraction of goals get one, from a team-mate ---
    assist_w = team.assist_weight[None, :] * mins_frac
    assist_w = np.where(assist_w.sum(axis=1, keepdims=True) > 0, assist_w, mins_frac + 1e-9)
    assist_picks, _ = allocate_slots(
        np.full(n, MAX_GOALS_ALLOC, dtype=np.int64), assist_w, MAX_GOALS_ALLOC, rng
    )
    assisted = goal_valid & (rng.random((n, MAX_GOALS_ALLOC)) < assist_fraction)
    assisted &= assist_picks != goal_picks  # a team-mate, not the scorer (solo goal otherwise)
    player_assists = counts_from_slots(assist_picks, assisted, p)

    # --- cards: referee/team yellow rate, allocate by foul propensity x minutes ---
    yellows_total = rng.poisson(np.full(n, yellow_rate, dtype=np.float64))
    card_w = team.foul_weight[None, :] * mins_frac
    card_w = np.where(card_w.sum(axis=1, keepdims=True) > 0, card_w, mins_frac + 1e-9)
    yc_picks, yc_valid = allocate_slots(yellows_total, card_w, MAX_CARDS_ALLOC, rng)
    player_yellows = counts_from_slots(yc_picks, yc_valid, p)

    reds_total = rng.poisson(np.full(n, red_rate_team, dtype=np.float64))
    rc_picks, rc_valid = allocate_slots(reds_total, card_w, MAX_CARDS_ALLOC, rng)
    player_reds = counts_from_slots(rc_picks, rc_valid, p)
    # second yellow -> red for an already-booked player
    second = (player_yellows >= 1) & (rng.random((n, p)) < second_yellow_prob)
    player_reds = player_reds + second.astype(np.int32)

    result = TeamSimResult(
        minutes=minutes,
        shots=player_shots,
        sot=player_sot,
        goals=player_goals,
        assists=player_assists,
        yellows=player_yellows,
        reds=player_reds,
        index=dict(team.index),
    )
    return result, (goal_picks, goal_valid)


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def simulate(mi: MatchInputs, n_sims: int = 50_000, seed: int | None = None) -> SimResult:
    """Simulate ``n_sims`` complete matches for the fixture described by ``mi``."""
    rng = np.random.default_rng(seed)

    # 1. final score - sample the DC-corrected joint PMF in one shot.
    sm = score_matrix(mi.lambda_home, mi.lambda_away, mi.rho, SCORE_GRID_MAX)
    idx = rng.choice(sm.size, size=n_sims, p=sm.ravel())
    home_goals, away_goals = np.divmod(idx, sm.shape[1])
    home_goals = home_goals.astype(np.int32)
    away_goals = away_goals.astype(np.int32)

    # 2. minutes.
    home_minutes = simulate_minutes(mi.home, n_sims, rng)
    away_minutes = simulate_minutes(mi.away, n_sims, rng)

    # 3. team shot volume (game-state adjusted by the drawn margin).
    home_shots = simulate_team_shots(
        mi.exp_shots_home,
        home_goals,
        away_goals,
        mi.shots_dispersion_k,
        mi.shots_game_state_beta,
        rng,
    )
    away_shots = simulate_team_shots(
        mi.exp_shots_away,
        away_goals,
        home_goals,
        mi.shots_dispersion_k,
        mi.shots_game_state_beta,
        rng,
    )

    # 4-8. per-team shots/SoT/scorers/assists/cards.
    red_rate_team = mi.red_rate_match / 2.0
    home_res, (_, h_goal_valid) = _team_result(
        mi.home,
        home_minutes,
        home_goals,
        home_shots,
        yellow_rate=mi.yellow_rate_home,
        red_rate_team=red_rate_team,
        second_yellow_prob=mi.second_yellow_prob,
        assist_fraction=mi.assist_fraction,
        rng=rng,
    )
    away_res, (_, a_goal_valid) = _team_result(
        mi.away,
        away_minutes,
        away_goals,
        away_shots,
        yellow_rate=mi.yellow_rate_away,
        red_rate_team=red_rate_team,
        second_yellow_prob=mi.second_yellow_prob,
        assist_fraction=mi.assist_fraction,
        rng=rng,
    )

    # 9. goal minutes -> half-time scores.
    h_goal_min = np.where(
        h_goal_valid, _minute_from_buckets(mi.goal_bucket_weights, h_goal_valid.shape, rng), -1.0
    )
    a_goal_min = np.where(
        a_goal_valid, _minute_from_buckets(mi.goal_bucket_weights, a_goal_valid.shape, rng), -1.0
    )
    home_ht_goals = ((h_goal_min >= 0) & (h_goal_min <= 45.0)).sum(axis=1).astype(np.int32)
    away_ht_goals = ((a_goal_min >= 0) & (a_goal_min <= 45.0)).sum(axis=1).astype(np.int32)

    # 10. corners (team-level, game-state adjusted).
    home_corners = simulate_team_shots(
        mi.exp_corners_home,
        home_goals,
        away_goals,
        mi.corners_dispersion_k,
        mi.corners_game_state_beta,
        rng,
    )
    away_corners = simulate_team_shots(
        mi.exp_corners_away,
        away_goals,
        home_goals,
        mi.corners_dispersion_k,
        mi.corners_game_state_beta,
        rng,
    )

    return SimResult(
        n_sims=n_sims,
        home_goals=home_goals,
        away_goals=away_goals,
        home_ht_goals=home_ht_goals,
        away_ht_goals=away_ht_goals,
        home_shots=home_shots.astype(np.int32),
        away_shots=away_shots.astype(np.int32),
        home_corners=home_corners.astype(np.int32),
        away_corners=away_corners.astype(np.int32),
        home_goal_minutes=h_goal_min,
        away_goal_minutes=a_goal_min,
        home=home_res,
        away=away_res,
        inputs=mi,
        seed=seed,
    )
