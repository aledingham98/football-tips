"""Data containers passed between the model layer and the simulation engine.

The model layer (``models/``) turns fitted parameters + squad data into a
:class:`MatchInputs`. The engine (:mod:`simulation.engine`) consumes that and
returns a :class:`SimResult` of NumPy arrays, one row per simulation. Nothing in
this module imports Streamlit or touches the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

NDArray = np.ndarray


class Pos(IntEnum):
    GK = 0
    DEF = 1
    MID = 2
    FWD = 3


@dataclass
class TeamSim:
    """Per-player simulation inputs for one team.

    Every array is 1-D of length ``n_players`` and aligned to ``player_names``.
    Rates are per-90 where noted; the engine scales them by simulated minutes.
    """

    player_names: list[str]
    position: NDArray  # int, Pos codes
    start_prob: NDArray  # P(named in the XI)
    play_full_prob: NDArray  # P(plays 90 | starts)
    sub_off_mean: NDArray  # mean minute subbed off | starts and not full 90
    sub_off_sd: NDArray
    sub_on_prob: NDArray  # P(comes off the bench | did not start)
    sub_on_mean: NDArray  # mean minute brought on | used as sub
    sub_on_sd: NDArray
    shot_share: NDArray  # share of team shots while on the pitch (per-90, outfield sums ~1)
    sot_rate: NDArray  # P(shot is on target), already shrunk
    finish_rate: NDArray  # P(goal | shot), heavily shrunk toward positional average
    assist_weight: NDArray  # relative creative weight for assist allocation
    foul_weight: NDArray  # relative foul propensity for card allocation

    _ARRAYS = (
        "position",
        "start_prob",
        "play_full_prob",
        "sub_off_mean",
        "sub_off_sd",
        "sub_on_prob",
        "sub_on_mean",
        "sub_on_sd",
        "shot_share",
        "sot_rate",
        "finish_rate",
        "assist_weight",
        "foul_weight",
    )

    def __post_init__(self) -> None:
        n = len(self.player_names)
        for name in self._ARRAYS:
            arr = np.asarray(getattr(self, name), dtype=np.float64)
            if arr.shape != (n,):
                raise ValueError(f"TeamSim.{name} has shape {arr.shape}, expected ({n},)")
            setattr(self, name, arr)
        self.index = {nm: i for i, nm in enumerate(self.player_names)}

    @property
    def n_players(self) -> int:
        return len(self.player_names)

    def player_id(self, name: str) -> int:
        try:
            return self.index[name]
        except KeyError as exc:
            raise KeyError(f"unknown player {name!r}; squad is {self.player_names}") from exc


@dataclass
class MatchInputs:
    """Everything the engine needs to simulate one fixture."""

    home_team: str
    away_team: str
    league: str

    # --- score (Dixon-Coles) ---
    lambda_home: float
    lambda_away: float
    rho: float

    # --- squads ---
    home: TeamSim
    away: TeamSim

    # --- team shot volume ---
    exp_shots_home: float
    exp_shots_away: float
    shots_dispersion_k: float
    shots_game_state_beta: float

    # --- cards (rates already fold in referee, team foul index, derby) ---
    yellow_rate_home: float
    yellow_rate_away: float
    red_rate_match: float
    second_yellow_prob: float

    # --- corners ---
    exp_corners_home: float
    exp_corners_away: float
    corners_dispersion_k: float
    corners_game_state_beta: float

    # --- assists ---
    assist_fraction: float = 0.78

    # --- score over-dispersion: shared per-match tempo multiplier variance
    #     (0 = plain independent-Poisson Dixon-Coles) ---
    tempo_var: float = 0.0

    # --- goal timing ---
    goal_bucket_weights: NDArray = field(
        default_factory=lambda: np.array([0.78, 0.95, 1.15, 1.0, 1.10, 1.30])
    )
    first_half_stoppage_mean: float = 2.0
    second_half_stoppage_mean: float = 5.0

    kickoff_iso: str | None = None

    def __post_init__(self) -> None:
        self.goal_bucket_weights = np.asarray(self.goal_bucket_weights, dtype=np.float64)
        if self.goal_bucket_weights.shape != (6,):
            raise ValueError("goal_bucket_weights must have length 6 (15-min buckets)")
        for k in ("lambda_home", "lambda_away", "exp_shots_home", "exp_shots_away"):
            if getattr(self, k) <= 0:
                raise ValueError(f"{k} must be positive, got {getattr(self, k)}")


@dataclass
class TeamSimResult:
    """Per-player outcome arrays, shape ``(n_sims, n_players)``."""

    minutes: NDArray
    shots: NDArray
    sot: NDArray
    goals: NDArray
    assists: NDArray
    yellows: NDArray
    reds: NDArray
    index: dict[str, int]

    def col(self, player: str) -> int:
        try:
            return self.index[player]
        except KeyError as exc:
            raise KeyError(f"unknown player {player!r}") from exc


@dataclass
class SimResult:
    """Output of one call to :func:`simulation.engine.simulate`.

    All arrays have ``n_sims`` rows. Markets are counted over these (see
    :mod:`simulation.markets`); nothing here is a probability yet.
    """

    n_sims: int
    home_goals: NDArray  # (N,) int
    away_goals: NDArray
    home_ht_goals: NDArray  # goals scored in the first half (incl. added time)
    away_ht_goals: NDArray
    home_shots: NDArray  # realised team totals
    away_shots: NDArray
    home_corners: NDArray
    away_corners: NDArray
    home_goal_minutes: NDArray  # (N, max_goals_alloc), -1 padded
    away_goal_minutes: NDArray
    home: TeamSimResult
    away: TeamSimResult
    inputs: MatchInputs
    seed: int | None = None

    def team(self, side: str) -> TeamSimResult:
        if side not in ("home", "away"):
            raise ValueError("side must be 'home' or 'away'")
        return self.home if side == "home" else self.away
