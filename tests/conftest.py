"""Shared fixtures. The engine tests run without any network or committed data."""

from __future__ import annotations

import numpy as np
import pytest

from simulation.engine import simulate
from simulation.state import MatchInputs
from simulation.synthetic import synthetic_match_inputs


@pytest.fixture(scope="session")
def match_inputs() -> MatchInputs:
    # Asymmetric rates + non-zero rho so correlation structure is exercised.
    return synthetic_match_inputs(lambda_home=1.7, lambda_away=1.1, rho=-0.08, seed=11)


@pytest.fixture(scope="session")
def sim_result(match_inputs: MatchInputs):
    return simulate(match_inputs, n_sims=40_000, seed=2024)


@pytest.fixture(scope="session")
def big_sim_result(match_inputs: MatchInputs):
    # Large N so Monte-Carlo error is small enough to assert against analytic values.
    return simulate(match_inputs, n_sims=200_000, seed=99)


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
