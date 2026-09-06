"""Dixon-Coles maths + the headline test: simulated marginals must match the
analytic Dixon-Coles probabilities to within Monte-Carlo error.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from models.dixon_coles import analytic_markets, expected_goals, score_matrix, tau
from simulation.engine import simulate
from simulation.markets import Leg, probability
from simulation.synthetic import synthetic_match_inputs


def test_tau_special_cells():
    lam, mu, rho = 1.6, 1.1, -0.09
    assert tau(0, 0, lam, mu, rho) == pytest.approx(1 - lam * mu * rho)
    assert tau(0, 1, lam, mu, rho) == pytest.approx(1 + lam * rho)
    assert tau(1, 0, lam, mu, rho) == pytest.approx(1 + mu * rho)
    assert tau(1, 1, lam, mu, rho) == pytest.approx(1 - rho)
    assert tau(2, 3, lam, mu, rho) == pytest.approx(1.0)


def test_score_matrix_normalised_and_mean_matches_lambda():
    lam, mu = 1.85, 0.95
    sm = score_matrix(lam, mu, rho=-0.1, max_goals=15)
    assert sm.sum() == pytest.approx(1.0, abs=1e-12)
    eh, ea = expected_goals(sm)
    # DC correction perturbs the mean very slightly; independent Poisson would be exact.
    assert eh == pytest.approx(lam, abs=0.03)
    assert ea == pytest.approx(mu, abs=0.03)


def test_rho_zero_recovers_independent_poisson():
    lam, mu = 1.4, 1.2
    sm = score_matrix(lam, mu, rho=0.0, max_goals=20)
    from scipy.stats import poisson

    indep = np.outer(poisson.pmf(np.arange(21), lam), poisson.pmf(np.arange(21), mu))
    assert np.allclose(sm, indep / indep.sum(), atol=1e-10)


@pytest.mark.parametrize(
    "leg, key",
    [
        (Leg("result", {"outcome": "home"}), "result_home"),
        (Leg("result", {"outcome": "draw"}), "result_draw"),
        (Leg("result", {"outcome": "away"}), "result_away"),
        (Leg("total_goals", {"line": 1.5, "side": "over"}), "over_1.5"),
        (Leg("total_goals", {"line": 2.5, "side": "over"}), "over_2.5"),
        (Leg("total_goals", {"line": 3.5, "side": "over"}), "over_3.5"),
        (Leg("total_goals", {"line": 2.5, "side": "under"}), "under_2.5"),
        (Leg("btts", {"yes": True}), "btts_yes"),
    ],
)
def test_simulated_marginals_match_analytic_within_mc_error(big_sim_result, leg, key):
    res = big_sim_result
    analytic = analytic_markets(
        score_matrix(res.inputs.lambda_home, res.inputs.lambda_away, res.inputs.rho, 15)
    )
    p_analytic = analytic[key]
    p_sim = probability(res, leg)
    se = math.sqrt(p_analytic * (1 - p_analytic) / res.n_sims)
    # 4.5 sigma: ~7e-6 chance of a spurious failure per assertion.
    assert abs(p_sim - p_analytic) < 4.5 * se, (
        f"{key}: sim={p_sim:.4f} analytic={p_analytic:.4f} "
        f"diff={abs(p_sim - p_analytic):.4f} tol={4.5 * se:.4f}"
    )


def test_result_probabilities_sum_to_one(big_sim_result):
    res = big_sim_result
    total = sum(probability(res, Leg("result", {"outcome": o})) for o in ("home", "draw", "away"))
    assert total == pytest.approx(1.0, abs=1e-9)


def test_higher_home_lambda_raises_home_win_prob():
    base = simulate(
        synthetic_match_inputs(lambda_home=1.3, lambda_away=1.3, seed=1), 60_000, seed=1
    )
    strong = simulate(
        synthetic_match_inputs(lambda_home=2.1, lambda_away=1.3, seed=1), 60_000, seed=1
    )
    p_base = probability(base, Leg("result", {"outcome": "home"}))
    p_strong = probability(strong, Leg("result", {"outcome": "home"}))
    assert p_strong > p_base + 0.10
