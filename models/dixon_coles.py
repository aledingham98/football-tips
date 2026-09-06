"""Dixon-Coles bivariate-Poisson score model.

This module has two halves:

* **Scoring maths** (:func:`tau`, :func:`score_matrix`, :func:`analytic_markets`)
  - pure functions used by the simulation engine and by the test-suite as the
  analytic ground truth that Monte-Carlo marginals must match.
* **Fitting** (:func:`fit`) - weighted maximum likelihood with exponential time
  decay, separate home/away attack & defence, a global home advantage and a
  hierarchical per-tier league-strength term. Added in the ingestion step.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

NDArray = np.ndarray


def tau(x: NDArray, y: NDArray, lam: float, mu: float, rho: float) -> NDArray:
    """Dixon-Coles low-score dependence correction.

    Adjusts only the 0-0, 1-0, 0-1 and 1-1 cells of the independent-Poisson
    joint distribution; everything else is multiplied by 1.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    out = np.ones(np.broadcast_shapes(x.shape, y.shape), dtype=np.float64)
    out = np.where((x == 0) & (y == 0), 1.0 - lam * mu * rho, out)
    out = np.where((x == 0) & (y == 1), 1.0 + lam * rho, out)
    out = np.where((x == 1) & (y == 0), 1.0 + mu * rho, out)
    out = np.where((x == 1) & (y == 1), 1.0 - rho, out)
    return np.maximum(out, 1e-9)  # keep probabilities non-negative for extreme rho


def score_matrix(lam: float, mu: float, rho: float, max_goals: int = 15) -> NDArray:
    """Joint PMF of (home goals, away goals) on a ``0..max_goals`` grid.

    Independent Poisson marginals with the Dixon-Coles ``tau`` correction,
    renormalised over the truncated grid so it sums to exactly 1.
    """
    g = np.arange(max_goals + 1)
    home_pmf = poisson.pmf(g, lam)
    away_pmf = poisson.pmf(g, mu)
    joint = np.outer(home_pmf, away_pmf)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    joint = joint * tau(xx, yy, lam, mu, rho)
    return joint / joint.sum()


def analytic_markets(sm: NDArray) -> dict[str, float]:
    """Exact market probabilities from a score matrix - the test ground truth."""
    n = sm.shape[0]
    xx, yy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    total = xx + yy
    out = {
        "result_home": float(sm[xx > yy].sum()),
        "result_draw": float(np.trace(sm)),
        "result_away": float(sm[xx < yy].sum()),
        "btts_yes": float(sm[(xx >= 1) & (yy >= 1)].sum()),
        "btts_no": float(sm[(xx == 0) | (yy == 0)].sum()),
    }
    for line in (0.5, 1.5, 2.5, 3.5, 4.5):
        out[f"over_{line}"] = float(sm[total > line].sum())
        out[f"under_{line}"] = float(sm[total < line].sum())
    for k in (0, 1, 2, 3):
        out[f"home_goals_{k}plus"] = float(sm[xx >= k].sum())
        out[f"away_goals_{k}plus"] = float(sm[yy >= k].sum())
    return out


def expected_goals(sm: NDArray) -> tuple[float, float]:
    """(E[home goals], E[away goals]) implied by a score matrix."""
    n = sm.shape[0]
    g = np.arange(n)
    return float((sm.sum(axis=1) * g).sum()), float((sm.sum(axis=0) * g).sum())
