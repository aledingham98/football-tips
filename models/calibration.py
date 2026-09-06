"""Post-hoc calibration map - the monitored safety net.

A per-market (optionally per-league) logistic recalibration:
``p' = sigmoid(a + b * logit(p))``, fit on held-out predictions. Identity
(a=0, b=1) where there isn't enough held-out data to fit one.

It is applied *after* the simulation, only to single-leg / marginal probabilities
where it has been validated. It is deliberately NOT applied inside joint bet
builder counting (that would break the correlation structure); builders get a
coarser whole-price adjustment at most. The bet log's calibration plots are what
keep this map honest over time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


def _logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def _fit_logistic(p: np.ndarray, o: np.ndarray) -> tuple[float, float]:
    """MLE of (a, b) in P(o=1) = sigmoid(a + b*logit(p))."""
    from scipy.optimize import minimize

    z = _logit(p)

    def nll(ab: np.ndarray) -> float:
        a, b = ab
        lin = a + b * z
        return float(np.mean(np.log1p(np.exp(-np.abs(lin))) + np.maximum(lin, 0.0) - o * lin))

    res = minimize(nll, np.array([0.0, 1.0]), method="Nelder-Mead")
    a, b = res.x
    # guard against pathological fits on thin / degenerate data
    if not np.isfinite(a) or not np.isfinite(b) or b <= 0 or b > 3:
        return 0.0, 1.0
    return float(a), float(b)


@dataclass
class CalibrationMap:
    params: dict[str, tuple[float, float]] = field(default_factory=dict)
    by_league: bool = True
    meta: dict = field(default_factory=dict)

    def _key(self, market: str, league: str | None) -> str:
        if self.by_league and league and f"{market}|{league}" in self.params:
            return f"{market}|{league}"
        return market

    def apply(self, market: str, p, league: str | None = None):
        a, b = self.params.get(self._key(market, league), (0.0, 1.0))
        out = _sigmoid(a + b * _logit(p))
        return float(out) if np.isscalar(p) else out

    def is_identity(self, market: str, league: str | None = None) -> bool:
        a, b = self.params.get(self._key(market, league), (0.0, 1.0))
        return abs(a) < 1e-6 and abs(b - 1.0) < 1e-6

    # --- persistence ---
    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(
                {
                    "by_league": self.by_league,
                    "params": {k: list(v) for k, v in self.params.items()},
                    "meta": self.meta,
                },
                indent=2,
            )
        )

    @classmethod
    def from_json(cls, path: str | Path) -> CalibrationMap:
        blob = json.loads(Path(path).read_text())
        return cls(
            params={k: tuple(v) for k, v in blob["params"].items()},
            by_league=blob.get("by_league", True),
            meta=blob.get("meta", {}),
        )


def fit_calibration_map(
    predictions: pd.DataFrame,
    *,
    by_league: bool = True,
    min_n: int = 200,
) -> CalibrationMap:
    """``predictions`` needs columns ``market, pred, outcome`` and, if ``by_league``,
    ``league``. Markets with fewer than ``min_n`` held-out rows get the identity map.
    """
    params: dict[str, tuple[float, float]] = {}
    for market, g in predictions.groupby("market"):
        if len(g) >= min_n:
            params[market] = _fit_logistic(g["pred"].to_numpy(), g["outcome"].to_numpy())
        if by_league and "league" in g.columns:
            for league, gl in g.groupby("league"):
                if len(gl) >= min_n:
                    params[f"{market}|{league}"] = _fit_logistic(
                        gl["pred"].to_numpy(), gl["outcome"].to_numpy()
                    )
    return CalibrationMap(
        params=params,
        by_league=by_league,
        meta={"n_rows": int(len(predictions)), "n_maps": len(params)},
    )
