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

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
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


# =========================================================================== #
# Fitting
# =========================================================================== #
# Parameter vector layout (all on the log-rate scale):
#   mu                      global intercept
#   gamma                   global home advantage
#   rho                     Dixon-Coles low-score correction
#   atk[t]      for t in teams     attack deviation
#   dfn[t]      for t in teams     defence deviation (higher = concedes fewer)
#   atk_ha[t]   for t in teams     home/away attack split (att_home = atk + atk_ha)
#   dfn_ha[t]   for t in teams     home/away defence split
#   atk_tier[k] for k in tiers     tier-level attack mean (hierarchy / league strength)
#   dfn_tier[k] for k in tiers     tier-level defence mean

_RHO_BOUNDS = (-0.2, 0.2)


@dataclass
class DixonColesConfig:
    time_decay_half_life_days: float = 180.0
    rho_init: float = -0.10
    ratings_l2: float = 0.05  # pull team atk/dfn toward their tier mean
    home_away_coupling: float = 0.35  # pull the home/away split toward 0
    league_strength_prior_sd: float = 0.40
    home_advantage_init: float = 0.25

    @classmethod
    def from_yaml(cls, model_yaml: dict) -> DixonColesConfig:
        dc = model_yaml.get("dixon_coles", {})
        ls = model_yaml.get("league_strength", {})
        return cls(
            time_decay_half_life_days=dc.get("time_decay_half_life_days", 180.0),
            rho_init=dc.get("rho_init", -0.10),
            ratings_l2=dc.get("ratings_l2", 0.05),
            home_away_coupling=dc.get("home_away_coupling", 0.35),
            league_strength_prior_sd=ls.get("prior_sd", 0.40),
            home_advantage_init=dc.get("home_advantage_init", 0.25),
        )


@dataclass
class TeamRatings:
    """Fitted parameters + the machinery to turn a fixture into (lambda_h, lambda_a, rho)."""

    table: pd.DataFrame  # index = team, cols: tier, atk, dfn, atk_ha, dfn_ha
    mu: float
    gamma: float
    rho: float
    tier_atk: dict[int, float]
    tier_dfn: dict[int, float]
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def _team_row(self, team: str, tier: int | None):
        if team in self.table.index:
            r = self.table.loc[team]
            return float(r.atk), float(r.dfn), float(r.atk_ha), float(r.dfn_ha), int(r.tier)
        # unknown team (promoted with no history here / cup opponent): tier mean
        t = tier if tier is not None else min(self.tier_atk)
        return self.tier_atk.get(t, 0.0), self.tier_dfn.get(t, 0.0), 0.0, 0.0, t

    def lambdas(
        self,
        home_team: str,
        away_team: str,
        *,
        home_tier: int | None = None,
        away_tier: int | None = None,
        neutral: bool = False,
    ) -> tuple[float, float]:
        atk_h, dfn_h, atk_ha_h, dfn_ha_h, _ = self._team_row(home_team, home_tier)
        atk_a, dfn_a, atk_ha_a, dfn_ha_a, _ = self._team_row(away_team, away_tier)
        ha = 0.0 if neutral else self.gamma
        log_lh = self.mu + ha + (atk_h + atk_ha_h) - (dfn_a - dfn_ha_a)
        log_la = self.mu + (atk_a - atk_ha_a) - (dfn_h + dfn_ha_h)
        return float(np.exp(log_lh)), float(np.exp(log_la))

    def score_matrix(self, home_team: str, away_team: str, max_goals: int = 15, **kw) -> NDArray:
        lh, la = self.lambdas(home_team, away_team, **kw)
        return score_matrix(lh, la, self.rho, max_goals)

    # ------------------------------------------------------------------ #
    def to_parquet(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.table.to_parquet(path)
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "mu": self.mu,
                    "gamma": self.gamma,
                    "rho": self.rho,
                    "tier_atk": {str(k): v for k, v in self.tier_atk.items()},
                    "tier_dfn": {str(k): v for k, v in self.tier_dfn.items()},
                    "meta": self.meta,
                },
                indent=2,
            )
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> TeamRatings:
        path = Path(path)
        blob = json.loads(path.with_suffix(".json").read_text())
        return cls(
            table=pd.read_parquet(path),
            mu=blob["mu"],
            gamma=blob["gamma"],
            rho=blob["rho"],
            tier_atk={int(k): v for k, v in blob["tier_atk"].items()},
            tier_dfn={int(k): v for k, v in blob["tier_dfn"].items()},
            meta=blob.get("meta", {}),
        )


def _decay_weights(dates: pd.Series, half_life_days: float) -> NDArray:
    ref = dates.max()
    age = (ref - dates).dt.total_seconds().to_numpy() / 86400.0
    return np.exp(-np.log(2.0) * age / half_life_days)


def fit(
    matches: pd.DataFrame,
    config: DixonColesConfig | None = None,
    *,
    team_tiers: dict[str, int] | None = None,
) -> TeamRatings:
    """Weighted MAP fit of the Dixon-Coles model.

    ``matches`` needs columns: ``date, home_team, away_team, fthg, ftag`` and
    either ``league`` (mapped to a tier via ``team_tiers``) or a ``tier`` column.
    Exponential time decay is applied relative to the most recent match.
    """
    cfg = config or DixonColesConfig()
    df = matches.dropna(subset=["home_team", "away_team", "fthg", "ftag"]).copy()
    df["date"] = pd.to_datetime(df["date"])

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    tidx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    # team -> tier
    tiers_map: dict[str, int] = {}
    if "tier" in df.columns:
        for t in teams:
            sub = df.loc[(df.home_team == t) | (df.away_team == t), "tier"]
            tiers_map[t] = int(sub.mode().iloc[0])
    elif team_tiers:
        tiers_map = {t: team_tiers.get(t, 1) for t in teams}
    else:
        tiers_map = dict.fromkeys(teams, 1)
    tiers = sorted(set(tiers_map.values()))
    tier_pos = {k: i for i, k in enumerate(tiers)}
    n_tiers = len(tiers)
    team_tier_idx = np.array([tier_pos[tiers_map[t]] for t in teams])

    hi = df["home_team"].map(tidx).to_numpy()
    ai = df["away_team"].map(tidx).to_numpy()
    x = df["fthg"].to_numpy(dtype=np.int64)
    y = df["ftag"].to_numpy(dtype=np.int64)
    w = _decay_weights(df["date"], cfg.time_decay_half_life_days)

    is00 = (x == 0) & (y == 0)
    is01 = (x == 0) & (y == 1)
    is10 = (x == 1) & (y == 0)
    is11 = (x == 1) & (y == 1)
    lgx = gammaln(x + 1.0)
    lgy = gammaln(y + 1.0)

    # unpack helpers
    def _unpack(p: NDArray):
        i = 0
        mu = p[i]
        i += 1
        gamma = p[i]
        i += 1
        rho = p[i]
        i += 1
        atk = p[i : i + n_teams]
        i += n_teams
        dfn = p[i : i + n_teams]
        i += n_teams
        atk_ha = p[i : i + n_teams]
        i += n_teams
        dfn_ha = p[i : i + n_teams]
        i += n_teams
        atk_tier = p[i : i + n_tiers]
        i += n_tiers
        dfn_tier = p[i : i + n_tiers]
        i += n_tiers
        return mu, gamma, rho, atk, dfn, atk_ha, dfn_ha, atk_tier, dfn_tier

    def neg_log_post(p: NDArray) -> float:
        mu, gamma, rho, atk, dfn, atk_ha, dfn_ha, atk_tier, dfn_tier = _unpack(p)
        log_lh = mu + gamma + (atk[hi] + atk_ha[hi]) - (dfn[ai] - dfn_ha[ai])
        log_la = mu + (atk[ai] - atk_ha[ai]) - (dfn[hi] + dfn_ha[hi])
        lh = np.exp(np.clip(log_lh, -4, 4))
        la = np.exp(np.clip(log_la, -4, 4))

        t = np.ones_like(lh)
        t = np.where(is00, 1.0 - lh * la * rho, t)
        t = np.where(is01, 1.0 + lh * rho, t)
        t = np.where(is10, 1.0 + la * rho, t)
        t = np.where(is11, 1.0 - rho, t)
        t = np.clip(t, 1e-6, None)

        ll = w * (np.log(t) + x * np.log(lh) - lh - lgx + y * np.log(la) - la - lgy)
        nll = -ll.sum()

        # priors (MAP)
        tier_center_atk = atk - atk_tier[team_tier_idx]
        tier_center_dfn = dfn - dfn_tier[team_tier_idx]
        pen = 0.0
        pen += (
            cfg.ratings_l2
            * w.sum()
            * (tier_center_atk @ tier_center_atk + tier_center_dfn @ tier_center_dfn)
        )
        pen += cfg.home_away_coupling * w.sum() * (atk_ha @ atk_ha + dfn_ha @ dfn_ha)
        s2 = cfg.league_strength_prior_sd**2
        pen += 0.5 / s2 * (atk_tier @ atk_tier + dfn_tier @ dfn_tier)
        pen += 0.5 * (atk @ atk + dfn @ dfn) * 1e-3  # weak anchor for identifiability
        pen += 0.5 * ((mu - np.log(1.35)) ** 2) / 0.25
        pen += 0.5 * ((gamma - cfg.home_advantage_init) ** 2) / 0.05
        pen += 0.5 * ((rho - cfg.rho_init) ** 2) / 0.02
        return nll + pen

    n_par = 3 + 4 * n_teams + 2 * n_tiers
    x0 = np.zeros(n_par)
    x0[0] = np.log(max(df[["fthg", "ftag"]].to_numpy().mean(), 0.5))
    x0[1] = cfg.home_advantage_init
    x0[2] = cfg.rho_init
    bounds = [(-2, 2), (-1, 1), _RHO_BOUNDS] + [(-3, 3)] * (4 * n_teams + 2 * n_tiers)

    res = minimize(
        neg_log_post, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500, "ftol": 1e-10}
    )
    mu, gamma, rho, atk, dfn, atk_ha, dfn_ha, atk_tier, dfn_tier = _unpack(res.x)

    table = pd.DataFrame(
        {
            "tier": [tiers_map[t] for t in teams],
            "atk": atk,
            "dfn": dfn,
            "atk_ha": atk_ha,
            "dfn_ha": dfn_ha,
        },
        index=pd.Index(teams, name="team"),
    )
    return TeamRatings(
        table=table,
        mu=float(mu),
        gamma=float(gamma),
        rho=float(rho),
        tier_atk={k: float(atk_tier[tier_pos[k]]) for k in tiers},
        tier_dfn={k: float(dfn_tier[tier_pos[k]]) for k in tiers},
        meta={
            "fitted_at": dt.datetime.now(dt.UTC).isoformat(),
            "n_matches": int(len(df)),
            "n_teams": n_teams,
            "date_max": str(df["date"].max().date()),
            "half_life_days": cfg.time_decay_half_life_days,
            "converged": bool(res.success),
            "neg_log_post": float(res.fun),
        },
    )
