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
from scipy.special import digamma, gammaln
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


def score_matrix(
    lam: float,
    mu: float,
    rho: float,
    max_goals: int = 15,
    tempo_var: float = 0.0,
    draw_adjust: float = 0.0,
) -> NDArray:
    """Joint PMF of (home goals, away goals) on a ``0..max_goals`` grid.

    Independent Poisson marginals with the Dixon-Coles ``tau`` correction,
    renormalised over the truncated grid so it sums to exactly 1.

    ``tempo_var`` > 0 adds a shared per-match "tempo" multiplier
    ``theta ~ Gamma(mean 1, var tempo_var)`` scaling both teams' rates before the
    Poisson draw. Marginalising it out gives a bivariate negative-binomial: fatter
    tails on the total and mild positive home/away dependence - the fix for the
    plain-Dixon-Coles under-dispersion of goals markets. ``tempo_var == 0``
    recovers the independent-Poisson form exactly.

    ``draw_adjust`` in (0, 1) shrinks the score-line diagonal by that fraction
    before renormalising - a small correction for the well-known tendency of a
    Poisson-family model to over-predict draws (teams play more decisively near
    parity than independent scoring implies).
    """
    g = np.arange(max_goals + 1)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    if tempo_var <= 0.0:
        joint = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    else:
        k = 1.0 / tempo_var
        s = lam + mu + k
        log_joint = (
            xx * np.log(lam)
            + yy * np.log(mu)
            + gammaln(xx + yy + k)
            - gammaln(k)
            - gammaln(xx + 1.0)
            - gammaln(yy + 1.0)
            + k * np.log(k)
            - (xx + yy + k) * np.log(s)
        )
        joint = np.exp(log_joint)
    joint = joint * tau(xx, yy, lam, mu, rho)
    if draw_adjust:
        np.fill_diagonal(joint, np.diagonal(joint) * (1.0 - draw_adjust))
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
# Parameter vector layout (all on the log-rate scale unless noted):
#   mu                      global intercept
#   gamma                   global home advantage
#   rho                     Dixon-Coles low-score correction
#   tempo_var               variance of the shared per-match tempo multiplier (>= 0)
#   atk[t]      for t in teams     attack deviation
#   dfn[t]      for t in teams     defence deviation (higher = concedes fewer)
#   atk_ha[t]   for t in teams     home/away attack split (att_home = atk + atk_ha)
#   dfn_ha[t]   for t in teams     home/away defence split
#   atk_tier[k] for k in tiers     tier-level attack mean (hierarchy / league strength)
#   dfn_tier[k] for k in tiers     tier-level defence mean

_RHO_BOUNDS = (-0.2, 0.2)
_TEMPO_BOUNDS = (0.0, 0.6)  # 0 == plain independent-Poisson Dixon-Coles


@dataclass
class DixonColesConfig:
    time_decay_half_life_days: float = 180.0
    rho_init: float = -0.10
    ratings_l2: float = 0.05  # pull team atk/dfn toward their tier mean
    home_away_coupling: float = 0.35  # pull the home/away split toward 0
    league_strength_prior_sd: float = 0.40
    home_advantage_init: float = 0.25
    tempo_var_init: float = 0.03  # shared per-match tempo multiplier variance
    tempo_var_prior_sd: float = 0.06
    draw_adjust: float = 0.0  # fraction to shrink the score-line diagonal (draw over-prediction)
    promotion_shrinkage_base: float = 0.25  # extra L2 pull for just-promoted/relegated sides
    promotion_shrinkage_decay_games: float = 10.0

    @classmethod
    def from_yaml(cls, model_yaml: dict) -> DixonColesConfig:
        dc = model_yaml.get("dixon_coles", {})
        ls = model_yaml.get("league_strength", {})
        prm = model_yaml.get("promotion_shrinkage", {})
        return cls(
            time_decay_half_life_days=dc.get("time_decay_half_life_days", 180.0),
            rho_init=dc.get("rho_init", -0.10),
            ratings_l2=dc.get("ratings_l2", 0.05),
            home_away_coupling=dc.get("home_away_coupling", 0.35),
            league_strength_prior_sd=ls.get("prior_sd", 0.40),
            home_advantage_init=dc.get("home_advantage_init", 0.25),
            tempo_var_init=dc.get("tempo_var_init", 0.03),
            tempo_var_prior_sd=dc.get("tempo_var_prior_sd", 0.06),
            draw_adjust=dc.get("draw_adjust", 0.0),
            promotion_shrinkage_base=prm.get("base", 0.6),
            promotion_shrinkage_decay_games=prm.get("decay_games", 12.0),
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
    tempo_var: float = 0.0
    draw_adjust: float = 0.0
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
        return score_matrix(lh, la, self.rho, max_goals, self.tempo_var, self.draw_adjust)

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
                    "tempo_var": self.tempo_var,
                    "draw_adjust": self.draw_adjust,
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
            tempo_var=blob.get("tempo_var", 0.0),
            draw_adjust=blob.get("draw_adjust", 0.0),
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

    # team -> tier for the hierarchical prior: the mode of the team's matches
    # (stable across the fit). Separately, count how many games the team has in
    # the tier of its *most recent* match - the app uses this low count to flag
    # just-promoted / just-relegated sides rather than the fit trying to correct
    # for them (which destabilises the ratings without squad-strength data).
    tiers_map: dict[str, int] = {}
    current_tier_games: dict[str, int] = {}
    if "tier" in df.columns:
        df_sorted = df.sort_values("date")
        for t in teams:
            sub = df_sorted.loc[(df_sorted.home_team == t) | (df_sorted.away_team == t), "tier"]
            tiers_map[t] = int(sub.mode().iloc[0])
            recent = int(sub.iloc[-1]) if len(sub) else tiers_map[t]
            current_tier_games[t] = int((sub == recent).sum())
    elif team_tiers:
        tiers_map = {t: team_tiers.get(t, 1) for t in teams}
        current_tier_games = dict.fromkeys(teams, 999)
    else:
        tiers_map = dict.fromkeys(teams, 1)
        current_tier_games = dict.fromkeys(teams, 999)
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
    xy = (x + y).astype(np.float64)

    # unpack helpers
    def _unpack(p: NDArray):
        i = 0
        mu = p[i]
        i += 1
        gamma = p[i]
        i += 1
        rho = p[i]
        i += 1
        tempo_var = p[i]
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
        return mu, gamma, rho, tempo_var, atk, dfn, atk_ha, dfn_ha, atk_tier, dfn_tier

    s2 = cfg.league_strength_prior_sd**2
    anchor = 1e-3  # weak L2 on atk/dfn for identifiability
    ln135 = np.log(1.35)

    def _obj(p: NDArray) -> tuple[float, NDArray]:
        """Negative log-posterior and its analytic gradient (jac=True)."""
        mu, gamma, rho, tempo_var, atk, dfn, atk_ha, dfn_ha, atk_tier, dfn_tier = _unpack(p)
        eta_h = mu + gamma + (atk[hi] + atk_ha[hi]) - (dfn[ai] - dfn_ha[ai])
        eta_a = mu + (atk[ai] - atk_ha[ai]) - (dfn[hi] + dfn_ha[hi])
        lh = np.exp(np.minimum(eta_h, 20.0))
        la = np.exp(np.minimum(eta_a, 20.0))

        t = np.ones_like(lh)
        t = np.where(is00, 1.0 - lh * la * rho, t)
        t = np.where(is01, 1.0 + lh * rho, t)
        t = np.where(is10, 1.0 + la * rho, t)
        t = np.where(is11, 1.0 - rho, t)
        t = np.clip(t, 1e-6, None)
        inv_t = 1.0 / t

        # bivariate negative-binomial (shared Gamma tempo). k capped so the
        # objective stays smooth as tempo_var -> 0 (that limit is Poisson).
        k = 1.0 / max(tempo_var, 1e-4)
        s = lh + la + k
        base = (
            x * np.log(lh)
            + y * np.log(la)
            + gammaln(xy + k)
            - gammaln(k)
            - lgx
            - lgy
            + k * np.log(k)
            - (xy + k) * np.log(s)
        )
        nll = -(w * (np.log(t) + base)).sum()

        # --- gradient of nll wrt eta_h, eta_a, rho, tempo_var ---
        dbase_h = x - (xy + k) * lh / s
        dbase_a = y - (xy + k) * la / s
        dlt_h = np.where(is00, -lh * la * rho * inv_t, 0.0) + np.where(is01, lh * rho * inv_t, 0.0)
        dlt_a = np.where(is00, -lh * la * rho * inv_t, 0.0) + np.where(is10, la * rho * inv_t, 0.0)
        dlt_rho = (
            np.where(is00, -lh * la * inv_t, 0.0)
            + np.where(is01, lh * inv_t, 0.0)
            + np.where(is10, la * inv_t, 0.0)
            + np.where(is11, -inv_t, 0.0)
        )
        g_h = w * (dbase_h + dlt_h)  # d(log-lik)/d eta_h per match
        g_a = w * (dbase_a + dlt_a)

        grad = np.zeros_like(p)
        grad[0] = -(g_h.sum() + g_a.sum()) + (mu - ln135) / 0.25
        grad[1] = -g_h.sum() + (gamma - cfg.home_advantage_init) / 0.05
        grad[2] = -(w * dlt_rho).sum() + (rho - cfg.rho_init) / 0.02
        if tempo_var > 1e-4:
            dbase_dk = digamma(xy + k) - digamma(k) + np.log(k) + 1.0 - np.log(s) - (xy + k) / s
            grad[3] = -(w * dbase_dk * (-k * k)).sum()
        grad[3] += (tempo_var - cfg.tempo_var_init) / (cfg.tempo_var_prior_sd**2)

        bc_h_gh = np.bincount(hi, g_h, minlength=n_teams)
        bc_a_ga = np.bincount(ai, g_a, minlength=n_teams)
        bc_a_gh = np.bincount(ai, g_h, minlength=n_teams)
        bc_h_ga = np.bincount(hi, g_a, minlength=n_teams)

        tc_atk = atk - atk_tier[team_tier_idx]
        tc_dfn = dfn - dfn_tier[team_tier_idx]
        c1 = cfg.ratings_l2 * w.sum()
        c2 = cfg.home_away_coupling * w.sum()

        # eta_h = ... + atk_h + atk_ha_h - dfn_a + dfn_ha_a
        # eta_a = ... + atk_a - atk_ha_a - dfn_h - dfn_ha_h
        d_atk = -(bc_h_gh + bc_a_ga) + 2 * c1 * tc_atk + anchor * atk
        d_dfn = (bc_a_gh + bc_h_ga) + 2 * c1 * tc_dfn + anchor * dfn
        d_atkha = -(bc_h_gh - bc_a_ga) + 2 * c2 * atk_ha
        d_dfnha = (-bc_a_gh + bc_h_ga) + 2 * c2 * dfn_ha
        d_atktier = -2 * c1 * np.bincount(team_tier_idx, tc_atk, minlength=n_tiers) + atk_tier / s2
        d_dfntier = -2 * c1 * np.bincount(team_tier_idx, tc_dfn, minlength=n_tiers) + dfn_tier / s2

        o = 4
        for block in (d_atk, d_dfn, d_atkha, d_dfnha):
            grad[o : o + n_teams] = block
            o += n_teams
        grad[o : o + n_tiers] = d_atktier
        o += n_tiers
        grad[o : o + n_tiers] = d_dfntier

        pen = (
            c1 * (tc_atk @ tc_atk + tc_dfn @ tc_dfn)
            + c2 * (atk_ha @ atk_ha + dfn_ha @ dfn_ha)
            + 0.5 / s2 * (atk_tier @ atk_tier + dfn_tier @ dfn_tier)
            + 0.5 * anchor * (atk @ atk + dfn @ dfn)
            + 0.5 * (mu - ln135) ** 2 / 0.25
            + 0.5 * (gamma - cfg.home_advantage_init) ** 2 / 0.05
            + 0.5 * (rho - cfg.rho_init) ** 2 / 0.02
            + 0.5 * (tempo_var - cfg.tempo_var_init) ** 2 / (cfg.tempo_var_prior_sd**2)
        )
        return nll + pen, grad

    n_par = 4 + 4 * n_teams + 2 * n_tiers
    x0 = np.zeros(n_par)
    x0[0] = np.log(max(df[["fthg", "ftag"]].to_numpy().mean(), 0.5))
    x0[1] = cfg.home_advantage_init
    x0[2] = cfg.rho_init
    x0[3] = cfg.tempo_var_init
    bounds = [(-2, 2), (-1, 1), _RHO_BOUNDS, _TEMPO_BOUNDS] + [(-3, 3)] * (
        4 * n_teams + 2 * n_tiers
    )

    res = minimize(
        _obj,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 2000, "maxfun": 20000, "ftol": 1e-12, "gtol": 1e-7},
    )
    mu, gamma, rho, tempo_var, atk, dfn, atk_ha, dfn_ha, atk_tier, dfn_tier = _unpack(res.x)
    # status 0 = converged; 2 = a parameter resting on a bound (e.g. tempo_var at
    # its floor when the data wants no extra dispersion) - a valid solution for us.
    converged = res.status in (0, 2) and np.isfinite(res.fun)

    table = pd.DataFrame(
        {
            "tier": [tiers_map[t] for t in teams],
            "atk": atk,
            "dfn": dfn,
            "atk_ha": atk_ha,
            "dfn_ha": dfn_ha,
            "current_tier_games": [current_tier_games[t] for t in teams],
        },
        index=pd.Index(teams, name="team"),
    )
    return TeamRatings(
        table=table,
        mu=float(mu),
        gamma=float(gamma),
        rho=float(rho),
        tempo_var=float(tempo_var),
        draw_adjust=cfg.draw_adjust,
        tier_atk={k: float(atk_tier[tier_pos[k]]) for k in tiers},
        tier_dfn={k: float(dfn_tier[tier_pos[k]]) for k in tiers},
        meta={
            "fitted_at": dt.datetime.now(dt.UTC).isoformat(),
            "n_matches": int(len(df)),
            "n_teams": n_teams,
            "date_max": str(df["date"].max().date()),
            "half_life_days": cfg.time_decay_half_life_days,
            "converged": bool(converged),
            "opt_status": int(res.status),
            "opt_message": str(res.message),
            "neg_log_post": float(res.fun),
        },
    )
