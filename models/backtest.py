"""Walk-forward backtest + calibration for the Dixon-Coles score model.

For each month of the held-out period the model is refit on *only* the matches
before that month (no lookahead), then used to price every match in the month.
Predicted probabilities are compared with outcomes to produce reliability curves,
Brier score, log-loss and expected calibration error per market - and, where the
source has closing odds, against the closing line as the benchmark.

This is the Phase 2 gate: calibration curves on held-out data, before any UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from models.dixon_coles import DixonColesConfig, analytic_markets, fit

# market key -> (human label, function(row) -> outcome 0/1 from fthg/ftag)
_MARKETS: dict[str, tuple[str, callable]] = {
    "result_home": ("Home win", lambda r: int(r.fthg > r.ftag)),
    "result_draw": ("Draw", lambda r: int(r.fthg == r.ftag)),
    "result_away": ("Away win", lambda r: int(r.fthg < r.ftag)),
    "over_1.5": ("Over 1.5", lambda r: int(r.fthg + r.ftag > 1)),
    "over_2.5": ("Over 2.5", lambda r: int(r.fthg + r.ftag > 2)),
    "over_3.5": ("Over 3.5", lambda r: int(r.fthg + r.ftag > 3)),
    "btts_yes": ("BTTS", lambda r: int(r.fthg >= 1 and r.ftag >= 1)),
}

# closing-odds columns (proportional de-vig) for the benchmark, per market
_CLOSE_COLS = {
    "result_home": ("close_home_odds", "close_draw_odds", "close_away_odds", 0),
    "result_draw": ("close_home_odds", "close_draw_odds", "close_away_odds", 1),
    "result_away": ("close_home_odds", "close_draw_odds", "close_away_odds", 2),
    "over_2.5": ("close_over25_odds", "close_under25_odds", None, 0),
}


@dataclass
class CalibrationResult:
    predictions: pd.DataFrame  # match x market long form: pred, outcome, close_pred
    metrics: pd.DataFrame  # per-market Brier / logloss / ECE (model vs closing)
    reliability: pd.DataFrame  # per-market decile bins: mean_pred, mean_outcome, n

    def summary(self) -> str:
        return self.metrics.round(4).to_string()


def _devig_row(row: pd.Series, cols: tuple) -> float | None:
    a, b, c, which = cols
    odds = [row.get(a), row.get(b)] + ([row.get(c)] if c else [])
    odds = [o for o in odds if pd.notna(o) and o and o > 1]
    if len(odds) < (3 if c else 2):
        return None
    inv = [1.0 / o for o in odds]
    s = sum(inv)
    return inv[which] / s


def _reliability(df: pd.DataFrame, n_bins: int = 12) -> pd.DataFrame:
    """Equal-count (quantile) bins so the curve has resolution where the data is."""
    out = []
    for market, g in df.groupby("market"):
        g = g.sort_values("pred")
        edges = np.quantile(g["pred"], np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        idx = np.clip(np.digitize(g["pred"], edges[1:-1]), 0, len(edges) - 2)
        for b in range(len(edges) - 1):
            sel = idx == b
            if sel.sum() < 5:
                continue
            out.append(
                {
                    "market": market,
                    "bin": b,
                    "mean_pred": g["pred"][sel].mean(),
                    "mean_outcome": g["outcome"][sel].mean(),
                    "se": np.sqrt(
                        g["outcome"][sel].mean() * (1 - g["outcome"][sel].mean()) / sel.sum()
                    ),
                    "n": int(sel.sum()),
                }
            )
    return pd.DataFrame(out)


def _calibration_slope(p: np.ndarray, o: np.ndarray) -> tuple[float, float]:
    """Logistic fit of outcome on logit(pred). Perfect calibration -> slope 1, intercept 0."""
    from scipy.optimize import minimize

    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

    def nll(ab):
        a, b = ab
        lin = a + b * z
        return np.mean(np.log1p(np.exp(-np.abs(lin))) + np.maximum(lin, 0) - o * lin)

    r = minimize(nll, [0.0, 1.0], method="Nelder-Mead")
    return float(r.x[1]), float(r.x[0])  # slope, intercept


def _metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for market, g in df.groupby("market"):
        p = g["pred"].to_numpy()
        o = g["outcome"].to_numpy()
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        slope, intercept = _calibration_slope(p, o)
        row = {
            "market": market,
            "n": len(g),
            "base_rate": o.mean(),
            "mean_pred": p.mean(),
            "brier": np.mean((p - o) ** 2),
            "logloss": -np.mean(o * np.log(pc) + (1 - o) * np.log(1 - pc)),
            "ece": _ece(p, o),
            "cal_slope": slope,
            "cal_intercept": intercept,
        }
        cg = g.dropna(subset=["close_pred"])
        if len(cg) > 30:
            cp = np.clip(cg["close_pred"].to_numpy(), 1e-6, 1 - 1e-6)
            co = cg["outcome"].to_numpy()
            row["brier_close"] = np.mean((cp - co) ** 2)
            row["brier_vs_close"] = row["brier"] - row["brier_close"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("market")


def _ece(p: np.ndarray, o: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            e += sel.mean() * abs(p[sel].mean() - o[sel].mean())
    return e


def walk_forward_calibration(
    matches: pd.DataFrame,
    *,
    holdout_start: str,
    config: DixonColesConfig | None = None,
    min_train_matches: int = 400,
) -> CalibrationResult:
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    test = df[df["date"] >= pd.Timestamp(holdout_start)]
    months = sorted(test["date"].dt.to_period("M").unique())
    print(f"walk-forward: {len(test)} test matches across {len(months)} months")

    records: list[dict] = []
    for period in months:
        m_start = period.to_timestamp()
        m_end = (period + 1).to_timestamp()
        train = df[df["date"] < m_start]
        if len(train) < min_train_matches:
            continue
        month_matches = df[(df["date"] >= m_start) & (df["date"] < m_end)]
        ratings = fit(train, config)
        for row in month_matches.itertuples(index=False):
            tier = getattr(row, "tier", 1)
            sm = ratings.score_matrix(row.home_team, row.away_team, home_tier=tier, away_tier=tier)
            am = analytic_markets(sm)
            for market, (_, outcome_fn) in _MARKETS.items():
                rec = {
                    "date": row.date,
                    "league": row.league,
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "market": market,
                    "pred": am[market],
                    "outcome": outcome_fn(row),
                    "close_pred": (
                        _devig_row(pd.Series(row._asdict()), _CLOSE_COLS[market])
                        if market in _CLOSE_COLS
                        else None
                    ),
                }
                records.append(rec)
        print(f"  {period}: fit on {len(train)} -> priced {len(month_matches)} matches")

    preds = pd.DataFrame.from_records(records)
    return CalibrationResult(
        predictions=preds,
        metrics=_metrics(preds),
        reliability=_reliability(preds),
    )


def plot_calibration(
    result: CalibrationResult, out_dir: str | Path = "reports/calibration"
) -> list[Path]:
    """Reliability diagrams (one panel per market) + a summary bar of Brier vs close."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rel = result.reliability
    markets = list(_MARKETS)

    fig, axes = plt.subplots(3, 3, figsize=(11, 11))
    for ax, market in zip(axes.flat, markets, strict=False):
        g = rel[rel["market"] == market]
        lo = float(min(g["mean_pred"].min(), g["mean_outcome"].min())) - 0.05
        hi = float(max(g["mean_pred"].max(), g["mean_outcome"].max())) + 0.05
        lo, hi = max(0.0, lo), min(1.0, hi)
        ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=1)
        ax.errorbar(
            g["mean_pred"],
            g["mean_outcome"],
            yerr=1.96 * g["se"],
            fmt="o-",
            ms=4,
            lw=1,
            capsize=2,
            alpha=0.85,
        )
        m = result.metrics.loc[market] if market in result.metrics.index else None
        title = _MARKETS[market][0]
        if m is not None:
            title += f"\nBrier {m.brier:.3f}  ECE {m.ece:.3f}  slope {m.cal_slope:.2f}"
            if "brier_close" in result.metrics.columns and pd.notna(m.get("brier_close")):
                title += f"\n(closing line Brier {m.brier_close:.3f})"
        ax.set_title(title, fontsize=8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("predicted")
        ax.set_ylabel("observed")
    for ax in axes.flat[len(markets) :]:
        ax.axis("off")
    n = int(result.predictions.groupby("market").size().iloc[0])
    fig.suptitle(
        f"Dixon-Coles walk-forward calibration - {n} held-out PL+Championship matches\n"
        "monthly refit, no lookahead   (95% CI bars from held-out counts)",
        fontsize=11,
    )
    fig.tight_layout()
    p1 = out_dir / "reliability.png"
    fig.savefig(p1, dpi=110)
    plt.close(fig)
    return [p1]
