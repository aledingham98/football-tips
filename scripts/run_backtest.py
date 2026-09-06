"""Phase 2 gate: walk-forward calibration of the Dixon-Coles model on held-out data.

    uv run python -m scripts.run_backtest [--holdout 2025-08-01] [--half-life 180]

Writes reports/calibration/reliability.png and data/processed/calibration_*.parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from models.backtest import plot_calibration, walk_forward_calibration
from models.dixon_coles import DixonColesConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", default="data/processed/matches.parquet")
    ap.add_argument("--holdout", default="2025-08-01", help="first date of the held-out period")
    ap.add_argument("--half-life", type=float, default=180.0)
    ap.add_argument("--l2", type=float, default=None, help="override config ratings_l2")
    ap.add_argument("--out", default="reports/calibration")
    args = ap.parse_args()

    matches = pd.read_parquet(args.matches)
    from config.loader import model_config

    cfg = DixonColesConfig.from_yaml(model_config())
    cfg.time_decay_half_life_days = args.half_life
    if args.l2 is not None:
        cfg.ratings_l2 = args.l2
    result = walk_forward_calibration(matches, holdout_start=args.holdout, config=cfg)

    cols = ["n", "base_rate", "mean_pred", "brier", "logloss", "ece", "cal_slope", "cal_intercept"]
    print("\n" + result.metrics[cols].round(4).to_string())

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    result.predictions.to_parquet("data/processed/calibration_predictions.parquet")
    result.metrics.to_parquet("data/processed/calibration_metrics.parquet")
    paths = plot_calibration(result, out_dir=args.out)
    print("\nwrote:", *[str(p) for p in paths], "+ calibration_{predictions,metrics}.parquet")


if __name__ == "__main__":
    main()
