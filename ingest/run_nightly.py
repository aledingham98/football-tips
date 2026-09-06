"""Nightly data job - run by .github/workflows/nightly-data.yml.

Order (each step independent; a later failure never discards earlier output):

1. match history      football-data.co.uk (+ football-data.org)  -> data/processed/matches.parquet
2. team ratings       weighted Dixon-Coles fit                   -> data/model/ratings.parquet
3. calibration        walk-forward on the last complete season   -> data/processed/calibration_metrics.parquet
                                                                    reports/calibration/reliability.png
4. calibration map    logistic recalibration on held-out preds   -> data/model/calibration_map.json
5. FBref (best effort) player + team stats, all four tiers       -> data/processed/fbref/*.parquet

The workflow commits whatever changed back to main; Streamlit Cloud redeploys.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import traceback

import pandas as pd

from config.loader import leagues_config, model_config
from models.backtest import plot_calibration, walk_forward_calibration
from models.calibration import fit_calibration_map
from models.dixon_coles import DixonColesConfig, fit


def _default_holdout() -> str:
    today = dt.date.today()
    year = today.year if today.month >= 8 else today.year - 1
    return f"{year - 1}-08-01"


def _step(name: str, fn) -> bool:
    print(f"\n=== {name} ===")
    try:
        fn()
        print(f"--- {name}: ok")
        return True
    except Exception:  # noqa: BLE001
        print(f"--- {name}: FAILED")
        traceback.print_exc()
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=_default_holdout())
    # FBref is opt-in and run as its own workflow step (it is slow and can hang on
    # Cloudflare); the core pipeline below never waits on it.
    ap.add_argument("--with-fbref", action="store_true")
    ap.add_argument("--fbref-throttle", type=float, default=4.0)
    args = ap.parse_args()

    lc = leagues_config()
    all_leagues = list(lc["leagues"])
    seasons = lc["seasons"]["history"]
    mcfg = model_config()
    ok = {}

    def step_matches():
        from ingest.matches import build_match_history

        build_match_history(all_leagues, seasons)

    def step_fixtures():
        from ingest.fixtures import build_fixtures

        build_fixtures(all_leagues)

    def step_ratings():
        matches = pd.read_parquet("data/processed/matches.parquet")
        ratings = fit(matches, DixonColesConfig.from_yaml(mcfg))
        ratings.to_parquet("data/model/ratings.parquet")
        print(f"  ratings: {ratings.meta}")

    def step_calibration():
        matches = pd.read_parquet("data/processed/matches.parquet")
        result = walk_forward_calibration(
            matches, holdout_start=args.holdout, config=DixonColesConfig.from_yaml(mcfg)
        )
        result.predictions.to_parquet("data/processed/calibration_predictions.parquet")
        result.metrics.to_parquet("data/processed/calibration_metrics.parquet")
        plot_calibration(result)
        print("\n" + result.metrics.round(4).to_string())

    def step_calibration_map():
        preds = pd.read_parquet("data/processed/calibration_predictions.parquet")
        cmap = fit_calibration_map(preds, by_league=True)
        cmap.to_json("data/model/calibration_map.json")
        print(f"  calibration map: {cmap.meta}, maps for {sorted(cmap.params)}")

    def step_fbref():
        from ingest.fbref import ingest_fbref

        current = [lc["seasons"]["current"]]
        manifest = ingest_fbref(all_leagues, current, throttle=args.fbref_throttle)
        print(f"  fbref manifest: {manifest}")

    ok["matches"] = _step("1. match history", step_matches)
    if ok["matches"]:
        ok["ratings"] = _step("2. team ratings", step_ratings)
        ok["calibration"] = _step("3. calibration", step_calibration)
        if ok.get("calibration"):
            ok["calibration_map"] = _step("4. calibration map", step_calibration_map)
    ok["fixtures"] = _step("5. upcoming fixtures", step_fixtures)
    if args.with_fbref:
        ok["fbref"] = _step("6. FBref (best effort, current season)", step_fbref)

    print(f"\n=== summary: {ok} ===")
    # fail the job only if the core pipeline (matches -> ratings) broke
    return 0 if ok.get("matches") and ok.get("ratings") else 1


if __name__ == "__main__":
    sys.exit(main())
