from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.calibration import CalibrationMap, fit_calibration_map


def _biased_predictions(n=4000, bias=0.08, seed=0):
    """Model that is systematically `bias` too low; true outcome follows pred+bias."""
    rng = np.random.default_rng(seed)
    pred = rng.uniform(0.15, 0.85, n)
    true_p = np.clip(pred + bias, 0.01, 0.99)
    outcome = (rng.random(n) < true_p).astype(int)
    return pd.DataFrame({"market": "over_2.5", "league": "ECH", "pred": pred, "outcome": outcome})


def test_map_corrects_a_systematic_bias():
    df = _biased_predictions(bias=0.08)
    cmap = fit_calibration_map(df, by_league=False)
    raw_gap = df["outcome"].mean() - df["pred"].mean()
    corrected = cmap.apply("over_2.5", df["pred"].to_numpy())
    corr_gap = df["outcome"].mean() - corrected.mean()
    assert abs(corr_gap) < abs(raw_gap) / 2
    assert abs(corr_gap) < 0.02


def test_identity_when_too_little_data():
    df = _biased_predictions(n=50)
    cmap = fit_calibration_map(df, by_league=False, min_n=200)
    assert cmap.is_identity("over_2.5")
    assert cmap.apply("over_2.5", 0.5) == pytest.approx(0.5)


def test_well_calibrated_input_stays_put():
    rng = np.random.default_rng(1)
    pred = rng.uniform(0.1, 0.9, 4000)
    outcome = (rng.random(4000) < pred).astype(int)
    df = pd.DataFrame({"market": "result_home", "pred": pred, "outcome": outcome})
    cmap = fit_calibration_map(df, by_league=False)
    corrected = cmap.apply("result_home", pred)
    assert np.mean(np.abs(corrected - pred)) < 0.03


def test_roundtrip_json(tmp_path):
    df = _biased_predictions()
    cmap = fit_calibration_map(df, by_league=True)
    p = tmp_path / "cal.json"
    cmap.to_json(p)
    loaded = CalibrationMap.from_json(p)
    assert loaded.apply("over_2.5", 0.4, "ECH") == pytest.approx(cmap.apply("over_2.5", 0.4, "ECH"))


def test_apply_is_monotonic():
    df = _biased_predictions(bias=0.1)
    cmap = fit_calibration_map(df, by_league=False)
    xs = np.linspace(0.05, 0.95, 50)
    ys = cmap.apply("over_2.5", xs)
    assert np.all(np.diff(ys) > 0)
