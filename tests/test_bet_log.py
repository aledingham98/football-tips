from __future__ import annotations

import pandas as pd
import pytest

from storage.bet_log import BetLog, calibration_table, summarise


@pytest.fixture
def log(tmp_path):
    return BetLog(path=tmp_path / "bl.db")


def test_add_update_all_roundtrip(log):
    bid = log.add(
        placed_at="2026-09-06T12:00",
        fixture="A v B",
        market="Home win",
        stake=10.0,
        price_taken=2.0,
        model_prob=0.55,
    )
    df = log.all()
    assert len(df) == 1 and df.iloc[0]["result"] == "pending"
    log.update(bid, result="won", closing_price=1.85)
    assert log.all().iloc[0]["closing_price"] == pytest.approx(1.85)


def test_summarise_profit_roi_and_clv(log):
    log.add(
        placed_at="t1",
        fixture="A v B",
        market="m",
        stake=10,
        price_taken=2.0,
        model_prob=0.5,
        closing_price=1.8,
        result="won",
    )
    log.add(
        placed_at="t2",
        fixture="C v D",
        market="m",
        stake=10,
        price_taken=3.0,
        model_prob=0.3,
        closing_price=3.2,
        result="lost",
    )
    s = summarise(log.all())
    assert s["n"] == 2
    assert s["profit"] == pytest.approx(10 * 1.0 - 10)  # +10 win, -10 loss = 0
    assert s["staked"] == pytest.approx(20)
    assert s["n_clv"] == 2
    # took 2.0 vs close 1.8 => +11%; took 3.0 vs close 3.2 => -6.25%
    assert s["clv_mean"] == pytest.approx(((2.0 / 1.8 - 1) + (3.0 / 3.2 - 1)) / 2)
    assert s["beat_close_rate"] == pytest.approx(0.5)


def test_calibration_table_buckets(log):
    for i in range(40):
        won = "won" if i % 2 == 0 else "lost"
        log.add(
            placed_at=f"t{i}",
            fixture="x",
            market="m",
            stake=1,
            price_taken=2.0,
            model_prob=0.3 + 0.01 * (i % 30),
            result=won,
        )
    cal = calibration_table(log.all(), bins=5)
    assert not cal.empty
    assert {"n", "model", "actual"} <= set(cal.columns)


def test_import_csv_appends(log):
    log.add(placed_at="t1", fixture="A v B", market="m", stake=5, price_taken=2.0)
    df = pd.DataFrame(
        [
            {
                "placed_at": "t2",
                "fixture": "C v D",
                "market": "m",
                "stake": 7,
                "price_taken": 1.9,
                "model_prob": 0.6,
                "result": "pending",
            }
        ]
    )
    n = log.import_csv(df)
    assert n == 1
    assert len(log.all()) == 2
