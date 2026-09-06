"""Bet log persistence.

SQLite for now (works locally + drives the analysis views). Streamlit Cloud's
filesystem is ephemeral, so the page also offers CSV export / import. Swapping to
Supabase later is a matter of reimplementing this class's five methods.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PATH = Path("data/bet_log.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    placed_at     TEXT NOT NULL,
    fixture       TEXT NOT NULL,
    market        TEXT NOT NULL,
    legs          TEXT,
    selection     TEXT,
    stake         REAL NOT NULL,
    price_taken   REAL NOT NULL,
    model_prob    REAL,
    closing_price REAL,
    result        TEXT DEFAULT 'pending',   -- pending | won | lost | void
    notes         TEXT
);
"""

COLUMNS = [
    "id",
    "placed_at",
    "fixture",
    "market",
    "legs",
    "selection",
    "stake",
    "price_taken",
    "model_prob",
    "closing_price",
    "result",
    "notes",
]


@dataclass
class BetLog:
    path: Path = DEFAULT_PATH

    def _conn(self) -> sqlite3.Connection:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.execute(_SCHEMA)
        return conn

    def add(self, **fields) -> int:
        keys = [k for k in COLUMNS if k != "id" and k in fields]
        with self._conn() as c:
            cur = c.execute(
                f"INSERT INTO bets ({','.join(keys)}) VALUES ({','.join('?' * len(keys))})",
                [fields[k] for k in keys],
            )
            return int(cur.lastrowid)

    def update(self, bet_id: int, **fields) -> None:
        keys = [k for k in fields if k in COLUMNS and k != "id"]
        if not keys:
            return
        with self._conn() as c:
            c.execute(
                f"UPDATE bets SET {','.join(k + '=?' for k in keys)} WHERE id=?",
                [*(fields[k] for k in keys), bet_id],
            )

    def delete(self, bet_id: int) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM bets WHERE id=?", (bet_id,))

    def all(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query("SELECT * FROM bets ORDER BY placed_at DESC", c)

    def import_csv(self, df: pd.DataFrame) -> int:
        keep = [col for col in COLUMNS if col != "id" and col in df.columns]
        n = 0
        for row in df[keep].itertuples(index=False):
            self.add(**dict(zip(keep, row, strict=True)))
            n += 1
        return n


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def _profit(row: pd.Series) -> float:
    if row["result"] == "won":
        return row["stake"] * (row["price_taken"] - 1.0)
    if row["result"] == "lost":
        return -row["stake"]
    return 0.0


def summarise(df: pd.DataFrame) -> dict:
    settled = df[df["result"].isin(["won", "lost"])].copy()
    if settled.empty:
        return {"n": 0}
    settled["profit"] = settled.apply(_profit, axis=1)
    staked = settled["stake"].sum()
    clv = settled.dropna(subset=["closing_price"])
    clv_pct = (
        (clv["price_taken"] / clv["closing_price"] - 1.0) if len(clv) else pd.Series(dtype=float)
    )
    return {
        "n": len(settled),
        "staked": float(staked),
        "profit": float(settled["profit"].sum()),
        "roi": float(settled["profit"].sum() / staked) if staked else 0.0,
        "win_rate": float((settled["result"] == "won").mean()),
        "n_clv": int(len(clv)),
        "clv_mean": float(clv_pct.mean()) if len(clv) else None,
        "beat_close_rate": float((clv_pct > 0).mean()) if len(clv) else None,
    }


def calibration_table(df: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    settled = df[df["result"].isin(["won", "lost"])].dropna(subset=["model_prob"]).copy()
    if settled.empty:
        return pd.DataFrame()
    settled["hit"] = (settled["result"] == "won").astype(int)
    settled["bucket"] = pd.cut(settled["model_prob"], np.linspace(0, 1, bins + 1))
    g = settled.groupby("bucket", observed=True).agg(
        n=("hit", "size"), model=("model_prob", "mean"), actual=("hit", "mean")
    )
    return g.reset_index()
