"""The Odds API credit budgeter.

Free Starter tier = 500 credits/month. Credits are consumed as
``markets x regions`` per call, so ``region=uk`` + ``markets=h2h,totals,btts`` is
**3 credits per league per call**. The budgeter:

* persists the remaining quota to a small JSON file (survives the ephemeral
  filesystem because the scheduled job commits it back to the repo),
* refuses any call that would take the remaining balance below ``floor``
  (default 50) - raising :class:`QuotaExceeded` *before* any HTTP request,
* takes the ``x-requests-remaining`` response header as the source of truth
  after every successful call.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

MARKET_CREDIT_COST = 1  # per market per region
DEFAULT_FLOOR = 50


class QuotaExceeded(RuntimeError):
    """Raised when a call would breach the configured quota floor."""


def call_cost(n_markets: int, n_regions: int = 1) -> int:
    return n_markets * n_regions * MARKET_CREDIT_COST


@dataclass
class CreditBudgeter:
    state_path: Path
    floor: int = DEFAULT_FLOOR
    monthly_allowance: int = 500

    def _read(self) -> dict:
        try:
            return json.loads(Path(self.state_path).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {"remaining": self.monthly_allowance, "updated": None, "source": "default"}

    def _write(self, state: dict) -> None:
        state["updated"] = dt.datetime.now(dt.UTC).isoformat()
        p = Path(self.state_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(state, indent=2, sort_keys=True))

    # --- queries ---------------------------------------------------------- #
    @property
    def remaining(self) -> int:
        return int(self._read()["remaining"])

    def can_afford(self, cost: int) -> bool:
        return self.remaining - cost >= self.floor

    def check(self, cost: int) -> None:
        """Raise :class:`QuotaExceeded` unless ``cost`` keeps us at/above the floor."""
        rem = self.remaining
        if rem - cost < self.floor:
            raise QuotaExceeded(
                f"call costs {cost} credits; {rem} remaining and floor is {self.floor}. "
                f"Refusing (would leave {rem - cost}). Skipping The Odds API this run."
            )

    # --- updates -------------------------------------------------------- #
    def update_from_header(self, remaining_header: str | int | None) -> None:
        """Authoritative update from ``x-requests-remaining``."""
        if remaining_header is None:
            return
        state = self._read()
        state["remaining"] = int(float(remaining_header))
        state["source"] = "header"
        self._write(state)

    def record_spend(self, cost: int) -> None:
        """Fallback decrement when no header was returned."""
        state = self._read()
        state["remaining"] = max(0, int(state["remaining"]) - cost)
        state["source"] = "estimated"
        self._write(state)

    def warn_if_low(self) -> str | None:
        rem = self.remaining
        if rem < self.floor:
            return f"The Odds API quota CRITICAL: {rem} credits left (floor {self.floor})."
        if rem < self.floor * 2:
            return f"The Odds API quota low: {rem} credits left."
        return None
