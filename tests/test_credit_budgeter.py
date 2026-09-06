"""The credit budgeter must refuse to exceed quota - and the provider must make
no HTTP request when it refuses.
"""

from __future__ import annotations

import json

import pytest

from ingest.odds.budget import CreditBudgeter, QuotaExceeded, call_cost
from ingest.odds.the_odds_api import TheOddsApiProvider


@pytest.fixture
def state_file(tmp_path):
    return tmp_path / "odds" / "quota.json"


def _seed(path, remaining):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"remaining": remaining}))


def test_call_cost_is_markets_times_regions():
    assert call_cost(3, 1) == 3
    assert call_cost(3, 2) == 6
    assert call_cost(1, 1) == 1


def test_allows_call_that_stays_at_or_above_floor(state_file):
    _seed(state_file, 55)
    b = CreditBudgeter(state_file, floor=50)
    assert b.can_afford(3) is True
    b.check(3)  # 55 - 3 = 52 >= 50, no raise


def test_refuses_call_that_would_breach_floor(state_file):
    _seed(state_file, 52)
    b = CreditBudgeter(state_file, floor=50)
    assert b.can_afford(3) is False
    with pytest.raises(QuotaExceeded):
        b.check(3)  # 52 - 3 = 49 < 50


def test_boundary_exactly_at_floor_is_allowed(state_file):
    _seed(state_file, 53)
    b = CreditBudgeter(state_file, floor=50)
    b.check(3)  # leaves exactly 50


def test_header_update_is_authoritative(state_file):
    _seed(state_file, 200)
    b = CreditBudgeter(state_file, floor=50)
    b.update_from_header("137")
    assert b.remaining == 137
    b.record_spend(3)
    assert b.remaining == 134


def test_missing_state_file_defaults_to_full_allowance(tmp_path):
    b = CreditBudgeter(tmp_path / "nope.json", floor=50, monthly_allowance=500)
    assert b.remaining == 500


class _ExplodingSession:
    """Any HTTP call is a test failure."""

    def get(self, *a, **k):  # noqa: D401,ANN001
        raise AssertionError("HTTP request made despite quota block")


class _FakeResponse:
    def __init__(self, payload, remaining):
        self._payload = payload
        self.headers = {"x-requests-remaining": str(remaining)}
        self.status_code = 200

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class _RecordingSession:
    def __init__(self, response):
        self._response = response
        self.calls: list[dict] = []

    def get(self, url, params=None, timeout=None):  # noqa: ANN001
        self.calls.append({"url": url, "params": params})
        return self._response


def test_provider_makes_no_request_when_quota_blocks(state_file):
    _seed(state_file, 51)  # 51 - 3 = 48 < floor 50
    b = CreditBudgeter(state_file, floor=50)
    provider = TheOddsApiProvider("KEY", b, session=_ExplodingSession())
    with pytest.raises(QuotaExceeded):
        provider.fetch(["EPL"])


def test_provider_happy_path_updates_quota_from_header(state_file):
    _seed(state_file, 400)
    b = CreditBudgeter(state_file, floor=50)
    payload = [
        {
            "id": "evt1",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "commence_time": "2026-09-06T14:00:00Z",
            "bookmakers": [
                {
                    "key": "skybet",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Arsenal", "price": 1.9},
                                {"name": "Chelsea", "price": 4.2},
                                {"name": "Draw", "price": 3.6},
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": 1.95, "point": 2.5},
                                {"name": "Under", "price": 1.9, "point": 2.5},
                            ],
                        },
                    ],
                }
            ],
        }
    ]
    session = _RecordingSession(_FakeResponse(payload, remaining=397))
    provider = TheOddsApiProvider("KEY", b, session=session)
    quotes = provider.fetch(["EPL"])

    assert len(session.calls) == 1
    assert b.remaining == 397
    markets = {q.market for q in quotes}
    assert {"result", "total_goals"} <= markets
    home = next(q for q in quotes if q.market == "result" and q.selection == "home")
    assert home.decimal_odds == pytest.approx(1.9)
    assert home.league == "EPL"
