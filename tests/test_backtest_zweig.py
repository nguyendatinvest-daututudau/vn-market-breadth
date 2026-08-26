"""Tests for the event-aligned Zweig VN-Index backtest."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import backtest_zweig as bt
import backfill_zweig as bf


def _history(closes):
    start = datetime(2026, 1, 1)
    return [
        {
            "date": (start + timedelta(days=i)).strftime("%d/%m/%Y"),
            "index_closes": {"VNI": close},
            "markets": {},
        }
        for i, close in enumerate(closes)
    ]


def test_evaluate_events_aligns_forward_returns_by_exact_date():
    history = _history([100 + i for i in range(70)])
    event_date = history[0]["date"]

    report = bt.evaluate_events(history, [event_date])

    assert report["matched_events"] == 1
    assert report["horizons"]["T+5"]["events"]["n"] == 1
    assert report["horizons"]["T+5"]["events"]["avg_return_pct"] == 5.0


def test_evaluate_events_does_not_substitute_missing_index_date():
    history = _history([100 + i for i in range(70)])
    report = bt.evaluate_events(history, ["31/12/2025"])
    assert report["matched_events"] == 0
    assert report["horizons"]["T+20"]["events"]["n"] == 0


def test_backfill_zweig_preserves_existing_and_fills_missing_hose_ad():
    history = [
        {"date": "01/01/2026", "markets": {"HOSE": {"advances": 10, "declines": 20}}},
        {"date": "02/01/2026", "markets": {"HOSE": {}}},
    ]

    class Client:
        def daily_index(self, index_id, from_date, to_date):
            assert index_id == "VNINDEX"
            return [{"Advances": 30, "Declines": 15, "Nochanges": 5}]

    filled, failed = bf.backfill(history, Client(), sleep_seconds=0, checkpoint=0)

    assert (filled, failed) == (1, 0)
    assert history[0]["markets"]["HOSE"]["advances"] == 10
    assert history[1]["markets"]["HOSE"]["advances"] == 30
    assert history[1]["markets"]["HOSE"]["total_symbols"] == 50
