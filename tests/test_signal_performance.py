"""Unit tests for signal_performance — pure helpers, no OHLC cache needed."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import signal_performance as sp


def _frame(prices, highs=None, lows=None, start="2026-08-01"):
    dates = pd.date_range(start, periods=len(prices), freq="B")
    df = pd.DataFrame({
        "TradingDate": dates.strftime("%Y-%m-%d"),
        "Open": prices, "High": highs or prices,
        "Low": lows or prices, "Close": prices,
        "Volume": [1_000_000] * len(prices),
    })
    df["_d"] = pd.to_datetime(df["TradingDate"])
    return df


def test_is_buy_sell_status():
    assert sp._is_buy({"symbol": "X"}) is True
    assert sp._is_buy({"status": "VALID_BUY"}) is True
    assert sp._is_buy({"status": "BUY"}) is True
    assert sp._is_buy({"status": "WATCHLIST"}) is False
    assert sp._is_buy({"status": "SELL_WARNING"}) is False
    assert sp._is_sell({"status": "SELL"}) is True
    assert sp._is_sell({"status": "BUY"}) is False


def test_entry_price_prefers_buy_price():
    assert sp._entry_price({"buy_price": 100.0, "last_price": 90.0}) == 100.0
    assert sp._entry_price({"last_price": 90.0}) == 90.0
    assert sp._entry_price({"last_price": -5}) is None
    assert sp._entry_price({}) is None


def test_evaluate_signal_forward_returns():
    prices = [100.0] * 5 + [103.0] * 20  # +3% tu T+5 tro di
    df = _frame(prices)
    ev = sp.evaluate_signal(df, 4, 100.0)
    assert ev["matured"][5] is True
    assert abs(ev["fwd"][5] - 0.03) < 1e-9
    assert abs(ev["fwd"][10] - 0.03) < 1e-9


def test_evaluate_signal_immature():
    df = _frame([100.0] * 6)
    ev = sp.evaluate_signal(df, 4, 100.0)
    assert ev["matured"][5] is False
    assert ev["matured"][10] is False
    assert ev["fwd"].get(10) is None


def test_plan_target_before_stop():
    prices = [100.0] * 30
    highs = [100.0] * 30
    lows = [100.0] * 30
    highs[6] = 110.0  # target hit o T+2 (entry=100, ATR fallback=2 -> target=104)
    df = _frame(prices, highs, lows)
    ev = sp.evaluate_signal(df, 4, 100.0)
    assert ev["plan"]["outcome"] == "target_first"
    assert ev["plan"]["bars"] == 2


def test_plan_stop_before_target():
    prices = [100.0] * 30
    highs = [100.0] * 30
    lows = [100.0] * 30
    lows[7] = 90.0  # stop hit o T+3 (stop=96)
    df = _frame(prices, highs, lows)
    ev = sp.evaluate_signal(df, 4, 100.0)
    assert ev["plan"]["outcome"] == "stop_first"
    assert ev["plan"]["bars"] == 3


def test_summarize_hit_rates():
    s = sp._summarize([0.03, 0.03, -0.05, 0.10])
    assert s["n"] == 4
    assert s["hit2"] == 0.75
    assert s["hit7"] == 0.25
    assert s["hit0"] == 0.75
    assert sp._summarize([]) is None
