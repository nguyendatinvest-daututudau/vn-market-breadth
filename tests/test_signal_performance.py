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


def test_first_touch_up_down_tie_timeout():
    import numpy as np
    n = 30
    base_h = np.array([100.0] * n)
    base_l = np.array([100.0] * n)
    # up first o bar 2
    h = base_h.copy(); h[2] = 106.0
    assert sp._first_touch(h, base_l, n, 0, 105.0, 95.0, 20) == ("up_first", 2)
    # down first o bar 3
    l = base_l.copy(); l[3] = 94.0
    assert sp._first_touch(base_h, l, n, 0, 105.0, 95.0, 20) == ("down_first", 3)
    # cung bar = tie
    h2 = base_h.copy(); l2 = base_l.copy(); h2[1] = 106.0; l2[1] = 94.0
    assert sp._first_touch(h2, l2, n, 0, 105.0, 95.0, 20) == ("tie", 1)
    # khong cham = timeout
    assert sp._first_touch(base_h, base_l, n, 0, 105.0, 95.0, 20) == ("timeout", None)
    # muc invalid
    assert sp._first_touch(base_h, base_l, n, 0, 95.0, 95.0, 20) == ("timeout", None)


def test_pm5_hit_p5_first():
    prices = [100.0] * 30
    highs = [100.0] * 30
    lows = [100.0] * 30
    highs[7] = 106.0  # +6% o T+3
    df = _frame(prices, highs, lows)
    ev = sp.evaluate_signal(df, 4, 100.0)
    assert ev["pm5"]["outcome"] == "hit_p5_first"
    assert ev["pm5"]["bars"] == 3
    assert ev["pm5"]["matured"] is True


def test_pm5_hit_m5_first():
    prices = [100.0] * 30
    highs = [100.0] * 30
    lows = [100.0] * 30
    lows[6] = 94.0  # -6% o T+2
    df = _frame(prices, highs, lows)
    ev = sp.evaluate_signal(df, 4, 100.0)
    assert ev["pm5"]["outcome"] == "hit_m5_first"
    assert ev["pm5"]["bars"] == 2


def test_pm5_immature_not_counted():
    df = _frame([100.0] * 10)  # idx=4 chi co 5 forward bars, khong cham
    ev = sp.evaluate_signal(df, 4, 100.0)
    assert ev["pm5"]["outcome"] == "timeout"
    assert ev["pm5"]["matured"] is False


def _trend_frame(n=250, start=100.0, step=0.5):
    import pandas as pd
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    closes = [start + i * step for i in range(n)]
    df = pd.DataFrame({
        "TradingDate": dates.strftime("%Y-%m-%d"),
        "Open": closes,
        "High": [c * 1.01 for c in closes],
        "Low": [c * 0.99 for c in closes],
        "Close": closes,
        "Volume": [1_000_000] * n,
    })
    df["_d"] = pd.to_datetime(df["TradingDate"])
    return df


def test_recompute_trade_plan_deterministic():
    df = _trend_frame()
    p1 = sp._recompute_trade_plan(df, 249)
    p2 = sp._recompute_trade_plan(df, 249)
    assert p1 is not None and p2 is not None
    assert p1 == p2
    assert p1["targets"]["r1"] > p1["entry_zone"]["mid"]
    assert p1["stop"] < p1["entry_zone"]["low"]
    assert p1["entry_type"] in ("base", "breakout", "pullback")


def test_plan_real_r1_first_on_uptrend():
    df = _trend_frame()
    ev = sp.evaluate_signal(df, 229, float(df["Close"].iloc[229]))
    pr = ev["plan_real"]
    assert pr["matured"] is True
    assert pr["outcome"] == "r1_first"
    assert pr["r1"] is not None and pr["stop"] is not None
