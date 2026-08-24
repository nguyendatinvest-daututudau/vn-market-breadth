"""Test stock_health: indicator tinh toan + scoring + guard cache rong."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from stock_health import (
    MIN_HISTORY,
    compute_symbol_health,
    osc_score,
    _macd,
    _pct_change,
    _stoch,
    _willr,
)


def make_df(n=300, seed=7, trend=0.5):
    rng = np.random.default_rng(seed)
    close = 20 + np.cumsum(rng.normal(trend / 100, 0.4, n))
    close = np.maximum(close, 5)
    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, n)
    dates = pd.bdate_range("2024-01-02", periods=n)
    return pd.DataFrame({
        "TradingDate": dates, "Open": open_, "High": high,
        "Low": low, "Close": close, "Volume": rng.integers(10_000, 500_000, n).astype(float),
    })


def test_compute_symbol_health_full():
    df = make_df()
    h = compute_symbol_health(df, "TEST", None)
    assert h is not None
    assert h["close"] > 0
    assert h["chg_pct"] is not None
    # MA stack day du voi 300 bars
    for k in ("sma10", "sma20", "sma50", "sma100", "sma200"):
        assert h["ma"][k] is not None
        assert h["ma"]["dist"][k] is not None
    assert isinstance(h["ma"]["cross_20_50"], bool)
    # Oscillators
    assert 0 <= h["osc"]["rsi14"] <= 100
    assert 0 <= h["osc"]["stoch_k"] <= 100
    assert -100 <= h["osc"]["willr"] <= 0
    assert h["osc"]["adx14"] is None or 0 <= h["osc"]["adx14"] <= 100
    # 52W
    assert h["w52"]["high"] >= h["close"] * 0.9
    assert 0 <= h["w52"]["pos"] <= 100
    # Perf + risk
    assert h["perf"]["ytd"] is not None
    assert h["risk"]["atr14"] is not None
    assert h["risk"]["atr_pct"] > 0
    # Pivots
    assert h["pivots"]["p"] is not None
    assert h["pivots"]["r1"] > h["pivots"]["p"] > h["pivots"]["s1"]
    # Spark 90 bars
    assert len(h["spark"]) == 90
    # Scores trong [-1, 1]
    for s in (h["ma_score"], h["osc_score"], h["overall"]):
        assert -1.0 <= s <= 1.0


def test_compute_symbol_health_short_history():
    df = make_df().head(30)
    assert compute_symbol_health(df, "TEST", None) is None


def test_beta_vs_index():
    from stock_health import _beta

    df = make_df()
    # Index dai hon va co nhan index goc lech hẳn so voi ma (mo phong VNI.csv 749 dong vs ma ~2000 dong)
    idx_long = pd.concat([make_df(seed=11)["Close"]] * 3 + [make_df(seed=11)["Close"].tail(100)], ignore_index=True)
    beta = _beta(df["Close"], idx_long)
    assert beta is not None
    assert -5 <= beta <= 5
    # Ca hai cung do dai nhung khac nhan index -> van phai tinh duoc
    b2 = _beta(df["Close"], make_df(seed=13).tail(200)["Close"])
    assert b2 is not None


def test_osc_score_extremes():
    assert osc_score({"rsi14": 20, "stoch_k": 10, "willr": -95, "cci20": -150}) == 1.0
    assert osc_score({"rsi14": 80, "stoch_k": 90, "willr": -5, "cci20": 150}) == -1.0
    assert osc_score({"rsi14": 50, "stoch_k": 50}) == 0.0


def test_macd_stoch_willr_shapes():
    df = make_df()
    macd, sig, hist = _macd(df["Close"])
    assert None not in (macd, sig, hist)
    assert abs((macd - sig) - hist) < 1e-6
    k, d = _stoch(df)
    assert 0 <= k <= 100 and 0 <= d <= 100
    wr = _willr(df)
    assert -100 <= wr <= 0


def test_pct_change_direction():
    close = pd.Series([10.0] * 50 + [11.0])
    assert abs(_pct_change(close, 5) - 10.0) < 1e-6
