"""Smoke test cho calibrate_conviction: build_frame + pipeline tren du lieu tong hop."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import calibrate_conviction as cc


def _write_cache(tmp_path: Path, symbol: str, n: int = 300, seed: int = 7):
    """CSV OHLC gia lap: xu the tang de co tin hieu momentum."""
    rng = np.random.default_rng(seed)
    drift = np.linspace(0, 40, n)
    noise = rng.normal(0, 0.8, n).cumsum() * 0.15
    close = 20 + drift * 0.5 + noise
    close = np.maximum(close, 5.0)
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = (high + low) / 2
    volume = rng.integers(600_000, 2_000_000, n).astype(float)
    dates = pd.bdate_range("2023-01-02", periods=n)
    df = pd.DataFrame({
        "TradingDate": dates.strftime("%d/%m/%Y"),
        "Open": open_, "High": high, "Low": low, "Close": close,
        "Volume": volume,
    })
    df.to_csv(tmp_path / f"{symbol}.csv", index=False)


def test_build_frame_produces_expected_columns(tmp_path):
    cc.CACHE_DIR  # module constant exists
    orig = cc.CACHE_DIR
    try:
        cc.CACHE_DIR = tmp_path
        _write_cache(tmp_path, "AAA")
        fr = cc.build_frame("AAA")
    finally:
        cc.CACHE_DIR = orig
    assert fr is not None
    for col in ("symbol", "date", "fwd10", "score", "strategy_ok", "eligible_a",
                "hot", "_valid", "vol_ratio", "dist_ma20", "atr_pct", "w52_pos"):
        assert col in fr.columns, col
    # conviction duoc them boi apply_regime_factor
    out = cc.apply_regime_factor(fr, {})
    assert "conviction" in out.columns
    assert out["factor"].notna().all()
    assert fr["_valid"].any()
    # valid chi nam trong khoang [MIN_SYMBOL_HISTORY, n - max_fwd)
    n = len(fr)
    start = fr.index[fr["_valid"]][0] if fr["_valid"].any() else None
    assert start == cc.MIN_SYMBOL_HISTORY
    last_valid = fr.index[fr["_valid"]][-1]
    assert last_valid <= n - 20


def test_apply_regime_factor_maps_tones():
    from _shared import MIN_SYMBOL_HISTORY
    idx = pd.date_range("2024-01-01", periods=260, freq="B")
    close = pd.Series(np.linspace(10, 50, 260))
    big = pd.DataFrame({
        "symbol": "AAA",
        "date": idx,
        "raw": np.where(np.arange(260) >= MIN_SYMBOL_HISTORY, 55.0, np.nan),
    })
    regime_map = {pd.Timestamp("2024-01-01"): "risk_on", pd.Timestamp("2024-12-01"): "risk_off"}
    out = cc.apply_regime_factor(big, regime_map)
    assert abs(out["factor"].iloc[0] - 1.0) < 1e-9
    assert out["factor"].isna().sum() == 0
    # ngay khong trong map -> 1.0
    mid = out["conviction"].dropna()
    assert (mid == 55.0).all()
    # ngay cuoi (risk_off) -> round(55*0.8)=44
    tail = pd.to_datetime(out["date"]).dt.normalize() == pd.Timestamp("2024-12-01").normalize()
    if tail.any():
        assert (out.loc[tail, "conviction"].dropna() == 44.0).all()


def test_hit_stats_basic():
    idx = pd.date_range("2020-01-01", periods=400, freq="B")[:400]
    df = pd.DataFrame({
        "symbol": "A",
        "date": list(idx[:200]) + list(pd.date_range("2023-01-01", periods=200, freq="B")),
        "fwd10": np.concatenate([np.full(200, 0.05), np.full(200, -0.01)]),
        "_valid": True,
    })
    m = pd.Series(True, index=df.index)
    s = cc.hit_stats(df, m)
    assert s["n"] == 400
    assert s["hit2"] == 0.5
    assert s["train_n"] == 200 and s["test_n"] == 200
    assert s["train_hit2"] == 1.0 and s["test_hit2"] == 0.0


def test_equivalence_check_skips_without_momentum_file(tmp_path):
    import calibrate_conviction as mod
    orig = mod.MOMENTUM_JSON
    try:
        mod.MOMENTUM_JSON = tmp_path / "missing.json"
        big = pd.DataFrame({"symbol": ["A"], "date": pd.to_datetime(["2024-01-02"]),
                            "strategy_ok": [1], "score": [45.0]})
        eq = mod.equivalence_check(big)
    finally:
        mod.MOMENTUM_JSON = orig
    assert eq["status"] == "skip"
