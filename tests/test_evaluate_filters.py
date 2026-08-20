"""Unit tests cho evaluate_filters (metrics, helpers, walk-forward) — không cần cache."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import numpy as np
import pandas as pd
import pytest

from evaluate_filters import (
    _obv, _rolling_slope_norm, metrics, sell_metrics, walk_forward,
    MIN_OBSERVATIONS,
)


def _frame(n=400, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", periods=n)
    close = 20 * np.cumprod(1 + rng.normal(0.0003, 0.02, n))
    fwd10 = (np.roll(close, -10) / close - 1.0)
    fwd10[-10:] = np.nan
    vol_ok = np.ones(n, dtype=np.int8)
    valid = np.zeros(n, dtype=np.int8)
    valid[220 : n - 20] = 1
    df = pd.DataFrame({
        "date": dates,
        "_valid": valid,
        "vol_ok": vol_ok,
        "fwd10": fwd10.astype(np.float32),
        "e_ma": rng.integers(0, 2, n).astype(np.int8),
        "e_pb": rng.integers(0, 2, n).astype(np.int8),
        "e_bo": rng.integers(0, 2, n).astype(np.int8),
        "e_mo": rng.integers(0, 2, n).astype(np.int8),
        "mom_any": rng.integers(0, 2, n).astype(np.int8),
        "ens_any": rng.integers(0, 2, n).astype(np.int8),
    })
    return df


def test_metrics_basic():
    frame = _frame()
    out = metrics(frame, np.ones(len(frame), dtype=bool))
    assert out is not None
    assert out["n"] > MIN_OBSERVATIONS
    assert 0.0 <= out["hit2"] <= 1.0
    assert "yearly" in out
    assert "train_hit2" in out
    assert "test_hit2" in out


def test_metrics_requires_min_obs():
    frame = _frame()
    mask = np.zeros(len(frame), dtype=bool)
    mask[:5] = True
    assert metrics(frame, mask) is None


def test_metrics_ignores_invalid_rows():
    frame = _frame()
    # All rows invalid -> None even if mask wide
    frame["_valid"] = np.zeros(len(frame), dtype=np.int8)
    assert metrics(frame, np.ones(len(frame), dtype=bool)) is None


def test_sell_metrics():
    frame = _frame()
    col = np.zeros(len(frame), dtype=np.int8)
    valid_idx = np.where(np.asarray(frame["_valid"], dtype=bool))[0][:200]
    col[valid_idx] = 1
    frame["test_sell"] = col
    out = sell_metrics(frame, "test_sell")
    assert out is not None
    assert set(out) == {"n", "p_drop0", "p_drop2", "median", "mean"}


def test_obv_up_only():
    close = pd.Series(np.arange(1.0, 6.0))
    vol = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    obv = _obv(close, vol)
    assert np.allclose(obv.to_numpy(), [0, 2, 5, 9, 14])


def test_rolling_slope_increasing_positive():
    s = pd.Series(np.arange(1.0, 31.0))
    slope = _rolling_slope_norm(s, 25)
    last = slope.iloc[-1]
    assert last > 0


def test_rolling_slope_decreasing_negative():
    s = pd.Series(np.arange(30.0, 0.0, -1.0))
    slope = _rolling_slope_norm(s, 25)
    assert slope.iloc[-1] < 0


def test_walk_forward_structure():
    frame = _frame(n=600)
    out = walk_forward(frame)
    assert "weights_train" in out
    assert "weights_current" in out
    assert set(out["weights_train"]) == {"ma_crossover", "pullback", "breakout", "momentum"}
    for key in ("current_strong", "oos_strong", "current_weak_low", "oos_weak_low"):
        assert key in out
        assert "train" in out[key] and "test" in out[key]