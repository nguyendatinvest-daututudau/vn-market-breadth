"""Sinh data/stock_health.json - the suc khoe co phieu thuan ky thuat cho tung ma.

Mo the ma bat ky tren dashboard se doc tu file nay: MA stack, RSI, Stoch, MACD,
ADX, CCI, Williams %R, 52W, performance, ATR/beta, pivot, sparkline 90 phien.

Chay trong fetch_and_compute.py sau khi cache OHLC da cap nhat.
Neu cache rong thi KHONG ghi de output cu (tranh mat du lieu nhu backtest).
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from _shared import CACHE_DIR, DATA_DIR, DOCS_DATA_DIR, vn_now
from cache_utils import compute_rsi_wilder, load_cache

OUTPUT_NAME = "stock_health.json"
MIN_HISTORY = 60
SPARK_BARS = 90
INDEX_FILES = {"VNI", "HNXINDEX"}  # chi so khong dua vao the co phieu


def _r(v, nd=2):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, nd)


def _sma(close: pd.Series, n: int):
    if len(close) < n:
        return None
    return float(close.tail(n).mean())


def _atr14(df: pd.DataFrame):
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1).dropna()
    if len(tr) < 14:
        return None
    vals = tr.to_numpy()
    atr = vals[:14].mean()
    for i in range(14, len(vals)):
        atr = (atr * 13.0 + vals[i]) / 14.0
    return float(atr)


def _stoch(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    if len(df) < k_period + d_period:
        return None, None
    hh = df["High"].rolling(k_period).max()
    ll = df["Low"].rolling(k_period).min()
    k = (df["Close"] - ll) / (hh - ll).replace(0, np.nan) * 100.0
    k = k.dropna()
    if len(k) < d_period:
        return None, None
    d = k.rolling(d_period).mean().dropna()
    return float(k.iloc[-1]), float(d.iloc[-1])


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    if len(close) < slow + signal:
        return None, None, None
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return float(line.iloc[-1]), float(sig.iloc[-1]), float(line.iloc[-1] - sig.iloc[-1])


def _adx(df: pd.DataFrame, period: int = 14):
    if len(df) < period * 2 + 1:
        return None, None, None
    h, l, c = df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy()
    n = len(df)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        up = h[i] - h[i - 1]
        dn = l[i - 1] - l[i]
        plus_dm[i] = up if up > dn and up > 0 else 0.0
        minus_dm[i] = dn if dn > up and dn > 0 else 0.0
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = tr[1 : period + 1].sum()
    pdm = plus_dm[1 : period + 1].sum()
    mdm = minus_dm[1 : period + 1].sum()
    if atr <= 0:
        return None, None, None
    dxs = []
    for i in range(period + 1, n):
        atr = atr - atr / period + tr[i]
        pdm = pdm - pdm / period + plus_dm[i]
        mdm = mdm - mdm / period + minus_dm[i]
        if atr <= 0:
            continue
        pdi = 100.0 * pdm / atr
        mdi = 100.0 * mdm / atr
        denom = pdi + mdi
        dxs.append(100.0 * abs(pdi - mdi) / denom if denom > 0 else 0.0)
        if len(dxs) == period:
            adx = sum(dxs) / period
        elif len(dxs) > period:
            adx = (adx * (period - 1) + dxs[-1]) / period
    if len(dxs) < period:
        return None, None, None
    return float(adx), float(pdi), float(mdi)


def _cci(df: pd.DataFrame, period: int = 20):
    if len(df) < period:
        return None
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    tp = tp.tail(period)
    sma = tp.mean()
    mad = (tp - sma).abs().mean()
    if mad == 0 or not math.isfinite(mad):
        return None
    return float((tp.iloc[-1] - sma) / (0.015 * mad))


def _willr(df: pd.DataFrame, period: int = 14):
    if len(df) < period:
        return None
    hh = float(df["High"].tail(period).max())
    ll = float(df["Low"].tail(period).min())
    if hh == ll:
        return None
    return float((hh - float(df["Close"].iloc[-1])) / (hh - ll) * -100.0)


def _beta(sym_close: pd.Series, idx_close: pd.Series, lookback: int = 252):
    if sym_close is None or idx_close is None:
        return None
    s = sym_close.tail(lookback).pct_change().dropna()
    i = idx_close.tail(lookback).pct_change().dropna()
    joined = pd.concat([s, i], axis=1, join="inner").dropna()
    if len(joined) < 60:
        return None
    var_i = float(joined.iloc[:, 1].var())
    if var_i <= 0:
        return None
    cov = float(joined.cov().iloc[0, 1])
    return cov / var_i


def _pct_change(close: pd.Series, bars: int):
    if len(close) <= bars:
        return None
    base = float(close.iloc[-1 - bars])
    if base <= 0:
        return None
    return (float(close.iloc[-1]) / base - 1.0) * 100.0


def _ytd_perf(close: pd.Series, dates: pd.Series):
    if len(close) < 2:
        return None
    year = dates.iloc[-1].year
    mask = dates.dt.year == year
    if not mask.any():
        return None
    base = float(close[mask].iloc[0])
    if base <= 0:
        return None
    return (float(close.iloc[-1]) / base - 1.0) * 100.0


def _pivots(df: pd.DataFrame):
    if len(df) < 2:
        return None
    prev = df.iloc[-2]
    h, l, c = float(prev["High"]), float(prev["Low"]), float(prev["Close"])
    if any(not math.isfinite(x) for x in (h, l, c)) or h == l:
        return None
    p = (h + l + c) / 3.0
    return {
        "p": _r(p),
        "r1": _r(2 * p - l),
        "r2": _r(p + (h - l)),
        "s1": _r(2 * p - h),
        "s2": _r(p - (h - l)),
    }


def osc_score(osc: dict) -> float:
    """Diem dao dong -1..+1, quy tac giong Technical Ratings (RSI/Stoch/MACD/ADX/CCI/W%R)."""
    votes = []
    rsi = osc.get("rsi14")
    if rsi is not None:
        votes.append(1 if rsi < 30 else (-1 if rsi > 70 else 0))
    k = osc.get("stoch_k")
    if k is not None:
        votes.append(1 if k < 20 else (-1 if k > 80 else 0))
    macd, sig = osc.get("macd"), osc.get("macd_signal")
    if macd is not None and sig is not None:
        votes.append(1 if macd > sig else -1)
    adx, pdi, mdi = osc.get("adx14"), osc.get("plus_di"), osc.get("minus_di")
    if adx is not None and pdi is not None and mdi is not None:
        if adx > 25:
            votes.append(1 if pdi > mdi else -1)
    cci = osc.get("cci20")
    if cci is not None:
        votes.append(1 if cci < -100 else (-1 if cci > 100 else 0))
    wr = osc.get("willr")
    if wr is not None:
        votes.append(1 if wr < -80 else (-1 if wr > -20 else 0))
    return _r(sum(votes) / len(votes)) if votes else 0.0


def compute_symbol_health(df: pd.DataFrame, symbol: str, idx_close: pd.Series | None) -> dict | None:
    if df is None or len(df) < MIN_HISTORY:
        return None
    df = df.sort_values("TradingDate").reset_index(drop=True)
    close = df["Close"]
    last, prev = float(close.iloc[-1]), float(close.iloc[-2])
    if last <= 0 or prev <= 0:
        return None

    smas = {n: _sma(close, n) for n in (10, 20, 50, 100, 200)}
    above = {f"sma{n}": (True if smas[n] is not None and last > smas[n] else False) for n in smas}
    dist = {
        f"sma{n}": _r((last / smas[n] - 1.0) * 100.0) if smas[n] else None
        for n in smas
    }
    ma_votes = [1 if above[f"sma{n}"] else -1 for n in smas if smas[n] is not None]
    ma_score = _r(sum(ma_votes) / len(ma_votes)) if ma_votes else 0.0

    osc = {
        "rsi14": _r(compute_rsi_wilder(close, 14), 1),
    }
    k, d = _stoch(df)
    osc["stoch_k"], osc["stoch_d"] = _r(k, 1), _r(d, 1)
    macd, sig, hist = _macd(close)
    osc["macd"], osc["macd_signal"], osc["macd_hist"] = _r(macd), _r(sig), _r(hist)
    adx, pdi, mdi = _adx(df)
    osc["adx14"], osc["plus_di"], osc["minus_di"] = _r(adx, 1), _r(pdi, 1), _r(mdi, 1)
    osc["cci20"] = _r(_cci(df), 1)
    osc["willr"] = _r(_willr(df), 1)
    o_score = osc_score(osc)

    w = 252
    win = df.tail(w)
    hi52 = float(win["High"].max()) if len(win) and math.isfinite(float(win["High"].max())) else None
    lo52 = float(win["Low"].min()) if len(win) and math.isfinite(float(win["Low"].min())) else None
    pos52 = pct_from_high = None
    if hi52 and lo52 and hi52 > lo52:
        pos52 = _r((last - lo52) / (hi52 - lo52) * 100.0, 1)
        pct_from_high = _r((last / hi52 - 1.0) * 100.0)
    new_high = bool(hi52 is not None and last >= hi52)
    new_low = bool(lo52 is not None and last <= lo52)

    vol20 = float(df["Volume"].tail(20).mean()) if df["Volume"].notna().any() else None
    last_vol = float(df["Volume"].iloc[-1]) if math.isfinite(float(df["Volume"].iloc[-1])) else None
    rel_vol = _r(last_vol / vol20, 2) if last_vol and vol20 and vol20 > 0 else None

    dates = df["TradingDate"]
    atr = _atr14(df)
    rets = close.pct_change().dropna()
    vol_w = _r(float(rets.tail(5).std() * math.sqrt(5) * 100.0), 1) if len(rets) >= 5 else None
    vol_m = _r(float(rets.tail(21).std() * math.sqrt(21) * 100.0), 1) if len(rets) >= 21 else None

    return {
        "date": dates.iloc[-1].strftime("%d/%m/%Y"),
        "close": _r(last), "prev_close": _r(prev),
        "chg_pct": _r((last / prev - 1.0) * 100.0),
        "open": _r(float(df["Open"].iloc[-1])), "high": _r(float(df["High"].iloc[-1])), "low": _r(float(df["Low"].iloc[-1])),
        "gap_pct": _r((float(df["Open"].iloc[-1]) / prev - 1.0) * 100.0),
        "volume": int(last_vol) if last_vol else None,
        "avg_vol20": int(vol20) if vol20 else None,
        "rel_vol": rel_vol,
        "value_m": _r(last * (last_vol or 0) / 1e9, 2),
        "ma": {
            "sma10": _r(smas[10]), "sma20": _r(smas[20]), "sma50": _r(smas[50]),
            "sma100": _r(smas[100]), "sma200": _r(smas[200]),
            "above": above, "dist": dist,
            "cross_20_50": bool(smas[20] and smas[50] and smas[20] > smas[50]),
            "cross_50_200": bool(smas[50] and smas[200] and smas[50] > smas[200]),
            "score": ma_score,
        },
        "osc": osc,
        "osc_score": o_score,
        "ma_score": ma_score,
        "overall": _r((ma_score + o_score) / 2.0),
        "w52": {"high": _r(hi52), "low": _r(lo52), "pct_from_high": pct_from_high, "pos": pos52, "new_high": new_high, "new_low": new_low},
        "perf": {
            "w1": _r(_pct_change(close, 5), 1), "m1": _r(_pct_change(close, 21), 1),
            "m3": _r(_pct_change(close, 63), 1), "m6": _r(_pct_change(close, 126), 1),
            "ytd": _r(_ytd_perf(close, dates), 1), "y1": _r(_pct_change(close, 252), 1),
        },
        "risk": {
            "atr14": _r(atr), "atr_pct": _r(atr / last * 100.0, 1) if atr else None,
            "vol_w": vol_w, "vol_m": vol_m,
            "beta": _r(_beta(close, idx_close)),
        },
        "pivots": _pivots(df),
        "spark": [_r(v) for v in close.tail(SPARK_BARS).tolist()],
    }


def main():
    files = [p for p in CACHE_DIR.glob("*.csv") if p.stem not in INDEX_FILES]
    if not files:
        print("KHONG ghi de stock_health: cache OHLC rong.")
        return
    idx_df = load_cache("VNI", CACHE_DIR)
    idx_close = idx_df["Close"] if idx_df is not None and len(idx_df) else None

    symbols = {}
    for path in sorted(files):
        sym = path.stem.upper()
        try:
            df = load_cache(sym, CACHE_DIR)
        except Exception:
            continue
        try:
            health = compute_symbol_health(df, sym, idx_close)
        except Exception as exc:
            print(f"Loi tinh {sym}: {exc}")
            continue
        if health:
            symbols[sym] = health

    if not symbols:
        print("KHONG ghi de stock_health: khong du du lieu ma.")
        return

    output = {
        "generated_at": vn_now().isoformat(),
        "total_symbols": len(symbols),
        "date": next(iter(symbols.values()))["date"],
        "symbols": symbols,
    }
    for d in (DATA_DIR, DOCS_DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)
        (d / OUTPUT_NAME).write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    print(f"Da ghi {OUTPUT_NAME}: {len(symbols)} ma.")


if __name__ == "__main__":
    main()
