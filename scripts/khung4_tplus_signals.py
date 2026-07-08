"""Khung4/Tplus buy signals based on Diep AFL logic.

This is intentionally separate from Luc Mach. It only tracks the Khung4/Tplus
state line `d` and reports buy points where state flips from 0 to 1.
"""
from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from cache_utils import load_cache as _load_cache
from _shared import tqdm, DATA_DIR, CACHE_DIR, DOCS_DATA_DIR, list_symbols, json_default as _json_default

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

SIGNALS_JSON = DATA_DIR / "khung4_tplus_signals.json"
DOCS_SIGNALS_JSON = DOCS_DATA_DIR / "khung4_tplus_signals.json"

MIN_VOLUME = int(os.environ.get("KHUNG4_TPLUS_MIN_VOLUME", "20000"))
MIN_HISTORY = int(os.environ.get("KHUNG4_TPLUS_MIN_HISTORY", "20"))


def _last_bool(series: pd.Series) -> bool:
    return bool(series.iloc[-1]) if len(series) else False


def has_ohlcv(df: pd.DataFrame) -> bool:
    required = ("Open", "High", "Low", "Close", "Volume")
    if len(df) < MIN_HISTORY or not all(col in df.columns for col in required):
        return False
    return not df[list(required)].tail(MIN_HISTORY).isna().any().any()


def compute_khung4_tplus(df: pd.DataFrame) -> dict:
    high = df["High"].reset_index(drop=True)
    low = df["Low"].reset_index(drop=True)
    close = df["Close"].reset_index(drop=True)
    n = len(df)

    d = pd.Series(np.nan, index=range(n), dtype=float)
    for i in range(4, n):
        recent_high = high.iloc[i - 4:i].max()
        recent_low = low.iloc[i - 4:i].min()
        if close.iloc[i] > recent_high:
            d.iloc[i] = low.iloc[i - 3:i + 1].min()
        elif close.iloc[i] < recent_low:
            d.iloc[i] = high.iloc[i - 3:i + 1].max()
        else:
            d.iloc[i] = d.iloc[i - 1]

    cross_up = ((close > d) & (close.shift(1) <= d.shift(1))).fillna(False)
    cross_down = ((d > close) & (d.shift(1) <= close.shift(1))).fillna(False)

    state = pd.Series(0, index=range(n), dtype=int)
    for i in range(1, n):
        if cross_up.iloc[i]:
            state.iloc[i] = 1
        elif cross_down.iloc[i]:
            state.iloc[i] = 0
        else:
            state.iloc[i] = state.iloc[i - 1]

    buy = (state > state.shift(1)).fillna(False)
    sell = (state < state.shift(1)).fillna(False)

    return {
        "buy_series": buy,
        "sell_series": sell,
        "buy": _last_bool(buy),
        "sell": _last_bool(sell),
        "state": int(state.iloc[-1]) if n else 0,
        "d": None if not n or pd.isna(d.iloc[-1]) else round(float(d.iloc[-1]), 2),
        "buy_price": float(close.iloc[-1]) if _last_bool(buy) else None,
        "sell_price": float(close.iloc[-1]) if _last_bool(sell) else None,
    }


def analyze_symbol(symbol: str) -> dict | None:
    df = _load_cache(symbol, CACHE_DIR).sort_values("TradingDate").reset_index(drop=True)
    if not has_ohlcv(df):
        return None

    last_volume = df["Volume"].iloc[-1]
    if pd.isna(last_volume) or float(last_volume) <= MIN_VOLUME:
        return None

    signal = compute_khung4_tplus(df)
    if not signal["buy"]:
        return None

    close = df["Close"]
    return {
        "symbol": symbol,
        "status": "BUY",
        "signal_type": "khung4_tplus_buy",
        "score": 100,
        "khung4_tplus_buy": True,
        "khung4_tplus_sell": bool(signal["sell"]),
        "khung4_tplus_state": int(signal["state"]),
        "khung4_tplus_d": signal["d"],
        "buy_price": signal["buy_price"],
        "last_price": float(close.iloc[-1]),
        "last_volume": float(last_volume),
        "strategies": ["khung4_tplus_buy"],
    }


def get_filtered_symbols() -> list[str]:
    symbols = list_symbols(CACHE_DIR, min_history=MIN_HISTORY)
    return [s for s in symbols if len(s) <= 3 and not any(c.isdigit() for c in s)]


def main():
    tqdm.write("=" * 60)
    tqdm.write("Khung4/Tplus Buy Signals")
    tqdm.write("=" * 60)

    symbols = get_filtered_symbols()
    tqdm.write(f"\nPhan tich {len(symbols)} ma...\n")

    signals = []
    skipped_ohlc = 0
    bar = tqdm(symbols, desc="[K4] Khung4/Tplus", unit="sym")
    for sym in bar:
        bar.set_postfix_str(sym, refresh=True)
        result = analyze_symbol(sym)
        if result:
            signals.append(result)
        else:
            df = _load_cache(sym, CACHE_DIR)
            if len(df) < MIN_HISTORY or not has_ohlcv(df):
                skipped_ohlc += 1

    signals.sort(key=lambda x: (x["score"], x["last_volume"]), reverse=True)
    now = datetime.now(timezone.utc) + timedelta(hours=7)
    output = {
        "mode": "khung4_tplus_original",
        "generated_at": now.isoformat(),
        "date": now.strftime("%d/%m/%Y"),
        "min_volume": MIN_VOLUME,
        "min_history": MIN_HISTORY,
        "total_symbols_analyzed": len(symbols),
        "total_signals": len(signals),
        "skipped_missing_ohlc": skipped_ohlc,
        "buy": signals,
        "all_signals": signals,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SIGNALS_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    DOCS_SIGNALS_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    tqdm.write(f"\nDa ghi: {SIGNALS_JSON.name}")
    tqdm.write(f"Tin hieu mua Khung4/Tplus: {len(signals)}")
    if skipped_ohlc:
        tqdm.write(f"Bo qua {skipped_ohlc} ma do cache chua co OHLCV day du.")


if __name__ == "__main__":
    main()
