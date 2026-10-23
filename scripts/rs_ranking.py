"""
RS Ranking — Xep hang suc manh tuong doi (Relative Strength) so voi VN-Index.

RS Rating (1-99, kieu IBD) = percentile rank cua diem RS tho:
    rs_raw = 0.4*rel20 + 0.2*rel60 + 0.2*rel120 + 0.2*rel250
trong do relN = return N phien cua ma - return N phien cua benchmark.

Benchmark: VN-Index (VNI.csv trong cache). Neu thieu thi fallback ve trung vi
gia quyen bang nhau cua toan bo ma (giong Accumulation Radar).

Bo loc "RS manh":
    rs_rating >= RS_STRONG_MIN  VA
    Close > MA20                VA
    Close > MA50

Output: data/rs_ranking.json (+ dong bo docs/data/).
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from cache_utils import load_cache as _load_cache
from _shared import (
    CACHE_DIR,
    DATA_DIR,
    DOCS_DATA_DIR,
    format_market_date,
    is_market_data_fresh,
    json_default,
    list_symbols,
    signal_market_date,
    tqdm,
    vn_now,
)

warnings.filterwarnings("ignore", category=FutureWarning)

OUTPUT_JSON = DATA_DIR / "rs_ranking.json"
DOCS_OUTPUT_JSON = DOCS_DATA_DIR / "rs_ranking.json"

BENCHMARK_SYMBOL = "VNI"
INDEX_SYMBOLS = {"VNI", "VNINDEX", "HNXINDEX", "HNX", "VN30", "HNX30", "UPCOM"}

MIN_AVG_VOLUME = 300_000
MIN_HISTORY = 260
WINDOWS = (20, 60, 120, 250)
WINDOW_WEIGHTS = {20: 0.4, 60: 0.2, 120: 0.2, 250: 0.2}
RS_STRONG_MIN = 80
MA_TREND_WINDOWS = (20, 50)


def _pct_change(close: pd.Series, days: int) -> float | None:
    if len(close) <= days:
        return None
    start = close.iloc[-days - 1]
    end = close.iloc[-1]
    if pd.isna(start) or start <= 0 or pd.isna(end):
        return None
    return float((end / start - 1.0) * 100.0)


def _benchmark_returns(reference_date=None) -> tuple[dict[int, float | None], str]:
    """Tra ve return benchmark theo tung cua so. Uu tien VN-Index trong cache."""
    df = _load_cache(BENCHMARK_SYMBOL, CACHE_DIR)
    if not df.empty:
        df = df.sort_values("TradingDate").reset_index(drop=True)
        ref = reference_date if reference_date is not None else signal_market_date()
        fresh = ref is None or is_market_data_fresh(df["TradingDate"].iloc[-1], ref)
        if fresh and len(df) > max(WINDOWS):
            close = df["Close"]
            rets = {w: _pct_change(close, w) for w in WINDOWS}
            if any(v is not None for v in rets.values()):
                return rets, f"VNINDEX({BENCHMARK_SYMBOL})"

    # Fallback: trung vi return cua toan bo ma (proxy quyen bang nhau).
    symbols = _stock_symbols()
    buckets: dict[int, list[float]] = {w: [] for w in WINDOWS}
    for symbol in symbols:
        df = _load_cache(symbol, CACHE_DIR)
        if len(df) < MIN_HISTORY:
            continue
        close = df.sort_values("TradingDate")["Close"]
        for w in WINDOWS:
            value = _pct_change(close, w)
            if value is not None:
                buckets[w].append(value)
    rets = {w: (round(float(np.nanmedian(buckets[w])), 2) if buckets[w] else None) for w in WINDOWS}
    return rets, "equal_weight_median"


def _stock_symbols() -> list[str]:
    symbols = list_symbols(min_history=MIN_HISTORY, min_volume=MIN_AVG_VOLUME)
    return [s for s in symbols if s.upper() not in INDEX_SYMBOLS and len(s) <= 4 and s.isalpha()]


def _analyze_symbol(symbol: str, bench: dict[int, float | None], reference_date=None) -> dict | None:
    df = _load_cache(symbol, CACHE_DIR).sort_values("TradingDate").reset_index(drop=True)
    if len(df) < MIN_HISTORY:
        return None
    ref = reference_date if reference_date is not None else signal_market_date()
    if not is_market_data_fresh(df["TradingDate"].iloc[-1], ref):
        return None

    close = df["Close"]
    rel: dict[int, float | None] = {}
    for w in WINDOWS:
        stock_ret = _pct_change(close, w)
        bench_ret = bench.get(w)
        rel[w] = (stock_ret - bench_ret) if (stock_ret is not None and bench_ret is not None) else None

    if rel[20] is None or rel[60] is None:
        return None

    total_w = 0.0
    weighted = 0.0
    for w in WINDOWS:
        if rel[w] is None:
            continue
        weighted += WINDOW_WEIGHTS[w] * rel[w]
        total_w += WINDOW_WEIGHTS[w]
    if total_w <= 0:
        return None
    rs_raw = weighted / total_w

    ma_values = {}
    for w in MA_TREND_WINDOWS:
        if len(close) >= w:
            ma_values[w] = float(close.iloc[-w:].mean())
        else:
            ma_values[w] = None

    window_52w = close.iloc[-250:] if len(close) >= 250 else close
    high_52w = float(window_52w.max()) if len(window_52w) else None
    last = float(close.iloc[-1])
    pct_from_high = round((last / high_52w - 1.0) * 100.0, 1) if high_52w and high_52w > 0 else None

    avg_vol = df["Volume"].dropna().iloc[-20:].mean() if "Volume" in df else np.nan

    return {
        "symbol": symbol,
        "rs_raw": round(float(rs_raw), 3),
        "rel20": round(rel[20], 2) if rel[20] is not None else None,
        "rel60": round(rel[60], 2) if rel[60] is not None else None,
        "rel120": round(rel[120], 2) if rel[120] is not None else None,
        "rel250": round(rel[250], 2) if rel[250] is not None else None,
        "close": round(last, 2),
        "ma20": round(ma_values[20], 2) if ma_values.get(20) is not None else None,
        "ma50": round(ma_values[50], 2) if ma_values.get(50) is not None else None,
        "pct_from_high": pct_from_high,
        "avg_volume": int(avg_vol) if not pd.isna(avg_vol) else None,
    }


def _percentile_ratings(rows: list[dict]) -> None:
    """Gan rs_rating 1-99 bang percentile rank cua rs_raw (in-place)."""
    if not rows:
        return
    values = sorted(r["rs_raw"] for r in rows)
    n = len(values)
    for row in rows:
        rank = np.searchsorted(values, row["rs_raw"], side="right")
        pct = rank / n * 100.0
        row["rs_rating"] = int(max(1, min(99, round(pct))))


def main() -> None:
    tqdm.write("=" * 60)
    tqdm.write("RS Ranking — Suc manh tuong doi so voi VN-Index")
    tqdm.write("=" * 60)

    reference_date = signal_market_date()
    bench, bench_label = _benchmark_returns(reference_date)
    tqdm.write(f"Benchmark: {bench_label} | returns: {bench}\n")

    symbols = _stock_symbols()
    tqdm.write(f"Phan tich {len(symbols)} ma...\n")

    rows: list[dict] = []
    bar = tqdm(symbols, desc="[RS] Ranking", unit="sym")
    for symbol in bar:
        bar.set_postfix_str(symbol, refresh=True)
        result = _analyze_symbol(symbol, bench, reference_date)
        if result:
            rows.append(result)

    _percentile_ratings(rows)

    def is_strong(row: dict) -> bool:
        return (
            row["rs_rating"] >= RS_STRONG_MIN
            and row["ma20"] is not None
            and row["ma50"] is not None
            and row["close"] > row["ma20"]
            and row["close"] > row["ma50"]
        )

    ranked = sorted(rows, key=lambda r: (r["rs_rating"], r["rel20"] or 0.0), reverse=True)
    strong = [r for r in ranked if is_strong(r)]

    now = vn_now()
    market_date = format_market_date(signal_market_date()) or now.strftime("%d/%m/%Y")

    output = {
        "generated_at": now.isoformat(),
        "date": market_date,
        "benchmark": bench_label,
        "benchmark_returns": {
            str(w): (round(bench[w], 2) if bench.get(w) is not None else None) for w in WINDOWS
        },
        "criteria": f"RS>={RS_STRONG_MIN} & Close>MA20 & Close>MA50",
        "strong_threshold": RS_STRONG_MIN,
        "total_symbols_analyzed": len(rows),
        "strong_count": len(strong),
        "strong": strong,
        "ranked": ranked,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8"
    )
    DOCS_OUTPUT_JSON.write_bytes(OUTPUT_JSON.read_bytes())

    tqdm.write(f"\nDa ghi: {OUTPUT_JSON.name}")
    tqdm.write(f"RS manh (rating>={RS_STRONG_MIN} + tren MA20/MA50): {len(strong)} ma")
    for row in strong[:8]:
        tqdm.write(
            f"  {row['symbol']:6s} | RS: {row['rs_rating']:3d} | rel20: {row['rel20']:+6.2f}% "
            f"| Close: {row['close']} | MA20: {row['ma20']} | MA50: {row['ma50']}"
        )


if __name__ == "__main__":
    main()
