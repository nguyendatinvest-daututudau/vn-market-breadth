"""Backfill breadth_history.json tu 01/01/2016 den phien gan nhat.

Cach chay (can SSI credentials):
    cd scripts
    python backfill_history.py

Quy trinh:
  1. Lay universe HOSE/HNX tu symbol_universes.json (hoac fetch tu API).
  2. Voi moi ma: neu cache chua du tu ~03/2015 den nay thi fetch lai full
     (co phan trang trong daily_ohlc) va luu cache. Neu cache da du thi bo qua
     (resume duoc neu lan truoc bi ngat).
  3. Tinh per-date: % tren MA10/20/50/200 + RSI14 pulse, ap dung filter
     thanh khoan (TB20 volume >= MIN_AVG_VOLUME) nhu pipeline chinh.
  4. Gop HOSE + HNX thanh ALL, chi giu compact (pct + rsi_pulse) de nhe.
  5. Ghi data/breadth_history.json va dong bo sang docs/data/.
"""
from __future__ import annotations

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

from _shared import CACHE_DIR, DATA_DIR, DOCS_DATA_DIR, DATE_FMT, parse_market_date, tqdm, vn_now
from cache_utils import compute_rsi_wilder_series, load_cache
from fetch_and_compute import (
    MA_WINDOWS,
    MARKETS,
    MIN_AVG_VOLUME,
    REQUEST_SLEEP_SEC,
    HISTORY_JSON,
    _load_exchange_universe,
    _save_exchange_universe,
    _write_json,
    update_ohlc,
)
from ssi_client import SSIClient

# Fetch tu day de MA200 co the tinh ngay tu dau 2016 (~200 phien truoc 01/01/2016).
FETCH_START = datetime(2015, 3, 1)
# Chi xuat ban du lieu tu moc hien thi nay tro di.
DISPLAY_START = datetime(2016, 1, 1)
RSI_PERIOD = 14
# Neu phien gan nhat trong cache tre hon ngay nay qua so ngay nay thi coi la chua du.
MAX_END_GAP_DAYS = 7


def _needs_fetch(symbol: str, end_date: datetime) -> bool:
    """True neu cache cua symbol chua phu tu FETCH_START den end_date."""
    df = load_cache(symbol, CACHE_DIR)
    if df.empty or "TradingDate" not in df.columns:
        return True
    dates = pd.to_datetime(df["TradingDate"], errors="coerce").dropna()
    if dates.empty:
        return True
    dmin, dmax = dates.min(), dates.max()
    return not (
        dmin <= pd.Timestamp(FETCH_START) + pd.Timedelta(days=10)
        and dmax >= pd.Timestamp(end_date) - pd.Timedelta(days=MAX_END_GAP_DAYS)
    )


def _symbol_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Tra ve khung per-date {elig{w}, above{w}, rsi_*} cho 1 symbol."""
    df = df.dropna(subset=["TradingDate", "Close"])
    df = df.sort_values("TradingDate").reset_index(drop=True)
    n = len(df)
    if n < 20:
        return pd.DataFrame()
    dates = pd.to_datetime(df["TradingDate"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")

    if "Volume" in df.columns:
        vol20 = pd.to_numeric(df["Volume"], errors="coerce").rolling(20, min_periods=20).mean()
    else:
        vol20 = pd.Series(np.nan, index=range(n))
    scope = vol20.notna() & (vol20 >= MIN_AVG_VOLUME)

    rows: dict[str, np.ndarray] = {}
    for w in MA_WINDOWS:
        ma = close.rolling(w).mean()
        elig = ma.notna() & scope
        above = elig & (close >= ma)
        rows[f"elig{w}"] = elig.fillna(False).to_numpy().astype("int8")
        rows[f"above{w}"] = above.fillna(False).to_numpy().astype("int8")

    rsi_series = compute_rsi_wilder_series(close, RSI_PERIOD).to_numpy()
    valid = (np.arange(n) >= RSI_PERIOD) & scope.to_numpy()
    rows["rsi_total"] = valid.astype("int8")
    rows["rsi_u30"] = (valid & (rsi_series < 30)).astype("int8")
    rows["rsi_o70"] = (valid & (rsi_series > 70)).astype("int8")
    rows["rsi_o50"] = (valid & (rsi_series > 50)).astype("int8")

    frame = pd.DataFrame(rows, index=dates)
    return frame[~frame.index.isna()]


def _market_summary(total_frame: pd.DataFrame) -> tuple[dict, dict]:
    """Tra ve (counts, pcts) keyed theo date string."""
    counts: dict[str, dict] = {}
    pcts: dict[str, dict] = {}
    for date, row in total_frame.iterrows():
        dstr = date.strftime(DATE_FMT)
        c = {w: (int(row[f"above{w}"]), int(row[f"elig{w}"])) for w in MA_WINDOWS}
        rsi = {
            "under_30": int(row["rsi_u30"]),
            "over_70": int(row["rsi_o70"]),
            "over_50": int(row["rsi_o50"]),
            "total": int(row["rsi_total"]),
            "period": RSI_PERIOD,
        }
        counts[dstr] = c
        pcts[dstr] = {
            f"pct_above_ma{w}": round(c[w][0] / c[w][1] * 100, 1) if c[w][1] else 0.0
            for w in MA_WINDOWS
        }
        pcts[dstr]["rsi_pulse"] = rsi
    return counts, pcts


def _empty_pct() -> dict:
    return {
        f"pct_above_ma{w}": 0.0 for w in MA_WINDOWS
    } | {"rsi_pulse": {"under_30": 0, "over_70": 0, "over_50": 0, "total": 0, "period": RSI_PERIOD}}


def _combine_counts(c1: dict, c2: dict | None) -> dict:
    out = {}
    for w in MA_WINDOWS:
        a, b = c1[w]
        out[w] = (a, b)
    if c2:
        for w in MA_WINDOWS:
            a, b = out[w]
            c, d = c2[w]
            out[w] = (a + c, b + d)
    return out


def _counts_to_pct(counts: dict, rsi: dict | None) -> dict:
    entry = {
        f"pct_above_ma{w}": round(counts[w][0] / counts[w][1] * 100, 1) if counts[w][1] else 0.0
        for w in MA_WINDOWS
    }
    entry["rsi_pulse"] = rsi or {"under_30": 0, "over_70": 0, "over_50": 0, "total": 0, "period": RSI_PERIOD}
    return entry


def main() -> None:
    os.environ["PIPELINE_FORCE_FULL_HISTORY"] = "1"
    os.environ["PIPELINE_HISTORY_START"] = FETCH_START.strftime(DATE_FMT)

    client = SSIClient()
    end_date = vn_now().replace(tzinfo=None).replace(hour=0, minute=0, second=0, microsecond=0)
    print(f"Fetch tu {FETCH_START.strftime(DATE_FMT)} den {end_date.strftime(DATE_FMT)}")
    print(f"Xuat ban tu {DISPLAY_START.strftime(DATE_FMT)} tro di\n")

    counts_by_market: dict[str, dict] = {}
    pcts_by_market: dict[str, dict] = {}

    for market in MARKETS:
        symbols = _load_exchange_universe(market)
        if not symbols:
            print(f"[{market}] Universe cache rong, fetch tu API...")
            symbols = client.common_stock_symbols(market)
            if symbols:
                _save_exchange_universe(market, symbols)
        if not symbols:
            print(f"[{market}] WARN: khong co universe, bo qua san nay.")
            continue
        print(f"[{market}] {len(symbols)} ma")

        total_frame = None
        for sym in tqdm(symbols, desc=f"[{market}] fetch+compute", unit="ma", ncols=80):
            try:
                if _needs_fetch(sym, end_date):
                    df = update_ohlc(client, sym, end_date)
                    if df.attrs.get("api_called"):
                        time.sleep(REQUEST_SLEEP_SEC)
                else:
                    df = load_cache(sym, CACHE_DIR)
            except Exception as exc:
                tqdm.write(f"  [WARN] {sym}: loi {exc}")
                continue
            if df.empty:
                continue
            frame = _symbol_frame(df)
            if frame.empty:
                continue
            total_frame = frame if total_frame is None else total_frame.add(frame, fill_value=0)

        if total_frame is None:
            print(f"[{market}] Khong co du lieu nao, bo qua.")
            continue
        counts_by_market[market], pcts_by_market[market] = _market_summary(total_frame)
        print(f"[{market}] Xong: {len(counts_by_market[market])} ngay\n")

    if not counts_by_market:
        print("Khong co du lieu san nao, thoat.")
        return

    all_dates = set().union(*(set(p) for p in pcts_by_market.values()))
    all_dates = [d for d in all_dates if parse_market_date(d) is not None and parse_market_date(d) >= DISPLAY_START]
    all_dates.sort(key=lambda d: parse_market_date(d) or datetime.min)

    history = []
    for dstr in all_dates:
        markets: dict = {}
        for market in MARKETS:
            markets[market] = pcts_by_market.get(market, {}).get(dstr, _empty_pct())
        all_counts = None
        rsi_sum = None
        for market in MARKETS:
            mc = counts_by_market.get(market, {}).get(dstr)
            if mc:
                all_counts = _combine_counts(all_counts, mc) if all_counts is not None else dict(mc)
            mrsi = pcts_by_market.get(market, {}).get(dstr, {}).get("rsi_pulse")
            if mrsi:
                rsi_sum = {
                    "under_30": (rsi_sum or {}).get("under_30", 0) + mrsi["under_30"],
                    "over_70": (rsi_sum or {}).get("over_70", 0) + mrsi["over_70"],
                    "over_50": (rsi_sum or {}).get("over_50", 0) + mrsi["over_50"],
                    "total": (rsi_sum or {}).get("total", 0) + mrsi["total"],
                    "period": RSI_PERIOD,
                }
        markets["ALL"] = _counts_to_pct(all_counts, rsi_sum) if all_counts is not None else _empty_pct()
        history.append({"date": dstr, "markets": markets})

    if not history:
        print("Khong co ngay nao du lieu, thoat.")
        return

    _write_json(HISTORY_JSON, history)
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DATA_DIR / HISTORY_JSON.name).write_bytes(HISTORY_JSON.read_bytes())

    print(f"\nDa ghi {HISTORY_JSON.name}: {len(history)} phien ({history[0]['date']} -> {history[-1]['date']})")
    print(f"Dong bo sang {DOCS_DATA_DIR / HISTORY_JSON.name}")


if __name__ == "__main__":
    main()
