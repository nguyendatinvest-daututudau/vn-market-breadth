"""
Luc Mach Signals — core buy/sell filter from VUDD + Tplus + trend/volume setups.

Version dau tap trung vao du lieu OHLCV trong cache. RS-line va sector strength
duoc de null vi project chua co benchmark/sector mapping on dinh cho tung ma.
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from cache_utils import load_cache as _load_cache
from _shared import tqdm, DATA_DIR, CACHE_DIR, DOCS_DATA_DIR, list_symbols, json_default as _json_default

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

SIGNALS_JSON = DATA_DIR / "luc_mach_signals.json"
DOCS_SIGNALS_JSON = DOCS_DATA_DIR / "luc_mach_signals.json"

VUDD_PERIODS = (13, 20, 35, 55, 65)
LUC_MACH_THRESHOLD = 3
MIN_AVG_VOLUME = 300_000
MIN_TRADING_VALUE = 10_000_000_000


def _last_bool(series: pd.Series) -> bool:
    return bool(series.iloc[-1]) if len(series) else False


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=1).mean()


def _tema(series: pd.Series, period: int) -> pd.Series:
    ema1 = _ema(series, period)
    ema2 = _ema(ema1, period)
    ema3 = _ema(ema2, period)
    return 3 * ema1 - 3 * ema2 + ema3


def _zero_lag_tema(series: pd.Series, period: int) -> pd.Series:
    tma1 = _tema(series, period)
    tma2 = _tema(tma1, period)
    return tma1 + (tma1 - tma2)


def _cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def _cross_down(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def has_ohlcv(df: pd.DataFrame) -> bool:
    required = ("Open", "High", "Low", "Close", "Volume")
    return all(col in df.columns for col in required) and not df[list(required)].tail(80).isna().any().any()


def compute_vudd(df: pd.DataFrame, period: int) -> dict:
    open_ = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    ha_close_raw = (open_ + high + low + close) / 4
    ha_open = ha_close_raw.shift(1).ewm(alpha=0.5, adjust=False).mean()
    ha_high = pd.concat([high, ha_close_raw, ha_open], axis=1).max(axis=1)
    ha_low = pd.concat([low, ha_close_raw, ha_open], axis=1).min(axis=1)
    ha_close = (ha_close_raw + ha_open + ha_high + ha_low) / 4
    avg_price = (high + low + close) / 3

    zl_ha = _zero_lag_tema(ha_close, period)
    zl_typ = _zero_lag_tema(avg_price, period)
    buy = _cross_up(zl_typ, zl_ha)
    sell = _cross_up(zl_ha, zl_typ)
    return {
        "buy_series": buy.fillna(False),
        "sell_series": sell.fillna(False),
        "buy": _last_bool(buy),
        "sell": _last_bool(sell),
        "zl_ha": None if pd.isna(zl_ha.iloc[-1]) else round(float(zl_ha.iloc[-1]), 2),
        "zl_typ": None if pd.isna(zl_typ.iloc[-1]) else round(float(zl_typ.iloc[-1]), 2),
    }


def compute_tplus(df: pd.DataFrame) -> dict:
    high = df["High"].reset_index(drop=True)
    low = df["Low"].reset_index(drop=True)
    close = df["Close"].reset_index(drop=True)
    n = len(df)
    d = pd.Series(np.nan, index=range(n), dtype=float)

    for i in range(n):
        if i < 4:
            d.iloc[i] = close.iloc[i]
            continue
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
        "state": int(state.iloc[-1]),
        "d": None if pd.isna(d.iloc[-1]) else round(float(d.iloc[-1]), 2),
    }


def compute_common(df: pd.DataFrame) -> dict:
    close = df["Close"]
    open_ = df["Open"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    ma5 = _sma(close, 5)
    ma20 = _sma(close, 20)
    ma50 = _sma(close, 50)
    ma200 = _sma(close, 200)
    vol_ma50 = _sma(volume, 50)
    trading_value_ma20 = _sma(close * volume, 20)

    trend_up_basic = (close > ma50) & (ma20 > ma50)
    trend_up_strong = trend_up_basic & (ma50 > ma200)
    trend_weak = (close < ma20) | (close < ma50) | (ma20 < ma50)
    volume_ok = volume >= 1.2 * vol_ma50
    strong_volume_ok = volume >= 1.5 * vol_ma50
    liquidity_ok = trading_value_ma20 >= MIN_TRADING_VALUE

    ma5_cross_up_ma20 = _cross_up(ma5, ma20)
    close_cross_up_ma20 = _cross_up(close, ma20)
    close_cross_down_ma20 = _cross_down(close, ma20)
    close_below_ma50 = close < ma50
    pullback_buy = trend_up_basic & (ma5.shift(1) < ma20.shift(1)) & (ma5_cross_up_ma20 | close_cross_up_ma20)
    pullback_buy_valid = pullback_buy & liquidity_ok & volume_ok

    high_20_prev = high.rolling(20, min_periods=20).max().shift(1)
    low_20_prev = low.rolling(20, min_periods=20).min().shift(1)
    breakout_20 = close > high_20_prev
    breakout_buy = breakout_20 & strong_volume_ok & trend_up_basic & liquidity_ok
    darvas_buy_valid = breakout_buy
    darvas_sell = close < low_20_prev

    day_range = (high - low).replace(0, np.nan)
    close_position = (close - low) / day_range
    bearish_high_volume = (volume > vol_ma50) & (close < open_) & (close_position < 0.4)

    return {
        "ma5": ma5, "ma20": ma20, "ma50": ma50, "ma200": ma200,
        "vol_ma50": vol_ma50, "trading_value_ma20": trading_value_ma20,
        "trend_up_basic": trend_up_basic.fillna(False),
        "trend_up_strong": trend_up_strong.fillna(False),
        "trend_weak": trend_weak.fillna(False),
        "volume_ok": volume_ok.fillna(False),
        "strong_volume_ok": strong_volume_ok.fillna(False),
        "liquidity_ok": liquidity_ok.fillna(False),
        "pullback_buy": pullback_buy.fillna(False),
        "pullback_buy_valid": pullback_buy_valid.fillna(False),
        "breakout_buy": breakout_buy.fillna(False),
        "darvas_buy_valid": darvas_buy_valid.fillna(False),
        "darvas_sell": darvas_sell.fillna(False),
        "close_cross_down_ma20": close_cross_down_ma20.fillna(False),
        "close_below_ma50": close_below_ma50.fillna(False),
        "bearish_high_volume": bearish_high_volume.fillna(False),
        "high_20_prev": high_20_prev,
        "low_20_prev": low_20_prev,
    }


def analyze_symbol(symbol: str) -> dict | None:
    df = _load_cache(symbol, CACHE_DIR).sort_values("TradingDate").reset_index(drop=True)
    if len(df) < 210 or not has_ohlcv(df):
        return None

    vudds = {p: compute_vudd(df, p) for p in VUDD_PERIODS}
    tplus = compute_tplus(df)
    common = compute_common(df)

    buy_score = sum(1 for p in VUDD_PERIODS if vudds[p]["buy"]) + int(tplus["buy"])
    sell_score = sum(1 for p in VUDD_PERIODS if vudds[p]["sell"]) + int(tplus["sell"])
    luc_mach_buy = buy_score >= LUC_MACH_THRESHOLD
    luc_mach_sell = sell_score >= LUC_MACH_THRESHOLD
    vudd_sell_any = any(vudds[p]["sell"] for p in VUDD_PERIODS)

    liquidity_ok = _last_bool(common["liquidity_ok"])
    volume_ok = _last_bool(common["volume_ok"])
    strong_volume_ok = _last_bool(common["strong_volume_ok"])
    trend_up_basic = _last_bool(common["trend_up_basic"])
    trend_up_strong = _last_bool(common["trend_up_strong"])
    pullback_buy = _last_bool(common["pullback_buy"])
    pullback_buy_valid = _last_bool(common["pullback_buy_valid"])
    breakout_buy = _last_bool(common["breakout_buy"])
    darvas_buy_valid = _last_bool(common["darvas_buy_valid"])
    darvas_sell = _last_bool(common["darvas_sell"])
    close_cross_down_ma20 = _last_bool(common["close_cross_down_ma20"])
    close_below_ma50 = _last_bool(common["close_below_ma50"])
    bearish_high_volume = _last_bool(common["bearish_high_volume"])

    rs_ok = None
    sector_ok = None
    early_watchlist = liquidity_ok and (
        vudds[13]["buy"] or vudds[20]["buy"] or tplus["buy"] or buy_score >= 2
    )
    valid_buy = liquidity_ok and trend_up_basic and volume_ok and (
        pullback_buy or breakout_buy or darvas_buy_valid or luc_mach_buy
    )
    strong_buy = liquidity_ok and trend_up_strong and strong_volume_ok and luc_mach_buy and (
        breakout_buy or darvas_buy_valid or pullback_buy_valid
    )
    sell_warning = luc_mach_sell or close_cross_down_ma20 or close_below_ma50 or bearish_high_volume or darvas_sell

    if strong_buy:
        status = "STRONG_BUY"
        signal_type = "strong"
    elif valid_buy:
        status = "VALID_BUY"
        signal_type = "valid"
    elif early_watchlist:
        status = "WATCHLIST"
        signal_type = "watch"
    elif sell_warning:
        status = "SELL_WARNING"
        signal_type = "sell_warning"
    else:
        return None

    setup_score = 0
    setup_score += min(buy_score, 6) * 10
    setup_score += 15 if trend_up_basic else 0
    setup_score += 10 if trend_up_strong else 0
    setup_score += 10 if volume_ok else 0
    setup_score += 10 if strong_volume_ok else 0
    setup_score += 15 if (pullback_buy or breakout_buy or darvas_buy_valid) else 0
    setup_score -= 25 if sell_warning else 0
    setup_score = max(0, min(100, setup_score))

    strategies = []
    if luc_mach_buy: strategies.append("luc_mach")
    if pullback_buy: strategies.append("pullback")
    if breakout_buy: strategies.append("breakout")
    if darvas_buy_valid: strategies.append("darvas")
    if tplus["buy"]: strategies.append("tplus")
    for p in VUDD_PERIODS:
        if vudds[p]["buy"]:
            strategies.append(f"vudd{p}")
    if sell_warning: strategies.append("sell_warning")

    close = df["Close"]
    volume = df["Volume"]
    vol_ma50 = common["vol_ma50"].iloc[-1]
    trading_value_ma20 = common["trading_value_ma20"].iloc[-1]
    vol_ratio = float(volume.iloc[-1] / vol_ma50) if pd.notna(vol_ma50) and vol_ma50 > 0 else None

    return {
        "symbol": symbol,
        "status": status,
        "signal_type": signal_type,
        "score": int(round(setup_score)),
        "buy_score": int(buy_score),
        "sell_score": int(sell_score),
        "luc_mach_buy": bool(luc_mach_buy),
        "luc_mach_sell": bool(luc_mach_sell),
        "vudd_sell_any": bool(vudd_sell_any),
        "tplus_buy": bool(tplus["buy"]),
        "tplus_sell": bool(tplus["sell"]),
        "trend_up_basic": trend_up_basic,
        "trend_up_strong": trend_up_strong,
        "volume_ok": volume_ok,
        "strong_volume_ok": strong_volume_ok,
        "liquidity_ok": liquidity_ok,
        "pullback_buy": pullback_buy,
        "pullback_buy_valid": pullback_buy_valid,
        "breakout_buy": breakout_buy,
        "darvas_buy_valid": darvas_buy_valid,
        "sell_warning": bool(sell_warning),
        "rs_ok": rs_ok,
        "sector_ok": sector_ok,
        "strategies": strategies,
        "vudd_buy_periods": [p for p in VUDD_PERIODS if vudds[p]["buy"]],
        "vudd_sell_periods": [p for p in VUDD_PERIODS if vudds[p]["sell"]],
        "ma20": None if pd.isna(common["ma20"].iloc[-1]) else round(float(common["ma20"].iloc[-1]), 1),
        "ma50": None if pd.isna(common["ma50"].iloc[-1]) else round(float(common["ma50"].iloc[-1]), 1),
        "ma200": None if pd.isna(common["ma200"].iloc[-1]) else round(float(common["ma200"].iloc[-1]), 1),
        "vol_ratio": None if vol_ratio is None else round(vol_ratio, 2),
        "trading_value_ma20_billion": None if pd.isna(trading_value_ma20) else round(float(trading_value_ma20) / 1_000_000_000, 1),
        "last_price": float(close.iloc[-1]),
        "last_volume": float(volume.iloc[-1]) if not pd.isna(volume.iloc[-1]) else None,
    }


def get_filtered_symbols() -> list[str]:
    return list_symbols(CACHE_DIR, min_history=210, min_volume=MIN_AVG_VOLUME)


def main():
    tqdm.write("=" * 60)
    tqdm.write("Luc Mach Signals — VUDD + Tplus + setup filter")
    tqdm.write("=" * 60)

    symbols = get_filtered_symbols()
    tqdm.write(f"\nPhan tich {len(symbols)} ma...\n")

    signals = []
    skipped_ohlc = 0
    bar = tqdm(symbols, desc="[LM] Luc Mach", unit="sym")
    for sym in bar:
        bar.set_postfix_str(sym, refresh=True)
        result = analyze_symbol(sym)
        if result:
            signals.append(result)
        else:
            df = _load_cache(sym, CACHE_DIR)
            if len(df) >= 210 and not has_ohlcv(df):
                skipped_ohlc += 1

    rank = {"STRONG_BUY": 4, "VALID_BUY": 3, "WATCHLIST": 2, "SELL_WARNING": 1}
    signals.sort(key=lambda x: (rank.get(x["status"], 0), x["score"]), reverse=True)

    strong = [s for s in signals if s["status"] == "STRONG_BUY"]
    valid = [s for s in signals if s["status"] == "VALID_BUY"]
    watch = [s for s in signals if s["status"] == "WATCHLIST"]
    sell = [s for s in signals if s["status"] == "SELL_WARNING"]
    now = datetime.now(timezone.utc) + timedelta(hours=7)

    output = {
        "generated_at": now.isoformat(),
        "date": now.strftime("%d/%m/%Y"),
        "total_symbols_analyzed": len(symbols),
        "total_signals": len(signals),
        "skipped_missing_ohlc": skipped_ohlc,
        "strong_buy": len(strong),
        "valid_buy": len(valid),
        "watch_count": len(watch),
        "sell_warning_count": len(sell),
        "strong": strong,
        "valid": valid,
        "watch": watch,
        "sell_warning": sell,
        "all_signals": signals,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SIGNALS_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    DOCS_SIGNALS_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    tqdm.write(f"\nDa ghi: {SIGNALS_JSON.name}")
    tqdm.write(f"Tin hieu: {output['total_signals']} (Strong: {len(strong)}, Valid: {len(valid)}, Watch: {len(watch)}, Sell: {len(sell)})")
    if skipped_ohlc:
        tqdm.write(f"Bo qua {skipped_ohlc} ma do cache chua co OHLCV day du.")


if __name__ == "__main__":
    main()
