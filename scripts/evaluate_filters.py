"""
Evaluate filter logic effectiveness over full OHLC history (2015 -> now).

Chay tuong tu cac backtest co san (vectorized, per-symbol), nhung do luong
HIEN QUA cua TUNG logic loc, khong chi cua tung chi dan moc.

Dau ra: data/evaluation_filters.json (+ sync docs/data). Doc cung luc:
  - baseline (toan bo vu tru thanh khoan) de tinh lift
  - 8 nhom filter: momentum score brackets / permutation common filters /
    bonus conditions / ensemble brackets + agreement / confluence xuyen he /
    luc-mach, khung4-tplus, pre-breakout / regime gating / sell signals
  - walk-forward OOS: re-tune ensemble weights tren nua dau, test nua sau

Cach chay (can OHLC cache, thuong la tren GitHub Actions):
    cd scripts && python evaluate_filters.py [--limit-symbols N] [--skip-heavy]
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

from _shared import (
    CACHE_DIR, DATA_DIR, DOCS_DATA_DIR, MIN_SYMBOL_HISTORY, WEIGHTS_PATH,
    DEFAULT_WEIGHTS, SCORE_BREAKOUT, SCORE_HYBRID, SCORE_MA, SCORE_ROC, vn_now,
)
from cache_utils import load_cache as _load_cache, compute_rsi_wilder_series
from ensemble_signals import compute_breakout_signal_series, compute_momentum_signal_series
from market_regime import (
    compose_regime, score_ad_ratio, score_index_state,
    score_pct_above, score_rsi_pulse, score_volume_ud,
)
from backtest_momentum import (
    compute_adx_series, compute_bonuses_vectorized, compute_breakout_signal,
    compute_hybrid_signal, compute_ma_crossover_signal, compute_roc_momentum_signal,
)
from luc_mach_signals import compute_vudd
from khung4_tplus_signals import compute_khung4_tplus
from mama_positional_signals import compute_mama_positional_system
from advanced_trailstop_signals import compute_advanced_trailstop

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

OUTPUT_JSON = DATA_DIR / "evaluation_filters.json"
DOCS_OUTPUT_JSON = DOCS_DATA_DIR / "evaluation_filters.json"
HISTORY_JSON = DATA_DIR / "breadth_history.json"

FWD_OPTIONS = (5, 10, 20)
MIN_AVG_VOLUME = 300_000
MIN_OBSERVATIONS = 50
OOS_CUTOFF = pd.Timestamp("2022-01-01")

VUDD_PERIODS = (13, 20, 35, 55, 65)
LUC_MACH_THRESHOLD = int(os.environ.get("LUC_MACH_THRESHOLD", "3"))


# --- Helpers -----------------------------------------------------------------

def _clamp(value, lo: float = 0.0, hi: float = 1.0):
    return max(lo, min(hi, value))


def _rolling_slope_norm(series: pd.Series, window: int) -> pd.Series:
    """Rolling linear-regression slope of `series` over `window`, normalized by mean."""
    s = series.astype(float).reset_index(drop=True)
    idx = np.arange(len(s))
    n = window
    Sx = n * (n - 1) / 2.0
    Sxx = n * (n - 1) * (2 * n - 1) / 6.0
    Sy = s.rolling(window).sum()
    Sxy_all = (idx * s).rolling(window).sum()
    start_idx = pd.Series(idx - (window - 1), index=s.index)
    Sxy = Sxy_all - start_idx * Sy
    denom = n * Sxx - Sx * Sx
    slope = (n * Sxy - Sx * Sy) / denom
    mean = Sy / n
    return (slope / mean).where(mean > 0, 0.0)


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff()).fillna(0.0)
    return (sign * volume).cumsum()


def _vudd_series(df: pd.DataFrame, period: int) -> tuple[pd.Series, pd.Series]:
    out = compute_vudd(df, period)
    return out["buy_series"].astype(bool), out["sell_series"].astype(bool)


def _khung4_series(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    out = compute_khung4_tplus(df)
    return out["buy_series"].astype(bool), out["sell_series"].astype(bool)


def _mama_series(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    out = compute_mama_positional_system(df)
    return out["buy_series"].astype(bool), out["sell_series"].astype(bool)


def _ats_series(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    out = compute_advanced_trailstop(df)
    return out["buy_series"].astype(bool), out["sell_series"].astype(bool)


def _pre_breakout_series(df: pd.DataFrame) -> dict:
    """Rolling approximation of strategy_signals.detect_base_quality + composite."""
    close = df["Close"].astype(float)
    volume = df["Volume"].fillna(0.0).astype(float)
    w = 25

    base_hi = close.rolling(w).max()
    base_lo = close.rolling(w).min()
    mid = (base_hi + base_lo) / 2
    range_pct = ((base_hi - base_lo) / mid * 100).where(mid > 0, 999.0)
    price_pos = ((close - base_lo) / (base_hi - base_lo)).where(base_hi > base_lo, 0.0)

    range_score = ((8.0 - range_pct) / 8.0 * 1.5).clip(0, 1)
    pos_score = ((price_pos - 0.5) / 0.5).clip(0, 1)
    vol_score = (-_rolling_slope_norm(volume, w) * 20).clip(0, 1)

    obv = _obv(close, volume)
    obv_start = obv.shift(w - 1)
    obv_trend = ((obv - obv_start) / obv_start.abs()).where(obv_start.abs() > 1, 0.0)
    obv_score = (obv_trend * 50).clip(0, 1)

    base_score = (0.40 * range_score + 0.30 * pos_score
                  + 0.20 * vol_score + 0.10 * obv_score)

    mom5 = (obv - obv.shift(5)) / obv.abs().where(obv.abs() > 1, 1.0)
    mom10 = (obv - obv.shift(10)) / obv.abs().where(obv.abs() > 1, 1.0)
    mom21 = (obv - obv.shift(21)) / obv.abs().where(obv.abs() > 1, 1.0)
    obv_trend_score = pd.Series(0.0, index=df.index)
    obv_trend_score += (mom5 > 0).astype(float) * 0.4
    obv_trend_score += (mom10 > 0).astype(float) * 0.3
    obv_trend_score += (mom21 > 0).astype(float) * 0.3
    obv_trend_score += ((mom5 > 0) & (mom10 > 0) & (mom21 > 0)).astype(float) * 0.2
    obv_trend_score = obv_trend_score.clip(0, 1)

    logret = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    vol10 = logret.rolling(10).std()
    vol63 = logret.rolling(63).std()
    ratio = (vol10 / vol63).replace([np.inf, -np.inf], np.nan)
    grad = ((ratio - ratio.shift(6)) / ratio.shift(6)).replace([np.inf, -np.inf], 0.0)
    grad_score = (grad * 3).clip(0, 1).where(grad > 0, 0.0)

    composite = (0.50 * base_score + 0.30 * obv_trend_score + 0.20 * grad_score) * 100

    base_ok = base_score >= 0.60
    pos_ok = price_pos >= 0.75
    mom_ok = mom5 > 0
    active = base_ok & pos_ok & mom_ok

    return {
        "pre_any": (active & (composite >= 60)).astype(np.int8),
        "pre_strong": (active & (composite >= 75)).astype(np.int8),
        "pre_moderate": (active & (composite >= 60) & (composite < 75)).astype(np.int8),
    }


# --- Per-symbol computation --------------------------------------------------

def backtest_symbol(symbol: str, skip_heavy: bool) -> pd.DataFrame | None:
    """Return one row per valid observation, tagged with every filter flag."""
    df = _load_cache(symbol, CACHE_DIR)
    if len(df) < MIN_SYMBOL_HISTORY:
        return None
    df = df.sort_values("TradingDate").reset_index(drop=True)
    n = len(df)
    if n < MIN_SYMBOL_HISTORY:
        return None

    close = df["Close"].astype(float)
    volume = df["Volume"].fillna(0.0).astype(float)

    # --- Forward returns ---
    out = {"symbol": symbol, "date": df["TradingDate"], "_close": close.to_numpy(dtype=np.float32)}
    for lf in FWD_OPTIONS:
        out[f"fwd{lf}"] = (close.shift(-lf) / close - 1.0).to_numpy(dtype=np.float32)

    # --- Indicators ---
    ma10 = close.rolling(10).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    rsi = compute_rsi_wilder_series(close, 14)
    adx = compute_adx_series(close, 14)
    vol_avg20 = volume.rolling(20).mean()

    vol_ok = (vol_avg20 >= MIN_AVG_VOLUME).astype(np.int8)
    trend_ok = ((close > ma50) & (ma50 > ma200)).astype(np.int8)
    rsi_band = ((rsi >= 48) & (rsi <= 72)).astype(np.int8)
    adx_ok = (adx >= 20).astype(np.int8)

    # --- Momentum system (mirror backtest_momentum.py) ---
    sig_ma = compute_ma_crossover_signal(close, volume, rsi, adx)
    sig_bo = compute_breakout_signal(close, volume, rsi)
    sig_roc = compute_roc_momentum_signal(close, volume, rsi)
    sig_hy = compute_hybrid_signal(close, volume, rsi, adx)
    bonuses = compute_bonuses_vectorized(close, volume, rsi, adx)

    strategy_ok = (trend_ok & rsi_band & adx_ok & vol_ok).astype(np.int8)
    score = (sig_ma * SCORE_MA + sig_bo * SCORE_BREAKOUT
             + sig_roc * SCORE_ROC + sig_hy * SCORE_HYBRID
             + bonuses["bonus_total"])
    score_f = score.to_numpy(dtype=np.float32)

    def bucket(lo, hi):
        return (strategy_ok & (score >= lo) & (score < hi)).astype(np.int8)

    out["score"] = score_f
    out["mom_any"] = (strategy_ok & (score >= 30)).astype(np.int8)
    out["mom_strong"] = (strategy_ok & (score >= 60)).astype(np.int8)
    out["mom_watch"] = (strategy_ok & (score >= 30) & (score < 60)).astype(np.int8)
    for name, (lo, hi) in {"b30_40": (30, 40), "b40_50": (40, 50), "b50_60": (50, 60),
                           "b60_70": (60, 70), "b70_80": (70, 80), "b80_plus": (80, 999)}.items():
        out[name] = bucket(lo, hi)

    out["bonus_vol"] = (strategy_ok & (bonuses["vol_surge"] == 1)).astype(np.int8)
    out["bonus_adx"] = (strategy_ok & (bonuses["adx_strong"] == 1)).astype(np.int8)
    out["bonus_rsi"] = (strategy_ok & (bonuses["rsi_gold"] == 1)).astype(np.int8)

    # --- Common-filter permutation ---
    out["perm_trend"] = (trend_ok & vol_ok).astype(np.int8)
    out["perm_trend_rsi"] = (trend_ok & rsi_band & vol_ok).astype(np.int8)
    out["perm_trend_adx"] = (trend_ok & adx_ok & vol_ok).astype(np.int8)
    out["perm_full"] = strategy_ok

    # --- Ensemble (mirror ensemble_signals.py + backtest_weights.py) ---
    sig_ma_e = ((ma10 > ma50) & (close > ma10) & (rsi > 50)).astype(np.int8)
    near_ma50 = ((close / ma50 >= 0.93) & (close / ma50 <= 1.00)).astype(np.int8)
    sig_pb = ((close > ma200) & near_ma50 & (rsi > 45)).astype(np.int8)
    sig_bo_e = compute_breakout_signal_series(close, volume).astype(np.int8)
    sig_mo_e = compute_momentum_signal_series(close, volume)[0].astype(np.int8)

    try:
        w = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8")).get("weights", DEFAULT_WEIGHTS)
    except (OSError, ValueError):
        w = dict(DEFAULT_WEIGHTS)
    ens_total = (sig_ma_e * float(w.get("ma_crossover", 0.25))
                 + sig_pb * float(w.get("pullback", 0.25))
                 + sig_bo_e * float(w.get("breakout", 0.25))
                 + sig_mo_e * float(w.get("momentum", 0.25)))
    ens_liq = vol_ok.astype(np.int8)

    out["e_ma"] = sig_ma_e
    out["e_pb"] = sig_pb
    out["e_bo"] = sig_bo_e
    out["e_mo"] = sig_mo_e
    out["ens_total"] = ens_total.to_numpy(dtype=np.float32)
    out["ens_any"] = (ens_liq & (ens_total >= 0.35)).astype(np.int8)
    out["ens_strong"] = (ens_liq & (ens_total >= 0.65)).astype(np.int8)
    out["ens_weak"] = (ens_liq & (ens_total >= 0.35) & (ens_total < 0.65)).astype(np.int8)
    for name, (lo, hi) in {"e35_50": (0.35, 0.50), "e50_65": (0.50, 0.65),
                           "e65_80": (0.65, 0.80), "e80_plus": (0.80, 9.99)}.items():
        out[name] = (ens_liq & (ens_total >= lo) & (ens_total < hi)).astype(np.int8)
    ens_agree = (sig_ma_e + sig_pb + sig_bo_e + sig_mo_e).astype(np.int8)
    out["ens_2of4"] = (ens_liq & (ens_agree >= 2)).astype(np.int8)
    out["ens_3of4"] = (ens_liq & (ens_agree >= 3)).astype(np.int8)
    out["ens_4of4"] = (ens_liq & (ens_agree >= 4)).astype(np.int8)

    # --- luc-mach (VUDD + Tplus) ---
    luc_buy = pd.Series(0, index=df.index, dtype=np.int8)
    luc_sell = pd.Series(0, index=df.index, dtype=np.int8)
    for period in VUDD_PERIODS:
        v_buy, v_sell = _vudd_series(df, period)
        luc_buy += v_buy.astype(np.int8)
        luc_sell += v_sell.astype(np.int8)
    k_buy, k_sell = _khung4_series(df)
    luc_buy += k_buy.astype(np.int8)
    luc_sell += k_sell.astype(np.int8)

    out["luc_valid"] = ((luc_buy >= LUC_MACH_THRESHOLD)
                        & (luc_sell < LUC_MACH_THRESHOLD)).astype(np.int8)
    out["luc_sell"] = ((luc_sell >= LUC_MACH_THRESHOLD)
                       & (luc_buy < LUC_MACH_THRESHOLD)).astype(np.int8)
    out["luc_conflict"] = ((luc_buy >= LUC_MACH_THRESHOLD)
                           & (luc_sell >= LUC_MACH_THRESHOLD)).astype(np.int8)
    out["luc_watch"] = (((luc_buy + luc_sell) >= 2)
                        & (out["luc_valid"] == 0) & (out["luc_sell"] == 0)
                        & (out["luc_conflict"] == 0)).astype(np.int8)

    # --- khung4-tplus (buy/sell) ---
    out["k4_buy"] = k_buy.astype(np.int8)
    out["k4_sell"] = k_sell.astype(np.int8)

    # --- mama / ats (heavy) ---
    if skip_heavy:
        out["mama_buy"] = out["mama_sell"] = out["ats_buy"] = out["ats_sell"] = None
    else:
        m_buy, m_sell = _mama_series(df)
        a_buy, a_sell = _ats_series(df)
        out["mama_buy"] = m_buy.astype(np.int8)
        out["mama_sell"] = m_sell.astype(np.int8)
        out["ats_buy"] = a_buy.astype(np.int8)
        out["ats_sell"] = a_sell.astype(np.int8)

    # --- pre-breakout ---
    pre = _pre_breakout_series(df)
    out["pre_any"] = pre["pre_any"]
    out["pre_strong"] = pre["pre_strong"]
    out["pre_moderate"] = pre["pre_moderate"]

    # --- Confluence (count of independent buy systems) ---
    src = pd.Series(0, index=df.index, dtype=np.int8)
    src += out["mom_any"]
    src += out["ens_any"]
    src += out["luc_valid"]
    if not skip_heavy:
        src += out["mama_buy"]
        src += out["ats_buy"]
    src += out["k4_buy"]
    src += out["pre_any"]
    out["src_count"] = src.to_numpy(dtype=np.int8)
    out["confluence_2"] = (src >= 2).astype(np.int8)
    out["confluence_3"] = (src >= 3).astype(np.int8)
    out["confluence_4"] = (src >= 4).astype(np.int8)

    # --- Liquidity + valid-window mask ---
    out["vol_ok"] = vol_ok.to_numpy(dtype=np.int8)
    valid_start = MIN_SYMBOL_HISTORY
    valid_end = n - max(FWD_OPTIONS)
    out["_valid"] = np.zeros(n, dtype=np.int8)
    out["_valid"][valid_start:valid_end] = 1

    return pd.DataFrame(out)


# --- Regime series (breadth-only + A/D from OHLC + index position) ----------

def build_regime_map(frames: list[pd.DataFrame]) -> dict[pd.Timestamp, str]:
    """Per-date regime tone from breadth_history + per-date A/D accumulated on the fly."""
    hist = []
    if HISTORY_JSON.exists():
        try:
            hist = json.loads(HISTORY_JSON.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            hist = []
    breadth = {}
    for entry in hist:
        m = (entry.get("markets") or {}).get("ALL") or {}
        if not m:
            continue
        d = pd.to_datetime(entry.get("date"), dayfirst=True, errors="coerce")
        if pd.isna(d):
            continue
        breadth[pd.Timestamp(d.date())] = {
            "pct20": m.get("pct_above_ma20"),
            "pct50": m.get("pct_above_ma50"),
            "pulse": m.get("rsi_pulse"),
        }

    # Per-date A/D from same-day close change across all valid rows
    ad = {}
    for frame in frames:
        ok = np.asarray(frame["_valid"].astype(bool), dtype=bool)
        dates = pd.to_datetime(frame["date"].to_numpy())
        closes = frame["_close"].to_numpy(dtype=np.float32)
        for i in range(len(frame)):
            if not ok[i] or i == 0 or not np.isfinite(closes[i]) or not np.isfinite(closes[i - 1]):
                continue
            d = pd.Timestamp(dates[i].date())
            delta = closes[i] - closes[i - 1]
            if delta > 0:
                ad.setdefault(d, [0, 0])[0] += 1
            elif delta < 0:
                ad.setdefault(d, [0, 0])[1] += 1

    regime_map = {}
    for d in breadth:
        b = breadth[d]
        comps = {
            "ad_ratio": {"value": None, "points": 50.0},
            "pct_above_ma20": {"value": b["pct20"], "points": score_pct_above(b["pct20"])},
            "pct_above_ma50": {"value": b["pct50"], "points": score_pct_above(b["pct50"])},
            "index_position": {"value": None, "points": 50.0},
            "rsi_pulse": {"value": b["pulse"], "points": score_rsi_pulse(b["pulse"])},
            "volume_ud": {"value": None, "points": 50.0},
        }
        ab = ad.get(d)
        if ab and ab[1] > 0:
            ratio = ab[0] / ab[1]
            comps["ad_ratio"] = {"value": ratio, "points": score_ad_ratio(ratio)}
        out = compose_regime(comps)
        regime_map[d] = out["tone"]
    return regime_map


# --- Metrics -----------------------------------------------------------------

def metrics(frame: pd.DataFrame, mask, fwd_key: str = "fwd10") -> dict | None:
    """Hit rates + median/mean forward return for rows where mask is true & valid."""
    mask_np = np.asarray(mask, dtype=bool)
    valid_np = np.asarray(frame["_valid"].astype(bool), dtype=bool)
    idx = np.where(mask_np & valid_np)[0]
    if idx.size == 0:
        return None
    r = frame[fwd_key].to_numpy(dtype=np.float32)[idx]
    r = r[np.isfinite(r)]
    if len(r) < MIN_OBSERVATIONS:
        return None
    n = len(r)
    out = {
        "n": int(n),
        "hit0": round(float(np.mean(r >= 0.0)), 4),
        "hit2": round(float(np.mean(r >= 0.02)), 4),
        "hit7": round(float(np.mean(r >= 0.07)), 4),
        "median": round(float(np.median(r)), 4),
        "mean": round(float(np.mean(r)), 4),
    }
    if "date" in frame.columns:
        years = pd.to_datetime(frame["date"]).dt.year.to_numpy()[idx]
        r = frame[fwd_key].to_numpy(dtype=np.float32)[idx]
        keep = np.isfinite(r)
        years, r = years[keep], r[keep]
        if keep.sum() >= MIN_OBSERVATIONS:
            yearly = {}
            for y in np.unique(years):
                ysub = r[years == y]
                if len(ysub) >= 10:
                    yearly[int(y)] = round(float(np.mean(ysub >= 0.02)), 4)
            out["yearly"] = yearly
            train = years < OOS_CUTOFF.year
            test = ~train
            out["train_hit2"] = round(float(np.mean(r[train] >= 0.02)), 4) if train.sum() >= 10 else None
            out["test_hit2"] = round(float(np.mean(r[test] >= 0.02)), 4) if test.sum() >= 10 else None
    return out


def evaluate_candidates(frame: pd.DataFrame, candidates: list[dict], baseline_liq: dict) -> dict:
    results = {}
    for cand in candidates:
        col = cand["col"]
        if col not in frame.columns or frame[col].isna().all():
            continue
        mask = frame[col].astype(bool)
        m = metrics(frame, mask)
        m_liq = metrics(frame, mask & frame["vol_ok"].astype(bool))
        if m is None:
            continue
        entry = {
            "group": cand["group"], "label": cand["label"], "desc": cand.get("desc", ""),
            "metrics": m,
        }
        if m_liq is not None:
            entry["metrics_liq"] = m_liq
            entry["lift2_liq"] = round(m_liq["hit2"] - baseline_liq["hit2"], 4)
            entry["lift2_liq_train"] = (
                round(m_liq["train_hit2"] - baseline_liq["train_hit2"], 4)
                if m_liq.get("train_hit2") is not None and baseline_liq.get("train_hit2") is not None else None)
            entry["lift2_liq_test"] = (
                round(m_liq["test_hit2"] - baseline_liq["test_hit2"], 4)
                if m_liq.get("test_hit2") is not None and baseline_liq.get("test_hit2") is not None else None)
        results[cand["key"]] = entry
    return results


def define_candidates() -> list[dict]:
    g = "momentum_score_brackets"
    g2 = "common_filter_permutation"
    g3 = "bonus_conditions"
    g4 = "ensemble_brackets"
    g5 = "confluence"
    g6 = "unvalidated_systems"
    g7 = "regime_gating"
    g8 = "sell_signals"
    return [
        # Group 1
        {"key": "mom_any", "col": "mom_any", "group": g, "label": "Momentum score >= 30 (common filters on)", "desc": "Baseline buy gate of the momentum system"},
        {"key": "mom_watch", "col": "mom_watch", "group": g, "label": "Momentum watch (30-59)"},
        {"key": "mom_strong", "col": "mom_strong", "group": g, "label": "Momentum strong (>=60)"},
        {"key": "b30_40", "col": "b30_40", "group": g, "label": "Momentum score 30-40"},
        {"key": "b40_50", "col": "b40_50", "group": g, "label": "Momentum score 40-50"},
        {"key": "b50_60", "col": "b50_60", "group": g, "label": "Momentum score 50-60"},
        {"key": "b60_70", "col": "b60_70", "group": g, "label": "Momentum score 60-70"},
        {"key": "b70_80", "col": "b70_80", "group": g, "label": "Momentum score 70-80"},
        {"key": "b80_plus", "col": "b80_plus", "group": g, "label": "Momentum score 80+"},
        # Group 2
        {"key": "perm_trend", "col": "perm_trend", "group": g2, "label": "Trend only (close>ma50>ma200) + liquid"},
        {"key": "perm_trend_rsi", "col": "perm_trend_rsi", "group": g2, "label": "Trend + RSI band (48-72)"},
        {"key": "perm_trend_adx", "col": "perm_trend_adx", "group": g2, "label": "Trend + ADX>=20"},
        {"key": "perm_full", "col": "perm_full", "group": g2, "label": "Trend + RSI + ADX + liquid (full common filter)"},
        # Group 3
        {"key": "bonus_vol", "col": "bonus_vol", "group": g3, "label": "Momentum + vol surge (>2.0x)"},
        {"key": "bonus_adx", "col": "bonus_adx", "group": g3, "label": "Momentum + ADX>28"},
        {"key": "bonus_rsi", "col": "bonus_rsi", "group": g3, "label": "Momentum + RSI gold (50-68)"},
        # Group 4
        {"key": "ens_any", "col": "ens_any", "group": g4, "label": "Ensemble total >= 0.35 (current gate)"},
        {"key": "ens_weak", "col": "ens_weak", "group": g4, "label": "Ensemble weak (0.35-0.65)", "desc": "Currently shown as 'buy' on the dashboard"},
        {"key": "ens_strong", "col": "ens_strong", "group": g4, "label": "Ensemble strong (>=0.65)"},
        {"key": "e35_50", "col": "e35_50", "group": g4, "label": "Ensemble 0.35-0.50"},
        {"key": "e50_65", "col": "e50_65", "group": g4, "label": "Ensemble 0.50-0.65"},
        {"key": "e65_80", "col": "e65_80", "group": g4, "label": "Ensemble 0.65-0.80"},
        {"key": "e80_plus", "col": "e80_plus", "group": g4, "label": "Ensemble 0.80+"},
        {"key": "ens_2of4", "col": "ens_2of4", "group": g4, "label": "Ensemble 2-of-4 strategies agree"},
        {"key": "ens_3of4", "col": "ens_3of4", "group": g4, "label": "Ensemble 3-of-4 strategies agree"},
        {"key": "ens_4of4", "col": "ens_4of4", "group": g4, "label": "Ensemble 4-of-4 strategies agree"},
        # Group 5
        {"key": "confluence_2", "col": "confluence_2", "group": g5, "label": "2+ independent systems agree", "desc": "The dashboard 'Đồng thuận' metric"},
        {"key": "confluence_3", "col": "confluence_3", "group": g5, "label": "3+ independent systems agree"},
        {"key": "confluence_4", "col": "confluence_4", "group": g5, "label": "4+ independent systems agree"},
        # Group 6
        {"key": "luc_valid", "col": "luc_valid", "group": g6, "label": "Luc-Mach valid (>=3/6 VUDD+Tplus aligned)"},
        {"key": "luc_watch", "col": "luc_watch", "group": g6, "label": "Luc-Mach watch"},
        {"key": "k4_buy", "col": "k4_buy", "group": g6, "label": "Khung4-Tplus buy"},
        {"key": "pre_any", "col": "pre_any", "group": g6, "label": "Pre-breakout composite >= 60 (approx)", "desc": "Rolling approximation, vol-gradient approximated"},
        {"key": "pre_strong", "col": "pre_strong", "group": g6, "label": "Pre-breakout strong (>=75, approx)"},
        # Group 8
        {"key": "luc_sell", "col": "luc_sell", "group": g8, "label": "Luc-Mach sell_warning", "desc": "Probability of drop after signal"},
        {"key": "k4_sell", "col": "k4_sell", "group": g8, "label": "Khung4-Tplus sell"},
        {"key": "mama_buy", "col": "mama_buy", "group": g6, "label": "MAMA positional buy", "desc": "Backtest win rate ~17% (T+10 +7%)"},
        {"key": "ats_buy", "col": "ats_buy", "group": g6, "label": "ATS buy", "desc": "Backtest win rate ~17% (T+10 +7%)"},
        {"key": "mama_sell", "col": "mama_sell", "group": g8, "label": "MAMA sell", "desc": "Probability of drop after signal"},
        {"key": "ats_sell", "col": "ats_sell", "group": g8, "label": "ATS sell", "desc": "Probability of drop after signal"},
    ]


def sell_metrics(frame: pd.DataFrame, col: str) -> dict | None:
    """For sell signals: probability of a drop after the signal (T+10)."""
    mask_np = np.asarray(frame[col].astype(bool), dtype=bool)
    valid_np = np.asarray(frame["_valid"].astype(bool), dtype=bool)
    idx = np.where(mask_np & valid_np)[0]
    if idx.size < MIN_OBSERVATIONS:
        return None
    r = frame["fwd10"].to_numpy(dtype=np.float32)[idx]
    r = r[np.isfinite(r)]
    if len(r) < MIN_OBSERVATIONS:
        return None
    return {
        "n": int(len(r)),
        "p_drop0": round(float(np.mean(r < 0.0)), 4),
        "p_drop2": round(float(np.mean(r < -0.02)), 4),
        "median": round(float(np.median(r)), 4),
        "mean": round(float(np.mean(r)), 4),
    }


# --- Walk-forward / OOS ------------------------------------------------------

def walk_forward(frame: pd.DataFrame) -> dict:
    """Re-tune ensemble weights on train, apply on test; compare vs current weights."""
    fwd10 = frame["fwd10"].to_numpy(dtype=np.float32)
    valid = np.asarray(frame["_valid"].astype(bool), dtype=bool)
    is_test = pd.to_datetime(frame["date"]).dt.year.to_numpy() >= OOS_CUTOFF.year
    keep = valid & np.isfinite(fwd10)
    hit = fwd10 >= 0.02

    def bracket_hits(mask_key: str) -> dict:
        m = np.asarray(frame[mask_key].astype(bool), dtype=bool)
        out = {}
        for tag, sel in (("train", ~is_test), ("test", is_test)):
            s = keep & m & sel
            out[tag] = (round(float(hit[s].mean()), 4), int(s.sum())) if s.sum() >= 10 else (None, int(s.sum()))
        return out

    sig_cols = {"ma_crossover": "e_ma", "pullback": "e_pb", "breakout": "e_bo", "momentum": "e_mo"}
    train_hit = {}
    for name, col in sig_cols.items():
        s = keep & np.asarray(frame[col].astype(bool), dtype=bool) & ~is_test
        train_hit[name] = float(hit[s].mean()) if s.sum() >= MIN_OBSERVATIONS else 0.0
    total_raw = sum(train_hit.values())
    w_train = {k: round(v / total_raw, 4) for k, v in train_hit.items()} if total_raw > 0 else dict(DEFAULT_WEIGHTS)

    try:
        w_cur = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8")).get("weights", DEFAULT_WEIGHTS)
    except (OSError, ValueError):
        w_cur = dict(DEFAULT_WEIGHTS)

    ens_cur = (frame["e_ma"].astype(float) * float(w_cur.get("ma_crossover", 0.25))
               + frame["e_pb"].astype(float) * float(w_cur.get("pullback", 0.25))
               + frame["e_bo"].astype(float) * float(w_cur.get("breakout", 0.25))
               + frame["e_mo"].astype(float) * float(w_cur.get("momentum", 0.25)))
    ens_oos = (frame["e_ma"].astype(float) * w_train["ma_crossover"]
               + frame["e_pb"].astype(float) * w_train["pullback"]
               + frame["e_bo"].astype(float) * w_train["breakout"]
               + frame["e_mo"].astype(float) * w_train["momentum"])

    result = {
        "cutoff_year": int(OOS_CUTOFF.year),
        "weights_train": w_train,
        "weights_current": w_cur,
    }
    for label, s in [("strong", 0.65), ("weak_low", 0.35)]:
        result[f"current_{label}"] = {
            "train": bracket_hits_for(ens_cur, s, keep, is_test, hit)["train"],
            "test": bracket_hits_for(ens_cur, s, keep, is_test, hit)["test"],
        }
        result[f"oos_{label}"] = {
            "train": bracket_hits_for(ens_oos, s, keep, is_test, hit)["train"],
            "test": bracket_hits_for(ens_oos, s, keep, is_test, hit)["test"],
        }
    return result


def bracket_hits_for(ens: pd.Series, lower: float, keep, is_test, hit) -> dict:
    m = np.asarray(ens.to_numpy(dtype=np.float32) >= lower, dtype=bool)
    out = {}
    for tag, sel in (("train", ~is_test), ("test", is_test)):
        s = keep & m & sel
        out[tag] = (round(float(hit[s].mean()), 4), int(s.sum())) if s.sum() >= 10 else (None, int(s.sum()))
    return out


# --- Main --------------------------------------------------------------------

def main() -> int:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Evaluate filter logic effectiveness over full OHLC history")
    ap.add_argument("--limit-symbols", type=int, default=0, help="Only evaluate first N symbols (for smoke tests)")
    ap.add_argument("--skip-heavy", action="store_true", help="Skip MAMA/ATS/pre-breakout (faster, for smoke tests)")
    args = ap.parse_args()

    print("=" * 64)
    print("Evaluate Filter Logic — full OHLC history")
    print("=" * 64)

    symbols = sorted(CACHE_DIR.glob("*.csv"))
    symbols = [p.stem for p in symbols if p.stem != ".gitkeep" and not p.stem.startswith(("FU", "E1"))]
    if args.limit_symbols > 0:
        symbols = symbols[: args.limit_symbols]
    print(f"\nSymbols: {len(symbols)}")

    frames = []
    for i, sym in enumerate(symbols):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(symbols)} symbols processed...")
        f = backtest_symbol(sym, args.skip_heavy)
        if f is not None:
            frames.append(f)
    if not frames:
        print("No cache data. Run pipeline (fetch) first.")
        return 0

    print(f"\nBuilding regime map...")
    regime_map = build_regime_map(frames)
    print(f"  regime dates: {len(regime_map)}")

    for f in frames:
        f["regime"] = f["date"].map(regime_map).fillna("unknown")

    frame = pd.concat(frames, ignore_index=True)
    print(f"Observations (rows): {len(frame)}")
    frame["date"] = pd.to_datetime(frame["date"])
    print(f"Date range: {frame['date'].min().date()} -> {frame['date'].max().date()}  "
          f"({int((frame['date'].max() - frame['date'].min()).days / 365.25)} years)")

    # Baselines (T+10, hit2)
    baseline_all = metrics(frame, np.ones(len(frame), dtype=bool))
    mask_liq = np.asarray(frame["vol_ok"].astype(bool), dtype=bool)
    baseline_liq = metrics(frame, mask_liq)

    print(f"Baseline (all):     hit2={baseline_all['hit2']:.3f} n={baseline_all['n']}")
    print(f"Baseline (liquid):  hit2={baseline_liq['hit2']:.3f} n={baseline_liq['n']}")

    # Candidates
    candidates = define_candidates()
    results = evaluate_candidates(frame, candidates, baseline_liq)

    # Sell signals (drop probability)
    sells = {}
    for col in ("luc_sell", "k4_sell", "mama_sell", "ats_sell"):
        if col in frame.columns:
            sm = sell_metrics(frame, col)
            if sm:
                sells[col] = sm

    # Regime gating
    regime_gating = {}
    for tone in ("risk_off", "neutral", "risk_on", "overheated"):
        m_sel = np.asarray(frame["regime"].to_numpy() == tone, dtype=bool)
        base_tone = metrics(frame, m_sel & mask_liq)
        row = {"baseline_liq": base_tone}
        for key in ("mom_any", "mom_strong", "ens_any", "ens_strong", "confluence_2", "confluence_3"):
            if key not in frame.columns:
                continue
            cand_mask = np.asarray(frame[key].astype(bool), dtype=bool)
            m = metrics(frame, cand_mask & m_sel & mask_liq)
            if m is None:
                continue
            row[key] = {
                "hit2": m["hit2"],
                "n": m["n"],
                "lift2_vs_baseline_in_regime": (
                    round(m["hit2"] - base_tone["hit2"], 4)
                    if base_tone and base_tone["hit2"] is not None else None),
            }
        regime_gating[tone] = row

    # Walk-forward
    print("Running walk-forward OOS...")
    wf = walk_forward(frame)

    now = vn_now()
    output = {
        "generated_at": now.isoformat(),
        "date": now.strftime("%d/%m/%Y"),
        "num_symbols_tested": len(frames),
        "observations": len(frame),
        "lookforward_days": list(FWD_OPTIONS),
        "baseline_all": baseline_all,
        "baseline_liquid": baseline_liq,
        "filters": results,
        "sell_signals": sells,
        "regime_gating": regime_gating,
        "regime_series": {"n_dates": len(regime_map),
                          "buckets": {t: sum(1 for v in regime_map.values() if v == t) for t in ("risk_off", "neutral", "risk_on", "overheated")}},
        "walk_forward": wf,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    DOCS_OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    DOCS_OUTPUT_JSON.write_bytes(OUTPUT_JSON.read_bytes())

    print(f"\nSaved: {OUTPUT_JSON}")
    print(f"Synced: {DOCS_OUTPUT_JSON}")

    # Console summary
    print("\n--- Filter effectiveness (T+10, hit2 = +2%) ---")
    header = f"{'filter':34s} {'n':>7s} {'hit2':>6s} {'lift':>6s} {'med':>7s} {'train':>6s} {'test':>6s}"
    print(header)
    print("-" * len(header))
    for key, r in results.items():
        m = r.get("metrics_liq") or r["metrics"]
        lift = r.get("lift2_liq")
        tr = m.get("train_hit2")
        te = m.get("test_hit2")
        print(f"{key:34s} {m['n']:7d} {m['hit2']:6.3f} {('%.3f' % lift) if lift is not None else '  -':>6s} "
              f"{m['median']:7.4f} {('%.3f' % tr) if tr is not None else '   -':>6s} {('%.3f' % te) if te is not None else '   -':>6s}")
    print("\n--- Sell signals (probability of drop T+10) ---")
    for col, sm in sells.items():
        print(f"{col:12s} n={sm['n']:6d} p_drop0={sm['p_drop0']:.3f} p_drop2={sm['p_drop2']:.3f} median={sm['median']:+.4f}")
    print("\n--- Walk-forward (cutoff %d) ---" % OOS_CUTOFF.year)
    for k in ("current_strong", "oos_strong", "current_weak_low", "oos_weak_low"):
        if k in wf:
            v = wf[k]
            print(f"  {k:16s} train={v['train']} test={v['test']}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())