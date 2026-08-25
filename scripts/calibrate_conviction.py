"""
Calibrate Buy Conviction thresholds tren TOAN BO lich su OHLC (2015 -> nay).

Phuong phap: tai tao diem momentum tung ngay theo cach vectorized giong
evaluate_filters.py (mirror dung cong thuc momentum_signals.py), sau do:
  1. Tinh conviction cho moi ngay-ma (cong thuc v2 trong buy_conviction.py)
  2. Do hit rate T+5/T+10/T+20 theo tung nguong hang A/B (grid search),
     split train (<2022) / test (>=2022) giong walk-forward cua evaluate_filters
  3. Kiem tra tuong duong voi he that: so sanh score tai ngay moi nhat cua
     cache voi momentum_signals.json thuc te
  4. Thuoc tinh tung thanh phan (vol/adx/duoi/shadow) trong band mom_watch

Chay tren CI (co day du ohlc_cache qua backfill_history.py):
    python scripts/calibrate_conviction.py [--limit-symbols N]

Dau ra: data/buy_conviction_calibration.json + bang ket qua ra console (ASCII).
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd

from _shared import (
    CACHE_DIR, DATA_DIR, MIN_SYMBOL_HISTORY, SCORE_BREAKOUT, SCORE_HYBRID,
    SCORE_MA, SCORE_ROC, vn_now,
)
from cache_utils import load_cache as _load_cache, compute_rsi_wilder_series
from backtest_momentum import (
    compute_adx_series, compute_breakout_signal, compute_hybrid_signal,
    compute_ma_crossover_signal, compute_roc_momentum_signal,
)
from buy_conviction import (
    ADX_BONUS_LEVEL, ADX_BONUS_POINTS, EXT_DIST_MA20_MAX, EXT_PENALTY_POINTS,
    TIER_A_MIN, TIER_B_MIN, VOL_BONUS_POINTS, VOL_BONUS_RATIO,
    STREAK_FATIGUE_DAYS, ATR_OVER_BAND, W52_HEALTHY_POS, W52_HEALTHY_DIST_MAX,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

OUTPUT_JSON = DATA_DIR / "buy_conviction_calibration.json"
MOMENTUM_JSON = DATA_DIR / "momentum_signals.json"

# Momentum dung nguong 500K (momentum_signals.MIN_AVG_VOLUME);
# baseline "liquid" cua evaluation la 300K - giu ca hai de so sanh.
MOM_MIN_AVG_VOLUME = 500_000
LIQ_MIN_AVG_VOLUME = 300_000
OOS_YEAR = 2022
FWD_OPTIONS = (5, 10, 20)

REGIME_FACTOR_BY_TONE = {
    "risk_on": 1.0, "neutral": 0.9, "overheated": 0.85, "risk_off": 0.8,
}


def _norm_tone(tone: str | None) -> str | None:
    if not tone:
        return None
    return str(tone).lower().replace("-", "_").replace(" ", "_")


def build_frame(symbol: str) -> pd.DataFrame | None:
    """Mot dong moi ngay voi conviction inputs - mirror cong thuc momentum."""
    try:
        df = _load_cache(symbol, CACHE_DIR)
    except Exception:
        return None
    if df is None or len(df) < MIN_SYMBOL_HISTORY:
        return None
    df = df.sort_values("TradingDate").reset_index(drop=True)
    n = len(df)

    close = df["Close"].astype(float)
    volume = df["Volume"].fillna(0.0).astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    fwd = {lf: (close.shift(-lf) / close - 1.0) for lf in FWD_OPTIONS}

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    rsi = compute_rsi_wilder_series(close, 14)
    adx = compute_adx_series(close, 14)
    vol_avg20 = volume.rolling(20).mean()

    trend_ok = (close > ma50) & (ma50 > ma200)
    rsi_band = (rsi >= 48) & (rsi <= 72)
    adx_ok = adx >= 20
    vol_mom = vol_avg20 >= MOM_MIN_AVG_VOLUME
    strategy_ok = (trend_ok & rsi_band & adx_ok & vol_mom).fillna(False)

    sig_ma = compute_ma_crossover_signal(close, volume, rsi, adx)
    sig_bo = compute_breakout_signal(close, volume, rsi)
    sig_roc = compute_roc_momentum_signal(close, volume, rsi)
    sig_hy = compute_hybrid_signal(close, volume, rsi, adx)
    score = (sig_ma * SCORE_MA + sig_bo * SCORE_BREAKOUT
             + sig_roc * SCORE_ROC + sig_hy * SCORE_HYBRID)

    vol_ratio = np.where(vol_avg20 > 0, volume / vol_avg20, 0.0)
    dist_ma20 = (close / ma20 - 1.0) * 100.0

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_pct = tr.ewm(alpha=1.0 / 14.0, adjust=False).mean() / close * 100.0

    hi52 = close.rolling(252, min_periods=120).max()
    lo52 = close.rolling(252, min_periods=120).min()
    w52_pos = (close - lo52) / (hi52 - lo52) * 100.0

    rsi_gold = (rsi >= 50) & (rsi <= 68)
    has_signal = strategy_ok & (score >= 30)
    # raw: base + bonuses (single-count) + rsi_gold. Live double-counts vol/adx via buy_conviction, so keep single here for calibration clarity
    raw = (score
           + np.where(vol_ratio >= VOL_BONUS_RATIO, VOL_BONUS_POINTS, 0.0)
           + np.where(adx > ADX_BONUS_LEVEL, ADX_BONUS_POINTS, 0.0)
           + np.where(rsi_gold, 10.0, 0.0)
           + np.where(dist_ma20 > EXT_DIST_MA20_MAX, EXT_PENALTY_POINTS, 0.0))

    out = pd.DataFrame({
        "symbol": symbol,
        "date": pd.to_datetime(df["TradingDate"]),
        "fwd5": fwd[5],
        "fwd10": fwd[10],
        "fwd20": fwd[20],
        "score": score,
        "strategy_ok": strategy_ok.astype(np.int8),
        "has_signal": has_signal.astype(np.int8),
        "eligible_a": (strategy_ok & (score >= 30) & (score < 60)).astype(np.int8),
        "hot": (strategy_ok & (score >= 60)).astype(np.int8),
        "vol_ratio": vol_ratio,
        "dist_ma20": dist_ma20,
        "adx": adx,
        "atr_pct": atr_pct,
        "w52_pos": w52_pos,
        "raw": raw,
        "liq300": (vol_avg20 >= LIQ_MIN_AVG_VOLUME).astype(np.int8),
    })
    out["_valid"] = False
    out["_close"] = close
    start = MIN_SYMBOL_HISTORY
    end = n - max(FWD_OPTIONS)
    if end <= start:
        return None
    out.iloc[start:end, out.columns.get_loc("_valid")] = True
    return out


def apply_regime_factor(big: pd.DataFrame, regime_map: dict[pd.Timestamp, str]) -> pd.DataFrame:
    date_factor = {
        d: REGIME_FACTOR_BY_TONE.get(_norm_tone(tone), 1.0)
        for d, tone in regime_map.items()
    }
    big = big.copy()
    big["factor"] = pd.to_datetime(big["date"]).dt.normalize().map(date_factor).fillna(1.0)
    big["conviction"] = np.where(big["raw"].notna(), (big["raw"] * big["factor"]).round(), np.nan)
    return big


def hit_stats(df: pd.DataFrame, mask: pd.Series, fwd: str = "fwd10") -> dict | None:
    sub = df.loc[mask & df["_valid"], ["date", fwd]].dropna()
    if len(sub) < 30:
        return None
    years = sub["date"].dt.year
    train = sub.loc[years < OOS_YEAR, fwd]
    test = sub.loc[years >= OOS_YEAR, fwd]
    return {
        "n": int(len(sub)),
        "hit2": round(float((sub[fwd] >= 0.02).mean()), 4),
        "mean": round(float(sub[fwd].mean()), 4),
        "median": round(float(sub[fwd].median()), 4),
        "train_n": int(len(train)),
        "train_hit2": round(float((train >= 0.02).mean()), 4) if len(train) >= 30 else None,
        "test_n": int(len(test)),
        "test_hit2": round(float((test >= 0.02).mean()), 4) if len(test) >= 30 else None,
    }


def fmt_row(name: str, m: dict | None, base_hit2: float | None) -> str:
    if m is None:
        return f"  {name:28s} (du lieu khong du)"
    lift = (m["hit2"] - base_hit2) if base_hit2 is not None else 0.0
    tlift = (m["test_hit2"] - base_hit2) if (base_hit2 is not None and m["test_hit2"] is not None) else None
    tstr = f"{tlift:+.4f}" if tlift is not None else "   -   "
    return (f"  {name:28s} n={m['n']:6d} hit2={m['hit2']:.4f} lift={lift:+.4f} "
            f"test_lift={tstr} mean={m['mean']:+.4f}")


def equivalence_check(big: pd.DataFrame) -> dict:
    """So score tai ngay moi nhat cua cache voi momentum_signals.json that."""
    mom = None
    try:
        mom = json.loads(MOMENTUM_JSON.read_text(encoding="utf-8"))
    except Exception:
        pass
    if not mom:
        return {"status": "skip", "reason": "khong doc duoc momentum_signals.json"}
    actual = {r.get("symbol"): r.get("score") for r in mom.get("all_signals") or []}
    if not actual:
        return {"status": "skip", "reason": "momentum rong"}
    # Thu nhieu ngay de tim nhieu symbol chung
    dates_sorted = sorted(big["date"].unique(), reverse=True)
    best_common: dict[str, tuple[int, int]] = {}
    best_date = None
    for d in dates_sorted[:5]:
        tail = big[(big["date"] == d) & big["strategy_ok"].astype(bool) & (big["score"] >= 30)]
        recomputed = dict(zip(tail["symbol"], tail["score"].round().astype(int)))
        common = set(recomputed) & set(actual)
        if len(common) > len(best_common):
            best_common = {s: (int(recomputed[s]), int(actual[s])) for s in common}
            best_date = d
    if not best_common:
        return {"status": "warn", "reason": "khong giao nhau trong 5 ngay gan nhat",
                "cache_last_date": str(dates_sorted[0].date()) if dates_sorted else "?"}
    common = best_common
    diffs = {s: v[0] - v[1] for s, v in common.items()}
    exact = sum(1 for d in diffs.values() if abs(d) <= 0)
    near = sum(1 for d in diffs.values() if abs(d) <= 2)
    worst = sorted(diffs.items(), key=lambda kv: -abs(kv[1]))[:8]
    return {
        "status": "ok",
        "cache_last_date": str(best_date.date()),
        "compared": len(common),
        "exact_match_rate": round(exact / len(common), 4),
        "within2_match_rate": round(near / len(common), 4),
        "max_abs_diff": max(abs(d) for d in diffs.values()),
        "worst": worst,
    }


def main():
    ap = argparse.ArgumentParser(description="Calibrate buy conviction thresholds")
    ap.add_argument("--limit-symbols", type=int, default=0, help="gioi han so ma (debug)")
    args = ap.parse_args()

    symbols = sorted(p.stem for p in CACHE_DIR.glob("*.csv") if p.stem != ".gitkeep")
    if args.limit_symbols:
        symbols = symbols[: args.limit_symbols]
    print(f"Tai tao lich su cho {len(symbols)} ma ...")

    frames = []
    for i, sym in enumerate(symbols):
        fr = build_frame(sym)
        if fr is not None:
            frames.append(fr)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(symbols)}")
    if not frames:
        print("Khong co frame nao - kiem tra ohlc_cache.")
        return 1

    # Regime theo ngay (tan dung bo ton tai cua evaluate_filters)
    from evaluate_filters import build_regime_map
    regime_map = build_regime_map(frames)
    print(f"Regime map: {len(regime_map)} ngay")

    big = pd.concat(frames, ignore_index=True)
    del frames
    big = apply_regime_factor(big, regime_map)

    eq = equivalence_check(big)
    print(f"Tuong duong he that: {eq.get('status')} (exact={eq.get('exact_match_rate')}, "
          f"within2={eq.get('within2_match_rate')}, maxdiff={eq.get('max_abs_diff')})")

    results = {"generated_at": vn_now().isoformat(),
               "oos_year": OOS_YEAR,
               "symbols": len(symbols),
               "equivalence": eq}

    # --- Baseline liquid ---
    base_mask = big["liq300"].astype(bool)
    base = hit_stats(big, base_mask)
    base_train = base["train_hit2"] if base else None
    base_test = base["test_hit2"] if base else None
    print("\n== Baseline liquid (vol20>=300K) ==")
    print(fmt_row("baseline_liq", base, None))
    results["baseline_liq"] = base

    # --- Grid search nguong hang A/B ---
    band = big["eligible_a"].astype(bool)
    hot = big["hot"].astype(bool)
    any_sig = band | hot
    print("\n== Grid search nguong hang A (trong so hien tai, regime factor ap dung) ==")
    grid = []
    for a_min in (55, 58, 60, 62, 65, 68, 70):
        tier_a = band & (big["conviction"] >= a_min)
        m = hit_stats(big, tier_a)
        row = {"a_min": a_min, **(m or {})}
        grid.append(row)
        print(fmt_row(f"A>={a_min}", m, base["hit2"] if base else None))
    results["grid_tier_a"] = grid

    print("\n== Hang B theo nguong (A>=60 co dinh) ==")
    grid_b = []
    for b_min in (40, 45, 48, 50):
        tier_a = band & (big["conviction"] >= TIER_A_MIN)
        tier_b = any_sig & (big["conviction"] >= b_min) & ~tier_a
        m = hit_stats(big, tier_b)
        grid_b.append({"b_min": b_min, **(m or {})})
        print(fmt_row(f"B>={b_min}", m, base["hit2"] if base else None))
    results["grid_tier_b"] = grid_b

    # --- Thuoc tinh thanh phan trong band mom_watch ---
    print("\n== Thuoc tinh thanh phan (trong band 30-59) ==")
    attr = {}
    vr = big["vol_ratio"]
    adx = big["adx"]
    dm = big["dist_ma20"]
    conds = {
        "band_all": band,
        "band_vol_ge2": band & (vr >= VOL_BONUS_RATIO),
        "band_vol_lt2": band & (vr < VOL_BONUS_RATIO),
        "band_adx_gt28": band & (adx > ADX_BONUS_LEVEL),
        "band_adx_le28": band & (adx <= ADX_BONUS_LEVEL),
        "band_extended": band & (dm > EXT_DIST_MA20_MAX),
        "band_not_extended": band & (dm <= EXT_DIST_MA20_MAX),
        "band_clean_best": band & (vr >= VOL_BONUS_RATIO) & (adx > ADX_BONUS_LEVEL) & (dm <= EXT_DIST_MA20_MAX),
        "band_hot_ge60": hot,
    }
    for name, mask in conds.items():
        m = hit_stats(big, mask)
        attr[name] = m
        print(fmt_row(name, m, base["hit2"] if base else None))
    results["attribution"] = attr

    # --- Shadow candidates ---
    print("\n== Shadow candidates (weight=0, danh gia tren band 30-59) ==")
    shadow_out = {}
    sig_any_day = big.loc[big["has_signal"].astype(bool), ["symbol", "date"]].sort_values(["symbol", "date"])
    grp = sig_any_day.groupby("symbol")["date"]
    diff_days = grp.diff().dt.days
    new_run = (diff_days.isna() | (diff_days > 7))
    run_id = new_run.cumsum()
    streak_col = (sig_any_day.groupby(["symbol", run_id]).cumcount() + 1).astype(int)
    streaks_df = pd.DataFrame({
        "symbol": sig_any_day["symbol"],
        "_dnorm": pd.to_datetime(sig_any_day["date"]).dt.normalize(),
        "streak": streak_col,
    })
    big["_dnorm"] = pd.to_datetime(big["date"]).dt.normalize()
    big = big.merge(streaks_df, on=["symbol", "_dnorm"], how="left")
    big["streak"] = big["streak"].fillna(0).astype(int)
    big = big.drop(columns=["_dnorm"])

    sh_conds = {
        "shadow_streak_ge4": band & (big["streak"] >= STREAK_FATIGUE_DAYS),
        "shadow_streak_lt4": band & (big["streak"] < STREAK_FATIGUE_DAYS),
        "shadow_atr_over8": band & (big["atr_pct"] > ATR_OVER_BAND),
        "shadow_atr_le8": band & (big["atr_pct"] <= ATR_OVER_BAND),
        "shadow_w52_healthy": band & (big["w52_pos"] >= W52_HEALTHY_POS) & (big["dist_ma20"] <= W52_HEALTHY_DIST_MAX),
        "shadow_w52_other": band & ~((big["w52_pos"] >= W52_HEALTHY_POS) & (big["dist_ma20"] <= W52_HEALTHY_DIST_MAX)),
    }
    for name, mask in sh_conds.items():
        m = hit_stats(big, mask)
        shadow_out[name] = m
        print(fmt_row(name, m, base["hit2"] if base else None))
    results["shadow"] = shadow_out

    # --- Khuyen nghi nguong A: uu tien test_lift cao, n>=300 ---
    best = None
    for row in grid:
        if not row or row.get("test_hit2") is None:
            continue
        if (row.get("test_n") or 0) < 300:
            continue
        tl = row["test_hit2"] - (base_test or 0)
        if best is None or tl > (best["test_hit2"] - (base_test or 0)):
            best = row
    if best:
        rec = {"recommended_a_min": best["a_min"], "test_lift": round(best["test_hit2"] - (base_test or 0), 4),
              "note": "chon theo test_lift cao nhat voi test_n>=300; nguoi duyet chot cuoi"}
        print(f"\nKhuyen nghi: A>={best['a_min']} (test_lift {rec['test_lift']:+.4f})")
        results["recommendation"] = rec

    OUTPUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\nDa ghi: {OUTPUT_JSON.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
