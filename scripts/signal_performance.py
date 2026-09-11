"""
Signal Performance — do hieu qua tin hieu MUA that da phat trong signals_history.

- Chi do tin hieu THAT trong data/signals_history.json (toi da 120 phien),
  khong tai hien logic sinh tin hieu.
- Cac cach do:
  (a) hit-rate gia dong T+5/10/20 dat +2% (hit2), kem hit0/hit7/median/mean;
  (b) plan proxy ATR: stop = entry - 2*ATR14, target = entry + 2*ATR14,
      first-touch window 20 phien (High/Low);
  (c) pm5: cham entry*1.05 truoc entry*0.95 trong 20 phien (High/Low);
  (d) plan_real: TAI TINH compute_trade_plan() tai dung ngay tin hieu tu
      OHLC cache (SMA/ATR14/pivot classic/W52/High20 theo dung cong thuc
      stock_health), do first-touch R1 (chinh) vs stop trong 20 phien,
      R2 touch rate lam cot phu.
- Decay/reversal: duong cong hit2 T+1..T+20, MFE/MAE, so phien trung vi toi
  +2% / -2% dau tien, va so phien toi sell dau tien cung-he (neu co).

Cach chay:
    cd scripts && python signal_performance.py [--limit-days N]
"""
from __future__ import annotations

import argparse
import json
import math
import warnings

import numpy as np
import pandas as pd

from _shared import (
    CACHE_DIR,
    DATA_DIR,
    DOCS_DATA_DIR,
    json_default,
    parse_market_date,
    vn_now,
)
from cache_utils import load_cache as _load_cache
from buy_conviction import compute_trade_plan
from stock_health import _atr14 as _sh_atr14, _pivots as _sh_pivots, _sma as _sh_sma

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

HISTORY_JSON = DATA_DIR / "signals_history.json"
OUTPUT_JSON = DATA_DIR / "signal_performance.json"
DOCS_OUTPUT_JSON = DOCS_DATA_DIR / "signal_performance.json"

HORIZONS = (5, 10, 20)
DECAY_MAX = 20
PLAN_BARS = 20
PM5_BARS = 20
PLAN_REAL_BARS = 20
RECENT_CAP = 300

SYSTEMS = {
    "strategy": "Pre-Breakout",
    "ensemble": "Ensemble",
    "momentum": "Momentum",
    "luc_mach": "Luc-Mach",
    "khung4_tplus": "Khung4-Tplus",
    "mama_positional": "MAMA",
    "advanced_trailstop": "ATS",
}

BUY_STATUSES = {"BUY", "VALID_BUY", "STRONG_BUY", "VALID", "STRONG", "HOT"}
SELL_STATUSES = {"SELL", "SELL_WARNING", "SELL_WARN", "SELLWATCH"}


def _is_buy(sig: dict) -> bool:
    st = str(sig.get("status") or sig.get("signal_type") or "").upper()
    if not st:
        return True  # he khong co status (strategy/ensemble/momentum cu): all_signals = buy
    if "SELL" in st:
        return False
    if "WATCH" in st or "CONFLICT" in st:
        return False
    return True


def _is_sell(sig: dict) -> bool:
    st = str(sig.get("status") or sig.get("signal_type") or "").upper()
    return "SELL" in st


def _entry_price(sig: dict) -> float | None:
    for key in ("buy_price", "bprice", "last_buy_price", "last_price"):
        v = sig.get(key)
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f) and f > 0:
            return f
    return None


def _atr14(df: pd.DataFrame, idx: int) -> float | None:
    try:
        hi = df["High"].astype(float).to_numpy()
        lo = df["Low"].astype(float).to_numpy()
        cl = df["Close"].astype(float).to_numpy()
    except (KeyError, ValueError, TypeError):
        return None
    lo_i = max(1, idx - 13)
    trs = []
    for i in range(lo_i, idx + 1):
        if i <= 0 or i >= len(cl):
            continue
        tr = max(
            hi[i] - lo[i],
            abs(hi[i] - cl[i - 1]),
            abs(lo[i] - cl[i - 1]),
        )
        if math.isfinite(tr) and tr > 0:
            trs.append(tr)
    if not trs:
        return None
    return float(sum(trs) / len(trs))


def _load_symbol_frame(symbol: str) -> pd.DataFrame | None:
    try:
        df = _load_cache(symbol, CACHE_DIR)
    except Exception:
        return None
    if df is None or df.empty or "TradingDate" not in df.columns:
        return None
    need = {"Open", "High", "Low", "Close"}
    if not need.issubset(set(df.columns)):
        return None
    dates = pd.to_datetime(df["TradingDate"], dayfirst=True, errors="coerce")
    df = df.copy()
    df["_d"] = dates
    df = df.dropna(subset=["_d"]).sort_values("_d").reset_index(drop=True)
    return df


def _find_idx(frame: pd.DataFrame, sig_date) -> int | None:
    arr = frame["_d"].to_numpy()
    target = pd.Timestamp(sig_date)
    pos = int(np.searchsorted(arr.astype("datetime64[ns]"), target.to_datetime64()))
    if pos >= len(arr):
        return None
    return pos


def _first_touch(highs, lows, n: int, idx: int, up: float, down: float,
                 window: int) -> tuple[str, int | None]:
    """First-touch up vs down trong `window` phien forward (dung High/Low).

    Tra ve (outcome, bars) voi outcome trong
    {"up_first", "down_first", "tie", "timeout"}.
    """
    outcome, bars = "timeout", None
    if not (math.isfinite(up) and math.isfinite(down)) or up <= down:
        return outcome, bars
    for k in range(1, window + 1):
        j = idx + k
        if j >= n:
            break
        hi, lo = highs[j], lows[j]
        if not (math.isfinite(hi) and math.isfinite(lo)):
            continue
        hit_up = hi >= up
        hit_dn = lo <= down
        if hit_up and hit_dn:
            return "tie", k
        if hit_up:
            return "up_first", k
        if hit_dn:
            return "down_first", k
    return outcome, bars


def _recompute_trade_plan(frame: pd.DataFrame, idx: int) -> dict | None:
    """Tai tinh trade plan that tai dung ngay tin hieu.

    Mirror dung cong thuc stock_health (SMA/ATR14/pivot classic/W52/High20)
    tren lich su den het phien tin hieu, roi goi compute_trade_plan() hien tai.
    """
    try:
        hist = frame.iloc[:idx + 1]
        if len(hist) < 15:
            return None
        close = float(hist["Close"].iloc[-1])
        if not math.isfinite(close) or close <= 0:
            return None
        closes = hist["Close"]
        sma10 = _sh_sma(closes, 10)
        sma20 = _sh_sma(closes, 20)
        sma50 = _sh_sma(closes, 50)
        sma200 = _sh_sma(closes, 200)
        atr14 = _sh_atr14(hist)
        dist_ma20 = (close / sma20 - 1.0) * 100.0 if sma20 else None
        pivots = _sh_pivots(hist)
        win = hist.tail(252)
        try:
            w52_high = float(win["High"].max())
            w52_low = float(win["Low"].min())
        except (ValueError, TypeError):
            w52_high = w52_low = None
        if w52_high is not None and not math.isfinite(w52_high):
            w52_high = None
        if w52_low is not None and not math.isfinite(w52_low):
            w52_low = None
        try:
            high20 = float(hist["High"].tail(20).max()) if len(hist) >= 20 else None
        except (ValueError, TypeError):
            high20 = None
        if high20 is not None and not math.isfinite(high20):
            high20 = None
        return compute_trade_plan(
            close=close, sma10=sma10, sma20=sma20, sma50=sma50,
            sma200=sma200, atr14=atr14, dist_ma20=dist_ma20,
            pivots=pivots, w52_high=w52_high, w52_low=w52_low,
            high20=high20,
        )
    except Exception:
        return None


def evaluate_signal(frame: pd.DataFrame, idx: int, entry: float) -> dict | None:
    n = len(frame)
    closes = frame["Close"].astype(float).to_numpy()
    highs = frame["High"].astype(float).to_numpy()
    lows = frame["Low"].astype(float).to_numpy()
    out: dict = {"matured": {}, "fwd": {}, "mfe": None, "mae": None,
                 "bars_to_p2": None, "bars_to_m2": None,
                 "plan": None, "decay": {}, "pm5": None, "plan_real": None}
    any_matured = False
    for h in HORIZONS:
        j = idx + h
        if j < n and math.isfinite(closes[j]) and entry > 0:
            r = closes[j] / entry - 1.0
            if math.isfinite(r):
                out["fwd"][h] = float(r)
                out["matured"][h] = True
                any_matured = True
                continue
        out["matured"][h] = False
    # decay T+1..20 (close basis)
    for h in range(1, DECAY_MAX + 1):
        j = idx + h
        if j < n and entry > 0 and math.isfinite(closes[j]):
            r = closes[j] / entry - 1.0
            out["decay"][h] = float(r) if math.isfinite(r) else None
        else:
            out["decay"][h] = None
    # MFE/MAE + bars to +/-2% trong 20 phien (dung High/Low)
    mfe = mae = None
    b_p2 = b_m2 = None
    for k in range(1, PLAN_BARS + 1):
        j = idx + k
        if j >= n:
            break
        hi, lo = highs[j], lows[j]
        if not (math.isfinite(hi) and math.isfinite(lo)) or entry <= 0:
            continue
        mfe = max(mfe or -1e9, hi / entry - 1.0)
        mae = min(mae or 1e9, lo / entry - 1.0)
        if b_p2 is None and hi / entry - 1.0 >= 0.02:
            b_p2 = k
        if b_m2 is None and lo / entry - 1.0 <= -0.02:
            b_m2 = k
    out["mfe"] = float(mfe) if mfe is not None and mfe > -1e8 else None
    out["mae"] = float(mae) if mae is not None and mae < 1e8 else None
    out["bars_to_p2"] = b_p2
    out["bars_to_m2"] = b_m2
    # plan proxy ATR (giu nguyen ngu nghia cu)
    atr = _atr14(frame, idx)
    if atr is None or not math.isfinite(atr) or atr <= 0:
        atr = entry * 0.02
    stop = entry - 2 * atr
    target = entry + 2 * atr
    raw_outcome, bars = _first_touch(highs, lows, n, idx, target, stop, PLAN_BARS)
    outcome = {"up_first": "target_first", "down_first": "stop_first"}.get(raw_outcome, raw_outcome)
    matured_plan = (idx + 1) < n  # co it nhat 1 forward bar
    out["plan"] = {"stop": round(stop, 2), "target": round(target, 2),
                   "atr": round(float(atr), 2), "outcome": outcome,
                   "bars": bars, "matured": bool(matured_plan)}
    # (c) pm5: cham +5% truoc -5% trong 20 phien
    pm5_raw, pm5_bars = _first_touch(highs, lows, n, idx, entry * 1.05,
                                     entry * 0.95, PM5_BARS)
    pm5_outcome = {"up_first": "hit_p5_first",
                   "down_first": "hit_m5_first"}.get(pm5_raw, pm5_raw)
    full_window = (idx + PM5_BARS) < n
    pm5_matured = bool(pm5_raw in ("up_first", "down_first", "tie") or full_window)
    out["pm5"] = {"outcome": pm5_outcome, "bars": pm5_bars,
                  "matured": pm5_matured}
    # (d) plan_real: tai tinh trade plan that + first-touch R1 vs stop
    plan_real = {"outcome": None, "bars": None, "matured": False,
                 "r1": None, "r2": None, "stop": None,
                 "entry_mid": None, "entry_type": None, "r2_hit": None}
    rp = _recompute_trade_plan(frame, idx)
    if rp:
        r1 = (rp.get("targets") or {}).get("r1")
        r2 = (rp.get("targets") or {}).get("r2")
        rstop = rp.get("stop")
        entry_mid = (rp.get("entry_zone") or {}).get("mid")
        if (r1 is not None and math.isfinite(r1)
                and rstop is not None and math.isfinite(rstop) and r1 > rstop):
            rr_raw, rr_bars = _first_touch(highs, lows, n, idx, r1, rstop,
                                           PLAN_REAL_BARS)
            rr_outcome = {"up_first": "r1_first",
                          "down_first": "stop_first"}.get(rr_raw, rr_raw)
            full_real = (idx + PLAN_REAL_BARS) < n
            matured_real = bool(rr_raw in ("up_first", "down_first", "tie")
                                or full_real)
            r2_hit = None
            if r2 is not None and math.isfinite(r2):
                r2_hit = False
                for k in range(1, PLAN_REAL_BARS + 1):
                    j = idx + k
                    if j >= n:
                        break
                    hi = highs[j]
                    if math.isfinite(hi) and hi >= r2:
                        r2_hit = True
                        break
            plan_real = {"outcome": rr_outcome, "bars": rr_bars,
                         "matured": bool(matured_real),
                         "r1": round(float(r1), 2),
                         "r2": round(float(r2), 2) if r2 is not None and math.isfinite(r2) else None,
                         "stop": round(float(rstop), 2),
                         "entry_mid": round(float(entry_mid), 2) if entry_mid is not None and math.isfinite(entry_mid) else None,
                         "entry_type": rp.get("entry_type"),
                         "r2_hit": r2_hit}
    out["plan_real"] = plan_real
    out["any_matured"] = bool(any_matured or matured_plan
                              or pm5_matured or plan_real["matured"])
    return out


def _summarize(values: list[float]) -> dict | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return None
    arr = np.asarray(vals, dtype=float)
    return {
        "n": int(len(arr)),
        "hit0": round(float(np.mean(arr >= 0.0)), 4),
        "hit2": round(float(np.mean(arr >= 0.02)), 4),
        "hit7": round(float(np.mean(arr >= 0.07)), 4),
        "median": round(float(np.median(arr)), 4),
        "mean": round(float(np.mean(arr)), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Do hieu qua tin hieu mua that (signals_history)")
    ap.add_argument("--limit-days", type=int, default=0, help="Chi lay N phien gan nhat (smoke test)")
    args = ap.parse_args()

    if not HISTORY_JSON.exists():
        print(f"Khong thay {HISTORY_JSON}. Chay pipeline truoc.")
        return 0
    try:
        history = json.loads(HISTORY_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"Loi doc history: {e}")
        return 1
    if args.limit_days > 0:
        history = history[-args.limit_days:]

    frames: dict[str, pd.DataFrame] = {}
    sell_dates: dict[str, dict[str, list]] = {k: {} for k in SYSTEMS}

    # pass 1: sell map (de do reversal)
    for entry in history:
        d = parse_market_date(entry.get("date"))
        if d is None:
            continue
        for key in SYSTEMS:
            obj = entry.get(key) or {}
            for sig in obj.get("all_signals") or []:
                if not isinstance(sig, dict):
                    continue
                if _is_sell(sig):
                    sym = str(sig.get("symbol") or "").upper()
                    if sym:
                        sell_dates[key].setdefault(sym, []).append(d)

    records: list[dict] = []
    skipped_no_cache = 0
    for entry in history:
        d = parse_market_date(entry.get("date"))
        if d is None:
            continue
        for key in SYSTEMS:
            obj = entry.get(key) or {}
            for sig in obj.get("all_signals") or []:
                if not isinstance(sig, dict) or not _is_buy(sig):
                    continue
                sym = str(sig.get("symbol") or "").upper()
                if not sym or sym.startswith(("FU", "E1")):
                    continue
                entry_px = _entry_price(sig)
                if entry_px is None:
                    continue
                frame = frames.get(sym)
                if frame is None:
                    if sym in frames:
                        continue
                    frame = _load_symbol_frame(sym)
                    frames[sym] = frame  # type: ignore[assignment]
                    if frame is None:
                        skipped_no_cache += 1
                        continue
                idx = _find_idx(frame, d)
                if idx is None:
                    continue
                ev = evaluate_signal(frame, idx, entry_px)
                if ev is None or not ev.get("any_matured"):
                    continue
                # reversal: sell dau tien cung-he sau D
                bars_to_sell = None
                for sd in sell_dates[key].get(sym, []):
                    if sd > d:
                        try:
                            j = _find_idx(frame, sd)
                            if j is not None and j > idx:
                                bars_to_sell = j - idx
                                break
                        except Exception:
                            continue
                records.append({
                    "date": entry.get("date"),
                    "system": key,
                    "symbol": sym,
                    "entry": round(entry_px, 2),
                    **{f"fwd{h}": ev["fwd"].get(h) for h in HORIZONS},
                    "matured": ev["matured"],
                    "mfe": ev["mfe"],
                    "mae": ev["mae"],
                    "bars_to_p2": ev["bars_to_p2"],
                    "bars_to_m2": ev["bars_to_m2"],
                    "plan_outcome": ev["plan"]["outcome"] if ev.get("plan") else None,
                    "plan_bars": ev["plan"]["bars"] if ev.get("plan") else None,
                    "plan_matured": ev["plan"]["matured"] if ev.get("plan") else False,
                    "pm5_outcome": ev["pm5"]["outcome"] if ev.get("pm5") else None,
                    "pm5_bars": ev["pm5"]["bars"] if ev.get("pm5") else None,
                    "pm5_matured": ev["pm5"]["matured"] if ev.get("pm5") else False,
                    "plan_real_outcome": ev["plan_real"]["outcome"] if ev.get("plan_real") else None,
                    "plan_real_bars": ev["plan_real"]["bars"] if ev.get("plan_real") else None,
                    "plan_real_matured": ev["plan_real"]["matured"] if ev.get("plan_real") else False,
                    "plan_real_r2_hit": ev["plan_real"]["r2_hit"] if ev.get("plan_real") else None,
                    "plan_real_entry_type": ev["plan_real"]["entry_type"] if ev.get("plan_real") else None,
                    "decay": ev["decay"],
                    "bars_to_sell": bars_to_sell,
                })

    per_system: dict = {}
    for key, label in SYSTEMS.items():
        rows = [r for r in records if r["system"] == key]
        horizons = {}
        for h in HORIZONS:
            vals = [r[f"fwd{h}"] for r in rows if r["matured"].get(h) and r[f"fwd{h}"] is not None]
            horizons[f"T{h}"] = _summarize(vals) or {"n": 0}
        plan_rows = [r for r in rows if r["plan_matured"]]
        outcomes = {"target_first": 0, "stop_first": 0, "tie": 0, "timeout": 0}
        for r in plan_rows:
            outcomes[r["plan_outcome"]] = outcomes.get(r["plan_outcome"], 0) + 1
        n_plan = len(plan_rows) or 1
        plan = {
            "n": len(plan_rows),
            "target_first_rate": round(outcomes["target_first"] / n_plan, 4),
            "stop_first_rate": round(outcomes["stop_first"] / n_plan, 4),
            "tie_rate": round(outcomes["tie"] / n_plan, 4),
            "timeout_rate": round(outcomes["timeout"] / n_plan, 4),
            "outcomes": outcomes,
        }
        mfe = _summarize([r["mfe"] for r in rows])
        mae = _summarize([r["mae"] for r in rows])
        b2 = [r["bars_to_p2"] for r in rows if r["bars_to_p2"] is not None]
        bm2 = [r["bars_to_m2"] for r in rows if r["bars_to_m2"] is not None]
        bsell = [r["bars_to_sell"] for r in rows if r["bars_to_sell"] is not None]
        # (c) pm5 aggregate
        pm5_rows = [r for r in rows if r["pm5_matured"]]
        pm5_out = {"hit_p5_first": 0, "hit_m5_first": 0, "tie": 0, "timeout": 0}
        for r in pm5_rows:
            pm5_out[r["pm5_outcome"]] = pm5_out.get(r["pm5_outcome"], 0) + 1
        n_pm5 = len(pm5_rows) or 1
        pm5_bars = [r["pm5_bars"] for r in pm5_rows
                    if r["pm5_bars"] is not None
                    and r["pm5_outcome"] in ("hit_p5_first", "hit_m5_first", "tie")]
        pm5 = {
            "n": len(pm5_rows),
            "hit_p5_first_rate": round(pm5_out["hit_p5_first"] / n_pm5, 4),
            "hit_m5_first_rate": round(pm5_out["hit_m5_first"] / n_pm5, 4),
            "tie_rate": round(pm5_out["tie"] / n_pm5, 4),
            "timeout_rate": round(pm5_out["timeout"] / n_pm5, 4),
            "outcomes": pm5_out,
            "median_bars": round(float(np.median(pm5_bars)), 1) if pm5_bars else None,
        }
        # (d) plan_real aggregate
        pr_rows = [r for r in rows if r["plan_real_matured"]]
        pr_out = {"r1_first": 0, "stop_first": 0, "tie": 0, "timeout": 0}
        for r in pr_rows:
            pr_out[r["plan_real_outcome"]] = pr_out.get(r["plan_real_outcome"], 0) + 1
        n_pr = len(pr_rows) or 1
        pr_bars = [r["plan_real_bars"] for r in pr_rows
                   if r["plan_real_bars"] is not None
                   and r["plan_real_outcome"] in ("r1_first", "stop_first", "tie")]
        r2_hits = [r for r in pr_rows if r["plan_real_r2_hit"] is True]
        r2_known = [r for r in pr_rows if r["plan_real_r2_hit"] is not None]
        plan_real = {
            "n": len(pr_rows),
            "r1_first_rate": round(pr_out["r1_first"] / n_pr, 4),
            "stop_first_rate": round(pr_out["stop_first"] / n_pr, 4),
            "tie_rate": round(pr_out["tie"] / n_pr, 4),
            "timeout_rate": round(pr_out["timeout"] / n_pr, 4),
            "outcomes": pr_out,
            "r2_touch_rate": round(len(r2_hits) / len(r2_known), 4) if r2_known else None,
            "median_bars": round(float(np.median(pr_bars)), 1) if pr_bars else None,
        }
        decay = {}
        for h in range(1, DECAY_MAX + 1):
            vals = [r["decay"].get(h) for r in rows if r["decay"].get(h) is not None]
            s = _summarize(vals)
            decay[f"T{h}"] = {"hit2": s["hit2"], "median": s["median"], "n": s["n"]} if s else {"n": 0}
        per_system[key] = {
            "label": label,
            "n_total": len(rows),
            "horizons": horizons,
            "plan": plan,
            "pm5": pm5,
            "plan_real": plan_real,
            "mfe_median": mfe["median"] if mfe else None,
            "mae_median": mae["median"] if mae else None,
            "median_bars_to_p2": round(float(np.median(b2)), 1) if b2 else None,
            "median_bars_to_m2": round(float(np.median(bm2)), 1) if bm2 else None,
            "median_bars_to_sell": round(float(np.median(bsell)), 1) if bsell else None,
            "sell_hit_n": len(bsell),
            "decay": decay,
        }

    leaderboard = sorted(
        ({"system": k, "label": v["label"],
          "n": (v["horizons"].get("T10") or {}).get("n", 0),
          "hit2_T10": (v["horizons"].get("T10") or {}).get("hit2"),
          "target_first_rate": v["plan"]["target_first_rate"],
          "hit_p5_first_rate": v["pm5"]["hit_p5_first_rate"],
          "pm5_n": v["pm5"]["n"],
          "r1_first_rate": v["plan_real"]["r1_first_rate"],
          "plan_real_n": v["plan_real"]["n"]}
         for k, v in per_system.items()),
        key=lambda x: (x["hit2_T10"] is not None, x["hit2_T10"] or -1),
        reverse=True,
    )

    recent = sorted(records, key=lambda r: (str(r["date"]), r["symbol"]))[-RECENT_CAP:]

    now = vn_now()
    output = {
        "generated_at": now.isoformat(),
        "date": now.strftime("%d/%m/%Y"),
        "history_sessions": len(history),
        "history_range": [history[0].get("date"), history[-1].get("date")] if history else [None, None],
        "total_buy_signals": len(records),
        "skipped_no_cache": skipped_no_cache,
        "horizons": [f"T{h}" for h in HORIZONS],
        "win_definition": {
            "hit2": "close T+N / entry - 1 >= +2% (snapshot, khong phai first-touch)",
            "plan_proxy": "stop = entry - 2*ATR14, target = entry + 2*ATR14, first-touch window 20 phien (High/Low)",
            "pm5": "cham entry*1.05 truoc entry*0.95 trong 20 phien (High/Low); cung phien = tie",
            "plan_real": "tai tinh compute_trade_plan() tai ngay tin hieu; R1 (chinh) vs stop first-touch 20 phien; R2 touch rate phu",
        },
        "per_system": per_system,
        "leaderboard": leaderboard,
        "recent": recent,
        "caveats": [
            "Chi do tin hieu THAT trong signals_history (toi da 120 phien); phien gan nhat chua du T+5/10/20 thi tu dong loai khoi mau tuong ung.",
            "pm5/plan_real chi tinh khi du 20 phien forward hoac da cham 1 ben; timeout non-window khong dem.",
            "plan_real TAI TINH tu OHLC cache theo logic compute_trade_plan hien tai — co the lech plan da hien live neu logic tung thay doi giua cac phien.",
            "Plan proxy ATR giu lai de so sanh dai han; dung de so he voi nhau, khong phai khuyen nghi giao dich.",
            "Survivorship bias: cache chi con ma dang niem yet; overlapping windows khong phai mau doc lap.",
            "Gia adjusted + corporate action co the lech entry/forward o ma chia tach.",
        ],
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    DOCS_OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    DOCS_OUTPUT_JSON.write_bytes(OUTPUT_JSON.read_bytes())

    print(f"Sessions: {len(history)} | buy matured: {len(records)} | skip no-cache: {skipped_no_cache}")
    print("--- Leaderboard: hit2 T+10 | +5% truoc -5% | R1 that truoc stop ---")
    for row in leaderboard:
        print(f"  {row['label']:14s} n={row['n']:5d} hit2={row['hit2_T10']} "
              f"pm5={row['hit_p5_first_rate']}(n={row['pm5_n']}) "
              f"r1={row['r1_first_rate']}(n={row['plan_real_n']})")
    print(f"Saved: {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
