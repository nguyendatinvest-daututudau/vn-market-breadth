"""
Signal Performance — do hieu qua tin hieu MUA that da phat trong signals_history.

Pham vi v1 (backend truoc, UI sau):
- Chi do tin hieu THAT trong data/signals_history.json (toi da 120 phien),
  khong tai hien logic sinh tin hieu.
- Thang tinh CA HAI cach (theo lua chon user):
  (a) hit-rate gia dong T+5/10/20 dat +2% (hit2), kem hit0/hit7/median/mean;
  (b) target cham truoc stop (proxy ATR: stop = entry - 2*ATR14,
      target = entry + 2*ATR14, window 20 phien, dung High/Low).
- Decay/reversal: duong cong hit2 T+1..T+20, MFE/MAE, so phien trung vi toi
  +2% / -2% dau tien, va so phien toi sell dau tien cung-he (neu co).

Proxy trade-plan duoc ghi ro trong output.caveats vi lich su hien tai
(buy_conviction_history) khong luu stop/target tung lenh. Khi nao lich su
luu san stop/target thi script se uu tien dung gia tri that.

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

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

HISTORY_JSON = DATA_DIR / "signals_history.json"
OUTPUT_JSON = DATA_DIR / "signal_performance.json"
DOCS_OUTPUT_JSON = DOCS_DATA_DIR / "signal_performance.json"

HORIZONS = (5, 10, 20)
DECAY_MAX = 20
PLAN_BARS = 20
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


def evaluate_signal(frame: pd.DataFrame, idx: int, entry: float) -> dict | None:
    n = len(frame)
    closes = frame["Close"].astype(float).to_numpy()
    highs = frame["High"].astype(float).to_numpy()
    lows = frame["Low"].astype(float).to_numpy()
    out: dict = {"matured": {}, "fwd": {}, "mfe": None, "mae": None,
                 "bars_to_p2": None, "bars_to_m2": None,
                 "plan": None, "decay": {}}
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
    # plan proxy ATR
    atr = _atr14(frame, idx)
    if atr is None or not math.isfinite(atr) or atr <= 0:
        atr = entry * 0.02
    stop = entry - 2 * atr
    target = entry + 2 * atr
    outcome = "timeout"
    bars = None
    for k in range(1, PLAN_BARS + 1):
        j = idx + k
        if j >= n:
            break
        hi, lo = highs[j], lows[j]
        if not (math.isfinite(hi) and math.isfinite(lo)):
            continue
        hit_t = hi >= target
        hit_s = lo <= stop
        if hit_t and hit_s:
            outcome = "tie"
            bars = k
            break
        if hit_t:
            outcome = "target_first"
            bars = k
            break
        if hit_s:
            outcome = "stop_first"
            bars = k
            break
    matured_plan = (idx + 1) < n  # co it nhat 1 forward bar
    out["plan"] = {"stop": round(stop, 2), "target": round(target, 2),
                   "atr": round(float(atr), 2), "outcome": outcome,
                   "bars": bars, "matured": bool(matured_plan)}
    out["any_matured"] = bool(any_matured or matured_plan)
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
          "target_first_rate": v["plan"]["target_first_rate"]}
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
            "hit2": "close T+N / entry - 1 >= +2%",
            "plan_proxy": "stop = entry - 2*ATR14, target = entry + 2*ATR14, window 20 phien (High/Low)",
        },
        "per_system": per_system,
        "leaderboard": leaderboard,
        "recent": recent,
        "caveats": [
            "Chi do tin hieu THAT trong signals_history (toi da 120 phien); phien gan nhat chua du T+5/10/20 thi tu dong loai khoi mau tuong ung.",
            "Plan la PROXY ATR (lich su chua luu stop/target that) — dung de so he voi nhau, khong phai khuyen nghi giao dich.",
            "Survivorship bias: cache chi con ma dang niem yet; overlapping T+10 khong phai mau doc lap.",
            "Gia adjusted + corporate action co the lech entry/forward o ma chia tach.",
        ],
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    DOCS_OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    DOCS_OUTPUT_JSON.write_bytes(OUTPUT_JSON.read_bytes())

    print(f"Sessions: {len(history)} | buy matured: {len(records)} | skip no-cache: {skipped_no_cache}")
    print("--- Leaderboard T+10 hit2 ---")
    for row in leaderboard:
        print(f"  {row['label']:14s} n={row['n']:5d} hit2={row['hit2_T10']} target_first={row['target_first_rate']}")
    print(f"Saved: {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
