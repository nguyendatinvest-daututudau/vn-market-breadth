"""Backtest Zweig Breadth Thrust 16 thresholds + OOS, luu data/backtest_zweig.json."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from _shared import CACHE_DIR, DATA_DIR, DOCS_DATA_DIR, vn_now

HISTORY_JSON = DATA_DIR / "breadth_history.json"
OUTPUT_JSON = DATA_DIR / "backtest_zweig.json"
DOCS_OUTPUT = DOCS_DATA_DIR / "backtest_zweig.json"

LOWERS = [0.35, 0.38, 0.40, 0.42]
UPPERS = [0.60, 0.615, 0.62, 0.65]
LOOKFORWARDS = [5, 10, 20, 60]
OOS_CUTOFF = pd.Timestamp("2022-01-01")
MIN_OBS = 5


def _load_history() -> list:
    if not HISTORY_JSON.exists():
        return []
    try:
        return json.loads(HISTORY_JSON.read_text(encoding="utf-8"))
    except Exception:
        return []


def _load_vni_close() -> pd.Series | None:
    for sym in ("VNI", "VNINDEX", "VN30"):
        p = CACHE_DIR / f"{sym}.csv"
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p, encoding="utf-8")
            d = pd.to_datetime(df["TradingDate"], dayfirst=True, errors="coerce")
            c = pd.to_numeric(df["Close"], errors="coerce")
            s = pd.Series(c.values, index=d)
            s = s.sort_index().dropna()
            if len(s) > 100:
                return s
        except Exception:
            continue
    return None


def _compute_ema_ratios(history: list, market: str = "HOSE") -> tuple[list[str], list[float | None], pd.Series]:
    ratios: list[float | None] = []
    dates: list[str] = []
    for entry in history:
        m = (entry.get("markets") or {}).get(market) or {}
        a, d = m.get("advances"), m.get("declines")
        dates.append(entry.get("date"))
        if a is not None and d is not None and (a + d) > 0:
            ratios.append(float(a) / float(a + d))
        else:
            ratios.append(None)
    s = pd.Series(ratios)
    ema = s.ewm(span=10, adjust=False, min_periods=10).mean()
    return dates, ratios, ema


def _find_thrusts(ema: pd.Series, lower: float, upper: float) -> list[int]:
    thrusts = []
    for i in range(10, len(ema)):
        if pd.isna(ema.iloc[i]) or pd.isna(ema.iloc[i - 10]):
            continue
        window = ema.iloc[i - 10:i]
        if window.isna().any():
            continue
        if float(window.min()) < lower and float(ema.iloc[i]) > upper:
            # De-dup: ignore if thrust within 20d of prior
            if thrusts and i - thrusts[-1] < 20:
                continue
            thrusts.append(i)
    return thrusts


def _metrics(returns: list[float]) -> dict:
    if not returns:
        return {"n": 0, "hit0": None, "hit2": None, "mean": None, "median": None}
    s = pd.Series(returns)
    return {
        "n": int(len(s)),
        "hit0": round(float((s >= 0).mean()), 4),
        "hit2": round(float((s >= 0.02).mean()), 4),
        "mean": round(float(s.mean()), 4),
        "median": round(float(s.median()), 4),
    }


def main() -> int:
    history = _load_history()
    vni = _load_vni_close()
    if not history or vni is None or vni.empty:
        print("Thieu history hoac VNI cache")
        payload = {"generated_at": vn_now().isoformat(), "error": "thieu du lieu", "grid": []}
        OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        DOCS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        DOCS_OUTPUT.write_bytes(OUTPUT_JSON.read_bytes())
        return 1

    # Baseline: all days
    all_returns = {}
    for lf in LOOKFORWARDS:
        rets = []
        for i in range(len(vni) - lf):
            try:
                r = float(vni.iloc[i + lf] / vni.iloc[i] - 1)
                if abs(r) < 5:  # filter crazy
                    rets.append(r)
            except Exception:
                continue
        all_returns[lf] = rets

    baseline = {lf: _metrics(all_returns[lf]) for lf in LOOKFORWARDS}
    # Also split train/test for baseline
    # Map vni index dates to history dates for alignment not needed; use vni index directly for baseline split by date
    # For thrust, we need to map history index to vni date

    # Build date -> vni close map for thrust forward
    vni_dates = vni.index
    # History dates are dd/mm/yyyy, convert to Timestamp
    hist_dates = [pd.to_datetime(d, dayfirst=True, errors="coerce") for d in [h.get("date") for h in history]]
    # For each thrust, find corresponding vni date and compute forward
    dates, ratios, ema = _compute_ema_ratios(history, market="HOSE")

    grid = []
    for lo in LOWERS:
        for hi in UPPERS:
            thrusts = _find_thrusts(ema, lo, hi)
            thrust_dates = [hist_dates[i] for i in thrusts if pd.notna(hist_dates[i])]
            # Forward returns for each thrust
            per_lf = {}
            for lf in LOOKFORWARDS:
                rets = []
                rets_train = []
                rets_test = []
                for idx, d in zip(thrusts, thrust_dates):
                    try:
                        # Find vni position for date d
                        # vni index is Timestamp normalized
                        vni_pos = vni.index.get_indexer([pd.Timestamp(d.date())], method="nearest")[0]
                        # Check date close enough (within 5 days)
                        if abs((vni.index[vni_pos] - pd.Timestamp(d.date())).days) > 5:
                            continue
                        if vni_pos + lf >= len(vni):
                            continue
                        r = float(vni.iloc[vni_pos + lf] / vni.iloc[vni_pos] - 1)
                        if abs(r) >= 5:
                            continue
                        rets.append(r)
                        if pd.Timestamp(d.date()) < OOS_CUTOFF:
                            rets_train.append(r)
                        else:
                            rets_test.append(r)
                    except Exception:
                        continue
                # Filter min obs
                def _m(rets):
                    if len(rets) < MIN_OBS:
                        return {"n": len(rets), "hit2": None, "mean": None, "lift": None}
                    m = _metrics(rets)
                    base_hit2 = baseline[lf].get("hit2")
                    lift = round(m["hit2"] - base_hit2, 4) if m["hit2"] is not None and base_hit2 is not None else None
                    return {**m, "lift": lift}
                per_lf[str(lf)] = {"all": _m(rets), "train": _m(rets_train), "test": _m(rets_test)}

            grid.append({
                "lower": lo,
                "upper": hi,
                "n_thrusts": len(thrusts),
                "thrust_dates": [d.strftime("%d/%m/%Y") for d in thrust_dates[:10]],
                "per_forward": per_lf,
                "is_classic": lo == 0.40 and hi == 0.615,
            })

    # Sort grid by T+20 test lift desc where available
    def _key(g):
        v = g["per_forward"].get("20", {}).get("test", {}).get("lift")
        return v if v is not None else -99
    grid_sorted = sorted(grid, key=_key, reverse=True)
    best = grid_sorted[0] if grid_sorted else None

    payload = {
        "generated_at": vn_now().isoformat(),
        "method": "zweig_ema10_grid",
        "history_entries": len(history),
        "vni_points": len(vni),
        "market": "HOSE",
        "baseline": baseline,
        "grid": grid,
        "best": {"lower": best["lower"], "upper": best["upper"], "n": best["n_thrusts"]} if best else None,
        "classic": next((g for g in grid if g["is_classic"]), None),
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    DOCS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DOCS_OUTPUT.write_bytes(OUTPUT_JSON.read_bytes())
    print(f"Da ghi: {OUTPUT_JSON} grid={len(grid)} best={best['lower']}/{best['upper']} n={best['n_thrusts']}" if best else "Da ghi")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
