"""Trung tam danh gia thi truong — regime gauge + phan ky + breadth momentum.

Doc du lieu: data/breadth_latest.json, data/breadth_history.json, data/ohlc_cache.
Xuat:        data/market_regime.json (+ sync docs/data).

Cac thanh phan:
  A. Market Regime Gauge  : diem 0-100 tu A/D, %MA20/%MA50, index position,
                            RSI pulse, ty le KL tang/giam.
  B. Phan ky Gia vs Breadth: so dinh/day VNI voi %MA20 -> canh bao dao chieu.
  C. Breadth Momentum      : McClellan-style tu (%MA20 - 50).

Cach chay:
  python scripts/market_regime.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from _shared import CACHE_DIR, DATA_DIR, DOCS_DATA_DIR, DATE_FMT, parse_market_date, vn_now
from cache_utils import load_cache

LATEST_JSON = DATA_DIR / "breadth_latest.json"
HISTORY_JSON = DATA_DIR / "breadth_history.json"
OUTPUT_JSON = DATA_DIR / "market_regime.json"

INDEX_CACHE = {"VNI": "VNI.csv", "HNXINDEX": "HNXINDEX.csv"}
MARKET = "ALL"

# Trong so cac thanh phan regime (tong = 1.0)
WEIGHTS = {
    "ad_ratio": 0.25,
    "pct_above_ma20": 0.25,
    "pct_above_ma50": 0.15,
    "index_position": 0.20,
    "rsi_pulse": 0.10,
    "volume_ud": 0.05,
}


# --- Load --------------------------------------------------------------------

def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_latest() -> dict:
    return _load_json(LATEST_JSON) or {}


def load_history() -> list:
    return _load_json(HISTORY_JSON) or []


def load_index_frame(name: str) -> pd.DataFrame:
    return load_cache(name, CACHE_DIR)


# --- Score mapping (thuan, de test) ------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _piecewise(x: float, points: list[tuple[float, float]]) -> float:
    """Noi suy tuyen tinh giua cac diem (x, y) da sap xep."""
    if x is None:
        return 50.0
    pts = sorted(points)
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y1
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]


def score_ad_ratio(ad) -> float:
    return _piecewise(ad, [
        (0.5, 10.0), (1.0, 50.0), (1.5, 70.0), (2.5, 90.0), (4.0, 95.0),
    ])


def score_pct_above(pct) -> float:
    return _piecewise(pct, [
        (10.0, 10.0), (30.0, 50.0), (50.0, 70.0), (80.0, 85.0),
    ])


def score_index_state(idx: dict | None) -> float:
    if not idx:
        return 50.0
    score = 50.0
    if idx.get("above_ma20"):
        score += 12
    else:
        score -= 12
    if idx.get("above_ma50"):
        score += 10
    else:
        score -= 10
    if idx.get("above_ma200"):
        score += 8
    else:
        score -= 8
    if idx.get("macd_up"):
        score += 10
    else:
        score -= 10
    rsi = idx.get("rsi")
    if rsi is not None:
        if rsi < 30:
            score -= 15
        elif rsi < 45:
            score -= 8
        elif rsi < 55:
            score += 0
        elif rsi < 70:
            score += 8
        else:
            score += 5
    return _clamp(score)


def score_rsi_pulse(pulse: dict | None) -> float:
    if not pulse:
        return 50.0
    total = float(pulse.get("total") or 0)
    if total <= 0:
        return 50.0
    net = ((pulse.get("over_70") or 0) - (pulse.get("under_30") or 0)) / total
    return _clamp(50.0 + 60.0 * net)


def score_volume_ud(ratio) -> float:
    return _piecewise(ratio, [
        (0.4, 15.0), (0.8, 40.0), (1.2, 60.0), (2.0, 80.0), (3.0, 95.0),
    ])


def compose_regime(components: dict) -> dict:
    score = 0.0
    detail = {}
    for key, weight in WEIGHTS.items():
        value = components.get(key, {}).get("value")
        points = components.get(key, {}).get("points")
        if points is None:
            points = 50.0
        score += weight * points
        detail[key] = {"value": value, "points": round(float(points), 1)}
    score = round(_clamp(score), 1)
    if score < 30:
        label, tone = "Risk-Off", "risk_off"
    elif score < 55:
        label, tone = "Trung lập", "neutral"
    elif score < 80:
        label, tone = "Risk-On", "risk_on"
    else:
        label, tone = "Quá nóng", "overheated"
    return {"score": score, "label": label, "tone": tone,
            "components": detail, "weights": WEIGHTS}


# --- Index technical ---------------------------------------------------------

def index_technical(frame: pd.DataFrame) -> dict | None:
    if frame is None or len(frame) < 60:
        return None
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if len(close) < 60:
        return None
    last = float(close.iloc[-1])
    out = {"close": round(last, 2)}
    for window in (20, 50, 200):
        if len(close) >= window:
            ma = float(close.rolling(window).mean().iloc[-1])
            out[f"ma{window}"] = round(ma, 2)
            out[f"above_ma{window}"] = bool(last >= ma)
        else:
            out[f"above_ma{window}"] = None
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    out["rsi"] = round(float(rsi.iloc[-1]), 1)
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = float((macd_line - signal).iloc[-1])
    out["macd_hist"] = round(hist, 2)
    out["macd_up"] = bool(hist > 0)
    return out


# --- Volume up/down ----------------------------------------------------------

def volume_updown_ratio(cache_dir: Path = CACHE_DIR) -> float | None:
    """Ty le tong KL cac ma tang / tong KL cac ma giam (2 phien cuoi cung)."""
    up_vol = down_vol = 0.0
    for path in cache_dir.glob("*.csv"):
        sym = path.stem
        if sym in ("VNI", "HNXINDEX", "VNINDEX", "HNX") or not sym.isalpha():
            continue
        try:
            df = pd.read_csv(path)
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            if "Volume" in df.columns:
                df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
            else:
                df["Volume"] = float("nan")
            df = df.dropna(subset=["Close"])
            if len(df) < 2:
                continue
            prev = float(df["Close"].iloc[-2])
            last = float(df["Close"].iloc[-1])
            vol = float(df["Volume"].iloc[-1]) or 0.0
            if prev <= 0:
                continue
            if last > prev:
                up_vol += vol
            elif last < prev:
                down_vol += vol
        except Exception:
            continue
    if down_vol <= 0:
        return None
    return round(up_vol / down_vol, 3)


# --- Phan ky Gia vs Breadth --------------------------------------------------

def _series_aligned(history: list, index_frame: pd.DataFrame):
    """Map ngay -> pct_above_ma20 (ALL) va ngay -> close index."""
    pct_map: dict[datetime, float] = {}
    for entry in history:
        markets = entry.get("markets") or {}
        all_m = markets.get(MARKET) or {}
        pct = all_m.get("pct_above_ma20")
        date = parse_market_date(entry.get("date"))
        if date is not None and pct is not None:
            pct_map[date] = float(pct)
    close_map: dict[datetime, float] = {}
    if index_frame is not None and "TradingDate" in index_frame.columns:
        for _, row in index_frame.dropna(subset=["Close"]).iterrows():
            close_map[row["TradingDate"].to_pydatetime()] = float(row["Close"])
    common = sorted(set(pct_map) & set(close_map))
    return common, pct_map, close_map


def detect_divergence(history: list, index_frame: pd.DataFrame,
                      lookback: int = 20, px_tol: float = 0.02,
                      bh_tol: float = 8.0) -> dict:
    """So dinh/day index voi dinh/day %MA20. State: bearish | bullish | none."""
    common, pct_map, close_map = _series_aligned(history, index_frame)
    if len(common) < lookback + 1:
        return {"state": "none", "note": "Chưa đủ dữ liệu lịch sử.",
                "detail": {"available_sessions": len(common)}}
    window = common[-(lookback + 1):]
    closes = [close_map[d] for d in window]
    pcts = [pct_map[d] for d in window]
    last_close = closes[-1]
    last_pct = pcts[-1]
    price_high, price_low = max(closes), min(closes)
    pct_high, pct_low = max(pcts), min(pcts)

    state = "none"
    if last_close >= price_high * (1 - px_tol) and last_pct <= pct_high - bh_tol:
        state = "bearish"
    elif last_close <= price_low * (1 + px_tol) and last_pct >= pct_low + bh_tol:
        state = "bullish"

    if state == "bearish":
        note = ("Giá ở vùng đỉnh nhưng độ rộng (% trên MA20) không xác nhận — "
                "rủi ro đảo chiều giảm.")
    elif state == "bullish":
        note = ("Giá ở vùng đáy nhưng độ rộng đã cải thiện sớm — "
                "dấu hiệu tích lũy, có thể đảo chiều tăng.")
    else:
        note = "Không có phân kỳ rõ rệt giữa giá và độ rộng."

    return {
        "state": state,
        "note": note,
        "detail": {
            "sessions": len(window),
            "last_close": round(last_close, 2),
            "price_high": round(price_high, 2),
            "price_low": round(price_low, 2),
            "last_pct_above_ma20": round(last_pct, 1),
            "pct_high": round(pct_high, 1),
            "pct_low": round(pct_low, 1),
        },
    }


# --- Breadth momentum (McClellan-style) --------------------------------------

def breadth_momentum(history: list, lookback: int = 500) -> dict:
    """EMA19 - EMA39 cua (%MA20 - 50)."""
    series = []
    for entry in history[-lookback:]:
        all_m = (entry.get("markets") or {}).get(MARKET) or {}
        pct = all_m.get("pct_above_ma20")
        if pct is not None:
            series.append(float(pct))
    if len(series) < 60:
        return {"available": False}
    s = pd.Series(series) - 50.0
    ema19 = s.ewm(span=19, adjust=False).mean()
    ema39 = s.ewm(span=39, adjust=False).mean()
    osc = ema19 - ema39
    signal = osc.ewm(span=9, adjust=False).mean()
    hist = osc - signal
    o = float(osc.iloc[-1])
    si = float(signal.iloc[-1])
    hi = float(hist.iloc[-1])
    if o >= 10:
        extreme = "overbought"
    elif o <= -10:
        extreme = "oversold"
    else:
        extreme = "none"
    return {
        "available": True,
        "oscillator": round(o, 2),
        "signal": round(si, 2),
        "histogram": round(hi, 2),
        "extreme": extreme,
        "last_pct_above_ma20": round(series[-1], 1),
    }


def compute_zweig(history: list, market: str = "HOSE", window: int = 10) -> dict:
    """Zweig Breadth Thrust EMA10 HOSE-only. lower 0.40 -> upper 0.615 trong 10 ngay."""
    ratios: list[float | None] = []
    dates: list[str | None] = []
    for entry in history:
        m = (entry.get("markets") or {}).get(market) or {}
        a, d = m.get("advances"), m.get("declines")
        if a is not None and d is not None and (a + d) > 0:
            ratios.append(float(a) / float(a + d))
        else:
            ratios.append(None)
        dates.append(entry.get("date"))
    # EMA10
    s = pd.Series(ratios)
    ema = s.ewm(span=window, adjust=False, min_periods=window).mean()
    # Tim thrust gan nhat
    active = False
    thrust_date = None
    score = float(ema.iloc[-1]) if len(ema) and pd.notna(ema.iloc[-1]) else None
    prior_low = None
    thrust_size = None
    if len(ema) >= window + 10:
        # Duyet tu cuoi ve truoc tim lan thrust gan nhat trong 30 ngay
        for i in range(len(ema) - 1, max(-1, len(ema) - 30), -1):
            if pd.isna(ema.iloc[i]) or pd.isna(ema.iloc[max(0, i - 10)]):
                continue
            window_ema = ema.iloc[max(0, i - 10):i]
            if len(window_ema) < 10 or window_ema.isna().any():
                continue
            low = float(window_ema.min())
            cur = float(ema.iloc[i])
            if low < 0.40 and cur > 0.615:
                active = (i == len(ema) - 1) or (len(ema) - 1 - i) <= 10  # kich hoat trong 10 phien gan nhat
                # Chi lay thrust gan nhat
                if i == len(ema) - 1 or active:
                    thrust_date = dates[i]
                    prior_low = round(low, 3)
                    thrust_size = round(cur - low, 3)
                    score = round(cur, 3)
                    break
        # Neu khong co thrust gan nhat, lay score hien tai
        if score is not None:
            score = round(score, 3)
    return {
        "available": len([r for r in ratios if r is not None]) >= window,
        "active": active,
        "score": score,
        "date": thrust_date,
        "prior_low": prior_low,
        "thrust_size": thrust_size,
        "market": market,
        "window": window,
        "lower": 0.40,
        "upper": 0.615,
    }


# --- Build output ------------------------------------------------------------

def build_regime(latest: dict, history: list) -> dict:
    all_m = (latest.get("markets") or {}).get(MARKET) or {}
    components = {
        "ad_ratio": {"value": all_m.get("ad_ratio"),
                     "points": score_ad_ratio(all_m.get("ad_ratio"))},
        "pct_above_ma20": {"value": all_m.get("pct_above_ma20"),
                           "points": score_pct_above(all_m.get("pct_above_ma20"))},
        "pct_above_ma50": {"value": all_m.get("pct_above_ma50"),
                           "points": score_pct_above(all_m.get("pct_above_ma50"))},
        "rsi_pulse": {"value": all_m.get("rsi_pulse"),
                      "points": score_rsi_pulse(all_m.get("rsi_pulse"))},
    }
    vni = index_technical(load_index_frame("VNI"))
    components["index_position"] = {
        "value": vni, "points": score_index_state(vni),
    }
    ud = volume_updown_ratio()
    components["volume_ud"] = {"value": ud, "points": score_volume_ud(ud)}
    return compose_regime(components)


def main() -> int:
    latest = load_latest()
    history = load_history()
    if not latest or not history:
        print("Thieu breadth_latest.json / breadth_history.json. Chay pipeline truoc.")
        return 1

    date = (latest.get("markets") or {}).get(MARKET, {}).get("date", "")
    regime = build_regime(latest, history)
    divergence = detect_divergence(history, load_index_frame("VNI"))
    momentum = breadth_momentum(history)
    zweig = compute_zweig(history)

    indexes = {}
    for name in INDEX_CACHE:
        tech = index_technical(load_index_frame(name))
        if tech:
            indexes[name] = tech

    output = {
        "generated_at": vn_now().isoformat(),
        "date": date,
        "regime": regime,
        "divergence": divergence,
        "breadth_momentum": momentum,
        "zweig": zweig,
        "index": indexes,
    }
    OUTPUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DATA_DIR / OUTPUT_JSON.name).write_bytes(OUTPUT_JSON.read_bytes())

    print(f"Regime: {regime['score']} - {regime['label']} ({regime['tone']}) | {date}")
    print(f"Divergence: {divergence['state']}")
    print(f"Momentum: osc={momentum.get('oscillator')} ({momentum.get('extreme')})")
    print(f"Zweig: {'KICH HOAT ' + str(zweig.get('date')) if zweig.get('active') else 'khong kich hoat'} (score={zweig.get('score')})")
    print(f"Da ghi: {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
