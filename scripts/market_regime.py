"""Trung tam danh gia thi truong — regime gauge + phan ky + breadth momentum.

Doc du lieu: data/breadth_latest.json, data/breadth_history.json, data/ohlc_cache.
Xuat:        data/market_regime.json (+ sync docs/data).

Cac thanh phan:
  A. Market Regime Gauge  : diem 0-100 tu A/D, %MA20/%MA50, index position,
                            RSI pulse, ty le KL tang/giam.
  B. Phan ky Gia vs Breadth: so dinh/day VNI voi %MA20 -> canh bao dao chieu.
  C. Breadth Momentum      : McClellan-style tu (%MA20 - 50).
  D. Zweig Breadth Thrust  : EMA10 cua HOSE Adv/(Adv+Dec), 0.40 -> 0.615.

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
ZWEIG_MARKET = "HOSE"
ZWEIG_EMA_SPAN = 10
ZWEIG_LOWER_THRESHOLD = 0.40
ZWEIG_UPPER_THRESHOLD = 0.615
ZWEIG_WINDOW_SESSIONS = 10

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


# --- Zweig Breadth Thrust ----------------------------------------------------

def zweig_ema10_series(history: list, market: str = ZWEIG_MARKET,
                       span: int = ZWEIG_EMA_SPAN) -> list[dict]:
    """Tinh EMA adjust=False tu HOSE Adv/(Adv+Dec), bo unchanged."""
    by_date: dict[datetime, dict] = {}
    for entry in history:
        date = parse_market_date(entry.get("date"))
        if date is not None:
            by_date[date] = entry

    alpha = 2.0 / (span + 1.0)
    ema = None
    valid_count = 0
    series = []
    for date in sorted(by_date):
        market_data = ((by_date[date].get("markets") or {}).get(market) or {})
        ratio = None
        try:
            advances = float(market_data.get("advances"))
            declines = float(market_data.get("declines"))
            if advances >= 0 and declines >= 0 and advances + declines > 0:
                ratio = advances / (advances + declines)
        except (TypeError, ValueError):
            ratio = None

        valid = ratio is not None
        if valid:
            ema = ratio if ema is None else alpha * ratio + (1.0 - alpha) * ema
            valid_count += 1
        visible_ema = ema if valid_count >= span else None
        series.append({
            "date": date.strftime(DATE_FMT),
            "ratio": round(ratio, 6) if ratio is not None else None,
            "ema10": round(visible_ema, 6) if visible_ema is not None else None,
            "valid": valid,
        })
    return series


def _zweig_state_machine(
    series: list[dict],
    lower_threshold: float = ZWEIG_LOWER_THRESHOLD,
    upper_threshold: float = ZWEIG_UPPER_THRESHOLD,
    window_sessions: int = ZWEIG_WINDOW_SESSIONS,
) -> dict:
    """Theo doi setup; ngay arm la 0, xac nhan o ngay thu 10 van hop le."""
    setup = None
    terminal = None
    events = []
    last_row = None

    for pos, row in enumerate(series):
        last_row = row
        ema = row.get("ema10")
        valid_ema = bool(row.get("valid")) and ema is not None

        if setup is not None and pos - setup["position"] > window_sessions:
            terminal = {
                "state": "expired",
                "date": row.get("date"),
                "armed_date": setup["date"],
                "position": pos,
            }
            events.append({"date": row.get("date"), "kind": "expired", "ema10": ema})
            setup = None

        if valid_ema and ema < lower_threshold:
            if setup is None:
                events.append({"date": row.get("date"), "kind": "armed", "ema10": ema})
            setup = {"date": row.get("date"), "position": pos}
            terminal = None
            continue

        if setup is not None and valid_ema and ema > upper_threshold:
            age = pos - setup["position"]
            if age <= window_sessions:
                terminal = {
                    "state": "confirmed",
                    "date": row.get("date"),
                    "armed_date": setup["date"],
                    "position": pos,
                    "sessions_since_armed": age,
                }
                events.append({"date": row.get("date"), "kind": "confirmed", "ema10": ema})
                setup = None

    if not series or not any(row.get("ema10") is not None for row in series):
        return {
            "available": False,
            "state": "unavailable",
            "events": events,
        }

    current_ema = last_row.get("ema10") if last_row else None
    result = {
        "available": True,
        "current_observation_available": bool(last_row and last_row.get("valid")),
        "state": "inactive",
        "date": last_row.get("date") if last_row else None,
        "ratio": last_row.get("ratio") if last_row else None,
        "ema10": current_ema,
        "armed_date": None,
        "confirmed_date": None,
        "sessions_since_armed": None,
        "sessions_remaining": None,
        "progress_pct": None,
        "window_sessions": window_sessions,
        "thresholds": {"lower": lower_threshold, "upper": upper_threshold},
        "events": events,
    }
    if setup is not None:
        age = len(series) - 1 - setup["position"]
        result.update({
            "state": "armed" if current_ema is not None and current_ema < lower_threshold else "forming",
            "armed_date": setup["date"],
            "sessions_since_armed": age,
            "sessions_remaining": max(0, window_sessions - age),
            "progress_pct": round(_clamp(
                ((current_ema - lower_threshold) / (upper_threshold - lower_threshold)) * 100.0
            ), 1) if current_ema is not None else None,
        })
    elif terminal is not None:
        result["state"] = terminal["state"]
        result["armed_date"] = terminal.get("armed_date")
        if terminal["state"] == "confirmed":
            result["confirmed_date"] = terminal["date"]
            result["sessions_since_armed"] = terminal.get("sessions_since_armed")
            result["signal_age_sessions"] = len(series) - 1 - terminal["position"]
            result["progress_pct"] = 100.0
    return result


def zweig_breadth_thrust(history: list, market: str = ZWEIG_MARKET) -> dict:
    series = zweig_ema10_series(history, market=market)
    result = _zweig_state_machine(series)
    result.update({
        "market": market,
        "ema_period": ZWEIG_EMA_SPAN,
        "formula": "EMA10[Advances/(Advances+Declines)]",
        "valid_sessions": sum(row.get("valid", False) for row in series),
    })
    return result


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
    zweig = zweig_breadth_thrust(history)

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
        "zweig_breadth_thrust": zweig,
        "index": indexes,
    }
    OUTPUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DATA_DIR / OUTPUT_JSON.name).write_bytes(OUTPUT_JSON.read_bytes())

    print(f"Regime: {regime['score']} - {regime['label']} ({regime['tone']}) | {date}")
    print(f"Divergence: {divergence['state']}")
    print(f"Momentum: osc={momentum.get('oscillator')} ({momentum.get('extreme')})")
    print(f"Zweig EMA10: {zweig.get('state')} ({zweig.get('ema10')})")
    print(f"Da ghi: {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
