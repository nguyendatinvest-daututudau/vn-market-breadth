"""Backtest Zweig Classic EMA10 on HOSE breadth and VN-Index closes."""
from __future__ import annotations

import json
import statistics

from _shared import DATA_DIR, DOCS_DATA_DIR, parse_market_date, vn_now
from market_regime import zweig_breadth_thrust, zweig_ema10_series

HISTORY_JSON = DATA_DIR / "breadth_history.json"
OUTPUT_JSON = DATA_DIR / "backtest_zweig.json"
HORIZONS = (5, 10, 20, 60)


def _aligned_closes(history: list[dict]) -> list[tuple[str, float]]:
    rows = []
    for entry in history:
        date = entry.get("date")
        close = (entry.get("index_closes") or {}).get("VNI")
        if parse_market_date(date) is None or close is None:
            continue
        try:
            rows.append((date, float(close)))
        except (TypeError, ValueError):
            continue
    rows.sort(key=lambda row: parse_market_date(row[0]))
    dedup = {date: close for date, close in rows}
    return sorted(dedup.items(), key=lambda row: parse_market_date(row[0]))


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "avg_return_pct": None, "median_return_pct": None, "hit_rate_pct": None}
    return {
        "n": len(values),
        "avg_return_pct": round(statistics.fmean(values), 2),
        "median_return_pct": round(statistics.median(values), 2),
        "hit_rate_pct": round(sum(value > 0 for value in values) / len(values) * 100.0, 1),
    }


def evaluate_events(history: list[dict], event_dates: list[str]) -> dict:
    closes = _aligned_closes(history)
    positions = {date: i for i, (date, _close) in enumerate(closes)}
    event_set = set(event_dates)
    results = {}
    for horizon in HORIZONS:
        event_returns = []
        baseline_returns = []
        for i, (date, close) in enumerate(closes):
            if i + horizon >= len(closes) or close <= 0:
                continue
            forward = (closes[i + horizon][1] / close - 1.0) * 100.0
            baseline_returns.append(forward)
            if date in event_set:
                event_returns.append(forward)
        event_stats = _stats(event_returns)
        baseline_stats = _stats(baseline_returns)
        lift = None
        if event_stats["avg_return_pct"] is not None and baseline_stats["avg_return_pct"] is not None:
            lift = round(event_stats["avg_return_pct"] - baseline_stats["avg_return_pct"], 2)
        results[f"T+{horizon}"] = {
            "events": event_stats,
            "baseline": baseline_stats,
            "avg_lift_pct_points": lift,
        }
    return {
        "index_sessions": len(closes),
        "matched_events": sum(date in positions for date in event_set),
        "horizons": results,
    }


def build_report(history: list[dict]) -> dict:
    zweig = zweig_breadth_thrust(history)
    confirmed = [event for event in zweig.get("events", []) if event.get("kind") == "confirmed"]
    ema_series = zweig_ema10_series(history)
    return {
        "generated_at": vn_now().isoformat(),
        "method": "Zweig Classic EMA10",
        "formula": "EMA10[HOSE Advances/(Advances+Declines)]",
        "thresholds": {"lower": 0.40, "upper": 0.615, "window_sessions": 10},
        "data_quality": {
            "history_entries": len(history),
            "valid_ad_sessions": sum(row.get("valid", False) for row in ema_series),
            "ema_sessions": sum(row.get("ema10") is not None for row in ema_series),
        },
        "confirmed_events": confirmed,
        "event_count": len(confirmed),
        "performance": evaluate_events(history, [event["date"] for event in confirmed]),
    }


def main() -> int:
    try:
        history = json.loads(HISTORY_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        print("Khong doc duoc breadth_history.json")
        return 1
    report = build_report(history)
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DATA_DIR / OUTPUT_JSON.name).write_bytes(OUTPUT_JSON.read_bytes())
    print(f"Zweig backtest: {report['event_count']} events, {report['performance']['index_sessions']} index sessions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
