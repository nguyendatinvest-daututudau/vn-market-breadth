"""Backfill HOSE advances/declines needed by Zweig Breadth Thrust.

Only dates already present in breadth_history.json are requested. Existing A/D
rows are preserved by default, so repeated CI runs only fetch missing sessions.
"""
from __future__ import annotations

import argparse
import json
import time

from _shared import DATA_DIR, DOCS_DATA_DIR, parse_market_date
from ssi_client import SSIClient

HISTORY_JSON = DATA_DIR / "breadth_history.json"
DOCS_HISTORY_JSON = DOCS_DATA_DIR / HISTORY_JSON.name


def _write_history(history: list[dict]) -> None:
    HISTORY_JSON.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_HISTORY_JSON.write_bytes(HISTORY_JSON.read_bytes())


def _has_ad(entry: dict) -> bool:
    hose = (entry.get("markets") or {}).get("HOSE") or {}
    try:
        return int(hose.get("advances")) + int(hose.get("declines")) > 0
    except (TypeError, ValueError):
        return False


def backfill(history: list[dict], client: SSIClient, refresh: bool = False,
             sleep_seconds: float = 0.05, checkpoint: int = 25) -> tuple[int, int]:
    """Mutate history with authoritative HOSE A/D rows; return (filled, failed)."""
    filled = failed = pending = 0
    history.sort(key=lambda entry: parse_market_date(entry.get("date")) or parse_market_date("01/01/1900"))
    for entry in history:
        date = entry.get("date")
        if not date or (not refresh and _has_ad(entry)):
            continue
        pending += 1
        rows = client.daily_index("VNINDEX", date, date)
        row = rows[-1] if rows else None
        if not row:
            failed += 1
            continue
        try:
            advances = int(row.get("Advances") or 0)
            declines = int(row.get("Declines") or 0)
            unchanged = int(row.get("Nochanges") or 0)
        except (TypeError, ValueError):
            failed += 1
            continue
        if advances + declines <= 0:
            failed += 1
            continue
        hose = (entry.setdefault("markets", {})).setdefault("HOSE", {})
        hose.update({
            "advances": advances,
            "declines": declines,
            "unchanged": unchanged,
            "ad_ratio": round(advances / declines, 2) if declines else None,
            "total_symbols": advances + declines + unchanged,
        })
        filled += 1
        if checkpoint > 0 and filled % checkpoint == 0:
            _write_history(history)
            print(f"Checkpoint: {filled}/{pending} phien da dien A/D")
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return filled, failed


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill HOSE A/D cho Zweig EMA10")
    parser.add_argument("--refresh", action="store_true", help="Lay lai ca cac phien da co A/D")
    parser.add_argument("--sleep", type=float, default=0.05, help="Thoi gian nghi giua cac phien")
    args = parser.parse_args()

    if not HISTORY_JSON.exists():
        print("Khong co breadth_history.json")
        return 1
    try:
        history = json.loads(HISTORY_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        print("breadth_history.json khong hop le")
        return 1
    if not isinstance(history, list) or not history:
        print("breadth_history.json rong")
        return 1

    filled, failed = backfill(history, SSIClient(), refresh=args.refresh, sleep_seconds=args.sleep)
    _write_history(history)
    print(f"Zweig A/D backfill: them {filled} phien, khong lay duoc {failed} phien")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
