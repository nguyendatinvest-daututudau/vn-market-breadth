"""Backfill OHLCV lich su cho chi so (VNINDEX, HNXINDEX) tu SSI v3.

Nguon: get_securities_summary_by_index_historical (close/open/high/low/volume).
Ghi vao data/ohlc_cache/{symbol}.csv dang dd/mm/yyyy giong cache cua pipeline.

Cach chay:
  python scripts/backfill_index.py                 # 3 nam, ca hai chi so
  python scripts/backfill_index.py --years 1
  python scripts/backfill_index.py --only VNINDEX
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from _shared import CACHE_DIR, DATE_FMT, vn_now
from ssi_client import SSIClient

INDEXES = {
    "VNINDEX": "VNI.csv",
    "HNXINDEX": "HNXINDEX.csv",
}


def fetch_index_history(client: SSIClient, index_id: str, start: datetime, end: datetime) -> list[dict]:
    """Lay OHLCV hang ngay cua chi so theo tung khoang nam (v3 chan page size)."""
    rows: list[dict] = []
    day = start
    md = client._md()
    while day <= end:
        chunk_end = min(day + timedelta(days=365), end)
        from_str = day.strftime("%Y/%m/%d")
        to_str = chunk_end.strftime("%Y/%m/%d")
        try:
            bars = client._call(
                lambda f=from_str, t=to_str: md.get_securities_summary_by_index_historical(index_id, f, t)
            )
            for b in bars:
                rows.append({
                    "TradingDate": pd.to_datetime(getattr(b, "trading_date", ""), errors="coerce"),
                    "Open": getattr(b, "open_price", None),
                    "High": getattr(b, "high_price", None),
                    "Low": getattr(b, "low_price", None),
                    "Close": getattr(b, "close_price", None),
                    "Volume": getattr(b, "total_match", None),
                })
            print(f"[{index_id}] {from_str} -> {to_str}: {len(bars) if bars else 0} phien")
        except Exception as exc:
            print(f"[{index_id}] WARN {from_str}->{to_str}: {exc}")
        day = chunk_end + timedelta(days=1)
        time.sleep(0.3)
    return rows


def merge_and_write(index_id: str, filename: str, rows: list[dict]) -> int:
    """Gop voi cache cu (de xu ly idempotent), ghi file dd/mm/yyyy."""
    path: Path = CACHE_DIR / filename
    new_df = pd.DataFrame(rows)
    if new_df.empty or "TradingDate" not in new_df.columns:
        print(f"[{index_id}] Khong co du lieu moi. Bo qua.")
        return 0
    new_df = new_df.dropna(subset=["TradingDate", "Close"]).sort_values("TradingDate")
    new_df = new_df.drop_duplicates(subset=["TradingDate"], keep="last")

    frames = []
    if path.exists():
        try:
            old = pd.read_csv(path)
            text = old["TradingDate"].astype(str).str.strip()
            old["TradingDate"] = pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
            missing = old["TradingDate"].isna()
            if missing.any():
                old.loc[missing, "TradingDate"] = pd.to_datetime(
                    text.loc[missing], format="%Y-%m-%d", errors="coerce"
                )
            old = old.dropna(subset=["TradingDate"]).drop_duplicates(subset=["TradingDate"], keep="last")
            frames.append(old)
        except Exception:
            pass
    # Du lieu moi phai nam sau cung de thang trong drop_duplicates(keep='last').
    frames.append(new_df)

    merged = pd.concat(frames).sort_values("TradingDate").drop_duplicates(subset=["TradingDate"], keep="last")
    merged["TradingDate"] = merged["TradingDate"].dt.strftime(DATE_FMT)
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in merged.columns:
            merged[col] = float("nan")
    merged = merged[["TradingDate", "Open", "High", "Low", "Close", "Volume"]]
    merged.to_csv(path, index=False, encoding="utf-8")
    print(f"[{index_id}] Da ghi {len(merged)} phien: {path}")
    return len(merged)


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill index OHLCV tu SSI v3")
    ap.add_argument("--years", type=int, default=3, help="So nam backfill (mac dinh 3)")
    ap.add_argument("--only", choices=sorted(INDEXES), default=None,
                    help="Chi backfill mot chi so")
    args = ap.parse_args()

    client = SSIClient()
    client._ensure_auth()
    today = vn_now().replace(tzinfo=None)
    start = today - timedelta(days=args.years * 365)

    for index_id, filename in INDEXES.items():
        if args.only and args.only != index_id:
            continue
        print(f"\n=== {index_id} ({filename}) ===")
        rows = fetch_index_history(client, index_id, start, today)
        merge_and_write(index_id, filename, rows)


if __name__ == "__main__":
    sys.exit(main())
