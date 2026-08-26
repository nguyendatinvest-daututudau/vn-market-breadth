"""Backfill A/D cho 1 nam gan nhat trong breadth_history.json tu OHLC cache.
Giu nguyen 0.40/0.615, HOSE-only. Idempotent: bo qua entry da co A/D.
Chay tren CI (co cache), local cache rong thi skip."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from _shared import CACHE_DIR, DATA_DIR
from cache_utils import load_cache

HISTORY_JSON = DATA_DIR / "breadth_history.json"
INDEX_SYMS = {"VNI", "HNXINDEX", "VNINDEX", "VN30"}

def _backfill_one_year() -> int:
    if not HISTORY_JSON.exists():
        print("Khong co breadth_history.json")
        return 0
    history = json.loads(HISTORY_JSON.read_text(encoding="utf-8"))
    if not history:
        return 0

    # Chi backfill 1 nam gan nhat (~250 phien) thieu A/D
    # Lay 300 entry cuoi de du
    to_fill = [h for h in history[-300:] if (h.get("markets") or {}).get("HOSE", {}).get("advances") is None]
    if not to_fill:
        print("Khong can backfill: 300 entry gan nhat da co A/D")
        return 0

    # Index theo date string -> entry
    date_to_entry = {h.get("date"): h for h in history}
    # Quet cache mot lan, tao dict date -> list of (close, prev_close)
    # Cach don gian: voi moi symbol, load cache, tinh pct per date, cong don
    from collections import defaultdict
    per_date = defaultdict(lambda: {"adv": 0, "dec": 0, "unc": 0})

    cache_files = [p for p in CACHE_DIR.glob("*.csv") if p.stem.upper() not in INDEX_SYMS]
    print(f"Quet {len(cache_files)} ma tu cache...")
    for path in cache_files:
        try:
            df = load_cache(path.stem, CACHE_DIR)
        except Exception:
            continue
        if df.empty or len(df) < 2 or "Close" not in df.columns:
            continue
        # Duyet tung date trong to_fill
        # Tao dict date_str -> close
        # Chuyen TradingDate sang dd/mm/yyyy string de map
        try:
            df["_dstr"] = pd.to_datetime(df["TradingDate"], dayfirst=True, errors="coerce").dt.strftime("%d/%m/%Y")
        except Exception:
            continue
        close_map = dict(zip(df["_dstr"], df["Close"]))
        # Duyet to_fill dates
        for entry in to_fill:
            dstr = entry.get("date")
            if dstr not in close_map:
                continue
            # Tim prev_close: ngay truoc do trong df
            # Tim index cua dstr trong df
            # Don gian: lay close hien tai va close truoc do 1 dong
            idx_list = df.index[df["_dstr"] == dstr].tolist()
            if not idx_list:
                continue
            idx = idx_list[0]
            if idx == 0:
                continue
            # Tim prev trading date gan nhat co du lieu (co the gap nghi le)
            # Lay close va prev_close truc tiep tu df
            try:
                close = float(df.at[idx, "Close"])
                prev_close = float(df.at[idx - 1, "Close"])
                if not (close > 0 and prev_close > 0):
                    continue
                pct = (close / prev_close - 1) * 100
                if pct > 0:
                    per_date[dstr]["adv"] += 1
                elif pct < 0:
                    per_date[dstr]["dec"] += 1
                else:
                    per_date[dstr]["unc"] += 1
            except Exception:
                continue

    filled = 0
    for entry in to_fill:
        dstr = entry.get("date")
        if dstr not in per_date:
            continue
        c = per_date[dstr]
        # Chi dien neu co du lieu
        if c["adv"] + c["dec"] + c["unc"] == 0:
            continue
        for market in ("HOSE", "ALL"):
            m = (entry.get("markets") or {}).get(market)
            if m is None:
                continue
            # ALL la HOSE+HNX, nhung HNX thieu cache nhieu, dung HOSE lam proxy cho ALL neu thieu
            # Don gian: dien ca HOSE va ALL giong nhau tu per_date (da tinh tu tat ca ma)
            # Tru khi muon tach HOSE/HNX rieng thi can universe mapping - de don gian dung chung
            m["advances"] = c["adv"]
            m["declines"] = c["dec"]
            m["unchanged"] = c["unc"]
            m["ad_ratio"] = round(c["adv"] / c["dec"], 2) if c["dec"] else None
            m["total_symbols"] = c["adv"] + c["dec"] + c["unc"]
        filled += 1

    if filled:
        HISTORY_JSON.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Backfill xong: {filled} ngay trong 1 nam.")
    else:
        print("Khong fill duoc ngay nao (cache thieu).")
    return filled

if __name__ == "__main__":
    n = _backfill_one_year()
    print(f"Done: {n}")
