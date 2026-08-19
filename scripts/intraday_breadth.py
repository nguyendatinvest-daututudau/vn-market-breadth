"""Intraday breadth monitor — realtime A/D cho HOSE/HNX tu SSI v3 Streaming.

Nguon: WebSocket streaming (trade + OHLCV 1 phut + quote) cho toan bo universe.
Gia tham chieu (ref): close phien gan nhat trong data/ohlc_cache.
Ghi data/intraday_breadth.json (va docs/data) moi INTERVAL giay.

Cach chay:
  python scripts/intraday_breadth.py --once 90      # lay mau 90s roi thoat
  python scripts/intraday_breadth.py --watch        # chay lien tuc den Ctrl+C
  python scripts/intraday_breadth.py --interval 10  # ghi moi 10 giay (mac dinh)
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from _shared import CACHE_DIR, DATA_DIR, DOCS_DATA_DIR, DATE_FMT, vn_now
from ssi_client import SSIClient
from ssi_sdk import Stream
from ssi_sdk.enums import Timeframe

UNIVERSE_JSON = DATA_DIR / "symbol_universes.json"
OUTPUT_JSON = DATA_DIR / "intraday_breadth.json"
INDEX_CODES = ("VNINDEX", "HNXINDEX")
INDEX_TO_MARKET = {"VNINDEX": "HOSE", "HNXINDEX": "HNX"}
MARKETS = ("HOSE", "HNX")
BATCH = 100


# --- Ref price tu cache ------------------------------------------------------

def load_ref_prices() -> dict[str, float]:
    """close phien gan nhat cho tung ma trong universe (gia tham chieu hom nay)."""
    refs: dict[str, float] = {}
    for path in CACHE_DIR.glob("*.csv"):
        sym = path.stem.upper()
        if not sym.isalpha() or len(sym) > 3:
            continue
        try:
            dates, closes = [], []
            with open(path, encoding="utf-8") as fh:
                header = fh.readline().strip().split(",")
                try:
                    di = header.index("TradingDate")
                    ci = header.index("Close")
                except ValueError:
                    continue
                for line in fh:
                    parts = line.rstrip("\n").split(",")
                    if len(parts) <= max(di, ci):
                        continue
                    dates.append(parts[di])
                    closes.append(parts[ci])
            last_idx = len(dates) - 1
            if last_idx < 0:
                continue
            close = float(closes[last_idx])
            if close > 0:
                refs[sym] = close
        except Exception:
            continue
    return refs


# --- Universe ----------------------------------------------------------------

def load_universe() -> dict[str, list[str]]:
    if not UNIVERSE_JSON.exists():
        return {}
    try:
        payload = json.loads(UNIVERSE_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for market in MARKETS:
        syms = payload.get("exchanges", {}).get(market, {}).get("symbols", [])
        out[market] = sorted({s.upper() for s in syms if str(s).isalpha() and len(str(s)) <= 3})
    return out


# --- Breadth tracker ---------------------------------------------------------

class BreadthTracker:
    def __init__(self, universe: dict[str, list[str]], refs: dict[str, float]):
        self.universe = universe
        self.refs = refs
        self.prices: dict[str, float] = {}
        self.trade_count = 0

    def on_message(self, msg) -> None:
        sym = getattr(msg, "symbol", None)
        if not sym or sym in INDEX_CODES:
            return
        mtype = type(msg).__name__
        if mtype == "QuoteMessage":
            # Mid-price lam backup (neu chua co gia khop tu trade/ohlcv).
            if sym not in self.prices:
                bid = msg.bid_prices[0] if msg.bid_prices else 0
                ask = msg.ask_prices[0] if msg.ask_prices else 0
                if bid > 0 and ask > 0:
                    self.prices[sym] = round((bid + ask) / 2, 0)
            return
        price = getattr(msg, "price", None)
        if price is None:
            price = getattr(msg, "close", None)
        if price:
            self.prices[sym] = float(price)
            self.trade_count += 1

    def snapshot(self) -> dict:
        markets = {}
        for market in MARKETS:
            adv = dec = unc = no_data = ref_missing = 0
            for sym in self.universe[market]:
                ref = self.refs.get(sym)
                if ref is None:
                    ref_missing += 1
                    continue
                px = self.prices.get(sym)
                if px is None:
                    no_data += 1
                    continue
                if px > ref:
                    adv += 1
                elif px < ref:
                    dec += 1
                else:
                    unc += 1
            total = adv + dec + unc
            markets[market] = {
                "total_universe": len(self.universe[market]),
                "advances": adv,
                "declines": dec,
                "unchanged": unc,
                "no_data": no_data,
                "ref_missing": ref_missing,
                "tracked": total,
                "coverage": round(total / len(self.universe[market]), 3) if self.universe[market] else 0.0,
                "ad_ratio": round(adv / dec, 2) if dec else None,
                "advances_pct": round(adv / total * 100, 1) if total else 0.0,
            }
        return {
            "updated_at": vn_now().isoformat(timespec="seconds"),
            "session_date": vn_now().strftime(DATE_FMT),
            "source": "ssi_v3_streaming",
            "trade_messages": self.trade_count,
            "markets": markets,
        }

    def print_table(self) -> None:
        snap = self.snapshot()
        print(f"\n[{snap['session_date']} {snap['updated_at'][11:19]}] realtime A/D")
        for market in MARKETS:
            m = snap["markets"][market]
            adv, dec, unc = m["advances"], m["declines"], m["unchanged"]
            print(
                f"  {market:<5} ADV {adv:>3} | DEC {dec:>3} | UNC {unc:>3} | "
                f"no_data {m['no_data']:>3} | AD {m['ad_ratio']} | "
                f"AD% {m['advances_pct']}% | cover {m['coverage']:.0%}"
            )


def write_json(payload: dict) -> None:
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DATA_DIR / OUTPUT_JSON.name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# --- Main --------------------------------------------------------------------

def _subscribe_all(stream, universe: dict[str, list[str]]) -> None:
    def sub_batch(syms, fn) -> None:
        for i in range(0, len(syms), BATCH):
            fn(list(syms[i:i + BATCH]))
            time.sleep(0.2)

    all_syms = [s for market in MARKETS for s in universe[market]]
    sub_batch(all_syms, stream.streaming.subscribe_symbol_trade)
    sub_batch(all_syms, lambda s: stream.streaming.subscribe_symbol_ohlcv(s, Timeframe.MINUTE_1))
    sub_batch(all_syms, stream.streaming.subscribe_symbol_quote)
    print(f"Subscribed {len(all_syms)} symbols (trade + ohlcv 1m + quote).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Intraday breadth monitor (SSI v3 streaming)")
    ap.add_argument("--once", type=float, default=None, metavar="SECONDS",
                    help="Lay mau trong SECONDS roi thoat (mac dinh: chay lien tuc)")
    ap.add_argument("--interval", type=float, default=10.0,
                    help="Ghi file moi interval giay (mac dinh 10)")
    ap.add_argument("--markets", default="HOSE,HNX",
                    help="Danh sach san phan cach bang phay (mac dinh HOSE,HNX)")
    args = ap.parse_args()

    global MARKETS
    MARKETS = tuple(m.strip().upper() for m in args.markets.split(",") if m.strip())

    universe = load_universe()
    if not universe:
        print("Khong tai duoc universe tu data/symbol_universes.json. Chay pipeline truoc.")
        return
    refs = load_ref_prices()
    print(f"Universe: " + ", ".join(f"{m}={len(universe[m])}" for m in MARKETS))
    print(f"Ref prices loaded: {len(refs)}")

    client = SSIClient()
    auth = client._ensure_auth()
    stream = Stream(auth)
    tracker = BreadthTracker(universe, refs)
    stream.streaming.on_data = tracker.on_message
    stream.streaming.connect()
    _subscribe_all(stream, universe)

    last_write = 0.0
    start = time.time()
    deadline = start + args.once if args.once else None
    try:
        while True:
            stream.streaming.wait(timeout=min(args.interval, 5))
            now = time.time()
            if now - last_write >= args.interval:
                snap = tracker.snapshot()
                write_json(snap)
                tracker.print_table()
                last_write = now
            if deadline and now >= deadline:
                break
    except KeyboardInterrupt:
        print("\nDung lai (Ctrl+C).")
    finally:
        stream.streaming.disconnect()
        snap = tracker.snapshot()
        write_json(snap)
        print(f"Da ghi: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()