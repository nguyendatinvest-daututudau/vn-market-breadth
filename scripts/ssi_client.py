"""SSI FastConnect v3 client — Market Data qua SDK chinh thuc `ssi-sdk`.

Thay the hoan toan client v2 (fc-data.ssi.com.vn/api/v2). OAuth2 voi
client_id/api_key/api_secret; Market Data khong can OTP. Token duoc cache ra
scripts/token_cache.json (hoac nhan tu env SSI_TOKEN_CACHE dang base64 cho CI)
de tranh authenticate lai moi lan chay va tan dung refresh token.
"""

from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

from ssi_sdk import Auth, Config, Data
from ssi_sdk.enums import Board
from ssi_sdk.exceptions import AuthenticationError
from ssi_sdk.models import Token
from ssi_sdk.services.market_data import MarketDataService

from _shared import DATE_FMT, parse_market_date

SCRIPT_DIR = Path(__file__).resolve().parent
TOKEN_CACHE_PATH = SCRIPT_DIR / "token_cache.json"

# Map index code v2 -> Board, de fallback sang board summary neu index code doi.
INDEX_TO_BOARD = {
    "VNINDEX": Board.HOSE,
    "HNXINDEX": Board.HNX,
    "HNX": Board.HNX,
}


def _to_v3_datetime(value: str, end_of_day: bool = False) -> str:
    """dd/mm/yyyy -> 'YYYY/MM/DD HH:MM:SS' (dinh dang SDK v3)."""
    dt = datetime.strptime(value, DATE_FMT)
    return dt.strftime("%Y/%m/%d %H:%M:%S" if not end_of_day else "%Y/%m/%d 23:59:59")


class SSIClient:
    """Wrapper dong bo cho SSI FastConnect v3 (chi Market Data).

    Giu nguyen interface cua client v2 de fetch_and_compute.py,
    backfill_history.py va test khong can doi.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._auth: Auth | None = None
        self._data: MarketDataService | None = None

    # --- Auth (token cache + refresh + authenticate) ----------------------

    @staticmethod
    def _load_token() -> Token | None:
        raw = os.environ.get("SSI_TOKEN_CACHE")
        if raw:
            try:
                decoded = base64.b64decode(raw).decode("utf-8")
                return Token.from_dict(json.loads(decoded))
            except Exception:
                pass
        try:
            if TOKEN_CACHE_PATH.exists():
                return Token.from_dict(json.loads(TOKEN_CACHE_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
        return None

    @staticmethod
    def _save_token(token: Token | None) -> None:
        if token is None:
            return
        try:
            TOKEN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            TOKEN_CACHE_PATH.write_text(
                json.dumps(token.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _ensure_auth(self) -> Auth:
        with self._lock:
            if self._auth is not None and not self._auth.is_token_expired:
                return self._auth

            api_key = os.environ.get("SSI_API_KEY")
            api_secret = os.environ.get("SSI_API_SECRET")
            if not api_key or not api_secret:
                raise RuntimeError(
                    "Thieu SSI_API_KEY / SSI_API_SECRET (FastConnect v3). "
                    "Xem .env.example va README."
                )

            config = Config(
                client_id=os.environ.get("SSI_CLIENT_ID", ""),
                api_key=api_key,
                api_secret=api_secret,
                private_key=os.environ.get("SSI_PRIVATE_KEY", ""),
                log_level=os.environ.get("SSI_LOG_LEVEL", "INFO"),
            )
            auth = Auth(config)

            token = self._load_token()
            if token is not None:
                try:
                    auth.token_manager.set_token(token)
                except Exception:
                    pass

            try:
                if auth.is_token_expired:
                    if auth.has_refresh_token:
                        try:
                            print("[AUTH] Refresh token...")
                            auth.refresh()
                        except Exception:
                            # Refresh token het han -> authenticate lai.
                            print("[AUTH] Refresh that bai, authenticate lai...")
                            auth.authenticate()
                    else:
                        print("[AUTH] authenticate() (Market Data khong can OTP)...")
                        auth.authenticate()
            except AuthenticationError as exc:
                self._auth = None
                raise RuntimeError(
                    "Khong xac thuc duoc FastConnect v3. Neu lan dau can OTP, "
                    "chay `python scripts/ssi_auth_bootstrap.py`, roi dua ket qua "
                    "base64 vao bien env / GitHub secret SSI_TOKEN_CACHE.\n"
                    f"({exc})"
                ) from exc

            self._save_token(auth.token)
            self._auth = auth
            # Token moi -> phai tao lai MarketDataService de RestClient dung
            # access token moi (SDK khong tu cap nhat header sau khi refresh).
            self._data = None
            return auth

    def _call(self, fn):
        """Goi ham SDK; neu gap 401 thi xac thuc lai va goi lai mot lan."""
        try:
            return fn()
        except AuthenticationError:
            self._force_reauth()
            return fn()

    def _force_reauth(self) -> Auth:
        with self._lock:
            auth = self._auth
            if auth is None:
                return self._ensure_auth()
            try:
                if auth.has_refresh_token:
                    print("[AUTH] Refresh token (retry)...")
                    auth.refresh()
                else:
                    print("[AUTH] authenticate() (retry)...")
                    auth.authenticate()
            except Exception:
                try:
                    print("[AUTH] authenticate() (retry)...")
                    auth.authenticate()
                except Exception:
                    self._auth = None
                    raise
            self._data = None
            self._save_token(auth.token)
            return auth

    def _md(self) -> MarketDataService:
        auth = self._ensure_auth()
        if self._data is None:
            self._data = Data(auth).market_data
        return self._data

    # --- Securities --------------------------------------------------------

    def common_stock_symbols(self, market: str) -> list[str]:
        """Lay danh sach ma co phieu thuong theo san (HOSE/HNX)."""
        rows = self._call(lambda: self._md().get_securities_info_by_board(Board(market.upper())))
        symbols = []
        skipped = 0
        for r in rows:
            symbol = str(getattr(r, "symbol", "") or "").strip().upper()
            if not symbol:
                continue
            if getattr(r, "cw_underlying_symbol", None):  # chung quyen
                skipped += 1
                continue
            if re.search(r"\d", symbol):
                skipped += 1
                continue
            if len(symbol) > 3:
                skipped += 1
                continue
            symbols.append(symbol)
        print(f"[{market}] Securities v3: {len(rows)} -> sau loc {len(symbols)} (bo {skipped})")
        return symbols

    # --- OHLC ---------------------------------------------------------------

    def daily_ohlc(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
        page_size: int = 1000,
        max_pages: int = 50,
    ) -> list[dict]:
        """OHLCV theo ngay. Dinh dang ngay: dd/mm/yyyy. Tu dong phan trang."""
        f = _to_v3_datetime(from_date)
        t = _to_v3_datetime(to_date, end_of_day=True)
        all_rows = []
        page = 1
        md = self._md()
        while True:
            bars = self._call(lambda: md.get_ohlc_1day_historical(symbol, f, t, page=page, size=page_size))
            all_rows.extend(self._ohlc_to_dicts(bars))
            if len(bars) < page_size or page >= max_pages:
                break
            page += 1
        return all_rows

    @staticmethod
    def _ohlc_to_dicts(bars) -> list[dict]:
        out = []
        for b in bars:
            out.append({
                "TradingDate": getattr(b, "trading_date", ""),
                "Open": getattr(b, "open_price", None),
                "High": getattr(b, "high_price", None),
                "Low": getattr(b, "low_price", None),
                "Close": getattr(b, "close_price", None),
                "Volume": getattr(b, "volume", None),
            })
        return out

    # --- A/D theo chi so ----------------------------------------------------

    def daily_index(
        self,
        index_id: str,
        from_date: str,
        to_date: str,
        page_size: int = 100,
    ) -> list[dict]:
        """Advances/Declines/Nochanges theo chi so, lich su tung ngay giao dich.

        Tra ve list dict {TradingDate, Advances, Declines, Nochanges} giong
        client v2 de fetch_and_compute.get_advance_decline khong phai doi.
        Neu index summary khong co du lieu thi fallback sang board summary.
        """
        from_dt = parse_market_date(from_date)
        to_dt = parse_market_date(to_date)
        if from_dt is None or to_dt is None:
            return []
        board = INDEX_TO_BOARD.get(str(index_id).upper())
        md = self._md()
        rows = []
        day = from_dt
        while day <= to_dt:
            dstr = day.strftime("%Y/%m/%d")
            summary = None
            for attempt in range(3):
                try:
                    summary = self._call(lambda: md.get_index_summary_historical(index_id, dstr))
                except Exception:
                    summary = None
                if summary is not None:
                    break
                time.sleep(0.5 * (attempt + 1))
            if summary is None and board is not None:
                for attempt in range(3):
                    try:
                        summary = self._call(lambda: md.get_board_summary_historical(board, dstr))
                    except Exception:
                        summary = None
                    if summary is not None:
                        break
                    time.sleep(0.5 * (attempt + 1))
            if summary is not None:
                rows.append({
                    "TradingDate": getattr(summary, "trading_date", ""),
                    "Advances": getattr(summary, "total_advance_stock", 0) or 0,
                    "Declines": getattr(summary, "total_decline_stock", 0) or 0,
                    "Nochanges": getattr(summary, "total_steady_stock", 0) or 0,
                })
            day += timedelta(days=1)
        return rows
