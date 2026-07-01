"""
Client gọn nhẹ gọi SSI FastConnect Data API (FCData v2).
Tài liệu: https://guide.ssi.com.vn/ssi-products/tieng-viet/fastconnect-data
"""
import os
import time
import requests

BASE_URL = "https://fc-data.ssi.com.vn/api/v2/Market"


class SSIClient:
    def __init__(self, consumer_id: str | None = None, consumer_secret: str | None = None):
        self.consumer_id = consumer_id or os.environ["SSI_CONSUMER_ID"]
        self.consumer_secret = consumer_secret or os.environ["SSI_CONSUMER_SECRET"]
        self._token = None

    # ---------------- Auth ----------------
    def _get_token(self) -> str:
        if self._token:
            return self._token
        resp = requests.post(
            f"{BASE_URL}/AccessToken",
            json={"consumerID": self.consumer_id, "consumerSecret": self.consumer_secret},
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") not in (200, "Success", "success"):
            raise RuntimeError(f"SSI AccessToken failed: {payload}")
        self._token = payload["data"]["accessToken"]
        return self._token

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._get_token()}"}

    def _get(self, path: str, params: dict, retries: int = 3) -> dict:
        for attempt in range(retries):
            resp = requests.get(f"{BASE_URL}/{path}", params=params, headers=self._headers(), timeout=20)
            if resp.status_code == 401 and attempt == 0:
                # Token expired mid-run -> refresh once
                self._token = None
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"SSI GET {path} failed after {retries} retries")

    # ---------------- Endpoints ----------------
    def securities(self, market: str, page_size: int = 1000) -> list[dict]:
        """Danh sách mã theo sàn: HOSE | HNX | UPCOM"""
        out, page = [], 1
        while True:
            data = self._get("Securities", {"market": market, "pageIndex": page, "pageSize": page_size})
            rows = data.get("data", [])
            out.extend(rows)
            if len(rows) < page_size:
                break
            page += 1
            if page > 10:  # SSI giới hạn pageIndex 1..10
                break
        return out

    def daily_ohlc(self, symbol: str, from_date: str, to_date: str, page_size: int = 1000) -> list[dict]:
        """OHLCV theo ngày. Định dạng ngày: dd/mm/yyyy"""
        data = self._get(
            "DailyOhlc",
            {
                "symbol": symbol,
                "fromDate": from_date,
                "toDate": to_date,
                "pageIndex": 1,
                "pageSize": page_size,
                "ascending": "true",
            },
        )
        return data.get("data", [])

    def index_list(self, exchange: str | None = None) -> list[dict]:
        params = {"pageIndex": 1, "pageSize": 100}
        if exchange:
            params["exchange"] = exchange
        return self._get("IndexList", params).get("data", [])

    def daily_index(self, index_id: str, from_date: str, to_date: str, page_size: int = 1000) -> list[dict]:
        """Advances/Declines/Nochanges/Ceilings/Floors theo chỉ số (VNINDEX, HNXIndex, UPCOMIndex...)"""
        data = self._get(
            "DailyIndex",
            {
                "indexId": index_id,
                "fromDate": from_date,
                "toDate": to_date,
                "pageIndex": 1,
                "pageSize": page_size,
                "ascending": "true",
            },
        )
        return data.get("data", [])
