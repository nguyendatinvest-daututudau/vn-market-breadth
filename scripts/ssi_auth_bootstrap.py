"""Bootstrap xac thuc FastConnect v3 mot lan (chay local).

Luong:
  1. Doc client_id/api_key/api_secret tu env (SSI_CLIENT_ID, SSI_API_KEY, SSI_API_SECRET).
  2. Thu authenticate() khong OTP (Market Data khong can OTP).
  3. Neu server yeu cau OTP: goi request_otp() roi nhan mat ma tu user
     (SMS/email hoac Smart OTP tren app iBoard) de authenticate.
  4. Luu token xuong scripts/token_cache.json va in ra base64 de dat vao
     env / GitHub secret `SSI_TOKEN_CACHE` (dung cho CI khong can nhap OTP lai).

Cach chay:
    $env:SSI_CLIENT_ID="..."
    $env:SSI_API_KEY="..."
    $env:SSI_API_SECRET="..."
    python scripts/ssi_auth_bootstrap.py
"""

from __future__ import annotations

import base64
import json
import os
import sys

from ssi_sdk import Auth, Config
from ssi_sdk.exceptions import AuthenticationError

sys.path.insert(0, os.path.dirname(__file__))
from ssi_client import TOKEN_CACHE_PATH, SSIClient  # noqa: E402


def main() -> int:
    api_key = os.environ.get("SSI_API_KEY")
    api_secret = os.environ.get("SSI_API_SECRET")
    if not api_key or not api_secret:
        print("Thieu env SSI_API_KEY / SSI_API_SECRET.")
        print("Vi du (PowerShell):")
        print('  $env:SSI_CLIENT_ID="..."')
        print('  $env:SSI_API_KEY="..."')
        print('  $env:SSI_API_SECRET="..."')
        return 1

    config = Config(
        client_id=os.environ.get("SSI_CLIENT_ID", ""),
        api_key=api_key,
        api_secret=api_secret,
        private_key=os.environ.get("SSI_PRIVATE_KEY", ""),
        log_level=os.environ.get("SSI_LOG_LEVEL", "INFO"),
    )

    auth = Auth(config)
    try:
        print("[1] Thu authenticate() khong OTP (Market Data)...")
        auth.authenticate()
    except AuthenticationError as exc:
        print(f"    Can OTP: {exc}")
        print("[2] Goi request_otp()...")
        auth.request_otp()
        otp = input("[3] Nhap ma OTP (SMS/email hoac Smart OTP tren app): ").strip()
        if not otp:
            print("Khong co OTP, huy.")
            return 1
        auth.authenticate(otp=otp)
    print("[4] Xac thuc thanh cong.")

    token = auth.token
    SSIClient._save_token(token)
    b64 = base64.b64encode(json.dumps(token.to_dict(), ensure_ascii=False).encode("utf-8")).decode("ascii")
    print(f"\nToken da luu: {TOKEN_CACHE_PATH}")
    print("\nBase64 cho env / GitHub secret SSI_TOKEN_CACHE (copy dong duoi):")
    print(b64)
    print("\nMac dinh cac run sau se tu refresh token, khong can OTP.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
