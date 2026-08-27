"""Check frontend không bị revert - bắt lỗi vừa gặp.

Kiểm tra:
- buildUdVolumePanel / renderUdVolumeChart tồn tại
- pts không dùng trước khi khai báo (idxCnt sau pts)
- nav-cta duplicate phải =0 (chỉ giữ link giữa đỏ)
- data-sym trong phụ lục đủ
- Zweig chỉ dùng zweig_breadth_thrust

Thoát 1 nếu fail để CI chặn.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "docs" / "index.html"

def fail(msg: str):
    try:
        print(f"FAIL: {msg}")
    except UnicodeEncodeError:
        print("FAIL: frontend check failed")
    return False

def main() -> int:
    text = HTML.read_text(encoding="utf-8")
    ok = True

    # 1. UD Volume panel
    if "buildUdVolumePanel" not in text:
        ok = fail("thiếu buildUdVolumePanel") and False
    if "renderUdVolumeChart" not in text:
        ok = fail("thiếu renderUdVolumeChart") and False
    if "buildUdVolumePanel(m)" not in text:
        ok = fail("thiếu gọi buildUdVolumePanel(m) trong view-market") and False
    if text.count("renderUdVolumeChart") < 2:
        ok = fail("thiếu requestAnimationFrame(renderUdVolumeChart)") and False

    # 2. pts trước idxCnt - anti ReferenceError
    # Đảm bảo idxCnt khai báo sau pts
    idx_pos = text.find("const idxCnt = pts.filter")
    pts_pos = text.find("const pts = allPts.slice")
    if idx_pos != -1 and pts_pos != -1 and idx_pos < pts_pos:
        ok = fail("idxCnt dùng pts trước khi pts khai báo (ReferenceError chart trắng)") and False

    # 3. nav duplicate
    nav_cta = text.count('class="nav-cta"')
    if nav_cta != 0:
        ok = fail(f"nav-cta duplicate còn {nav_cta} cái, phải =0 (chỉ giữ link giữa đỏ)") and False
    if 'data-view="signals"' not in text:
        ok = fail("thieu CSS link giua do data-view signals") and False

    # 4. appendix data-sym
    # renderChips phải sinh data-sym
    if 'data-sym="${esc}"' not in text and "data-sym" not in text:
        ok = fail("phụ lục thiếu data-sym") and False
    # đếm data-sym trong file phải đủ
    if text.count("data-sym") < 10:
        ok = fail(f"data-sym quá ít ({text.count('data-sym')})") and False

    # 5. Zweig chỉ dùng current
    if "REGIME?.zweig" in text and "zweig_breadth_thrust" in text:
        # cho phép comment nhưng không cho fallback legacy
        if "legacy = REGIME?.zweig" in text:
            ok = fail("còn fallback legacy REGIME?.zweig") and False

    # 5b. Nhận xét nhanh phải xuống dòng
    if "shQuickTake" in text:
        if "join('<br>')" not in text:
            ok = fail("shQuickTake phải xuống dòng (join <br>)") and False

    # 5c. Kế hoạch lệnh phải có màu (card layout)
    if "tradePlanDetailHTML" in text:
        if "background:rgba(59,130,246" not in text or "R:R" not in text:
            ok = fail("tradePlanDetailHTML mất màu/card - phải có 3 card xanh/đỏ/xanh và R:R badge") and False

    # 6. cú pháp cơ bản: ngoặc cân bằng
    if text.count("{") != text.count("}"):
        print(f"WARN: {{ {text.count('{')} != }} {text.count('}')} - kiểm tra thủ công")

    if ok:
        print("check_frontend: OK")
        return 0
    print("check_frontend: FAILED")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
