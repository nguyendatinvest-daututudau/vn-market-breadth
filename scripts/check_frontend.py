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

    # 5d. Mã quan sát phải có lọc R:R >=1
    if "HUB_TABS" in text:
        if "value:'rr'" not in text and 'value:"rr"' not in text:
            ok = fail("HUB_TABS thiếu lọc R:R >=1") and False
        if "filter === 'rr'" not in text:
            ok = fail("applyFilters thiếu lọc rr") and False

    # 5e. Radar Vol Spike
    if 'data-filter="volSpike"' not in text:
        ok = fail("Radar thiếu nút Vol Spike") and False
    if 'vol_spike_ratio' not in text:
        ok = fail("Radar thiếu vol_spike_ratio") and False

    # 5f. Tab Hiệu quả tín hiệu
    if 'data-view="perf"' not in text:
        ok = fail("thiếu nav tab Hiệu quả (data-view perf)") and False
    if 'id="perfView"' not in text or 'id="perfContent"' not in text:
        ok = fail("thiếu perfView/perfContent") and False
    for fn in ("loadPerfData", "renderPerfContent", "perfVerdict"):
        if fn not in text:
            ok = fail(f"thiếu {fn}") and False
    if 'EMBEDDED_SIGNAL_PERFORMANCE' not in text:
        ok = fail("thiếu EMBEDDED_SIGNAL_PERFORMANCE") and False

    # 5g. Tìm mã toàn lịch sử tín hiệu
    if 'collectSearchSignals' not in text:
        ok = fail("thiếu collectSearchSignals (tìm mã mọi phiên)") and False
    if 'sigDate' not in text:
        ok = fail("thiếu cột Ngày (sigDate) cho tìm kiếm toàn lịch sử") and False

    # 5i. Dòng hành động trong ô Chi tiết
    for token in ("ensurePerfData", "actionForRow", "sig-action", "perfSymCache",
                  "sig-plan", "sp-buy", "sp-stop", "sp-tp", "sig-verdict"):
        if token not in text:
            ok = fail(f"thiếu {token} (dòng hành động bảng tín hiệu)") and False

    # 5j. Mua chuẩn gồm momentum strong + ensemble strong + conviction A/B
    if "r.source === 'ensemble' && r.signalType === 'strong') || r.convAB" not in text:
        ok = fail("tab Mua chuẩn thiếu ensemble strong / conviction A/B") and False
    if 'strongEmptyMsg' not in text:
        ok = fail("thiếu strongEmptyMsg (gợi ý khi Mua chuẩn trống)") and False
    if 'Báo mua rõ' not in text:
        ok = fail("thiếu nhãn Báo mua rõ trong actionForRow") and False

    # 5k. Tab tín hiệu gọn: why thu gọn, score hậu tố, sort gần chuẩn, giá gộp
    for token in ("toggleSigWhy", "vì sao? ▸", "watchNearness", "sigScoreText"):
        if token not in text:
            ok = fail(f"thiếu {token} (tab tín hiệu gọn)") and False
    if 'tr.tier-ref{opacity' in text:
        ok = fail("còn làm mờ dòng reference (tr.tier-ref opacity)") and False

    # 5h. Lịch sử tín hiệu dạng biểu đồ đường
    for token in ("sigHistChart", "renderSigHistChart", "sigHistPoints", "toggleSigHistSeries"):
        if token not in text:
            ok = fail(f"thiếu {token} (lịch sử tín hiệu chart)") and False
    if 'toggleHistoryDetail' in text or 'hist-detail' in text:
        ok = fail("còn bảng lịch sử cũ (toggleHistoryDetail/hist-detail)") and False

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
