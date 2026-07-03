"""Nhúng dữ liệu breadth + commentary vào dashboard HTML để mở trực tiếp không cần server."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LATEST_JSON = ROOT / "data" / "breadth_latest.json"
HISTORY_JSON = ROOT / "data" / "breadth_history.json"
MIDDAY_JSON = ROOT / "data" / "breadth_midday.json"
COMMENTARY_JSON = ROOT / "data" / "market_commentary.json"
SRC_HTML = ROOT / "docs" / "index.html"
OUT_HTML = ROOT / "docs" / "dashboard.html"

latest = json.loads(LATEST_JSON.read_text(encoding="utf-8"))
history = json.loads(HISTORY_JSON.read_text(encoding="utf-8") if HISTORY_JSON.exists() else "[]")
midday = json.loads(MIDDAY_JSON.read_text(encoding="utf-8")) if MIDDAY_JSON.exists() else None
commentary = json.loads(COMMENTARY_JSON.read_text(encoding="utf-8")) if COMMENTARY_JSON.exists() else None

html = SRC_HTML.read_text(encoding="utf-8")

# Inject dữ liệu JSON inline
inject_script = f"""
<script>
const EMBEDDED_LATEST = {json.dumps(latest, ensure_ascii=False)};
const EMBEDDED_HISTORY = {json.dumps(history, ensure_ascii=False)};
const EMBEDDED_MIDDAY = {json.dumps(midday, ensure_ascii=False)};
const EMBEDDED_COMMENTARY = {json.dumps(commentary, ensure_ascii=False)};
</script>
"""

# Inject script vào </head>
html = html.replace("</head>", inject_script + "</head>")

# Thay thế hàm loadData() — dùng markers rõ ràng
old_marker = "async function loadData(){"
new_func = """async function loadData(){
  LATEST = EMBEDDED_LATEST;
  HISTORY = EMBEDDED_HISTORY || [];
  MIDDAY = EMBEDDED_MIDDAY;
  COMMENTARY = EMBEDDED_COMMENTARY;
  if (MIDDAY && MIDDAY.markets && MIDDAY.markets.ALL.date !== LATEST.markets.ALL.date) { MIDDAY = null; }
  const sessionLabel = LATEST.session === 'midday'
    ? '<span class="session-badge midday">Phiên sáng 11:30</span>'
    : '<span class="session-badge close">Đóng cửa 15:10</span>';
  document.getElementById('metaLine').innerHTML =
    "Cập nhật: " + new Date(LATEST.generated_at).toLocaleString('vi-VN') + " · Ngày dữ liệu: " + LATEST.markets.ALL.date + " " + sessionLabel;
  MARKETS = ['ALL', ...Object.keys(LATEST.markets).filter(k => k !== 'ALL')];
  renderTabs();
  render();
}"""

if old_marker in html:
    idx = html.index(old_marker)
    # Tìm kết thúc hàm: dòng "loadData();" ngay trước "</script>"
    end_marker = "loadData();\n</script>"
    end_idx = html.index(end_marker, idx)
    # Kết thúc thay thế sau dấu } cuối cùng của hàm (trước loadData();)
    # Tìm } cuối trước end_marker
    before_end = html[idx:end_idx]
    # Lấy dấu } cuối cùng trong đoạn
    last_brace = before_end.rfind("}")
    if last_brace != -1:
        actual_end = idx + last_brace + 1
    else:
        actual_end = end_idx
    html = html[:idx] + new_func + "\n\n" + html[actual_end:]

OUT_HTML.write_text(html, encoding="utf-8")
print(f"Đã tạo: {OUT_HTML}")
print("Mở file này bằng double-click (file://) để xem dashboard.")
