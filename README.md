# VN Market Breadth Dashboard

Dashboard độ rộng thị trường chứng khoán Việt Nam (A/D Ratio, % cổ phiếu trên MA20/50/200),
lấy dữ liệu từ SSI FastConnect Data API, tự động cập nhật qua GitHub Actions, hiển thị qua GitHub Pages.

## Cấu trúc repo

```
scripts/
  ssi_client.py           # client gọi SSI FastConnect Data API
  fetch_and_compute.py    # pipeline chính: lấy dữ liệu -> tính breadth -> ghi JSON
  requirements.txt
data/
  breadth_latest.json     # snapshot mới nhất (được Actions ghi đè mỗi ngày)
  breadth_history.json    # lịch sử ~120 phiên gần nhất
  ohlc_cache/              # cache OHLC từng mã (không commit, dùng actions/cache)
docs/
  index.html               # dashboard, host qua GitHub Pages, đọc 2 file JSON ở trên
.github/workflows/update.yml   # cron job chạy script mỗi ngày sau giờ đóng cửa
```

## 1. Chạy thử ở máy local

```bash
cd scripts
pip install -r requirements.txt

export SSI_CONSUMER_ID="..."
export SSI_CONSUMER_SECRET="..."
python fetch_and_compute.py
```

Sau khi chạy xong, `data/breadth_latest.json` và `data/breadth_history.json` sẽ được tạo/cập nhật.
Mở `docs/index.html` bằng Live Server (VSCode extension) hoặc `python -m http.server` trong thư mục gốc
repo để dashboard fetch được 2 file JSON qua đường dẫn tương đối `../data/...`.

**Lưu ý:** mở trực tiếp `index.html` bằng cách double-click file (`file://`) sẽ bị chặn bởi CORS khi
`fetch()` file JSON. Luôn chạy qua local server khi test.

## 2. Đưa lên GitHub

```bash
git init
git add .
git commit -m "init: vn market breadth dashboard"
git branch -M main
git remote add origin https://github.com/<username>/<repo>.git
git push -u origin main
```

## 3. Khai báo Secrets cho GitHub Actions

Vào repo trên GitHub → **Settings → Secrets and variables → Actions → New repository secret**, thêm:

- `SSI_CONSUMER_ID`
- `SSI_CONSUMER_SECRET`

Workflow trong `.github/workflows/update.yml` sẽ đọc 2 secret này khi chạy, KHÔNG bao giờ hardcode
key vào code.

## 4. Bật GitHub Pages

Settings → Pages → Source: chọn branch `main`, thư mục `/docs` → Save.

Sau vài phút, dashboard sẽ có tại: `https://<username>.github.io/<repo>/`

## 5. Chạy thử workflow lần đầu

Vào tab **Actions** trên GitHub → chọn workflow "Update market breadth data" → **Run workflow**
(chạy tay lần đầu, không cần chờ tới lịch cron) để tạo `data/breadth_latest.json` thật từ SSI.

Sau đó workflow sẽ tự chạy theo lịch cron đã đặt (15:45 giờ VN các ngày thứ 2–6, sau giờ đóng cửa).

## Ghi chú quan trọng

- **Rate limit**: script tính MA breadth phải gọi API `DailyOhlc` cho *từng mã* trong danh sách sàn
  (~400 mã HOSE, tương tự HNX/UPCOM). Lần chạy đầu tiên sẽ chậm vì phải tải ~250 phiên lịch sử cho
  mỗi mã. Các lần sau chỉ tải bổ sung vài phiên gần nhất nhờ cơ chế cache
  (`data/ohlc_cache/`, được khôi phục giữa các lần chạy Actions qua `actions/cache`).
- Nếu SSI trả lỗi 401 giữa chừng, client sẽ tự lấy lại access token 1 lần trước khi báo lỗi.
- Cấu trúc JSON output (`breadth_latest.json`) độc lập với dashboard — có thể tái sử dụng cho
  Excel, Google Sheets, hoặc bot Telegram/Zalo nếu muốn mở rộng sau này.
- `data/breadth_latest.json` và `data/breadth_history.json` hiện đang chứa **dữ liệu mẫu** (khớp với
  ảnh chụp màn hình gốc, phiên 29/06/2026) để bạn xem trước giao diện ngay khi mới clone repo. Chạy
  `Run workflow` lần đầu trên GitHub Actions (mục 5 bên dưới) sẽ ghi đè bằng dữ liệu thật từ SSI.
