# VN Market Breadth Dashboard

Dashboard độ rộng thị trường chứng khoán Việt Nam (A/D Ratio, % cổ phiếu trên MA20/50/200),
lấy dữ liệu từ SSI FastConnect Data API, tự động cập nhật qua GitHub Actions, hiển thị qua GitHub Pages.

## Tính năng

- **Breadth Overview**: A/D Ratio, % trên MA20/MA50/MA200, breadth bar, newly above/below MA
- **Historical Chart**: Biểu đồ Canvas % trên MA20/MA50 theo thời gian với hover tooltip
- **Volume Breakout**: Phát hiện mã có volume đột biến (>=MA20 + KL > 1.3x TB20)
- **Pre-Breakout Signals**: Phát hiện cổ phiếu tích lũy sắp breakout (base score + OBV momentum + vol gradient)
- **Ensemble Signals**: 4-signal voting strategy (MA Crossover + Pullback + Breakout + Momentum)
- **Momentum Signals**: Bộ lọc momentum có điểm số, ADX/RSI/volume bonus
- **Lục Mạch Signals**: VUDD + Tplus + trend/volume/pullback/breakout/sell warning
- **Market Commentary**: Nhận định thị trường tự động (breadth + kỹ thuật VN-Index)
- **Session Compare**: So sánh phiên sáng vs đóng cửa

## Cấu trúc repo

```
scripts/
  ssi_client.py           # client gọi SSI FastConnect Data API
  cache_utils.py          # shared: load_cache + compute_rsi_wilder
  fetch_and_compute.py    # pipeline chính: fetch -> compute -> ghi JSON
  strategy_signals.py     # Pre-breakout detection (base + OBV + vol)
  ensemble_signals.py     # 4-signal voting strategy
  momentum_signals.py     # Momentum score + bonuses
  luc_mach_signals.py     # Lục Mạch: VUDD + Tplus + setup filter
  market_commentary.py    # Nhận định thị trường (breadth + technical)
  embed_data.py           # Tạo dashboard.html với embedded data
  requirements.txt
data/
  breadth_latest.json     # snapshot breadth mới nhất
  breadth_history.json    # lịch sử ~120 phiên
  breadth_midday.json     # snapshot phiên sáng
  strategy_signals.json   # tín hiệu pre-breakout
  ensemble_signals.json   # tín hiệu ensemble
  momentum_signals.json   # tín hiệu momentum
  luc_mach_signals.json   # tín hiệu Lục Mạch
  market_commentary.json  # nhận định thị trường
  ohlc_cache/             # cache OHLCV theo mã
docs/
  index.html              # dashboard source (fetch JSON từ data/)
  dashboard.html          # embedded version (mở trực tiếp không cần server)
  data/                   # JSON copies cho GitHub Pages
.github/workflows/update.yml  # cron: 15:10 VN (T2-T6)
```

## Chạy local

```bash
cd scripts
pip install -r requirements.txt

export SSI_CONSUMER_ID="..."
export SSI_CONSUMER_SECRET="..."
python fetch_and_compute.py
```

Sau khi chạy:
- `data/breadth_latest.json` — breadth snapshot
- `data/strategy_signals.json` — pre-breakout signals
- `data/ensemble_signals.json` — ensemble signals
- `data/momentum_signals.json` — momentum signals
- `data/luc_mach_signals.json` — Lục Mạch signals
- `data/market_commentary.json` — nhận định thị trường

Mở `docs/index.html` qua Live Server (VSCode) hoặc `python -m http.server`.
Mở `docs/dashboard.html` trực tiếp bằng double-click (file://) — đã embed data.

## Dashboard tabs

| Tab | Nội dung |
|-----|----------|
| ALL / HOSE / HNX | Breadth overview: A/D, MA stats, breadth bar, newly above/below |
| Lịch sử | Biểu đồ Canvas % trên MA20/MA50 theo thời gian |
| Breakout | Volume breakout symbols |
| Nhận định | Market commentary (breadth + VN-Index technical) |
| Pre-Breakout | Cổ phiếu tích lũy sắp breakout (base + OBV + vol gradient) |
| Ensemble | 4-signal voting: MA Crossover + Pullback + Breakout + Momentum |
| Lục Mạch | VUDD + Tplus + trend/volume/pullback/breakout/sell warning |

## Triển khai GitHub

1. Push lên GitHub
2. Settings → Secrets: `SSI_CONSUMER_ID`, `SSI_CONSUMER_SECRET`
3. Settings → Pages: branch `main`, folder `/docs`
4. Actions → Run workflow lần đầu

## Strategy Details

### Pre-Breakout (strategy_signals.py)
- Base detection: 25-session window, max range 8%, price in top 25%
- OBV momentum: 5/10/21-day OBV trend
- Volume regime: short/long volatility ratio + gradient
- Composite: 50% base + 30% OBV + 20% vol gradient
- Threshold: composite >= 60, OBV5 > 0, price near high <= 3%

### Ensemble (ensemble_signals.py)
- **MA Crossover**: MA10 > MA50 + Close > MA10 + RSI14 > 50
- **Pullback to Support**: uptrend (Close > MA200) + near MA50 (93-100%) + RSI > 45
- **Breakout**: new 20-day high + volume > 1.5x avg
- **Momentum ROC**: ROC10 > ROC20 + positive volume slope
- **Voting**: >= 3/4 = Strong Buy, 2/4 = Weak Buy

### Lục Mạch (luc_mach_signals.py)
- **VUDD**: Heikin Ashi + ZeroLag TEMA trên các chu kỳ 13/20/35/55/65
- **Tplus**: xác nhận phá vùng 4 phiên
- **Score**: 5 VUDD + 1 Tplus, threshold mặc định 3/6
- **Setup**: trend MA20/50/200, volume/GTGD, pullback MA20, breakout/Darvas 20 phiên
- **Status**: STRONG_BUY, VALID_BUY, WATCHLIST, SELL_WARNING
- **Lưu ý**: RS-line và sector strength chưa bật vì chưa có benchmark/sector mapping ổn định
