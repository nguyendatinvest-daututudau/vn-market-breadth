"""
Shared utilities: cache loading + RSI calculation.
Dung chung cho fetch_and_compute, strategy_signals, ensemble_signals, market_commentary.
"""
from __future__ import annotations
import logging
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def parse_trading_date(series: pd.Series) -> pd.Series:
    """Parse cache dates safely across dd/mm/YYYY and ISO YYYY-mm-dd formats."""
    text = series.astype(str).str.strip()
    parsed = pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text.loc[missing], format="%Y-%m-%d", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text.loc[missing], errors="coerce")
    return parsed


def load_cache(symbol: str, cache_dir: Path) -> pd.DataFrame:
    """Load OHLC cache for a symbol. Returns DataFrame with TradingDate, OHLC, Volume when available."""
    path = cache_dir / f"{symbol}.csv"
    if path.exists():
        try:
            df = pd.read_csv(path)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                df["TradingDate"] = parse_trading_date(df["TradingDate"])
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            for col in ("Open", "High", "Low", "Volume"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                else:
                    df[col] = float("nan")
            return df.dropna(subset=["Close"])
        except Exception as exc:
            logger.warning("load_cache(%s): %s", symbol, exc)
    return pd.DataFrame(columns=["TradingDate", "Open", "High", "Low", "Close", "Volume"])


def compute_rsi_wilder(close_series: pd.Series, period: int = 14) -> float:
    """RSI Wilder (exponential smoothing) — chinh xac hon simple MA."""
    if len(close_series) < period + 1:
        return 50.0
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return float(val) if not pd.isna(val) else 50.0


def compute_rsi_numpy(series: np.ndarray, period: int = 14) -> float:
    """RSI Wilder cho numpy array — dung chung voi ensemble_signals."""
    if len(series) < period + 1:
        return 50.0
    deltas = np.diff(series)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Wilder smoothing (alpha = 1/period)
    alpha = 1.0 / period
    avg_gain = gains[0]
    avg_loss = losses[0]
    for i in range(1, len(gains)):
        avg_gain = avg_gain * (1 - alpha) + gains[i] * alpha
        avg_loss = avg_loss * (1 - alpha) + losses[i] * alpha

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))
