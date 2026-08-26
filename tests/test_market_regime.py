"""Tests for market_regime (regime gauge, divergence, breadth momentum) + history fields."""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fetch_and_compute as pipeline
import market_regime as mr
from cache_utils import load_cache


# --- Score mapping -----------------------------------------------------------

def test_score_ad_ratio():
    assert mr.score_ad_ratio(0.5) == 10.0
    assert mr.score_ad_ratio(1.0) == 50.0
    assert mr.score_ad_ratio(1.5) == 70.0
    assert mr.score_ad_ratio(2.5) == 90.0
    assert mr.score_ad_ratio(5.0) == 95.0
    assert mr.score_ad_ratio(0.8) > 10.0
    assert mr.score_ad_ratio(0.8) < 50.0
    assert mr.score_ad_ratio(None) == 50.0


def test_score_pct_above():
    assert mr.score_pct_above(10.0) == 10.0
    assert mr.score_pct_above(50.0) == 70.0
    assert mr.score_pct_above(90.0) == 85.0
    assert mr.score_pct_above(30.0) == 50.0


def test_score_index_state():
    strong = mr.score_index_state({"above_ma20": True, "above_ma50": True,
                                   "above_ma200": True, "macd_up": True, "rsi": 62})
    weak = mr.score_index_state({"above_ma20": False, "above_ma50": False,
                                 "above_ma200": False, "macd_up": False, "rsi": 38})
    assert strong > weak
    assert mr.score_index_state(None) == 50.0


def test_score_rsi_pulse():
    bullish = mr.score_rsi_pulse({"over_70": 40, "under_30": 5, "total": 100})
    bearish = mr.score_rsi_pulse({"over_70": 5, "under_30": 40, "total": 100})
    assert bullish > 50.0
    assert bearish < 50.0
    assert mr.score_rsi_pulse(None) == 50.0


def test_compose_regime_labels():
    def comps(p):
        return {"ad_ratio": {"points": p}, "pct_above_ma20": {"points": p},
                "pct_above_ma50": {"points": p}, "index_position": {"points": p},
                "rsi_pulse": {"points": p}, "volume_ud": {"points": p}}
    risk_off = mr.compose_regime(comps(10))
    assert risk_off["tone"] == "risk_off"
    risk_on = mr.compose_regime(comps(70))
    assert risk_on["tone"] == "risk_on"
    overheated = mr.compose_regime(comps(90))
    assert overheated["tone"] == "overheated"
    assert 0.0 <= risk_on["score"] <= 100.0


# --- Divergence --------------------------------------------------------------

def _history_series(pcts, dates=None):
    out = []
    for i, p in enumerate(pcts):
        d = dates[i] if dates else (datetime(2026, 1, 1).toordinal() + i)
        out.append({"date": datetime.fromordinal(d).strftime("%d/%m/%Y"),
                    "markets": {"ALL": {"pct_above_ma20": p}}})
    return out


def _index_frame(closes, dates=None):
    rows = []
    for i, c in enumerate(closes):
        d = dates[i] if dates else datetime.fromordinal(datetime(2026, 1, 1).toordinal() + i)
        rows.append({"TradingDate": pd.Timestamp(d), "Close": c})
    return pd.DataFrame(rows)


def test_divergence_none():
    history = _history_series([40.0] * 25)
    frame = _index_frame([1000.0] * 25)
    result = mr.detect_divergence(history, frame, lookback=20)
    assert result["state"] == "none"


def test_divergence_bearish():
    closes = list(range(1000, 1025))
    pcts = [60.0] * 15 + [40.0] * 10  # gia len dinh moi nhung breadth sut giam
    history = _history_series(pcts)
    frame = _index_frame(closes)
    result = mr.detect_divergence(history, frame, lookback=20)
    assert result["state"] == "bearish"


def test_divergence_bullish():
    closes = list(range(1100, 1075, -1))
    pcts = [40.0] * 15 + [60.0] * 10  # gia xuong day nhung breadth cai thien som
    history = _history_series(pcts)
    frame = _index_frame(closes)
    result = mr.detect_divergence(history, frame, lookback=20)
    assert result["state"] == "bullish"


def test_divergence_insufficient_data():
    history = _history_series([40.0] * 5)
    frame = _index_frame([1000.0] * 5)
    result = mr.detect_divergence(history, frame, lookback=20)
    assert result["state"] == "none"


# --- Breadth momentum --------------------------------------------------------

def test_breadth_momentum():
    pcts = [50 + (i % 9) - 4 for i in range(200)]
    history = _history_series(pcts)
    mom = mr.breadth_momentum(history)
    assert mom["available"] is True
    for key in ("oscillator", "signal", "histogram"):
        assert key in mom
    assert mom["extreme"] in ("none", "overbought", "oversold")


# --- History compact fields (E) ---------------------------------------------

def test_compact_history_stores_ad_counts():
    snap = {
        "ALL": {"pct_above_ma20": 48.8, "advances": 131, "declines": 173,
                "unchanged": 80, "ad_ratio": 0.76, "total_symbols": 384,
                "ma_total_symbols": 160, "rsi_pulse": None},
        "HOSE": {"pct_above_ma20": 50.0, "advances": 124, "declines": 163,
                 "unchanged": 76, "ad_ratio": 0.76, "total_symbols": 363,
                 "ma_total_symbols": 139, "rsi_pulse": None},
    }
    compact = pipeline._compact_history_markets(snap)
    assert compact["ALL"]["advances"] == 131
    assert compact["ALL"]["declines"] == 173
    assert compact["ALL"]["unchanged"] == 80
    assert compact["HOSE"]["ma_total_symbols"] == 139
    assert compact["ALL"]["pct_above_ma20"] == 48.8


def _zweig_rows(values):
    dates = pd.bdate_range("2026-01-02", periods=len(values))
    return [
        {
            "date": date.strftime("%d/%m/%Y"),
            "ratio": value,
            "ema10": value,
            "valid": value is not None,
        }
        for date, value in zip(dates, values)
    ]


@pytest.mark.parametrize("values, expected", [
    ([0.50], "inactive"),
    ([0.39], "armed"),
    ([0.39, 0.50], "forming"),
    ([0.39, 0.62], "confirmed"),
    ([0.39] + [0.50] * 11, "expired"),
])
def test_zweig_state_machine_states(values, expected):
    assert mr._zweig_state_machine(_zweig_rows(values))["state"] == expected


def test_zweig_confirmation_allowed_at_ten_but_not_eleven_sessions():
    at_ten = mr._zweig_state_machine(_zweig_rows([0.39] + [0.50] * 9 + [0.62]))
    assert at_ten["state"] == "confirmed"
    assert at_ten["sessions_since_armed"] == 10

    at_eleven = mr._zweig_state_machine(_zweig_rows([0.39] + [0.50] * 10 + [0.62]))
    assert at_eleven["state"] == "expired"


def test_zweig_thresholds_are_strict():
    assert mr._zweig_state_machine(_zweig_rows([0.40]))["state"] == "inactive"
    assert mr._zweig_state_machine(_zweig_rows([0.39, 0.615]))["state"] == "forming"


def test_zweig_ema_uses_hose_ad_and_requires_ten_valid_observations():
    history = _history_series([50.0] * 11)
    for entry in history:
        entry["markets"]["HOSE"] = {"advances": 30, "declines": 70}
    history[-1]["markets"]["HOSE"] = {"advances": 70, "declines": 30}

    series = mr.zweig_ema10_series(history)

    assert series[8]["ema10"] is None
    assert series[9]["ema10"] == pytest.approx(0.3)
    assert series[10]["ema10"] == pytest.approx(0.3 + (2 / 11) * 0.4)


def test_zweig_missing_ad_carries_ema_but_consumes_setup_session():
    rows = _zweig_rows([0.39] + [0.50] * 8 + [None] + [0.62])
    result = mr._zweig_state_machine(rows)
    assert result["state"] == "confirmed"
    assert result["sessions_since_armed"] == 10


def test_main_preserves_nested_contract_and_adds_zweig(tmp_path, monkeypatch):
    monkeypatch.setattr(mr, "OUTPUT_JSON", tmp_path / "market_regime.json")
    monkeypatch.setattr(mr, "DOCS_DATA_DIR", tmp_path / "docs")
    monkeypatch.setattr(mr, "load_latest", lambda: {"markets": {"ALL": {"date": "17/07/2026"}}})
    monkeypatch.setattr(mr, "load_history", lambda: [{}])
    regime = {"score": 61.0, "label": "Risk-On", "tone": "risk_on", "components": {}, "weights": {}}
    monkeypatch.setattr(mr, "build_regime", lambda latest, history: regime)
    monkeypatch.setattr(mr, "detect_divergence", lambda history, frame: {"state": "none"})
    monkeypatch.setattr(mr, "breadth_momentum", lambda history: {"available": False})
    monkeypatch.setattr(mr, "zweig_breadth_thrust", lambda history: {"available": True, "state": "forming"})
    monkeypatch.setattr(mr, "load_index_frame", lambda name: name)
    monkeypatch.setattr(mr, "index_technical", lambda name: {"close": 1200.0 if name == "VNI" else 245.0})

    assert mr.main() == 0
    payload = json.loads((tmp_path / "market_regime.json").read_text(encoding="utf-8"))
    assert payload["regime"] == regime
    assert payload["index"]["VNI"]["close"] == 1200.0
    assert payload["zweig_breadth_thrust"]["state"] == "forming"
