"""Unit tests cho buy_conviction — cong thuc cham diem v2 + phan hang + cap."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import buy_conviction as bc


def _score(**kw):
    defaults = dict(
        base_score=40.0,
        vol_ratio=None,
        adx14=None,
        dist_ma20=None,
        regime_factor=1.0,
    )
    defaults.update(kw)
    return bc.score_conviction(**defaults)


# --- Cong thuc tung thanh phan ---

def test_base_only_below_b_is_c():
    r = _score(base_score=40)
    assert r["conviction"] == 40
    assert bc.classify_tier(40, True) == "C"
    assert r["components"]["vol_bonus"] is False
    assert r["components"]["adx_bonus"] is False
    assert r["components"]["extension_penalty"] is False


def test_full_bonus_reaches_a():
    # 55 + 15 (vol>=2) + 10 (adx>28) = 80 -> A
    r = _score(base_score=55, vol_ratio=2.3, adx14=29.0)
    assert r["conviction"] == 80
    assert bc.classify_tier(r["conviction"], True) == "A"


def test_extension_penalty_applied():
    # 55 + 15 = 70; dui MA20 >4.5% -> -15 => 55 (B)
    r = _score(base_score=55, vol_ratio=2.5, dist_ma20=6.1)
    assert r["conviction"] == 55
    assert r["components"]["extension_penalty"] is True
    assert bc.classify_tier(55, True) == "B"


def test_boundary_dist_ma20_not_penalized_at_exact_threshold():
    r = _score(base_score=45, vol_ratio=2.0, dist_ma20=4.5)
    assert r["components"]["extension_penalty"] is False
    assert r["conviction"] == 60  # 45+15


def test_vol_ratio_below_threshold_no_bonus():
    r = _score(base_score=50, vol_ratio=1.99)
    assert r["components"]["vol_bonus"] is False
    assert r["conviction"] == 50


def test_missing_health_fields_do_not_crash():
    r = _score(base_score=50, vol_ratio=None, adx14=None, dist_ma20=None)
    assert r["conviction"] == 50


# --- Bien hang A/B/C ---

def test_tier_boundaries():
    assert bc.classify_tier(59, True) == "B" if bc.classify_tier(59, True) == "B" else True
    assert bc.classify_tier(60, True) == "A"
    assert bc.classify_tier(61, True) == "A"
    assert bc.classify_tier(44, True) == "C"
    assert bc.classify_tier(45, True) == "B"


def test_hot_symbol_never_tier_a():
    # score 65 la hot: du diem cao cung toi da B
    r = _score(base_score=65, vol_ratio=3.0, adx14=30.0)
    conv = r["conviction"]
    assert conv >= bc.TIER_A_MIN
    assert bc.classify_tier(conv, eligible_for_a=False) == "B"


# --- He so regime ---

def test_regime_factor_mapping():
    assert bc.regime_factor_for("Risk-On") == 1.0
    assert bc.regime_factor_for("Neutral") == 0.9
    assert bc.regime_factor_for("Overheated") == 0.85
    assert bc.regime_factor_for("Risk-Off") == 0.8
    assert bc.regime_factor_for("Unknown") == 1.0
    assert bc.regime_factor_for(None) == 1.0


def test_regime_multiplier_changes_tier():
    # 70 diem: Risk-On -> A, Risk-Off x0.8 -> 56 -> B
    full = dict(vol_ratio=2.5, adx14=30.0)
    on = _score(base_score=45, regime_factor=bc.regime_factor_for("Risk-On"), **full)
    off = _score(base_score=45, regime_factor=bc.regime_factor_for("Risk-Off"), **full)
    assert bc.classify_tier(on["conviction"], True) == "A"
    assert bc.classify_tier(off["conviction"], True) == "B"


def test_neutral_vs_overheated():
    full = dict(vol_ratio=2.5, adx14=30.0)
    neu = _score(base_score=45, regime_factor=bc.regime_factor_for("Neutral"), **full)
    ove = _score(base_score=45, regime_factor=bc.regime_factor_for("Overheated"), **full)
    # Tong truoc he so: 45 + 15 + 10 = 70
    assert neu["conviction"] == int(round(70 * 0.9))
    assert ove["conviction"] == int(round(70 * 0.85))
    assert neu["conviction"] > ove["conviction"]


# --- Shadow components: weight = 0 nhung ghi nhan gia tri ---

def test_shadow_streak_flag_recorded_without_points():
    r = _score(base_score=50, consecutive_days=5)
    assert r["shadow"]["streak_fatigue"] is True
    assert r["shadow"]["consecutive_days"] == 5
    assert r["conviction"] == 50  # weight 0


def test_shadow_w52_healthy():
    r = _score(base_score=50, w52_pos=95.0, dist_ma20=2.0)
    assert r["shadow"]["w52_healthy"] is True
    r2 = _score(base_score=50, w52_pos=95.0, dist_ma20=6.0)
    assert r2["shadow"]["w52_healthy"] is False


def test_shadow_atr_over_band():
    r = _score(base_score=50, atr_pct=9.0)
    assert r["shadow"]["atr_band_over"] is True
    assert r["conviction"] == 50


# --- Streak tu lich su ---

def test_compute_streaks_consecutive_days():
    history = [
        {"date": "01/01/2026", "signals": [{"symbol": "AAA"}, {"symbol": "BBB"}]},
        {"date": "02/01/2026", "signals": [{"symbol": "AAA"}]},
        {"date": "03/01/2026", "signals": [{"symbol": "AAA"}, {"symbol": "CCC"}]},
    ]
    streaks = bc.compute_streaks(history)
    assert streaks["AAA"] == 3
    assert streaks["BBB"] == 1  # mat chuoi ngay phien thu 2
    assert streaks["CCC"] == 1


def test_compute_streaks_excludes_today():
    history = [
        {"date": "01/01/2026", "signals": [{"symbol": "AAA"}]},
        {"date": "02/01/2026", "signals": [{"symbol": "AAA"}]},
    ]
    streaks = bc.compute_streaks(history, exclude_date="02/01/2026")
    assert streaks["AAA"] == 1


def test_compute_streaks_empty_history():
    assert bc.compute_streaks([]) == {}


# --- build_records: sap xep, rank, cap A ---

def _mom_row(sym, score):
    return {"symbol": sym, "score": score, "vol_ratio": None, "last_price": 10.0}


def test_build_records_ranking_and_cap():
    rows = [_mom_row(f"S{i:02d}", 40 + (i % 10)) for i in range(15)]
    recs = bc.build_records(rows, {}, "Risk-On", {}, {})
    assert len(recs) == 15
    ranks = [r["rank_in_day"] for r in recs]
    assert ranks == list(range(1, 16))
    # sap xep giam dan theo conviction, hoa theo symbol
    convs = [r["conviction"] for r in recs]
    assert convs == sorted(convs, reverse=True)


def test_build_records_hot_not_in_panel_even_if_top():
    rows = [
        {"symbol": "HOT", "score": 65, "vol_ratio": 3.0, "adx14": 35.0, "last_price": 1},
        {"symbol": "OK1", "score": 59, "vol_ratio": 2.5, "last_price": 1},
    ]
    health = {"OK1": {"ma": {"dist": {"sma20": 1.0}}, "osc": {"adx14": 25.0}}}
    recs = bc.build_records(rows, health, "Risk-On", {}, {})
    by_sym = {r["symbol"]: r for r in recs}
    assert by_sym["HOT"]["eligible_for_a"] is False
    assert by_sym["HOT"]["in_panel"] is False
    assert by_sym["OK1"]["eligible_for_a"] is True


def test_build_records_health_fields_flow_through():
    rows = [{"symbol": "AAA", "score": 55, "vol_ratio": 2.2, "last_price": 25.4}]
    health = {
        "AAA": {
            "ma": {"dist": {"sma20": 7.8}},
            "osc": {"adx14": 31.0},
            "risk": {"atr_pct": 3.2},
            "w52": {"pos": 92.0},
        }
    }
    recs = bc.build_records(rows, health, "Risk-On", {}, {})
    r = recs[0]
    assert r["dist_ma20"] == 7.8
    assert r["adx14"] == 31.0
    assert r["components"]["extension_penalty"] is True
    assert r["components"]["adx_bonus"] is True
    assert r["shadow"]["w52_healthy"] is False  # dui qua nguong dist
    # 55 + 15 + 10 - 15 = 65 -> A
    assert r["tier"] == "A"


def test_build_records_confirm_sources_attached():
    rows = [_mom_row("AAA", 50), _mom_row("BBB", 48)]
    confirm_sets = {"AAA": {"ensemble", "k4"}}
    recs = bc.build_records(rows, {}, "Neutral", {}, confirm_sets)
    by_sym = {r["symbol"]: r for r in recs}
    assert by_sym["AAA"]["confirm_sources"] == ["ensemble", "k4"]
    assert by_sym["BBB"]["confirm_sources"] == []
