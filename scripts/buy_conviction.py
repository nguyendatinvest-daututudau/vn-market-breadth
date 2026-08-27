"""
Buy Conviction Score — Lop cham diem tin cay diem mua nam TREN cac he hien tai.

Nguyen tac bat bien: KHONG sua logic sinh tin hieu cua bat ky generator nao.
Module nay chi DOC ket qua co san (momentum + cac he khac + stock_health +
market_regime) roi ghep lai thanh diem conviction 0-100 va xep hang A/B/C.

Cong thuc v2 (trong so chot sau calibration 10 nam):
  - Nen: score momentum (band 30-59). Score >= 60 ("hot") van duoc tinh diem
         nhung bi loai khoi hang A (backtest: edge am OOS o bracket cao).
  - +15 neu vol_ratio >= 2.0   (calibration: mean +0.8pp, marginal)
  - +10 neu adx14 > 28         (calibration: hit2 47.0% vs 42.4%, OOS +6.2pp — MANH NHAT)
  - Extension penalty DA BO (calibration: dist>4.5% hit2 45.8% vs 41.0% — duoi thang hon)
  - Nhan he so regime: Risk-On x1.0 / Neutral x0.9 / Overheated x0.85 /
        Risk-Off x0.8 (thi truong xau -> nguong an len tu dong)

Shadow components (trong so = 0, chi ghi gia tri de calibration danh gia):
  consecutive_days (met moi tin hieu), atr_pct qua band, w52 gan dinh lanh manh,
  rs_vs_vni (chi tinh khi co du lieu VNI trong cache).

Hang: A >= TIER_A_MIN, B >= TIER_B_MIN, con lai C.
Cap: chi cap_a ma hang A dau tien duoc dua vao panel Mua chuan (in_panel=true).
"""
from __future__ import annotations

import json
import math
import os
import warnings

from _shared import DATA_DIR, DOCS_DATA_DIR, format_market_date, signal_market_date, tqdm, vn_now
from _shared import json_default as _json_default

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

MOMENTUM_JSON = DATA_DIR / "momentum_signals.json"
HEALTH_JSON = DATA_DIR / "stock_health.json"
REGIME_JSON = DATA_DIR / "market_regime.json"
ENSEMBLE_JSON = DATA_DIR / "ensemble_signals.json"
LUC_JSON = DATA_DIR / "luc_mach_signals.json"
K4_JSON = DATA_DIR / "khung4_tplus_signals.json"
MAMA_JSON = DATA_DIR / "mama_positional_signals.json"
ATS_JSON = DATA_DIR / "advanced_trailstop_signals.json"
HISTORY_JSON = DATA_DIR / "buy_conviction_history.json"
OUTPUT_JSON = DATA_DIR / "buy_conviction.json"

HISTORY_CAP = 260

# --- Trong so cong thuc (chot sau calibration) ---
BASE_MIN = 30          # diem toi thieu de mot row momentum duoc xet
BASE_HOT_MIN = 60      # >= muc nay la "hot" -> khong duoc len hang A
VOL_BONUS_RATIO = 2.0
VOL_BONUS_POINTS = 15
ADX_BONUS_LEVEL = 28.0
ADX_BONUS_POINTS = 10
EXT_DIST_MA20_MAX = 4.5
EXT_PENALTY_POINTS = 0   # da loai bo theo calibration: dist>4.5% actually +1.3pp lift
TIER_A_MIN = 50           # calibration: A>=55 -> n=95, hit2=42.1%; user request A>=50 for more signals
TIER_B_MIN = 40           # calibration: B>=40 -> hit2=42.8%, best tier B threshold
CAP_A = 12

REGIME_FACTOR = {
    "Risk-On": 1.0,
    "Neutral": 0.9,
    "Overheated": 0.85,
    "Risk-Off": 0.8,
}

# Shadow components: gia tri duoc tinh nhung trong so hien tai = 0.
SHADOW_WEIGHTS = {
    "streak_fatigue": 0,     # >= STREAK_FATIGUE_DAYS phien lien tiep co tin hieu
    "atr_band": 0,           # atr_pct > ATR_OVER_BAND -> phat nhe
    "w52_healthy": 0,        # gan dinh 52W nhung khong dui -> thuong nhe
    "rs_vs_vni": 0,          # RS 20 phien so voi VN-Index
}
STREAK_FATIGUE_DAYS = 4
ATR_OVER_BAND = 8.0
W52_HEALTHY_POS = 90.0
W52_HEALTHY_DIST_MAX = 3.0


def regime_factor_for(label: str | None) -> float:
    return REGIME_FACTOR.get(label or "", 1.0)


def classify_tier(conviction: int, eligible_for_a: bool) -> str:
    """A/B/C theo nguong; ma 'hot' (khong duoc len A) toi da B."""
    if conviction >= TIER_A_MIN and eligible_for_a:
        return "A"
    if conviction >= TIER_B_MIN:
        return "B"
    return "C"


def compute_streaks(history_entries: list[dict], exclude_date: str | None = None) -> dict[str, int]:
    """Dem so phien lien tiep GAN NHAT (tu hien tai nguoc ve truoc) ma tung ma
    xuat hien trong lich su conviction. Neu exclude_date khac None thi bo qua
    ngay do (dung khi ngay hom nay da duoc append truoc khi goi ham nay)."""
    streaks: dict[str, int] = {}
    active: set[str] | None = None
    for entry in reversed(history_entries or []):
        date = entry.get("date")
        if exclude_date and date == exclude_date:
            continue
        sigs = entry.get("signals") or []
        if not sigs:
            break
        syms = {s.get("symbol") for s in sigs}
        if not syms:
            break
        if active is None:
            active = set(syms)
            for sym in syms:
                streaks[sym] = 1
        else:
            # Chi giu streak cho ma van xuat hien lien tuc; gap thi giu nguyen streak hien tai
            new_active: set[str] = set()
            for sym in list(active):
                if sym in syms:
                    streaks[sym] = streaks.get(sym, 0) + 1
                    new_active.add(sym)
            active = new_active
            if not active:
                break
    return streaks


def score_conviction(
    base_score: float,
    *,
    vol_ratio: float | None,
    adx14: float | None,
    dist_ma20: float | None,
    regime_factor: float,
    consecutive_days: int = 0,
    atr_pct: float | None = None,
    w52_pos: float | None = None,
) -> dict:
    """Cham diem cho mot ma. Ham thuan, de test."""
    components: dict = {}
    total = float(base_score)
    components["base_score"] = base_score

    vol_ok = vol_ratio is not None and vol_ratio >= VOL_BONUS_RATIO
    if vol_ok:
        total += VOL_BONUS_POINTS
    components["vol_bonus"] = bool(vol_ok)

    adx_ok = adx14 is not None and adx14 > ADX_BONUS_LEVEL
    if adx_ok:
        total += ADX_BONUS_POINTS
    components["adx_bonus"] = bool(adx_ok)

    ext_ok = dist_ma20 is not None and dist_ma20 > EXT_DIST_MA20_MAX
    if ext_ok:
        total += EXT_PENALTY_POINTS
    components["extension_penalty"] = bool(ext_ok)

    shadow = {
        "consecutive_days": consecutive_days,
        "streak_fatigue": consecutive_days >= STREAK_FATIGUE_DAYS,
        "atr_pct": atr_pct,
        "atr_band_over": bool(atr_pct is not None and atr_pct > ATR_OVER_BAND),
        "w52_healthy": bool(
            w52_pos is not None
            and dist_ma20 is not None
            and w52_pos >= W52_HEALTHY_POS
            and dist_ma20 <= W52_HEALTHY_DIST_MAX
        ),
        "rs_vs_vni": None,
    }
    shadow_points = sum(
        SHADOW_WEIGHTS[k]
        for k, hit in (
            ("streak_fatigue", shadow["streak_fatigue"]),
            ("atr_band", shadow["atr_band_over"]),
            ("w52_healthy", shadow["w52_healthy"]),
        )
        if hit
    )
    total += shadow_points

    conviction = int(math.floor(total * regime_factor + 0.5))
    return {"conviction": conviction, "components": components, "shadow": shadow}


def compute_trade_plan(
    *,
    close: float | None,
    sma10: float | None,
    sma20: float | None,
    sma50: float | None,
    sma200: float | None,
    atr14: float | None,
    dist_ma20: float | None,
    pivots: dict | None,
    w52_high: float | None,
    w52_low: float | None,
) -> dict | None:
    """Ke hoach vao lenh cho tung diem mua hang A/B. Tra ve None neu thieu du lieu cot loi."""
    if close is None or not math.isfinite(close) or close <= 0:
        return None
    # Tim khang cu / ho tro gan nhat
    resistance = None
    support = None
    try:
        cands_res = [x for x in (sma20, sma50, sma200, (pivots or {}).get("r1"), (pivots or {}).get("r2"), w52_high) if x is not None and math.isfinite(x) and x > close]
        if cands_res:
            resistance = min(cands_res)
        cands_sup = [x for x in (sma20, sma50, sma200, (pivots or {}).get("s1"), (pivots or {}).get("s2"), w52_low) if x is not None and math.isfinite(x) and x < close]
        if cands_sup:
            support = max(cands_sup)
    except Exception:
        pass

    # Vung mua
    entry_type = "base"
    entry_low = entry_high = None
    entry_note = ""
    if dist_ma20 is not None and math.isfinite(dist_ma20) and dist_ma20 > 4.0 and sma10 is not None and sma20 is not None and math.isfinite(sma10) and math.isfinite(sma20):
        entry_type = "pullback"
        lo = min(sma10, sma20)
        hi = max(sma10, sma20)
        entry_low = round(lo * 0.99, 2)
        entry_high = round(hi * 1.01, 2)
        entry_note = "Duỗi >4% trên MA20 — chờ pullback về MA10-20 ±1% (không mua đuổi)"
    elif resistance is not None and (resistance - close) / close < 0.03:
        entry_type = "breakout"
        entry_low = round(resistance * 0.98, 2)
        entry_high = round(resistance * 1.02, 2)
        entry_note = f"Gần kháng cự {resistance:.1f} (<3%) — vùng phá vỡ, cần Vol×2 xác nhận"
    else:
        entry_low = round(close * 0.99, 2)
        entry_high = round(close * 1.01, 2)
        entry_note = "Vùng mua quanh giá hiện tại ±1%"

    if entry_low is None or entry_high is None or entry_low <= 0 or entry_high <= 0:
        return None
    entry_mid = (entry_low + entry_high) / 2.0

    # Cat lo: ho tro gan nhat duoi entry_low, floor toi thieu 3% va 2*ATR
    support_cands: list[float] = []
    for v in (sma20, sma50, sma200, (pivots or {}).get("s1"), (pivots or {}).get("s2"), w52_low):
        if v is not None and math.isfinite(v) and v < entry_low:
            support_cands.append(float(v))
    nearest_support = max(support_cands) if support_cands else None
    atr_stop = entry_low - 2 * atr14 if atr14 is not None and math.isfinite(atr14) and atr14 > 0 else None
    pct_stop = entry_low * 0.97
    # Stop = gan nhat, nhung phai dam bao cach it nhat 3% va 2*ATR
    if nearest_support is not None:
        stop = nearest_support
        # Enforce floor: stop phai <= min(entry_low*0.97, entry_low-2ATR) neu co
        floors = [x for x in (pct_stop, atr_stop) if x is not None and math.isfinite(x)]
        if floors:
            floor = min(floors)
            if stop > floor:
                stop = floor
    else:
        floors = [x for x in (atr_stop, pct_stop) if x is not None and math.isfinite(x)]
        stop = min(floors) if floors else round(entry_low * 0.97, 2)
    stop = round(float(stop), 2) if stop is not None and math.isfinite(stop) else None
    if stop is not None and stop >= entry_low:
        stop = round(min(pct_stop, atr_stop) if atr_stop is not None else pct_stop, 2)

    # Muc tieu: khang cu gan nhat tren entry_mid
    res_cands: list[float] = []
    for v in (sma20, sma50, sma200, (pivots or {}).get("r1"), (pivots or {}).get("r2"), w52_high):
        if v is not None and math.isfinite(v) and v > entry_mid:
            res_cands.append(float(v))
    nearest_res = min(res_cands) if res_cands else None
    r1 = float(nearest_res) if nearest_res is not None else None
    # r2 giu lai pivot r2 neu co va khac r1, de hien thi them muc tieu xa
    piv_r2 = (pivots or {}).get("r2")
    r2 = float(piv_r2) if piv_r2 is not None and math.isfinite(piv_r2) and piv_r2 != r1 else None
    if r2 is not None and r1 is not None and r2 <= r1:
        r2 = None

    def _pct(v: float | None) -> float | None:
        if v is None or not math.isfinite(v) or entry_mid == 0:
            return None
        return round((v - entry_mid) / entry_mid * 100.0, 2)

    rr1 = rr2 = None
    if stop is not None and entry_mid is not None and stop < entry_mid:
        risk = entry_mid - stop
        if risk > 0:
            if r1 is not None and r1 > entry_mid:
                rr1 = round((r1 - entry_mid) / risk, 2)
            if r2 is not None and r2 > entry_mid:
                rr2 = round((r2 - entry_mid) / risk, 2)

    return {
        "entry_type": entry_type,
        "entry_zone": {"low": entry_low, "high": entry_high, "mid": round(entry_mid, 2)},
        "entry_note": entry_note,
        "resistance": round(resistance, 2) if resistance is not None else None,
        "support": round(support, 2) if support is not None else None,
        "stop": stop,
        "stop_pct": _pct(stop),
        "targets": {"r1": r1, "r2": r2, "r1_pct": _pct(r1), "r2_pct": _pct(r2)},
        "rr": {"rr1": rr1, "rr2": rr2},
    }


def _load_json(path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _luc_buy_symbols() -> set[str]:
    data = _load_json(LUC_JSON)
    out: set[str] = set()
    for row in (data or {}).get("all_signals") or []:
        if row.get("status") == "VALID_BUY":
            sym = row.get("symbol")
            if sym:
                out.add(sym)
    return out


def _health_lookup() -> tuple[dict[str, dict], str | None]:
    data = _load_json(HEALTH_JSON)
    if not data:
        return {}, None
    return data.get("symbols") or {}, data.get("date")


def build_records(
    momentum_rows: list[dict],
    health: dict[str, dict],
    regime_label: str | None,
    streaks: dict[str, int] | None = None,
    confirm_sets: dict[str, set[str]] | None = None,
) -> list[dict]:
    streaks = streaks or {}
    confirm_sets = confirm_sets or {}
    factor = regime_factor_for(regime_label)
    records: list[dict] = []

    for row in momentum_rows:
        sym = row.get("symbol")
        if not sym:
            continue
        base_score = row.get("score") or 0
        if base_score < BASE_MIN:
            continue
        h = health.get(sym) or {}
        dist_ma20 = ((h.get("ma") or {}).get("dist") or {}).get("sma20")
        adx14_h = (h.get("osc") or {}).get("adx14")
        atr_pct = (h.get("risk") or {}).get("atr_pct")
        w52_pos = (h.get("w52") or {}).get("pos")

        scored = score_conviction(
            base_score,
            vol_ratio=row.get("vol_ratio"),
            adx14=adx14_h,
            dist_ma20=dist_ma20,
            regime_factor=factor,
            consecutive_days=streaks.get(sym, 0),
            atr_pct=atr_pct,
            w52_pos=w52_pos,
        )
        eligible_a = base_score < BASE_HOT_MIN
        tier = classify_tier(scored["conviction"], eligible_a)
        # Trade plan chi cho A/B (co du lieu health)
        trade_plan = None
        if tier in ("A", "B"):
            ma = h.get("ma") or {}
            piv = h.get("pivots")
            w52 = h.get("w52") or {}
            risk = h.get("risk") or {}
            trade_plan = compute_trade_plan(
                close=h.get("close"),
                sma10=ma.get("sma10"),
                sma20=ma.get("sma20"),
                sma50=ma.get("sma50"),
                sma200=ma.get("sma200"),
                atr14=risk.get("atr14"),
                dist_ma20=dist_ma20,
                pivots=piv if isinstance(piv, dict) else None,
                w52_high=w52.get("high"),
                w52_low=w52.get("low"),
            )
        records.append({
            "symbol": sym,
            "base_score": base_score,
            "conviction": scored["conviction"],
            "tier": tier,
            "eligible_for_a": eligible_a,
            "in_panel": False,  # gan sau khi sap xep + cap
            "components": scored["components"],
            "shadow": scored["shadow"],
            "dist_ma20": dist_ma20,
            "vol_ratio": row.get("vol_ratio"),
            "adx14": adx14_h,
            "last_price": row.get("last_price"),
            "confirm_sources": [],
            "trade_plan": trade_plan,
        })

    records.sort(key=lambda r: (-r["conviction"], r["symbol"]))
    a_rank = 0
    for i, rec in enumerate(records, start=1):
        rec["rank_in_day"] = i
        if rec["tier"] == "A":
            a_rank += 1
            rec["in_panel"] = a_rank <= CAP_A
        else:
            rec["in_panel"] = False
        srcs = confirm_sets.get(rec["symbol"]) or set()
        rec["confirm_sources"] = sorted(srcs)
    return records


def collect_confirm_sets() -> dict[str, set[str]]:
    """He dong thuan mua (chi hien thi, KHONG cong diem - backtest: edge am)."""
    sets: dict[str, set[str]] = {}

    def add(source: str, syms: set[str]):
        for sym in syms:
            sets.setdefault(sym, set()).add(source)

    ens = _load_json(ENSEMBLE_JSON) or {}
    add("ensemble", {r.get("symbol") for r in ens.get("all_signals") or [] if r.get("symbol")})
    add("luc_mach", _luc_buy_symbols())
    for path, name in ((K4_JSON, "k4"), (MAMA_JSON, "mama"), (ATS_JSON, "ats")):
        d = _load_json(path) or {}
        add(name, {r.get("symbol") for r in d.get("all_signals") or [] if r.get("symbol")})
    return sets


def append_history(date_str: str, records: list[dict]) -> None:
    history = []
    if HISTORY_JSON.exists():
        try:
            history = json.loads(HISTORY_JSON.read_text(encoding="utf-8"))
        except Exception:
            history = []
    slim = [{"symbol": r["symbol"], "conviction": r["conviction"], "tier": r["tier"]} for r in records]
    history = [h for h in history if h.get("date") != date_str]
    history.append({"date": date_str, "signals": slim})
    history = history[-HISTORY_CAP:]
    HISTORY_JSON.write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")


def main():
    tqdm.write("=" * 60)
    tqdm.write("Buy Conviction - Cham diem tin cay diem mua (lop tren)")
    tqdm.write("=" * 60)

    mom = _load_json(MOMENTUM_JSON)
    if not mom:
        tqdm.write("Khong co momentum_signals.json - bo qua conviction.")
        return
    rows = mom.get("all_signals") or []
    if not rows:
        tqdm.write("Momentum khong co tin hieu nao - ghi output rong.")
        date_str = mom.get("date") or format_market_date(signal_market_date())
        payload = {
            "generated_at": vn_now().isoformat(), "date": date_str,
            "regime": {"label": None, "factor": 1.0},
            "tier_thresholds": {"A": TIER_A_MIN, "B": TIER_B_MIN}, "cap_a": CAP_A,
            "counts": {"A": 0, "B": 0, "C": 0, "in_panel": 0}, "signals": [],
        }
        OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        append_history(date_str, [])
        return

    date_str = mom.get("date") or format_market_date(signal_market_date())
    regime_data = _load_json(REGIME_JSON) or {}
    regime_label = (regime_data.get("regime") or {}).get("label")
    factor = regime_factor_for(regime_label)

    health, _ = _health_lookup()
    prev_history = []
    if HISTORY_JSON.exists():
        try:
            prev_history = json.loads(HISTORY_JSON.read_text(encoding="utf-8"))
        except Exception:
            prev_history = []
    streaks = compute_streaks(prev_history)
    confirm_sets = collect_confirm_sets()

    records = build_records(rows, health, regime_label, streaks, confirm_sets)

    counts = {"A": 0, "B": 0, "C": 0, "in_panel": 0}
    for r in records:
        counts[r["tier"]] += 1
        if r["in_panel"]:
            counts["in_panel"] += 1

    payload = {
        "generated_at": vn_now().isoformat(),
        "date": date_str,
        "regime": {"label": regime_label, "factor": factor},
        "tier_thresholds": {"A": TIER_A_MIN, "B": TIER_B_MIN},
        "cap_a": CAP_A,
        "counts": counts,
        "signals": records,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
    OUTPUT_JSON.write_text(text, encoding="utf-8")
    DOCS_OUT = DOCS_DATA_DIR / OUTPUT_JSON.name
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_OUT.write_bytes(OUTPUT_JSON.read_bytes())
    append_history(date_str, records)

    tqdm.write(f"He so regime: {regime_label} x{factor}")
    tqdm.write(f"Hang A: {counts['A']} (vao panel: {counts['in_panel']}), B: {counts['B']}, C: {counts['C']}")
    for r in records[:10]:
        tqdm.write(f"  #{r['rank_in_day']:2d} {r['symbol']:6s} conv={r['conviction']:3d} [{r['tier']}] base={r['base_score']:.0f} vol={r['vol_ratio']} adx={r['adx14']} dMA20={r['dist_ma20']}")
    tqdm.write(f"Da ghi: {OUTPUT_JSON.name} + history")


if __name__ == "__main__":
    main()
