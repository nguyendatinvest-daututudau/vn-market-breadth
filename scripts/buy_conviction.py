"""
Buy Conviction Score — Lop cham diem tin cay diem mua nam TREN cac he hien tai.

Nguyen tac bat bien: KHONG sua logic sinh tin hieu cua bat ky generator nao.
Module nay chi DOC ket qua co san (momentum + cac he khac + stock_health +
market_regime) roi ghep lai thanh diem conviction 0-100 va xep hang A/B/C.

Cong thuc v2 (trong so khoi tao, se duoc chot lai sau buoc calibration):
  - Nen: score momentum (band 30-59). Score >= 60 ("hot") van duoc tinh diem
         nhung bi loai khoi hang A (backtest: edge am OOS o bracket cao).
  - +15 neu vol_ratio >= 2.0   (backtest lift +5.9pp)
  - +10 neu adx14 > 28         (backtest lift +4.1pp)
  - -15 neu dist_ma20 > 4.5%   (phat gia da dui - mua muon)
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
EXT_PENALTY_POINTS = -15
TIER_A_MIN = 60
TIER_B_MIN = 45
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
        for sym in syms:
            streaks[sym] = streaks.get(sym, 0) + 1
        # Chi dem chuoi lien tiep: dung o phien dau tien bi gian doan.
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

    conviction = int(round(total * regime_factor))
    return {"conviction": conviction, "components": components, "shadow": shadow}


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
        })

    records.sort(key=lambda r: (-r["conviction"], r["symbol"]))
    for i, rec in enumerate(records, start=1):
        rec["rank_in_day"] = i
        rec["in_panel"] = rec["tier"] == "A" and i <= CAP_A
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
