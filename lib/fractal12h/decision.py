"""decision — слой принятия решения над корзиной B-блоков (ASVK-portable).

Идея (метод Арденского → ASVK). Текущий basket = OR(B1..B9): любое одно
подтверждение делает пивот сигналом. На BTC это даёт WR ~74% — ниже, чем у
сильных одиночных блоков (B4 84%, B9 84%, B5 82%, B8 81%, B2 80%): OR
разбавляет их слабыми одиночными срабатываниями.

Методология Арденского прямо предписывает лечение: сетап — это НЕ одно условие,
а СЛИЯНИЕ нескольких обязательных факторов ("усиление PoT: 3 фактора → 90%",
"магия зон: слияние H1+H4+D1 → 70.5%", и все 7 сетапов перечисляют по несколько
обязательных факторов). Анатомия любого сетапа: ЗОНА + ТРИГГЕР(снятие ликвидности)
+ ПОДТВЕРЖДЕНИЕ. Здесь мы кодируем это как решение "сформировался сетап или нет"
в той же булевой логике пивота, что и basket — БЕЗ tp/sl, только попал/не попал.

Роли B-блоков (по методичкам order-block.pdf / imb.pdf / magia-zon.pdf / setapy.pdf):
    ЗОНА          b1 (FVG/имбаланс), b2 (Order Block)              — «зона интереса»
    ЛИКВИДНОСТЬ    b3 (fractal liquidity, maxV-sweep)              — «ложный вынос/снятие»
    ПОДТВЕРЖДЕНИЕ  b4 (MA), b5 (VWAP), b8 (power zone), b9 (momentum/climax)

Решения (все — булев признак на пивоте, WR меряется тем же confirmed/confirmable
Williams-n2, что и у basket — outcome «фрактал устоял»):
    decision_k2      confluence ≥ 2   (любые 2 блока)         — простой порог, для сравнения
    decision_k3      confluence ≥ 3   (любые 3 блока)
    decision_setup   ЗОНА & ПОДТВЕРЖДЕНИЕ                     — минимальный сетап Арденского (production)
    decision_pot     ЗОНА & ЛИКВИДНОСТЬ & ПОДТВЕРЖДЕНИЕ       — полный Power of Three (сильнейший)
    decision_hit  =  decision_setup                          — production-сигнал этого слоя

Reads:
  data/fractal12h/basket_hits_{SYM}_{start}_{end}.parquet   (все b*_hit + basket_hit)
Writes:
  data/fractal12h/decision_hits_{SYM}_{start}_{end}.parquet
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import DATA_OUT
from basket import HIT_COLS   # ["b1_hit".."b9_hit"], единый источник списка блоков


# ── Роли блоков в анатомии сетапа Арденского (зона + триггер + подтверждение) ──
ZONE_COLS    = ["b1_hit", "b2_hit"]                       # FVG / Order Block
LIQ_COLS     = ["b3_hit"]                                 # fractal liquidity (снятие / maxV-sweep)
CONFIRM_COLS = ["b4_hit", "b5_hit", "b8_hit", "b9_hit"]   # MA / VWAP / power zone / momentum


STRUCT_COLS = ["bstruct_bos", "bstruct_choch"]   # опционально, если b_structure посчитан
B6_COLS     = ["b6_hit"]                          # опционально, если b6_divergence посчитан
B7_COLS     = ["b7_hit"]                          # опционально, если b7_money_hands посчитан


def _merge_extra(h: pd.DataFrame, extra: pd.DataFrame | None, cols: list[str]) -> pd.DataFrame:
    if extra is not None:
        h = h.merge(extra[["pivot_open_ts_ms", "direction"] + cols],
                    on=["pivot_open_ts_ms", "direction"], how="left")
    for col in cols:
        if col not in h.columns:
            h[col] = False
        h[col] = h[col].fillna(False).astype(bool)
    return h


def compute_decision(hits: pd.DataFrame, bstruct: pd.DataFrame | None = None,
                     b6: pd.DataFrame | None = None, b7: pd.DataFrame | None = None) -> pd.DataFrame:
    """Добавляет колонки решения поверх basket_hits (b*_hit уже fillna(False)).

    Опционально подмешивает b_structure (BOS/CHoCH), b6_divergence (RSI) и
    b7_money_hands (объём) — их tiers появляются только если блок передан.
    b7 — полноценный confirm (климакс+поглощение), входит в decision_confirm."""
    h = hits.copy()
    for col in HIT_COLS:
        if col not in h.columns:
            h[col] = False
        h[col] = h[col].fillna(False).astype(bool)
    h = _merge_extra(h, bstruct, STRUCT_COLS)
    h = _merge_extra(h, b6, B6_COLS)
    h = _merge_extra(h, b7, B7_COLS)

    h["confluence"]       = h[HIT_COLS].sum(axis=1).astype("int16")
    h["decision_zone"]    = h[ZONE_COLS].any(axis=1)
    h["decision_liq"]     = h[LIQ_COLS].any(axis=1)
    h["decision_confirm"] = h[CONFIRM_COLS].any(axis=1)   # базовый confirm (высокий WR-запас)

    h["decision_k2"]      = h["confluence"] >= 2
    h["decision_k3"]      = h["confluence"] >= 3
    h["decision_setup"]   = h["decision_zone"] & h["decision_confirm"]
    h["decision_pot"]     = h["decision_zone"] & h["decision_liq"] & h["decision_confirm"]

    # ── структурно-усиленные tiers (b_structure) ──
    liq_ext               = h["decision_liq"] | h["bstruct_bos"]
    h["decision_pot_s"]   = h["decision_zone"] & liq_ext & h["decision_confirm"]
    h["decision_choch"]   = h["decision_zone"] & h["decision_confirm"] & h["bstruct_choch"]

    # ── дивергенция (b6) и money-hands (b7) как расширители охвата ──
    confirm_wide          = h["decision_confirm"] | h["b6_hit"] | h["b7_hit"]
    h["decision_setup_x"] = h["decision_zone"] & (h["decision_confirm"] | h["b6_hit"])  # +RSI
    h["decision_setup_w"] = h["decision_zone"] & confirm_wide                            # +RSI +money-hands (шире)
    h["decision_div"]     = h["decision_zone"] & h["decision_confirm"] & h["b6_hit"]     # дивергенция доп. фактором
    h["decision_mh"]      = h["decision_zone"] & h["decision_confirm"] & h["b7_hit"]     # money-hands доп. фактором

    # production: сетап зона&подтв, ПЛЮС ловим сильные CHoCH-развороты, что сетап пропустил
    h["decision_hit"]     = h["decision_setup"] | (h["decision_zone"] & h["bstruct_choch"])

    # ── ВОЗВРАТ сильных отсеянных сигналов (стратификация отсева, см. ROADMAP) ──
    # Инсайт: сильное подтверждение само по себе достаточно, зона не обязательна.
    # Сильные страты отсева: b9 без зоны (80.5%), ≥2 подтверждения (81%), CHoCH (90%).
    # Балласт (НЕ возвращаем): зона-без-подтв (58%), одиночная ликвидность b3 (61%), b5-alone (72%).
    nconf                 = h[CONFIRM_COLS].sum(axis=1)
    strong_confirm        = h["b9_hit"] | (nconf >= 2) | h["bstruct_choch"]
    recover_a             = (~h["decision_hit"]) & (~h["decision_zone"]) & strong_confirm
    h["decision_wide"]    = h["decision_hit"] | recover_a          # A: +72% сигналов, WR ~82%
    recover_b             = recover_a | ((~h["decision_hit"]) & (~h["decision_zone"]) & h["b4_hit"])
    h["decision_wide_b"]  = h["decision_hit"] | recover_b          # B: ~×2 сигналов, WR ~80%

    # ── ГРЕЙД УВЕРЕННОСТИ сигнала (по данным: чем больше совпавших факторов, тем выше WR) ──
    # 1 low ~74% · 2 med ~80% · 3 high ~85% · 4 very-high ~88% · 5 premium(CHoCH) ~96%
    conf = h["confluence"]
    grade = np.where(conf <= 1, 1, np.where(conf == 2, 2, np.where(conf == 3, 3, 4)))
    grade = np.where(h["bstruct_choch"], 5, grade)
    h["signal_grade"] = grade.astype("int8")
    h["signal_wr_est"] = pd.Series(grade, index=h.index).map(GRADE_WR).astype("float32")
    return h


# эмпирические ориентиры WR по грейду (pooled BTC/ETH/SOL, 2020-2026)
GRADE_WR = {1: 74.0, 2: 80.0, 3: 85.0, 4: 88.0, 5: 96.0}
GRADE_LABEL = {1: "LOW", 2: "MED", 3: "HIGH", 4: "V.HIGH", 5: "PREMIUM"}


def print_stats(hits: pd.DataFrame) -> None:
    def _wr(mask_col: str) -> tuple[int, int, int, float]:
        m = hits[mask_col]
        n = int(m.sum())
        cm = m & hits["confirmable"]
        n_c = int(cm.sum())
        n_conf = int(hits.loc[cm, "confirmed"].sum())
        wr = 100.0 * n_conf / n_c if n_c else 0.0
        return n, n_conf, n_c, wr

    base_m = hits["confirmable"]
    base_n = int(base_m.sum())
    base_wr = 100.0 * int(hits.loc[base_m, "confirmed"].sum()) / base_n if base_n else 0.0

    print(f"\n=== DECISION (слияние Арденского поверх B1..B9) ===", file=sys.stderr, flush=True)
    print(f"  A1 baseline   n={base_n:>4,}                 WR={base_wr:5.2f}%", file=sys.stderr, flush=True)
    order = [
        ("basket_hit",       "OR(B1..B9)"),
        ("decision_k2",      "confluence≥2"),
        ("decision_k3",      "confluence≥3"),
        ("decision_setup",   "зона&подтв (база)"),
        ("decision_setup_w", "+RSI+money (шире)"),
        ("decision_pot",     "зона&ликв&подтв"),
        ("decision_div",     "зона&подтв&дивер"),
        ("decision_mh",      "зона&подтв&money"),
        ("decision_choch",   "зона&подтв&CHoCH"),
        ("decision_hit",     "PRODUCTION (setup∪CHoCH)"),
        ("decision_wide",    "WIDE-A (+возврат b9/≥2/CHoCH)"),
        ("decision_wide_b",  "WIDE-B (+b4)"),
    ]
    for col, label in order:
        if col not in hits.columns:
            continue
        n, n_conf, n_c, wr = _wr(col)
        sel = 100.0 * n / base_n if base_n else 0.0
        mark = "  ✓>75" if wr > 75.0 else ""
        print(f"  {label:18s} n={n:>4,}  conf={n_conf:>3,}/{n_c:>3,}  "
              f"WR={wr:5.2f}%  Δ={wr-base_wr:+5.2f}pp  sel={sel:4.1f}%{mark}",
              file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", required=True)
    args = ap.parse_args()

    print(f"decision (Arden-confluence over B1..B9): {args.symbol} {args.start} → {args.end}",
          file=sys.stderr, flush=True)
    t0 = time.time()

    basket_path = DATA_OUT / f"basket_hits_{args.symbol}_{args.start}_{args.end}.parquet"
    if not basket_path.exists():
        raise FileNotFoundError(f"Missing: {basket_path} (запусти сначала basket / run_fractal12h)")
    hits = pd.read_parquet(basket_path)
    print(f"  loaded basket {len(hits):,} rows", file=sys.stderr, flush=True)

    def _opt(prefix: str) -> pd.DataFrame | None:
        p = DATA_OUT / f"{prefix}_hits_{args.symbol}_{args.start}_{args.end}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            print(f"  loaded {prefix} {len(df):,} rows", file=sys.stderr, flush=True)
            return df
        print(f"  ({prefix} не найден — его tiers пропущены)", file=sys.stderr, flush=True)
        return None

    bstruct = _opt("bstruct")
    b6 = _opt("b6")
    b7 = _opt("b7")
    hits = compute_decision(hits, bstruct, b6, b7)
    print_stats(hits)

    out = DATA_OUT / f"decision_hits_{args.symbol}_{args.start}_{args.end}.parquet"
    hits.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"\nwritten: {out.name}  ({len(hits):,} rows)  total {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
