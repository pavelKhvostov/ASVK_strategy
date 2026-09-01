"""run_fractal12h — оркестратор ASVK-portable фрактал-12h пайплайна для одного символа.

Запускает по очереди: A-cascade → B1 (FVG) → B2 (Order Block, B2C1) → B3 (Fractal
Liquidity) → B4 (HMA-200) → B5 (VWAP, B5C1) → B8 (Power Zone, B8C1) → B9 (Others) →
Basket (B1∪B2∪B3∪B4∪B5∪B8∪B9),
переиспользуя загруженные 1m/12h/15m данные между шагами (один load_1m() на весь
прогон вместо четырёх отдельных процессов, как в WSL-версии — экономит время старта
bundled Python при вызове из asvk_pipeline.py каждые 15 минут).

Usage:
  python run_fractal12h.py --symbol BTC --start 2020-01-01 --end 2026-07-24
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import load_1m, agg_12h, agg_15m, DATA_OUT
import a_cascade
import b1_fvg
import b2_ob
import b3_fractal_liquidity
import b4_hma
import b5_vwap
import b8_power_zone
import b9_others
import basket as basket_mod
import b_structure as bstruct_mod
import b6_divergence as b6_mod
import b7_money_hands as b7_mod
import decision as decision_mod
import verdict as verdict_mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", required=True)
    args = ap.parse_args()

    sym, start, end = args.symbol, args.start, args.end
    UTC = timezone.utc
    start_ms = int(datetime.fromisoformat(start).replace(tzinfo=UTC).timestamp() * 1000)
    end_ms   = int(datetime.fromisoformat(end).replace(tzinfo=UTC).timestamp() * 1000)

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"═══ run_fractal12h: {sym} {start} → {end} ═══", file=sys.stderr, flush=True)

    df_1m = load_1m(sym)
    df_12h = agg_12h(df_1m)
    df_15m = agg_15m(df_1m)
    print(f"  12h bars: {len(df_12h):,}   15m bars: {len(df_15m):,}", file=sys.stderr, flush=True)

    # ── A cascade ──
    print(f"\n── A cascade ──", file=sys.stderr, flush=True)
    cand = a_cascade.compute_a_cascade(df_12h)
    mask = (cand["pivot_open_ts_ms"] >= start_ms) & (cand["pivot_open_ts_ms"] < end_ms)
    cand = cand[mask].reset_index(drop=True)
    a_cascade.print_stage_stats(cand)
    a_path = DATA_OUT / f"a_candidates_{sym}_{start}_{end}.parquet"
    cand.to_parquet(a_path, index=False, compression="zstd", compression_level=9)
    print(f"  written: {a_path.name}  ({len(cand):,} rows)", file=sys.stderr, flush=True)

    # ── B1 FVG ──
    print(f"\n── B1 FVG ──", file=sys.stderr, flush=True)
    zones = b1_fvg.load_fvg_zones(sym)
    print(f"  FVG zones (TFs={b1_fvg.FVG_TFS}): {len(zones):,}", file=sys.stderr, flush=True)
    hits_b1 = b1_fvg.compute_b1(cand, zones, df_12h, sym)
    b1_fvg.print_stats(hits_b1)
    b1_path = DATA_OUT / f"b1_hits_{sym}_{start}_{end}.parquet"
    hits_b1.to_parquet(b1_path, index=False, compression="zstd", compression_level=9)
    print(f"  written: {b1_path.name}  ({len(hits_b1):,} rows)", file=sys.stderr, flush=True)

    # ── B2 Order Block (b2_hit = B2C1; B2C2 считается отдельно, в basket не входит) ──
    print(f"\n── B2 OB (B2C1 production + B2C2 отдельно) ──", file=sys.stderr, flush=True)
    bo_zones = b2_ob.load_block_orders_zones(sym)
    ol_zones = b2_ob.load_ob_liq_zones(sym)
    print(f"  block_orders zones (TFs={b2_ob.OB_TFS}): {len(bo_zones):,}", file=sys.stderr, flush=True)
    print(f"  ob_liq zones (TFs={b2_ob.OB_TFS}):       {len(ol_zones):,}", file=sys.stderr, flush=True)
    hits_b2 = b2_ob.compute_b2(cand, bo_zones, ol_zones, df_12h)
    b2_ob.print_stats(hits_b2)
    b2_path = DATA_OUT / f"b2_hits_{sym}_{start}_{end}.parquet"
    hits_b2.to_parquet(b2_path, index=False, compression="zstd", compression_level=9)
    print(f"  written: {b2_path.name}  ({len(hits_b2):,} rows)", file=sys.stderr, flush=True)

    # ── B3 Fractal Liquidity ──
    print(f"\n── B3 Fractal Liquidity ──", file=sys.stderr, flush=True)
    hits_b3 = b3_fractal_liquidity.compute_b3(cand, df_12h, sym)
    b3_fractal_liquidity.print_stats(hits_b3)
    b3_path = DATA_OUT / f"b3_hits_{sym}_{start}_{end}.parquet"
    hits_b3.to_parquet(b3_path, index=False, compression="zstd", compression_level=9)
    print(f"  written: {b3_path.name}  ({len(hits_b3):,} rows)", file=sys.stderr, flush=True)

    # ── B4 HMA (B4C1 + B4C2) ──
    print(f"\n── B4 HMA ──", file=sys.stderr, flush=True)
    hits_b4 = b4_hma.compute_b4(cand, df_12h, sym)
    b4_hma.print_stats(hits_b4)
    b4_path = DATA_OUT / f"b4_hits_{sym}_{start}_{end}.parquet"
    hits_b4.to_parquet(b4_path, index=False, compression="zstd", compression_level=9)
    print(f"  written: {b4_path.name}  ({len(hits_b4):,} rows)", file=sys.stderr, flush=True)

    # ── B5 VWAP (B5C1 only) ──
    print(f"\n── B5 VWAP (B5C1) ──", file=sys.stderr, flush=True)
    anchors = b5_vwap.load_vwap_anchors(sym)
    print(f"  W-aligned VWAP anchors: {len(anchors):,}", file=sys.stderr, flush=True)
    hits_b5 = b5_vwap.compute_b5(cand, df_1m, df_12h, anchors)
    b5_vwap.print_stats(hits_b5)
    b5_path = DATA_OUT / f"b5_hits_{sym}_{start}_{end}.parquet"
    hits_b5.to_parquet(b5_path, index=False, compression="zstd", compression_level=9)
    print(f"  written: {b5_path.name}  ({len(hits_b5):,} rows)", file=sys.stderr, flush=True)

    # ── B8 Power Zone (B8C1 only) ──
    print(f"\n── B8 Power Zone (B8C1) ──", file=sys.stderr, flush=True)
    s7d = b8_power_zone.load_s7d(sym)
    print(f"  s7d loaded: {len(s7d):,} rows", file=sys.stderr, flush=True)
    hits_b8 = b8_power_zone.compute_b8(cand, s7d)
    b8_power_zone.print_stats(hits_b8)
    b8_path = DATA_OUT / f"b8_hits_{sym}_{start}_{end}.parquet"
    hits_b8.to_parquet(b8_path, index=False, compression="zstd", compression_level=9)
    print(f"  written: {b8_path.name}  ({len(hits_b8):,} rows)", file=sys.stderr, flush=True)

    # ── B9 Others ──
    print(f"\n── B9 Others ──", file=sys.stderr, flush=True)
    hits_b9 = b9_others.compute_b9(cand, df_12h, df_15m, sym)
    b9_others.print_stats(hits_b9)
    b9_path = DATA_OUT / f"b9_hits_{sym}_{start}_{end}.parquet"
    hits_b9.to_parquet(b9_path, index=False, compression="zstd", compression_level=9)
    print(f"  written: {b9_path.name}  ({len(hits_b9):,} rows)", file=sys.stderr, flush=True)

    # ── Basket (B1 ∪ B2 ∪ B3 ∪ B4 ∪ B5 ∪ B8 ∪ B9) ──
    print(f"\n── Basket (B1∪B2∪B3∪B4∪B5∪B8∪B9) ──", file=sys.stderr, flush=True)
    merged = basket_mod.load_block_hits(sym, start, end)
    merged = basket_mod.compute_basket(merged)
    basket_mod.print_stats(merged)
    basket_path = DATA_OUT / f"basket_hits_{sym}_{start}_{end}.parquet"
    merged.to_parquet(basket_path, index=False, compression="zstd", compression_level=9)
    print(f"  written: {basket_path.name}  ({len(merged):,} rows)", file=sys.stderr, flush=True)

    # ── B-структура (BOS/CHoCH/sweep — ядро метода Арденского, причинно на 12h) ──
    print(f"\n── B-structure (BOS/CHoCH) ──", file=sys.stderr, flush=True)
    bstruct = bstruct_mod.compute_bstruct(cand, df_12h)
    bstruct_mod.print_stats(bstruct)
    bstruct_path = DATA_OUT / f"bstruct_hits_{sym}_{start}_{end}.parquet"
    bstruct.to_parquet(bstruct_path, index=False, compression="zstd", compression_level=9)
    print(f"  written: {bstruct_path.name}  ({len(bstruct):,} rows)", file=sys.stderr, flush=True)

    # ── B6 RSI-дивергенция (канон-блок B6) ──
    print(f"\n── B6 RSI-divergence ──", file=sys.stderr, flush=True)
    b6 = b6_mod.compute_b6(cand, df_12h)
    b6_mod.print_stats(b6)
    b6_path = DATA_OUT / f"b6_hits_{sym}_{start}_{end}.parquet"
    b6.to_parquet(b6_path, index=False, compression="zstd", compression_level=9)
    print(f"  written: {b6_path.name}  ({len(b6):,} rows)", file=sys.stderr, flush=True)

    # ── B7 Money Hands (канон-блок B7: климакс-объём + поглощение) ──
    print(f"\n── B7 Money Hands ──", file=sys.stderr, flush=True)
    b7 = b7_mod.compute_b7(cand, df_12h)
    b7_mod.print_stats(b7)
    b7_path = DATA_OUT / f"b7_hits_{sym}_{start}_{end}.parquet"
    b7.to_parquet(b7_path, index=False, compression="zstd", compression_level=9)
    print(f"  written: {b7_path.name}  ({len(b7):,} rows)", file=sys.stderr, flush=True)

    # ── Decision (слияние: зона + подтверждение(+B7) + структура + дивергенция) ──
    print(f"\n── Decision (Arden-confluence + structure + RSI + money-hands) ──", file=sys.stderr, flush=True)
    decided = decision_mod.compute_decision(merged, bstruct, b6, b7)
    decision_mod.print_stats(decided)
    decision_path = DATA_OUT / f"decision_hits_{sym}_{start}_{end}.parquet"
    decided.to_parquet(decision_path, index=False, compression="zstd", compression_level=9)
    print(f"  written: {decision_path.name}  ({len(decided):,} rows)", file=sys.stderr, flush=True)

    # ── Живой вердикт: LONG / SHORT / FLAT на текущей свече ──
    v = verdict_mod.compute_verdict(sym, start, end)
    verdict_mod.print_verdict(v)

    print(f"\n═══ run_fractal12h done in {time.time()-t0:.1f}s ═══", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
