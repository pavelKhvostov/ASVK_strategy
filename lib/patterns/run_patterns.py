"""run_patterns — оркестратор ASVK-portable "Block 5" (25 паттернов:
13 SHORT + 12 LONG антагонистов) для одного символа.

Usage:
  python run_patterns.py --symbol BTC --start 2020-01-01 --end 2026-07-26
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import load_1m, agg_1h, DATA_OUT
import hs_top
import sym_triangle
import descending_triangle
import rising_wedge
import rising_channel
import rectangle_top
import triple_top
import gartley
import bat
import butterfly
import crab
import gartley_long as gartley_long_mod
import bat_long as bat_long_mod
import butterfly_long as butterfly_long_mod
import crab_long as crab_long_mod
import ascending_triangle as ascending_triangle_mod
import sym_triangle_long as sym_triangle_long_mod
import falling_wedge as falling_wedge_mod
import falling_channel as falling_channel_mod
import rectangle_bottom as rectangle_bottom_mod
import triple_bottom as triple_bottom_mod
import hs_bottom as hs_bottom_mod
import patterns_basket

SHORT_PATTERNS = (
    ("sym_triangle",        sym_triangle.compute_sym_triangle),
    ("descending_triangle", descending_triangle.compute_descending_triangle),
    ("rising_wedge",        rising_wedge.compute_rising_wedge),
    ("rising_channel",      rising_channel.compute_rising_channel),
    ("rectangle_top",       rectangle_top.compute_rectangle_top),
    ("triple_top",          triple_top.compute_triple_top),
    ("gartley",             gartley.compute_gartley),
    ("bat",                 bat.compute_bat),
    ("butterfly",           butterfly.compute_butterfly),
    ("crab",                crab.compute_crab),
)

LONG_PATTERNS = (
    ("ascending_triangle",  ascending_triangle_mod.compute_ascending_triangle),
    ("sym_triangle_long",   sym_triangle_long_mod.compute_sym_triangle_long),
    ("falling_wedge",       falling_wedge_mod.compute_falling_wedge),
    ("falling_channel",     falling_channel_mod.compute_falling_channel),
    ("gartley_long",        gartley_long_mod.compute_gartley_long),
    ("bat_long",            bat_long_mod.compute_bat_long),
    ("butterfly_long",      butterfly_long_mod.compute_butterfly_long),
    ("crab_long",           crab_long_mod.compute_crab_long),
    # LONG группа 3 (bottom-family)
    ("rectangle_bottom",    rectangle_bottom_mod.compute_rectangle_bottom),
    ("triple_bottom",       triple_bottom_mod.compute_triple_bottom),
    ("hs_bottom",           hs_bottom_mod.compute_hs_bottom),
)


def _run_one(name, compute_fn, df_1h, sym, start, end, start_ms, end_ms):
    t1 = time.time()
    hits, stats = compute_fn(df_1h, sym)
    hits = hits[(hits["signal_ts"] >= start_ms) & (hits["signal_ts"] < end_ms)].reset_index(drop=True)
    print(f"  stats: {stats}", file=sys.stderr, flush=True)
    path = DATA_OUT / f"{name}_hits_{sym}_{start}_{end}.parquet"
    hits.to_parquet(path, index=False, compression="zstd", compression_level=9)
    print(f"  written: {path.name}  ({len(hits):,} rows)  [{time.time()-t1:.1f}s]",
          file=sys.stderr, flush=True)


def main():
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
    print(f"═══ run_patterns: {sym} {start} → {end} ═══", file=sys.stderr, flush=True)

    df_1m = load_1m(sym)
    df_1h = agg_1h(df_1m)
    print(f"  1h bars: {len(df_1h):,}", file=sys.stderr, flush=True)

    # ── H&S TOP ──
    print(f"\n── H&S TOP ──", file=sys.stderr, flush=True)
    hs_hits, hs_stats = hs_top.compute_hs(df_1h, sym)
    hs_hits = hs_hits[(hs_hits["signal_ts"] >= start_ms) & (hs_hits["signal_ts"] < end_ms)].reset_index(drop=True)
    hs_top.print_stats(hs_stats)
    hs_path = DATA_OUT / f"hs_hits_{sym}_{start}_{end}.parquet"
    hs_hits.to_parquet(hs_path, index=False, compression="zstd", compression_level=9)
    print(f"  written: {hs_path.name}  ({len(hs_hits):,} rows)", file=sys.stderr, flush=True)


    # ── SHORT-паттерны ──
    print(f"\n── SHORT patterns ──", file=sys.stderr, flush=True)
    for name, fn in SHORT_PATTERNS:
        print(f"\n  {name}", file=sys.stderr, flush=True)
        _run_one(name, fn, df_1h, sym, start, end, start_ms, end_ms)

    # ── LONG-паттерны (антагонисты) ──
    print(f"\n── LONG patterns (antagonists) ──", file=sys.stderr, flush=True)
    for name, fn in LONG_PATTERNS:
        print(f"\n  {name}", file=sys.stderr, flush=True)
        _run_one(name, fn, df_1h, sym, start, end, start_ms, end_ms)

    # ── Patterns basket (union, 21 паттернов) ──
    print(f"\n── Patterns basket ──", file=sys.stderr, flush=True)
    merged = patterns_basket.load_patterns_hits(sym, start, end)
    patterns_basket.print_stats(merged)
    basket_path = DATA_OUT / f"patterns_hits_{sym}_{start}_{end}.parquet"
    merged.to_parquet(basket_path, index=False, compression="zstd", compression_level=9)
    print(f"  written: {basket_path.name}  ({len(merged):,} rows)", file=sys.stderr, flush=True)

    print(f"\n═══ run_patterns done in {time.time()-t0:.1f}s ═══", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
