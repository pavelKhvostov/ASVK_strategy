"""run_patterns — оркестратор ASVK-portable "Block 5" (22 паттерна:
11 SHORT + 11 LONG антагонистов) для одного символа.

Инкрементальный режим: сканирует только последние LOOKBACK_BARS 1h-баров.
Старые хиты (до окна сканирования) сохраняются из предыдущего parquet.
Время: ~20-30s на символ вместо 1500s.

Usage:
  python run_patterns.py --symbol BTC --start 2020-01-01 --end 2026-07-28
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import time
from datetime import datetime, timezone

import pandas as pd

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

# Сколько 1h-баров считается «новыми» (граница merge).
LOOKBACK_BARS = 600
# Контекст для паттернов, у которых начало (X) до окна LOOKBACK_BARS.
# Harmonics: MAX_SPAN_XD = 504 bars. Загружаем LOOKBACK_BARS + CONTEXT_BARS суммарно.
CONTEXT_BARS = 504

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
    ("rectangle_bottom",    rectangle_bottom_mod.compute_rectangle_bottom),
    ("triple_bottom",       triple_bottom_mod.compute_triple_bottom),
    ("hs_bottom",           hs_bottom_mod.compute_hs_bottom),
)


def _merge_hits(name: str, sym: str, start: str, end: str,
                hits_new: pd.DataFrame, lookback_start_ms: int,
                start_ms: int, end_ms: int) -> pd.DataFrame:
    """Загружает старые хиты из предыдущего parquet (до окна сканирования),
    объединяет с новыми, фильтрует по [start_ms, end_ms)."""
    path = DATA_OUT / f"{name}_hits_{sym}_{start}_{end}.parquet"
    # Ищем любой существующий файл для этого паттерна+символа
    existing = list(DATA_OUT.glob(f"{name}_hits_{sym}_*.parquet"))
    if existing:
        latest = max(existing, key=lambda p: p.stem.split("_")[-1])
        hits_old = pd.read_parquet(latest)
        hits_old = hits_old[hits_old["signal_ts"] < lookback_start_ms]
        if not hits_new.empty:
            hits_new = hits_new[hits_new["signal_ts"] >= lookback_start_ms]
        hits = pd.concat([hits_old, hits_new], ignore_index=True)
    else:
        hits = hits_new
    if hits.empty or "signal_ts" not in hits.columns:
        hits = pd.DataFrame(columns=hits.columns if len(hits.columns) else ["signal_ts"])
        hits.to_parquet(path, index=False, compression="zstd", compression_level=9)
        return hits
    hits = (hits[(hits["signal_ts"] >= start_ms) & (hits["signal_ts"] < end_ms)]
            .sort_values("signal_ts")
            .reset_index(drop=True))
    hits.to_parquet(path, index=False, compression="zstd", compression_level=9)
    return hits


def _run_one(name, compute_fn, df_1h_scan, sym, start, end,
             start_ms, end_ms, lookback_start_ms):
    t1 = time.time()
    hits_new, stats = compute_fn(df_1h_scan, sym)
    hits = _merge_hits(name, sym, start, end, hits_new, lookback_start_ms, start_ms, end_ms)
    path = DATA_OUT / f"{name}_hits_{sym}_{start}_{end}.parquet"
    print(f"  stats: {stats}", file=sys.stderr, flush=True)
    print(f"  written: {path.name}  ({len(hits):,} rows, {len(hits_new):,} в окне)  [{time.time()-t1:.1f}s]",
          file=sys.stderr, flush=True)


def _needs_full_scan(sym: str, lookback_start_ms: int) -> bool:
    """Проверяет нужен ли full-scan.
    True если нет данных старше границы lookback — значит история утеряна
    (например удалены parquet-файлы или первый запуск после оптимизации).
    Использует bat_hits как reference — паттерн с наибольшим покрытием истории."""
    ref_files = list(DATA_OUT.glob(f"bat_hits_{sym}_*.parquet"))
    if not ref_files:
        return True
    try:
        latest = max(ref_files, key=lambda p: p.stat().st_mtime)
        df = pd.read_parquet(latest, columns=["signal_ts"])
        if df.empty:
            return True
        return int(df["signal_ts"].min()) >= lookback_start_ms
    except Exception:
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", required=True)
    ap.add_argument("--full-scan", action="store_true",
                    help="Сканировать всю историю (разовое восстановление, медленно)")
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
    print(f"  1h bars total: {len(df_1h):,}", file=sys.stderr, flush=True)

    # Определяем режим сканирования.
    # Инкрементальный по умолчанию, full-scan если:
    #   - передан флаг --full-scan, или
    #   - нет исторических данных вне lookback-окна (история утеряна / первый запуск)
    n_total = min(len(df_1h), LOOKBACK_BARS + CONTEXT_BARS)
    lookback_start_ms = int(df_1h.iloc[-LOOKBACK_BARS]["ts"])
    if args.full_scan or _needs_full_scan(sym, lookback_start_ms):
        scan_mode = "FULL-SCAN (--full-scan)" if args.full_scan else "FULL-SCAN (auto: нет истории вне lookback)"
        df_1h_scan = df_1h.reset_index(drop=True)
        lookback_start_ms = int(df_1h_scan["ts"].iloc[0])
    else:
        scan_mode = "INCREMENTAL"
        df_1h_scan = df_1h.iloc[-n_total:].reset_index(drop=True)
    print(f"  scan mode: {scan_mode}", file=sys.stderr, flush=True)
    print(f"  scan window: {len(df_1h_scan)} bars, merge boundary ts={lookback_start_ms}",
          file=sys.stderr, flush=True)

    # ── H&S TOP ──
    print(f"\n── H&S TOP ──", file=sys.stderr, flush=True)
    hs_hits_new, hs_stats = hs_top.compute_hs(df_1h_scan, sym)
    hs_top.print_stats(hs_stats)
    hs_hits = _merge_hits("hs", sym, start, end, hs_hits_new, lookback_start_ms, start_ms, end_ms)
    print(f"  written: hs_hits_{sym}_{start}_{end}.parquet  ({len(hs_hits):,} rows, {len(hs_hits_new):,} в окне)",
          file=sys.stderr, flush=True)

    # ── SHORT-паттерны ──
    print(f"\n── SHORT patterns ──", file=sys.stderr, flush=True)
    for name, fn in SHORT_PATTERNS:
        print(f"\n  {name}", file=sys.stderr, flush=True)
        _run_one(name, fn, df_1h_scan, sym, start, end, start_ms, end_ms, lookback_start_ms)

    # ── LONG-паттерны (антагонисты) ──
    print(f"\n── LONG patterns (antagonists) ──", file=sys.stderr, flush=True)
    for name, fn in LONG_PATTERNS:
        print(f"\n  {name}", file=sys.stderr, flush=True)
        _run_one(name, fn, df_1h_scan, sym, start, end, start_ms, end_ms, lookback_start_ms)

    # ── Patterns basket (union всех паттернов) ──
    print(f"\n── Patterns basket ──", file=sys.stderr, flush=True)
    merged = patterns_basket.load_patterns_hits(sym, start, end)
    patterns_basket.print_stats(merged)
    basket_path = DATA_OUT / f"patterns_hits_{sym}_{start}_{end}.parquet"
    merged.to_parquet(basket_path, index=False, compression="zstd", compression_level=9)
    print(f"  written: {basket_path.name}  ({len(merged):,} rows)", file=sys.stderr, flush=True)

    print(f"\n═══ run_patterns done in {time.time()-t0:.1f}s ═══", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
