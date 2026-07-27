"""patterns_basket — объединение сигналов всех 21 паттернов Block 5 в единый
список: H&S TOP + falling wedge (существовали раньше) + 11 SHORT-паттернов +
8 LONG-антагонистов (5 five-point: ascending_triangle, sym_triangle_long,
falling_wedge, falling_channel, rectangle_bottom; 3 bottom-family:
double_bottom, triple_bottom, hs_bottom).

Структура (не fractal12h/basket.py OR-маска над одним пулом кандидатов) —
каждый паттерн независим, разная геометрия и разное число сигналов,
объединяем просто как список разнородных событий (union list, не
OR-условие одного пивота). Общие поля — signal_ts, direction, pattern,
status; остальные (target/stop, X/A/B/C/D и т.п.) остаются каждый в своей
форме — TUI читает нужные поля напрямую из исходных *_hits parquet при
необходимости, здесь только сводный список.

Reads:
  G:\ASVK\data\patterns\{name}_hits_{SYM}_{start}_{end}.parquet
  для каждого name из PATTERN_FILE_NAMES ниже
Writes:
  G:\ASVK\data\patterns\patterns_hits_{SYM}_{start}_{end}.parquet
  columns: signal_ts, direction, pattern, status, symbol
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import time

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import DATA_OUT

COMMON_COLS = ["signal_ts", "direction", "pattern", "status"]

# Префиксы файлов *_hits_{symbol}_{start}_{end}.parquet — НЕ всегда совпадают
# с полем "pattern" внутри файла (hs_top.py исторически пишет в hs_hits_*,
# не hs_top_hits_*, чтобы не переименовывать уже существующий output).
PATTERN_FILE_NAMES = (
    # SHORT-паттерны
    "hs",
    "sym_triangle", "descending_triangle", "rising_wedge", "rising_channel", "rectangle_top",
    "triple_top",
    "gartley", "bat", "butterfly", "crab",
    # LONG группа 2 (five-point)
    "ascending_triangle", "sym_triangle_long", "falling_wedge", "falling_channel",
    # LONG группа 1 (гармоники)
    "gartley_long", "bat_long", "butterfly_long", "crab_long",
    # LONG группа 3 (bottom-family)
    "rectangle_bottom", "triple_bottom", "hs_bottom",
)


def load_patterns_hits(symbol: str, start: str, end: str) -> pd.DataFrame:
    frames = []
    for name in PATTERN_FILE_NAMES:
        path = DATA_OUT / f"{name}_hits_{symbol}_{start}_{end}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        df = pd.read_parquet(path)
        frames.append(df[COMMON_COLS])
    merged = pd.concat(frames, ignore_index=True)
    merged["symbol"] = symbol
    return merged.sort_values("signal_ts", ascending=False).reset_index(drop=True)


def print_stats(hits: pd.DataFrame) -> None:
    print(f"\n=== Patterns recap ({hits['symbol'].iloc[0] if len(hits) else '?'}) ===",
          file=sys.stderr, flush=True)
    for pattern in hits["pattern"].unique() if len(hits) else []:
        sub = hits[hits["pattern"] == pattern]
        counts = sub["status"].value_counts().to_dict()
        print(f"  {pattern:22s} n={len(sub):>4,}  {counts}", file=sys.stderr, flush=True)
    print(f"  total  n={len(hits):,}", file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-07-26")
    args = ap.parse_args()

    print(f"patterns_basket: {args.symbol} {args.start} → {args.end}", file=sys.stderr, flush=True)
    t0 = time.time()

    hits = load_patterns_hits(args.symbol, args.start, args.end)
    print_stats(hits)

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    out = DATA_OUT / f"patterns_hits_{args.symbol}_{args.start}_{args.end}.parquet"
    hits.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"written: {out.name}  ({len(hits):,} rows)  total {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
