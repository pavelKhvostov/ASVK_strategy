"""Symmetrical Triangle (BEARISH, SHORT) — Bulkowski Ch.66 "Triangles,
Symmetrical". Тонкая обёртка над five_point_core.py (общее ядро для всех 4
i1..i5-паттернов), см. канон-верифицированный
G:\\Claude\\patterns\\sym_triangle\\scan_triangle.py (101 уникальный паттерн,
BTCUSDT 1h 2020-01 → 2026-07).

Геометрия: опора (i1,i3,i5) растёт заметно (>0.05%/бар), сопротивление
(i2,i4) падает. TG (Table 66.10, "breakout price"): height=H2-L1,
TG=close[BR]-height.

Reads/Writes: см. block5_common.py / five_point_core.py.
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from common import load_1m, agg_1h, DATA_OUT
from five_point_core import compute_five_point, print_stats

PATTERN_NAME = "sym_triangle"


def compute_sym_triangle(df_1h, symbol: str):
    return compute_five_point('sym_triangle', PATTERN_NAME, df_1h, symbol)


def main() -> None:
    from datetime import datetime, timezone

    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-07-23")
    args = ap.parse_args()

    print(f"{PATTERN_NAME}: {args.symbol} {args.start} → {args.end}", file=sys.stderr, flush=True)
    t0 = time.time()

    df_1h = agg_1h(load_1m(args.symbol))
    hits, stats = compute_sym_triangle(df_1h, args.symbol)
    print_stats(stats, PATTERN_NAME.upper())

    UTC = timezone.utc
    start_ms = int(datetime.fromisoformat(args.start).replace(tzinfo=UTC).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(args.end).replace(tzinfo=UTC).timestamp() * 1000)
    hits = hits[(hits["signal_ts"] >= start_ms) & (hits["signal_ts"] < end_ms)].reset_index(drop=True)
    print(f"  in window: {len(hits):,} signals", file=sys.stderr, flush=True)

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    out = DATA_OUT / f"{PATTERN_NAME}_hits_{args.symbol}_{args.start}_{args.end}.parquet"
    hits.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"written: {out.name}  ({len(hits):,} rows)  total {time.time()-t0:.1f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
