"""Rectangle Bottom (LONG) — зеркало rectangle_top (Bulkowski Ch.51).

Тонкая обёртка над five_point_core_long.compute_five_point_long.
Геометрия, фильтры, параметры — в five_point_core_long.py.

Reads:
  G:\\ASVK\\data\\{SYM}USDT_1m.csv
  G:\\ASVK\\data\\events\\events_e12d_{SYM}_*.parquet   (bullish FVG, direction=long)
  G:\\ASVK\\data\\trendline\\trendline_{SYM}_1h78_*.parquet
  G:\\ASVK\\data\\wma\\wma_{SYM}_*.parquet
Writes:
  G:\\ASVK\\data\\patterns\\rectangle_bottom_hits_{SYM}_{start}_{end}.parquet
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from common import load_1m, agg_1h, DATA_OUT
from five_point_core_long import compute_five_point_long, print_stats

PATTERN_NAME = "rectangle_bottom"
MODE = "rectangle_bottom"


def compute_rectangle_bottom(df_1h, symbol):
    return compute_five_point_long(MODE, PATTERN_NAME, df_1h, symbol)


def main() -> None:
    from datetime import datetime, timezone
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-07-26")
    args = ap.parse_args()

    print(f"{PATTERN_NAME}: {args.symbol} {args.start} → {args.end}", file=sys.stderr, flush=True)
    t0 = time.time()

    df_1h = agg_1h(load_1m(args.symbol))
    hits, stats = compute_rectangle_bottom(df_1h, args.symbol)
    print_stats(stats, PATTERN_NAME.upper())

    UTC = timezone.utc
    start_ms = int(datetime.fromisoformat(args.start).replace(tzinfo=UTC).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(args.end).replace(tzinfo=UTC).timestamp() * 1000)
    hits = hits[(hits["signal_ts"] >= start_ms) & (hits["signal_ts"] < end_ms)].reset_index(drop=True)
    print(f"  in window: {len(hits):,} signals", file=sys.stderr, flush=True)

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    out = DATA_OUT / f"{PATTERN_NAME}_hits_{args.symbol}_{args.start}_{args.end}.parquet"
    hits.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"written: {out.name}  ({len(hits):,} rows)  total {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
