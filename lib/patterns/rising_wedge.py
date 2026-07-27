"""Rising Wedge (SHORT через downward breakout) — Bulkowski Ch.74 "Wedges,
Rising". Тонкая обёртка над five_point_core.py, см. канон-верифицированный
G:\\Claude\\patterns\\rising_wedge\\scan_rising_wedge.py (187 уникальных
паттернов, BTCUSDT 1h 2020-01 → 2026-07).

ВАЖНО: не путать с существующим wedge.py (Falling Wedge, LONG/бычий,
scipy-геометрия, отдельный неродственный паттерн — НЕ трогать, НЕ заменять).

Геометрия: обе линии (опора i1,i3,i5 и сопротивление i2,i4) растут, опора
круче (sup_slope > res_slope > 0), narrow_frac = 1-gap(i5)/gap(i1) > 0.10
(иначе — почти канал, см. rising_channel.py). TG (Table 74.10, "target is
the bottom of the wedge"): TG=min(L1,L3,L5), без вычитания height.

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

PATTERN_NAME = "rising_wedge"


def compute_rising_wedge(df_1h, symbol: str):
    return compute_five_point('rising_wedge', PATTERN_NAME, df_1h, symbol)


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
    hits, stats = compute_rising_wedge(df_1h, args.symbol)
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
